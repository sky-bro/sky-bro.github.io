
## batching 问题 {#batching-problem}

batching 是 LLM serving 系统让 GPU 忙起来的基本手段。单个请求通常无法充分利用 GPU，多个请求一起跑，才能把许多小矩阵运算变成更大的矩阵运算。问题是：请求不会同时结束。

在 continuous batching 之前，许多系统使用 **static batching**：先收集一批请求，一起跑模型，然后等这批请求全部生成结束，再接收下一批。

```text
time ------------------------------------------------------------>

Req A  [==Prefill==][D][D][D][D][D]  5 个 decode step 后完成
Req B  [==Prefill==][D][D][D][D][D][D][D][D][D][D]
Req C  [==Prefill==][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D]

Batch  |------------------ 必须等 Req C 结束 ------------------|
                     A 的 slot 空转          B 的 slot 空转
```

{{< figure src="/images/posts/continuous-batching/static-vs-continuous.svg" caption="<span class=\"figure-number\">Figure 1: </span>static batching（左）会在短请求结束后让 GPU slot 空转；continuous batching（右）在 slot 释放后立即插入新请求，从而维持高 GPU 利用率。" width="100%" >}}

短请求很早就释放了可用容量，但 static batching 不能在最长请求结束前复用这些容量。根因是调度粒度不匹配：

- static batching 在 **batch** 粒度调度
- 自回归推理天然按 **iteration** 前进
- 每个 decode iteration 为每个活跃请求生成一个新 token

所以调度器真正应该做决策的时机，不是一整批请求结束之后，而是每一次 iteration 结束之后。

## iteration-level scheduling {#iteration-level-scheduling}

**continuous batching** 也叫 *iteration-level scheduling* 或 *in-flight batching*。Orca 论文把这个思路系统化：每次 forward 之后，立刻移除已经完成的请求，并把等待队列里的新请求填进释放出来的 slot。

核心循环可以写成：

```python
while True:
    batch = scheduler.schedule()       # 为本轮 iteration 选择活跃请求
    outputs = model.forward(batch)

    for req, token in zip(batch, outputs):
        req.append(token)
        if token.is_eos or req.at_max_len:
            scheduler.finish(req)      # 释放 KV blocks
        else:
            scheduler.continue_(req)   # 留到下一轮 iteration
```

请求结束、取消或达到最大长度后，它占用的 [paged KV cache]({{< relref "paged-attention" >}}) blocks 才会回到 free pool。还在生成中的请求不会释放历史 KV，因为每个 decode step 都要继续读取这些 cache。

```text
time ------------------------------------------------------------>

Req A  [Pre][D][D][D][D][Done]
Req B  [Pre][D][D][D][D][D][D][D][Done]
Req C            [Pre][D][D][D][Done]       A 完成后插入
Req D                     [Pre][D][D][D]    C 完成后插入

       | iter | iter | iter | iter |
       每轮 iteration：完成旧请求，接纳新请求
```

static batching 问：“当前 batch 是否全部结束？”continuous batching 问：“在当前显存和 token budget 下，下一轮 iteration 应该填入哪些工作？”这个变化让调度器不断补充 active set，而不是让已经完成的 slot 空着。

## 把 prefill 和 decode 打包在一起 {#packing-prefill-decode}

continuous batching 的本质是 **每轮 forward 后重新调度 active set**。同一轮 iteration 里可能同时包含：

- **prefill**：新请求的一段 prompt，可能是完整 prompt，也可能是 [chunked prefill]({{< relref "chunked-prefill" >}}) 的一个 chunk
- **decode**：已有请求刚生成、准备用来预测下一个 token 的一个 token

采用 prefill/decode disaggregation（PD 分离）时，prefill worker 和 decode worker 可以各自维护 continuous batching 队列。本文只讨论最能暴露机制的一种情况：同一个 forward 同时打包 prefill rows 和 decode rows。

假设某轮 iteration 包含：

- request A：3 个 prefill tokens
- request B：1 个 decode token
- request C：2 个 prefill tokens

系统会把本轮要计算的新 token pack 成一个扁平输入：

$$X = [t_1^A, t_2^A, t_3^A, t_t^B, t_1^C, t_2^C]$$

然后一次性做线性投影：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

这里的“扁平”只描述物理计算布局，不表示这些 token 变成了一条长文本。每一行仍然带着自己的 request id、position id 和序列边界：

| packed index | token | request | position id |
|---:|---|---|---:|
| 0 | `t1(A)` | A | 1 |
| 1 | `t2(A)` | A | 2 |
| 2 | `t3(A)` | A | 3 |
| 3 | `t(B,t)` | B | `t` |
| 4 | `t1(C)` | C | 1 |
| 5 | `t2(C)` | C | 2 |

RoPE 或 learned positional embedding 看到的是每个请求自己的 position id。A 的第三个 token 在内存里紧挨着 B 的 token，但语义上仍然不是同一条序列。

{{< figure src="/images/posts/continuous-batching/mixed-forward-flow.svg" caption="<span class=\"figure-number\">Figure 2: </span>一轮 mixed iteration 的核心路径：调度器选择本轮新 token，线性层把它们作为 packed matrix 计算；attention 再用 request 边界、position id、mask 和 KV cache 恢复每个请求自己的上下文。" width="100%" >}}

### 防止跨请求 attention {#block-diagonal-mask}

packing 本身还不够。request B 不能 attend 到 request A 的 prompt。解决办法是 **block-diagonal causal mask**：每个请求内部允许 causal attention，不同请求之间全部屏蔽。

对 packed sequence 中的位置 `i` 和 `j`：

$$M_{ij}=0\ \text{when req}(i)=\text{req}(j)\ \text{and}\ j\le i,\quad M_{ij}=-\infty\ \text{otherwise}$$

{{< figure src="/images/posts/continuous-batching/packing-mask.svg" caption="<span class=\"figure-number\">Figure 3: </span>来自 A、B、C 三个请求的 tokens 被打包成一个扁平序列。block-diagonal causal mask 让每个请求只能 attend 到自己的前缀，跨请求位置在 softmax 后变成 0。" width="100%" >}}

这样得到的结果等价于为每个请求单独运行 attention，但它使用的是一次更大的 kernel launch 和一个 packed representation。实际系统里，FlashAttention 类的 varlen 接口会接收 cumulative sequence lengths（`cu_seqlens`），并在 attention kernel 内部应用这种边界。

### 与 KV cache 的关系 {#kv-cache-interaction}

packing 描述的是本轮要计算的新 token；历史上下文来自 KV cache。关键是分清两层：

- **线性层**只关心本轮 packed rows，所以多个 prefill rows、多个 decode rows 可以组成同一个 `X`。
- **attention 层**按 request id、position id、sequence boundary、causal mask 和 KV block table，为每一行 query 找到它自己的可见 keys。

{{< figure src="/images/posts/continuous-batching/packed-forward-kv-lifecycle.svg" caption="<span class=\"figure-number\">Figure 4: </span>一次 packed forward 的完整路径：调度器选择本轮新 token，线性层按 packed rows 做共享 matmul；attention kernel 再按每行的 request 边界读取对应 KV cache、写入本轮新 KV，并只在需要的位置采样 logits。" width="100%" >}}

对某个 prefill chunk 来说，本轮 chunk 里的 prompt tokens 是新的，因为它们对应的 KV entries 还没有写入 cache。如果启用 chunked prefill，后续 chunk 会读取前面 chunks 已经写好的 KV，同时追加本轮新 KV。

对 decode 请求来说，通常只有一行 query。以 request B 为例：

$$\operatorname{Attn}^{B,t}=\operatorname{softmax}\left(q^{B,t}\left[K^{B,\mathrm{cache}};k^{B,t}\right]^{T}/\sqrt{d^{\mathrm{k}}}\right)\left[V^{B,\mathrm{cache}};v^{B,t}\right]$$

这个公式只是从 B 的一行 query 视角看问题，不表示系统单独为 B 跑了一个小 kernel。真实实现里，多个 decode rows 和 prefill rows 仍然在同一个 packed batch 中；attention kernel 对每一行分别查自己的 KV block table。

prefill 结束时也会产生第一个输出 token。decoder-only transformer 会为 prompt 中每个位置都算 hidden state 和 logits，但 serving 系统一般只用最后一个 prompt token 的 logits 采样：

$$\text{first output token}=\operatorname{sample}\left(\operatorname{logits}(h^{\mathrm{last\ prompt}})\right)$$

如果 prompt 被切成多个 chunks，只有最后一个 chunk 结束时才会采样第一个输出 token；前面的 chunks 只是逐步填充 KV cache。

### 一个完整的 mixed iteration {#mixed-iteration-example}

用一个小例子串起来。假设 hidden size 是 `d`，本轮调度器选中：

- request A：新进来的 prompt chunk，有 3 个 tokens，还没有 KV cache
- request B：prompt 长度是 4，prefill 已经采样出第一个输出 token `B5`；本轮用 `B5` 做 decode，预测 `B6`
- request C：新进来的 prompt chunk，有 2 个 tokens，还没有 KV cache

本轮新 token 总数是 `3 + 1 + 2 = 6`，embedding 后得到：

$$X\in\mathbb{R}^{6\times d},\quad Q,K,V\in\mathbb{R}^{6\times d}$$

attention 按 request 分三块：

| request | 本轮新 token | position id | attention 可见范围 |
|---|---:|---|---|
| A | 3 | 1, 2, 3 | A 的 3 个 prompt tokens，causal |
| B | 1 | 5 | B 的 prompt cache `B1..B4` + 本轮 token `B5`，用来预测 `B6` |
| C | 2 | 1, 2 | C 的 2 个 prompt tokens，causal |

这轮 forward 可以压缩成一条链：

```text
scheduler
  -> pick A prefill rows, B5 decode row, C prefill rows
  -> pack rows into X
  -> shared projection: Q, K, V = XW
  -> attention:
       A rows see A chunk/cache only
       B5 row sees KV_B1..B4_cache + B5 current key, then predicts B6
       C rows see C chunk/cache only
  -> write current K/V into each request's KV cache
  -> sample logits from decode rows and final-prefill rows
```

B 的 query 不会看见 A/C 的 token。position id `5` 表示：`B1..B4` 已经在 cache 里，`B5` 是本轮 decode 输入；attention 看到 `B1..B5` 后输出 hidden state，再用 logits 预测 `B6`。

### 为什么 packed forward 等价 {#decoder-only-correctness}

正确性只依赖一个条件：

> packed row `i` 的可见集合，必须等于这个 request 单独运行时的可见集合。

线性层逐行独立：

$$q^{i}=x^{i}W^{Q},\quad k^{i}=x^{i}W^{K},\quad v^{i}=x^{i}W^{V}$$

attention 对 row `i` 的可见集合是：

$$S(i)=\lbrace\text{same request, earlier-or-current positions}\rbrace\cup\lbrace\text{that request's KV cache}\rbrace$$

如果 request id 不同，`j` 根本不在 `S(i)` 里；实现上等价于 mask 里给跨请求位置加负无穷。MLP、residual、layer norm 都是 row-wise 操作，也不会引入跨请求混合。

所以只要 position id、sequence boundary、causal mask 和 KV cache 索引正确，packed forward 的结果就等于把每个请求单独 forward 后再拼起来。共享的是物理计算形态，不共享语义上下文。

## token budget 与延迟 {#token-budgets-latency}

continuous batching 通过 **token budget** 控制 active set 占用的 KV cache。假设 GPU 显存是 `M`，模型权重占用 `W`，每个 cached token 需要 `k` bytes，粗略上限是：

$$N_{\max} = \frac{M-W}{k}$$

调度器试图维持：

$$\sum_{\text{active req}} L_{\text{req}} \approx N_{\max}$$

当一个请求释放 `ΔN` 个 token slots 时，调度器就接纳能放进这些 slots 的新工作。这也是 continuous batching 和 [paged attention]({{< relref "paged-attention" >}}) 天然配合的原因：显存可以按 block 粒度释放和复用。

注意两个粒度不同：

- **prompt chunk** 是调度粒度：本轮允许一个 prefill 请求带进来多少个 prompt tokens。
- **paged KV block** 是显存分配粒度：这些 tokens 的 KV 按固定大小 block 写入 cache，直到请求结束、取消或达到上限才释放。

所以，prefill 和 decode 被 pack 到一起，指的是**本轮被计算的新 token**被打包；它不要求 prefill 一定包含完整 prompt，也不等于一次性分配完整未来输出的 KV。

### 为什么 decode 是 bandwidth-bound {#decode-bandwidth}

decode 每次只生成一个 token，但每个 decode step 都必须读取该请求完整的 KV history。对于一个有 `L` 层、`n_h` 个 KV heads、head dimension 为 `d_h` 的模型：

$$\text{bytes per token}=2 \times L \times n_h \times d_h \times \text{sizeof(dtype)}$$

以 LLaMA-3 8B 为例，`L=32`，`n_h=8` 个 GQA KV heads，`d_h=128`，BF16：

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB per token}$$

4096-token context 大约需要 512 MB KV 数据。每个 decode step 都要从 HBM 流式读取这些 cache，所以瓶颈往往是内存带宽，而不是 tensor-core compute。

### TTFT 与 TPOT {#ttft-tpot}

两个延迟指标最重要：

| 指标 | 含义 |
|---|---|
| **TTFT** | time to first token |
| **TPOT** | time per output token |

continuous batching 主要改善吞吐和利用率。当 token budget 已满时，新请求可能需要排队，所以 TTFT 可能略有增加。单个孤立请求的 TPOT 不一定明显变化，但整体 TPOT 会改善，因为 GPU slot 很少空转。

## 下一个瓶颈 {#next-bottleneck}

continuous batching 解决的是**什么时候**接纳新请求。它没有解决：新接纳的请求在一次 iteration 里允许带来**多少工作**。

当新请求有很长 prompt 时，一个 2048-token prefill 可能独占某轮 iteration 数百毫秒。期间已有 decode 请求都要等待，于是 TPOT 会突然飙升。

这就是 **prefill-decode interference**：

- prefill 计算密集，喜欢大 chunk
- decode 对延迟敏感，希望 iteration 尽可能短且频繁
- continuous batching 把两者放在同一个调度循环中

下一步是 [chunked prefill]({{< relref "chunked-prefill" >}})：把长 prefill 切成多个 iteration，让 decode 请求持续前进。
