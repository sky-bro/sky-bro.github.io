
## batching 问题 {#batching-problem}

batching 是 LLM serving 系统让 GPU 忙起来的基本手段。单个请求通常无法充分利用 GPU，但多个请求放在一起，就能把很多小矩阵运算变成更大的矩阵运算。问题是：请求不会同时结束。

在 continuous batching 之前，许多 serving 系统使用 **static batching**：先收集一批请求，一起跑模型，然后等这批请求全部生成结束，再接收下一批。

```text
time ------------------------------------------------------------>

Req A  [==Prefill==][D][D][D][D][D]  5 个 decode step 后完成
Req B  [==Prefill==][D][D][D][D][D][D][D][D][D][D]
Req C  [==Prefill==][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D]

Batch  |------------------ 必须等 Req C 结束 ------------------|
                     A 的 slot 空转          B 的 slot 空转
```

{{< figure src="/images/posts/continuous-batching/static-vs-continuous.svg" caption="<span class=\"figure-number\">Figure 1: </span>static batching（左）会在短请求结束后让 GPU slot 空转；continuous batching（右）在 slot 释放后立即插入新请求，从而维持高 GPU 利用率。" width="100%" >}}

短请求很早就释放了可用容量，但 static batching 不能在最长请求结束前复用这些容量。真实 workload 中，这会让 GPU 利用率停留在 **30-50%** 左右，即使队列里还有等待中的请求。

根因是调度粒度不匹配：

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
            scheduler.finish(req)      # 立即释放 KV blocks
        else:
            scheduler.continue_(req)   # 继续留在下一轮 iteration
```

当某个请求完成时，它占用的 [paged KV cache]({{< relref "paged-attention" >}}) blocks 会回到 free pool。等待中的新请求可以在下一次 forward 立刻进入。

```text
time ------------------------------------------------------------>

Req A  [Pre][D][D][D][D][Done]
Req B  [Pre][D][D][D][D][D][D][D][Done]
Req C            [Pre][D][D][D][Done]       A 完成后插入
Req D                     [Pre][D][D][D]    C 完成后插入

       | iter | iter | iter | iter |
       每轮 iteration：完成旧请求，接纳新请求
```

这改变了系统维持的 invariant。static batching 问的是：“当前 batch 是否全部结束？”continuous batching 问的是：“在当前显存和 token budget 下，下一轮 iteration 应该填入哪些工作？”

这个看似很小的变化正是利用率提升的原因：调度器不断补充 active set，而不是让已经完成的 slot 空着。

## 把 prefill 和 decode 打包在一起 {#packing-prefill-decode}

continuous batching 会带来一个不那么直观的问题。同一轮 iteration 里可能同时包含两类工作：

- **prefill**：新请求需要处理很多 prompt tokens，或者一个长 prompt 的某个 [chunk]({{< relref "chunked-prefill" >}})
- **decode**：已有请求只贡献一个刚生成的新 token

这些序列长度不同。如果把所有请求 padding 到最长长度，大部分计算都会浪费在 pad token 上。因此 serving 系统会把所有“本轮新 token”**pack** 成一个扁平序列。

假设某轮 iteration 包含：

- request A：3 个 prefill tokens
- request B：1 个 decode token
- request C：2 个 prefill tokens

模型可以构造一个 packed input：

$$X = [t_1^A, t_2^A, t_3^A, t_t^B, t_1^C, t_2^C]$$

投影矩阵只需要在这个 packed sequence 上运行一次：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

没有 pad token 进入矩阵乘法。

{{< alert theme="info" >}}

对于 prefill 请求，prompt tokens 都是“新”的，因为它们对应的 KV entries 还不存在。对于 decode 请求，只有最新生成的 token 是新的；历史 token 已经在 KV cache 里。

{{< /alert >}}

### 防止跨请求 attention {#block-diagonal-mask}

仅仅 packing 还不正确：request B 不能 attend 到 request A 的 prompt。解决办法是 **block-diagonal causal mask**。在每个请求自己的 block 内部允许 causal attention；不同请求之间的 attention 用负无穷屏蔽。

对 packed sequence 中的位置 `i` 和 `j`：

$$M_{ij}=0\ \text{when req}(i)=\text{req}(j)\ \text{and}\ j\le i,\quad M_{ij}=-\infty\ \text{otherwise}$$

{{< figure src="/images/posts/continuous-batching/packing-mask.svg" caption="<span class=\"figure-number\">Figure 2: </span>来自 A、B、C 三个请求的 tokens 被打包成一个扁平序列。block-diagonal causal mask 让每个请求只能 attend 到自己的前缀，跨请求位置在 softmax 后变成 0。" width="100%" >}}

这样得到的结果在数学上等价于为每个请求单独运行 attention，但它使用的是一次更大的 kernel launch 和一个 packed representation。

| 方法 | 矩阵运算形态 | 浪费 |
|---|---|---|
| 每个请求单独跑 | 很多小操作 | GPU occupancy 差 |
| padded batch | 一个带 padding 的大操作 | pad token 计算 |
| packed batch | 一个有效的大操作 | 很少 |

实际系统里，FlashAttention 类的 varlen 接口会接收 cumulative sequence lengths（`cu_seqlens`），并在 attention kernel 内部应用这种 mask。

### 与 KV cache 的关系 {#kv-cache-interaction}

packing 描述的是当前 iteration 中被处理的新 token。历史上下文来自 KV cache。

对于 prefill 请求，新算出的 keys 和 values 会写入 cache，并在 prompt chunk 内部做 causal attention。对于 decode 请求，新 query 会 attend 到缓存历史以及刚追加的新 key：

$$\operatorname{attn}_t^B=\operatorname{softmax}\left(q_t^B [K_{\text{cache}}^B; k_t^B]^T / \sqrt{d_k}\right)[V_{\text{cache}}^B; v_t^B]$$

缓存中的 keys 和 values 不会重新进入 packed input。attention kernel 会从 paged KV cache 中读取它们。

这就是关键的不对称性：prefill 主要是 matrix-matrix work；decode 则是一个 query 读取很长的 KV history。

## token budget 与延迟 {#token-budgets-latency}

continuous batching 通过维护 **token budget** 来让 GPU 保持忙碌：active set 占用的 KV cache 不能超过这个上限。

如果 GPU 显存是 \(M\)，模型权重占用 \(W\)，每个 cached token 需要 \(k\) bytes，那么粗略上限是：

$$N_{\max} = \frac{M-W}{k}$$

调度器试图维持：

$$\sum_{\text{active req}} L_{\text{req}} \approx N_{\max}$$

当一个请求完成并释放 \(\Delta N\) 个 token slots 时，调度器就接纳能放进这些 slots 的新工作。这也是 continuous batching 和 [paged attention]({{< relref "paged-attention" >}}) 天然配合的原因：显存可以按 block 粒度释放和复用，而不是被固定的连续分配绑住。

### 为什么 decode 是 bandwidth-bound {#decode-bandwidth}

decode 看起来很便宜，因为它每次只生成一个 token。但每个 decode step 都必须读取该请求完整的 KV history。对于一个有 \(L\) 层、\(n_h\) 个 KV heads、head dimension 为 \(d_h\) 的模型：

$$\text{bytes per token}=2 \times L \times n_h \times d_h \times \text{sizeof(dtype)}$$

以 LLaMA-3 8B 为例，\(L=32\)，\(n_h=8\) 个 GQA KV heads，\(d_h=128\)，BF16：

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

continuous batching 解决的是**什么时候**接纳新请求。它本身没有解决：新接纳的请求在一次 iteration 里允许带来**多少工作**。

当新请求有很长 prompt 时，这个问题会变得严重。一个 2048-token prefill 可能独占某轮 iteration 数百毫秒。期间已有 decode 请求都要等待，于是它们的 TPOT 会突然飙升。

这就是 **prefill-decode interference**：

- prefill 计算密集，喜欢大 chunk
- decode 对延迟敏感，希望 iteration 尽可能短且频繁
- continuous batching 把两者放在同一个调度循环中

下一步是 [chunked prefill]({{< relref "chunked-prefill" >}})：把长 prefill 切成多个 iteration，让 decode 请求持续前进。
