
## 干扰问题 {#interference}

[continuous batching]({{< relref "continuous-batching" >}}) 通过按迭代粒度调度请求，让 GPU 尽量保持忙碌。但它有一个很容易破坏延迟体验的边界情况：**很长的 prefill**。

当一个带有 2048-token prompt 的请求到达时，朴素调度器会在一次迭代里把整个 prompt 跑完 prefill。以 A100 上的 7B 模型为例，2048-token prefill 大约需要 200 ms。在这 200 ms 里，当前 batch 里已经在流式输出的 decode 请求都要等待。

```text
time ──────────────────────────────────────────────────────────────────►

iter 1:  [Req A prefill: 2048 tokens — 200 ms                         ]
         ←────────────────────────────────────────────────────────────→
         Req B, C, D (decode) are ALL blocked for 200 ms

iter 2:  [A dec][B dec][C dec][D dec]  ← 5 ms
iter 3:  [A dec][B dec][C dec][D dec]  ← 5 ms
...
```

{{< figure src="/images/posts/chunked-prefill/prefill-blocking.svg" caption="<span class=\"figure-number\">Figure 1: </span>没有 chunked prefill 时（左），一次 2048-token prefill 会阻塞所有 decode 请求约 200 ms，让 TPOT 瞬间放大 41 倍。使用 C = 512 的 chunked prefill 后（右），decode 每个迭代都能继续运行，额外开销很小。" width="100%" >}}

从 Req B、C、D 的视角看，它们的 **TPOT**（time per output token，每输出一个 token 的时间）从 5 ms 突然跳到 205 ms。用户看到的就是流式输出卡了一下：前面 token 持续出来，突然停顿，然后又继续。

这就是 **prefill-decode interference**：prefill 是 compute-bound 的 GEMM 工作，一旦长 prompt 占住 GPU，就会饿死对延迟敏感的 decode GEMV。

两个指标天然拉扯：

| 优化方向 | TTFT（time to first token） | TPOT（time per output token） |
|---|---|---|
| 大 prefill，一次跑完 | ↓ 低：KV cache 很快准备好 | ↑ 高：decode 被阻塞 |
| 延后 prefill，优先 decode | ↑ 高：新请求等待更久 | ↓ 低：decode 不受影响 |

看起来不能同时最小化两者，除非我们把 prefill **切片**。

## chunked prefill 的核心思想 {#core-idea}

**chunked prefill** 把一个长 prompt 拆成大小为 \\(C\\) 的片段（chunk size），每个调度迭代只处理一个片段，并和 decode step 交错执行：

```text
without chunked prefill (C = 2048, full prompt):

  iter 1: [A prefill: 2048 tokens]
  iter 2: [A dec][B dec][C dec]
  iter 3: [A dec][B dec][C dec]

with chunked prefill (C = 512):

  iter 1: [A: tokens    0–511] [B dec][C dec]
  iter 2: [A: tokens  512–1023][B dec][C dec]
  iter 3: [A: tokens 1023–1535][B dec][C dec]
  iter 4: [A: tokens 1536–2047][B dec][C dec]
  iter 5: [A dec]              [B dec][C dec]  ← A now in decode
```

每个迭代有一个固定 **token budget** \\(T\\)：

$$
T = C\_{\text{prefill}} + N\_{\text{decode}}
$$

其中 \\(C_{\text{prefill}}\\) 是这个迭代处理的 prefill token 数，\\(N_{\text{decode}}\\) 是当前正在 decode 的请求数。调度器保证 \\(C_{\text{prefill}} + N_{\text{decode}} \leq T\\)。

decode 请求因此每个迭代都能继续前进。它们的 TPOT 大致变成：

$$
\text{TPOT} \approx \frac{\text{compute}(C\_{\text{prefill}} + N\_{\text{decode}})}{\text{compute}(N\_{\text{decode}})} \times \Delta t\_{\text{decode}}
$$

当 \\(C = 512\\)、\\(N_{\text{decode}} = 32\\) 时，一个 prefill chunk 带来的 TPOT 扰动很小：512 个 token 的 GEMM 远比完整 2048-token prefill 短，也就不会制造一次 200 ms 的长停顿。

## 正确性与成本模型 {#correctness-and-cost}

### 为什么切片不会改变结果 {#correctness}

**把 prefill 切开会改变模型输出吗？** 不会。chunked prefill 和一次性完整 prefill 在数学上等价。

原因来自 decoder-only Transformer 的 causal attention：位置 \\(i\\) 只能看见 \\(j \leq i\\) 的位置。

假设 prompt 是 \\([t_1, t_2, \ldots, t_L]\\)，按大小 \\(C\\) 切成多个 chunk。第 \\(s\\) 个 chunk 处理 token \\([(s-1)C+1, \ldots, sC]\\)。

对第 \\(s\\) 个 chunk 中的任意 token \\(t_i\\)：
- \\(j < (s-1)C + 1\\) 的 token 来自之前的 chunk，它们的 \\(k_j, v_j\\) 已经在前面的迭代中计算并写入 KV cache
- \\(j \in [(s-1)C+1, i]\\) 的 token 位于当前 chunk，它们的 \\(k_j, v_j\\) 在当前迭代中计算

所以 \\(t_i\\) 的 attention 可以拆成两部分：

$$
\begin{aligned}
\text{attn}\_{i} = \text{softmax\_merge}\Bigl(
  &\underbrace{\frac{q\_i \cdot K\_{\text{cache}}^T}{\sqrt{d\_k}}}\_{\text{attend to prior chunks}},\;
  \underbrace{\frac{q\_i \cdot K\_{\text{chunk}}^T}{\sqrt{d\_k}}}\_{\text{attend within current chunk}}
\Bigr) \cdot \begin{bmatrix} V\_{\text{cache}} \\ V\_{\text{chunk}} \end{bmatrix}
\end{aligned}
$$

这里的 `softmax_merge` 就是 online softmax merge，和 [paged attention]({{< relref "paged-attention" >}}) 中按 block 聚合 attention 的技巧是同一类思想。FlashAttention 的 `flash_attn_varlen_func` 原生支持这种形态：`cu_seqlens` 告诉 kernel 每个 token 的有效上下文长度，也就是历史 cache 加当前 chunk。

每个 chunk 结束后，新算出的 \\(k, v\\) 向量写入 KV cache：

$$
K\_{\text{cache}} \mathrel{+}= [k\_{(s-1)C+1}, \ldots, k\_{sC}]
$$

下一个 chunk 会看到扩展后的 cache。按归纳法，跑完全部 \\(\lceil L/C \rceil\\) 个 chunk 后，KV cache 的内容和一次性 prefill 完全相同；后续 decode 无法区分这两种执行方式。

### TTFT/TPOT 取舍与 chunk size 选择 {#tradeoff}

chunk size \\(C\\) 是最关键的调节旋钮：

$$
\text{TTFT} \approx \left\lceil \frac{L\_{\text{prompt}}}{C} \right\rceil \times \Delta t\_{\text{iter}}
$$

$$
\text{TPOT jitter} \propto \frac{C}{N\_{\text{decode}}} \times \frac{\text{FLOP}\_{\text{GEMM}}}{\text{FLOP}\_{\text{GEMV}}}
$$

- **更大的 \\(C\\)**：prefill 需要的迭代数更少，TTFT 更低；但每个 prefill chunk 更大，对 decode 的单次干扰更强，TPOT jitter 更高。
- **更小的 \\(C\\)**：decode 几乎不被干扰，TPOT 更稳定；但 prefill 被拆成更多迭代，TTFT 会升高。

甜点区间取决于 active decode 请求数和 prefill token 数之间的比例。常见生产默认值大致如下：

| engine | 默认 chunk size |
|---|---|
| vLLM (v0.4+) | 512 tokens |
| SGLang | 512 tokens |
| TensorRT-LLM | 1024 tokens |

以 \\(C = 512\\)、2048-token prompt 为例：prefill 需要 4 个迭代完成，每个迭代只给 decode step 额外加 512 个 token 的 GEMM 工作。相比一次性 full prefill，TTFT 只多出大约 \\(3 \times 5\text{ ms} = 15\text{ ms}\\)，对多数在线服务来说很容易接受。

### FLOPs 分析：切片不增加计算量 {#flops}

一个重要的 sanity check 是：chunking 会不会增加 FLOPs？答案是不会。

对单层 Transformer，causal attention 会访问 \\(L(L+1)/2\\) 个 query-key pair。把 \\(QK^T\\) 和 \\(\text{attn} \cdot V\\) 都算进去，attention FLOPs 可以写成：

$$
\text{FLOP}\_{\text{attn}}(L) \approx 4d \cdot \frac{L(L+1)}{2} + 4Ld^2
$$

第一项来自 causal attention pair；\\(4Ld^2\\) 来自四个投影矩阵。

**不切片**：一次调用处理 \\(L\\) 个 token。

**切片后**：调用 \\(\lceil L/C \rceil\\) 次，每次处理新的 prompt token，并 attention 到不断增长的 KV cache。所有 chunk 的总 attention FLOPs 是：

$$
\begin{aligned}
\text{FLOP}\_{\text{chunk-attn}}
&= 4Ld^2 + 4d \sum\_{i=1}^{L} i \\\\
&= 4Ld^2 + 4d \cdot \frac{L(L+1)}{2}
\end{aligned}
$$

这和不切片的情况相同。**chunking 只是把相同的 FLOPs 分散到更多迭代里，并没有制造额外计算。**

### IO 开销：实践中可以忽略 {#io-overhead}

真正额外需要关注的是每个 chunk 结束时，把新的 KV 向量写入 HBM。对 chunk size \\(C\\)、KV head 数 \\(n_h\\)、head dim \\(d_h\\)、层数 \\(L_{\text{layers}}\\)、BF16 存储：

$$
\text{write per chunk} = C \times 2 \times L\_{\text{layers}} \times n\_h \times d\_h \times 2 \text{ bytes}
$$

以 LLaMA-3 8B 为例（\\(L = 32, n_h = 8, d_h = 128\\)），当 \\(C = 512\\) 时：

$$
512 \times 2 \times 32 \times 8 \times 128 \times 2 = 67{,}108{,}864 \text{ bytes} \approx 64 \text{ MB}
$$

A100 的 HBM 带宽约 2 TB/s：

$$
\frac{64 \times 10^6}{2 \times 10^{12}} = 32 \text{ μs}
$$

32 微秒，相比约 5 ms 的迭代时间不到 1%。这部分 IO 开销在实践里通常可以忽略。

## 它如何融入推理服务栈 {#serving-stack}

### 与 prefix caching 的关系 {#prefix-cache-interaction}

chunked prefill 和 [prefix caching]({{< relref "prefix-caching" >}}) 可以自然组合。如果 prompt 的前 \\(k\\) 个 block 已经命中缓存，这些 block 可以完全跳过：

```text
prompt: [system prompt — 1024 tokens][user query — 1024 tokens]
             (cached — skip)              (must compute)

with prefix cache + chunked prefill (C = 512):
  iter 1: [user query tokens   0–511]   ← only 2 chunks instead of 4
  iter 2: [user query tokens 512–1023]
  iter 3: [decode]
```

cache hit 之后，有效 prefill 长度只剩下**未命中的后缀**。这会进一步降低 TTFT，也会减少 prefill 占用的调度迭代数。

### 调度器实现 {#implementation}

SGLang 里的调度逻辑大致可以抽象成：

```python
def schedule(self):
    budget = self.token_budget  # e.g., 2048 tokens

    # 1. running decode requests each consume 1 token
    for req in self.running:
        budget -= 1

    # 2. prefill requests consume up to chunk_size tokens
    for req in self.waiting:
        chunk = min(req.remaining_prefill, budget, self.chunk_size)
        if chunk == 0:
            break
        req.prefill_this_iter = chunk
        budget -= chunk

    return self.running + [r for r in self.waiting if r.prefill_this_iter > 0]
```

关键性质是：`prefill_this_iter` 可以小于 `remaining_prefill`，也就是允许一次 prefill 只完成一部分。下一轮调度再从上次停止的位置继续。

### 与 disaggregated prefill 的对比 {#vs-disaggregated}

chunked prefill 是解决 prefill-decode interference 的**原地方案**：prefill 和 decode 仍然共享同一张 GPU，只是调度器更细粒度地交错它们。

**disaggregated prefill** 更激进：把 prefill 路由到单独机器，decode GPU 完全看不到 prefill 流量。

| 维度 | chunked prefill | disaggregated prefill |
|---|---|---|
| 硬件要求 | 单 GPU / 单节点即可 | 需要独立的 prefill 池和 decode 池 |
| TPOT | 显著改善 | 最优，零干扰 |
| TTFT | 略微升高，chunking 多了迭代 | 通常更好，prefill 有专用资源 |
| 网络开销 | 无 | 需要跨节点迁移 KV cache |
| 实现复杂度 | 低，主要改调度器 | 高，需要集群协调 |
| 适用场景 | 通用生产 serving | 大规模、SLO 严格的部署 |

关于 disaggregated prefill 的完整讨论，会放在[下一篇文章]({{< relref "disaggregated-prefill" >}})里。

## 总结 {#summary}

chunked prefill 是 LLM serving 里性价比很高的优化：

- **零 FLOPs 开销**：chunking 分散的是同一份工作，不是增加工作
- **IO 开销可以忽略**：每个 chunk 约 32 μs 的 KV 写入，相比 5 ms 迭代时间很小
- **实现直接**：主要改变调度器，attention kernel 不需要重写
- **TPOT 改善明显**：decode 请求不再被长 prefill 整段阻塞
- **可组合**：可以和 prefix caching（跳过已缓存 chunk）、paged attention（按 block 写入 KV）、continuous batching（同一个迭代级调度循环）自然配合

唯一代价是 TTFT 会随着 chunk 数略微增加。在常见的 \\(C = 512\\)、几千 token prompt 场景里，这通常只是十几毫秒级别的代价，换来的是稳定得多的流式输出延迟。
