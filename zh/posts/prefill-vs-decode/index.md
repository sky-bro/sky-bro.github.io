
LLM 推理表面上像一个操作：输入 prompt，然后不断输出 token。底层其实是两个 workload 在共用同一套模型权重。

**prefill** 负责处理输入 prompt，并构建初始 KV cache。**decode** 负责逐 token 生成，每一步读取已经存在的 KV cache，再追加新 token 的 KV。权重是同一套，但硬件瓶颈完全不同：prefill 更像大批量矩阵乘法；decode 更像很多小 query 反复读取一张不断增长的内存表。

这个区分解释了为什么 serving engine 会把 **TTFT**（time to first token）和 **TPOT**（time per output token）分开看，为什么 [continuous batching]({{< relref "continuous-batching" >}}) 要把 prefill 和 decode 混在同一个 iteration 里，为什么需要 [chunked prefill]({{< relref "chunked-prefill" >}})，以及为什么 [disaggregated prefill]({{< relref "disaggregated-prefill" >}}) 会成为大规模生产系统里的自然架构。

{{< figure src="/images/posts/prefill-vs-decode/two-bottlenecks.svg" caption="<span class=\"figure-number\">Figure 1: </span>prefill 把很多 prompt tokens 一次性送进大 GEMM，主要影响 TTFT；decode 每个请求每次只前进一个 token，需要反复读取 KV cache，主要影响 TPOT。" width="100%" >}}

## 最小例子 {#small-example}

先看一个很小的 decoder-only Transformer 请求：

- prompt 长度：4 个 token，`A B C D`
- 输出长度：3 个 token，`x y z`
- 只有一层 attention、一个 KV head、很小的 hidden size

这个请求会经历两个阶段：

```text
prefill:
  input:  A B C D
  output: D 位置的 logits，以及 A B C D 的 KV cache

decode step 1:
  input:  x
  read:   KV(A B C D)
  output: x 的 logits，追加 KV(x)

decode step 2:
  input:  y
  read:   KV(A B C D x)
  output: y 的 logits，追加 KV(y)

decode step 3:
  input:  z
  read:   KV(A B C D x y)
  output: z 的 logits，追加 KV(z)
```

这里最重要的不对称，不是“prefill 用 prompt，decode 用生成文本”。真正重要的是：**一次 forward 里有多少新 token 进入模型**。

prefill 阶段，prompt 里的所有 token 都是新 token。如果 prompt 长度是 \\(S\\)，hidden state 矩阵形状是：

$$
X_{\text{prefill}} \in \mathbb{R}^{S \times d}
$$

线性层因此是矩阵乘矩阵：

$$
Y = X_{\text{prefill}} W,\quad
X_{\text{prefill}} \in \mathbb{R}^{S \times d},\ W \in \mathbb{R}^{d \times d}
$$

decode 阶段，每个请求每一步只贡献一个新 token：

$$
x_{\text{decode}} \in \mathbb{R}^{1 \times d}
$$

同一个线性层变成类似矩阵乘向量：

$$
y = x_{\text{decode}} W
$$

如果有 batching，decode 当然不一定真的是单个向量，而是 \\(B\\) 个活跃请求各贡献 1 行 token。但它在 sequence 维度上仍然远小于 prefill。正是这个形状差异改变了 GPU 瓶颈。

## 为什么 prefill 更吃算力 {#prefill-compute}

prefill 一次有很多 token 可以并行处理，所以 GPU 看到的是大而密的矩阵：

```text
prompt tokens S = 2048
hidden size   d = 4096

X: 2048 x 4096
W: 4096 x 4096
Y: 2048 x 4096
```

权重矩阵 \\(W\\) 被加载之后，可以在很多 prompt-token 行上复用。Tensor Core 喜欢这种大 GEMM，因为计算量足够大，可以摊薄内存移动和调度开销。

attention score 在 prefill 阶段也是类似的。虽然 causal mask 限制位置 \\(i\\) 只能看 \\(j \leq i\\) 的历史位置，但一次 forward 里仍然有很多 query 和很多 key：

$$
Q_{\text{prefill}} K_{\text{prefill}}^\top
\in \mathbb{R}^{S \times S}
$$

所以 prefill 经常被说成 **compute-bound**。瓶颈更接近 GPU 做大矩阵乘法的速度。prompt 很长时，prefill 当然很贵；但它也是最容易通过 batch 和 Tensor Core 吃满 GPU 的阶段。

从业务指标看，prefill 主要控制 **TTFT**：

```text
request arrives
  -> wait in queue
  -> run prefill
  -> sample first token
  -> user sees first token
```

如果 prefill 慢，用户会更久看不到第一个 token。prefix caching、prompt 压缩、chunked prefill 调参、disaggregated prefill 都会影响这条路径。

## 为什么 decode 更吃显存带宽 {#decode-bandwidth}

decode 的形状相反。每个活跃请求只增加一个新 query，但这个 query 要看完整历史。

对单个请求，假设当前上下文长度是 \\(T\\)：

$$
q_{\text{new}} K_{\leq T}^{\top}
\in \mathbb{R}^{1 \times T}
$$

query 很小，KV cache 不小。每一层都需要读取这个请求历史 token 的 keys 和 values。假设有 \\(L\\) 层、\\(n_{kv}\\) 个 KV heads、head dimension 是 \\(d_h\\)、dtype 是 \\(b\\) bytes、上下文长度是 \\(T\\)，单请求 KV cache 近似为：

$$
\text{KV bytes} = 2 \times L \times n_{kv} \times d_h \times T \times b
$$

其中 2 来自 keys 和 values。对一个 32 层、32 个 KV heads、head dimension 128、FP16 cache、4096-token context 的模型：

$$
2 \times 32 \times 32 \times 128 \times 4096 \times 2
\approx 2\ \text{GB}
$$

实际 kernel 和 cache 层级会影响每一步从 HBM 读多少，但这个数量级说明了问题：decode 大量时间花在读取历史 KV 数据上。上下文越长、并发越高，需要读和保存的 KV 越多。

所以 decode 经常是 **memory-bandwidth-bound**。GPU 可能还有 FLOPs 没用满，但下一个 token 必须等相关 KV cache 数据到位之后才能生成。

从业务指标看，decode 主要控制 **TPOT**：

```text
after first token:
  decode step
  sample token
  decode step
  sample token
  ...
```

如果 decode 慢，流式输出就会变慢。如果 decode 有抖动，用户会看到输出时快时慢，甚至突然停顿。

## 同一个模型想要两种 batch 形状 {#batch-shapes}

prefill 和 decode 都能从 batching 受益，但受益方式不一样。

| 维度 | prefill | decode |
|---|---|---|
| 每个请求的新 token 数 | 很多 prompt tokens | 一个 generated token |
| 主要矩阵形状 | 大 GEMM | 小 GEMM / 类 GEMV |
| attention 输入 | 很多 query 看 prompt | 一个 query 看长历史 |
| 常见瓶颈 | Tensor Core 算力 | HBM 带宽和 KV cache layout |
| 用户指标 | TTFT | TPOT |
| serving 压力 | 尽快接纳新请求 | 让活跃流稳定输出 |

这会制造 scheduler 冲突。长 prefill 希望一次跑大 chunk，因为这样计算效率最高，新请求 TTFT 也最低。正在 decode 的请求希望 iteration 短且频繁，因为每个用户都在等下一个流式 token。

如果 scheduler 一次性跑 2048-token prefill，这对新请求很高效，但所有已经在 decode 的请求都要等，TPOT 会突然升高。如果 scheduler 永远优先 decode，已有流式输出会很平滑，但新请求会更久拿不到第一个 token。

这就是基本的 **prefill-decode interference**：

```text
large prefill:
  Tensor Core 利用率好
  新请求 TTFT 好
  已有 decode 请求 TPOT jitter 差

decode-first scheduling:
  流式输出平滑
  新请求排队更久，TTFT 更差
```

因此没有脱离 workload 的“最佳 batch size”。短 prompt、长输出的聊天 workload 偏 decode-heavy；长 prompt、短回答的 RAG 或 agent workload 偏 prefill-heavy；代码助手如果有很长的共享 system prompt，只有在 prefix caching 命中时才会变得不那么 prefill-heavy。

## 对推理引擎意味着什么 {#serving-engines}

一旦把两个瓶颈分开看，很多 serving engine 特性就容易定位。

**KV cache** 是两个阶段之间的边界。prefill 写入初始 cache；decode 反复读取并追加 cache。所以 cache layout、block allocation、eviction 都是推理引擎问题，而不只是模型问题。

**PagedAttention** 让高并发 decode 可行。decode 很吃内存，所以显存碎片会直接降低系统能承载的 active streams 数量。

**Continuous batching** 把每个 iteration 填满。scheduler 调度的不是“完整请求”，而是下一轮模型工作：prefill 请求的一段 prompt tokens，以及 decode 请求的一个 token step。

**Chunked prefill** 把长 prefill 切开，让 decode 能继续前进。它本质上是用很小的 TTFT 增量换更低的 TPOT jitter。

**Prefix caching** 消除共享前缀的重复 prefill。它改善 TTFT，但请求进入 decode 后，并不会让后续每步生成更便宜。

**Disaggregated prefill** 是同一思想的大规模版本：既然 prefill 和 decode 想要不同的硬件运行点，就把它们放到不同 worker 上，再传输 KV cache。

## profiling 检查表 {#profiling-checklist}

serving benchmark 变慢时，先问是哪个阶段慢。

| 现象 | 可能阶段 | 优先检查 |
|---|---|---|
| TTFT 高，TPOT 正常 | prefill 或排队 | prompt 长度、prefix-cache 命中率、prefill batch size、queue delay |
| TTFT 正常，TPOT 高 | decode | KV cache 大小、显存带宽、active batch size、attention kernel |
| 前几个 token 正常，越生成越慢 | 长上下文 decode | context length、KV cache layout、KV quantization、eviction pressure |
| 流式输出周期性卡顿 | prefill-decode interference | 长 prompt 接入、chunked prefill 设置、token budget |
| 显存高、吞吐低 | decode capacity limit | max concurrent sequences、block fragmentation、KV dtype |

关键不是说“模型慢”，而是把 serving 系统看成 pipeline。prefill 和 decode 压力点不同，修复手段也必须对应阶段。

## 小结 {#conclusion}

prefill 和 decode 使用同一套 Transformer 权重，但它们不是同一个 workload。

- prefill 有很多新 token、大 GEMM、高 compute utilization，主要影响 TTFT
- decode 每个请求每步只有一个新 token，需要反复读取 KV cache，主要受显存带宽限制，影响 TPOT
- scheduler 难做，是因为 prefill 想要大 chunk，而 decode 想要短而频繁的 iteration

这是理解后续推理引擎机制的基本心智模型。把 prefill 和 decode 看成两个瓶颈之后，continuous batching、chunked prefill、prefix caching、PagedAttention、disaggregated prefill 就不再是零散技巧，而是在 compute、memory、latency 三者之间移动工作量的不同方式。
