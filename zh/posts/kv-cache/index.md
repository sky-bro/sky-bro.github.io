
## 引言 {#introduction}

大语言模型以**自回归**方式生成文本——每次生成一个 token，每个新 token 依赖于之前所有 token。这种串行特性带来了一个根本的优化机会：每一步中大部分计算是**冗余**的。

**KV cache**（键值缓存）正是利用这一冗余的技术。通过保存之前解码步骤中的 key 和 value 向量，我们避免了对它们的重复计算，将每步 \\(O(n^2)\\) 的开销降为 \\(O(n)\\)——代价是额外的显存占用。

## Transformer 注意力回顾 {#attention-recap}

在 Transformer decoder 中，自注意力机制如下。对每个 token，将其嵌入投影为三个向量：

\begin{align}
q_i &= W_q x_i \quad & \text{(query)} \\\\
k_i &= W_k x_i \quad & \text{(key)} \\\\
v_i &= W_v x_i \quad & \text{(value)}
\end{align}

token \\(i\\) 的注意力输出为：

\begin{equation}
\text{Attention}(q_i, K, V) = \text{softmax}\left(\frac{q_i K^T}{\sqrt{d_k}}\right) V
\end{equation}

其中 \\(K\\) 和 \\(V\\) 是**当前序列中所有 token** 的 key 和 value 的堆叠。causal masking 保证 token \\(i\\) 只能关注 \\(j \leq i\\) 的 token。

## 问题：冗余计算 {#redundant-computation}

考虑生成长度为 \\(n\\) 的序列。在第 \\(t\\) 步，模型需要对 token \\(1, \ldots, t\\) 计算注意力：

- 为新 token 计算 \\(q_t, k_t, v_t\\)
- 计算 \\(q_t\\) 与 \\(k_1, \ldots, k_t\\) 的注意力分数
- 用这些分数对 \\(v_1, \ldots, v_t\\) 加权求和

**如果没有缓存**，每一步我们都会重新计算之前所有 token 的 \\(k_j, v_j\\)：

- 第 1 步：计算 \\(k_1, v_1\\)
- 第 2 步：重新计算 \\(k_1, v_1\\)，再计算 \\(k_2, v_2\\)
- 第 3 步：重新计算 \\(k_1, v_1, k_2, v_2\\)，再计算 \\(k_3, v_3\\)
- ...
- 第 \\(n\\) 步：重新计算所有之前的，再计算 \\(k_n, v_n\\)

所有步的总计算量为：

\begin{equation}
\sum_{t=1}^{n} t = \frac{n(n+1)}{2} = O(n^2)
\end{equation}

每个之前 token 的 key 和 value 被重复计算了 \\(n - j\\) 次。这是浪费的，因为 \\(k_j\\) 和 \\(v_j\\) 是输入 token \\(x_j\\) 和固定权重 \\(W_k, W_v\\) 的**确定性函数**。

## 解决方案：KV 缓存 {#solution}

洞察很简单：**\\(k_j\\) 和 \\(v_j\\) 一旦计算出来就不会改变**。它们只依赖于 token 嵌入和模型权重，这两者在解码过程中都不变。

使用 KV cache 后：

- 第 1 步：计算 \\(k_1, v_1\\)，存入缓存
- 第 2 步：从缓存读取 \\(k_1, v_1\\)，计算 \\(k_2, v_2\\)，追加到缓存
- 第 3 步：从缓存读取 \\(k_1, v_1, k_2, v_2\\)，计算 \\(k_3, v_3\\)，追加到缓存
- ...
- 第 \\(t\\) 步：读取所有已缓存的 KV 对，只计算 \\(k_t, v_t\\)，追加

每步的 KV 投影工作从 \\(O(t)\\) 降为 \\(O(1)\\)。所有步的总 KV 投影工作量从 \\(O(n^2)\\) 降为 \\(O(n)\\)。

<img src="/images/posts/kv-cache/kv-cache-diagram.svg" alt="KV Cache: With vs Without" style="width:100%">

<span class="figure-number">Figure 1: </span>无 KV 缓存（上）每步都重新投影之前所有 token；有 KV 缓存（下）只投影新 token，之前的 KV 对从缓存读取。

### 每步 Decode 的具体操作 {#decode-step-detail}

在第 \\(t+1\\) 步，完整的四个操作为：

**① 只计算新 token 的投影**——唯一新增的 KV 计算：

\begin{equation}
Q_{t+1} = h_{t+1} W_Q, \quad K_{t+1} = h_{t+1} W_K, \quad V_{t+1} = h_{t+1} W_V
\end{equation}

**② 与缓存拼接**：

\begin{equation}
K_{1:t+1} = \text{concat}(K_{1:t}^{\text{cache}}, K_{t+1}), \quad V_{1:t+1} = \text{concat}(V_{1:t}^{\text{cache}}, V_{t+1})
\end{equation}

**③ 计算注意力**——query 是单行向量，attention 退化为 \\(1 \times (t+1)\\) 的点积：

\begin{equation}
\text{Output}\_{t+1} = \text{softmax}\left(\frac{Q_{t+1} K_{1:t+1}^T}{\sqrt{d_k}}\right) V_{1:t+1}
\end{equation}

**④ 更新缓存**：将 \\(K_{t+1}, V_{t+1}\\) 追加到缓存供下一步使用。

## 为什么缓存 K 和 V，而不缓存 Q？ {#why-not-queries}

为什么缓存 \\(K\\) 和 \\(V\\) 而不缓存 \\(Q\\)？

在每一步解码中，我们只需要**一个 query**——新生成 token 的 query。之前 token 的 query 不会再被使用，因为在因果自注意力中，只有_当前_ token 的 query 去 attend 之前的 key，之前 token 的 query 无关紧要。

相反，每个之前 token 的 **key** 都用于计算注意力分数，每个之前 token 的 **value** 都用于加权求和。所以我们需要缓存它们。

## KV 缓存的显存开销 {#memory-cost}

KV cache 并非免费。对于以下参数的模型：

- \\(L\\) 层
- 隐藏维度 \\(d\\)
- 序列长度 \\(n\\)
- batch size \\(b\\)

总缓存大小为：

\begin{equation}
\text{KV cache 大小} = 2 \times L \times b \times n \times d \times \text{bytes}
\end{equation}

乘以 2 是因为同时存 key 和 value。以 7B 模型为例，\\(L = 32\\)，\\(d = 4096\\)，batch size \\(b = 1\\)，序列长度 \\(n = 2048\\)：

\begin{equation}
2 \times 32 \times 1 \times 2048 \times 4096 \times 2 \approx 1\,\text{GB}
\end{equation}

（FP16，每元素 2 字节）。这随 batch size 和序列长度线性增长。对于 \\(b = 64\\)，\\(n = 8192\\)，缓存约 \\(32\\) GB——可能超过模型权重本身的显存。

{{< alert theme="warning" >}}

对于长序列和大批量，KV cache 可能成为显存的主要消费者——有时比模型权重本身还大。这就是为什么 PagedAttention、量化和淘汰策略对生产服务很重要。

{{< /alert >}}

## 缩减缓存：MQA 与 GQA {#mqa-gqa}

上述公式假设标准的**多头注意力（MHA）**，每个 head 有独立的 K 和 V。两种架构变体可以显著减小缓存：

**多查询注意力（MQA）** 让所有 query head 共享同一组 K、V，缓存缩小至 \\(1/n_\text{heads}\\)：

\begin{equation}
\text{Memory}_\text{MQA} = B \times S \times L \times 2 \times d_k \times \text{sizeof(dtype)}
\end{equation}

**分组查询注意力（GQA）** 是 MHA 与 MQA 的折中：\\(G\\) 组 K、V，每组被 \\(n_\text{heads}/G\\) 个 query head 共享：

\begin{equation}
\text{Memory}_\text{GQA} = B \times S \times L \times 2 \times d_k \times G \times \text{sizeof(dtype)}
\end{equation}

```text
MHA: Q [H1][H2]...[H64]   K [H1][H2]...[H64]   ← 64 个 KV head
GQA: Q [H1..H8][H9..H16]..[H57..H64]  K [G1]..[G8]  ← 8 个 KV head
MQA: Q [H1][H2]...[H64]   K [  共享   ]         ← 1 个 KV head
```

| 注意力类型 | KV head 数 | 缓存大小 |
| --- | --- | --- |
| MHA | \\(H\\) | \\(1\times\\) 基准 |
| GQA | \\(G\\) | \\(G/H \times\\) |
| MQA | \\(1\\) | \\(1/H \times\\) |

LLaMA-3-70B 使用 GQA，\\(H = 64\\) 个 query head，\\(G = 8\\) 个 KV head，实际缓存仅为等效 MHA 的 \\(1/8\\)。目前大多数主流 LLM（Mistral、Gemma、LLaMA-3）均已默认采用 GQA。

## 计算与显存的权衡 {#tradeoff}

KV cache 是经典的计算-显存权衡：

| 维度 | 无缓存 | 有缓存 |
| --- | --- | --- |
| **KV 投影 FLOPs** | 总计 \\(O(n^2)\\) | 总计 \\(O(n)\\) |
| **注意力 FLOPs** | 每步 \\(O(n^2)\\) | 每步 \\(O(n)\\) |
| **KV 显存** | 每步 \\(O(1)\\) | 累计 \\(O(n)\\) |
| **显存流量** | 每步重新计算 | 读缓存 + 写新 token |

实际上，自回归解码是**显存带宽受限**而非计算受限的。KV cache 减少了 FLOPs 但_增加了_显存流量，因为每步必须读取不断增长的缓存。这就是为什么 vLLM 等服务框架高度关注 KV cache 显存管理——瓶颈从重新计算转移到了缓存显存分配和带宽。

## LLM 推理的两个阶段 {#two-phases}

理解 KV cache 也有助于理解 LLM 推理的两个不同阶段：

**Prefill（提示处理）：** 整个输入提示被并行处理。所有 prompt token 的 KV 对被计算并缓存。这个阶段是计算受限的，因为我们同时处理许多 token。

**Decode（token 生成）：** 每次生成一个 token。每步读取整个 KV cache，计算注意力，追加一个新的 KV 对。这个阶段是显存带宽受限的，因为每步计算量相对较小但必须读取整个缓存。

<img src="/images/posts/kv-cache/prefill-vs-decode.svg" alt="Prefill vs Decode: Two Phases of LLM Inference" style="width:100%">

<span class="figure-number">Figure 2: </span>Prefill 并行处理所有 prompt token 并构建初始 KV cache；Decode 每步生成一个 token，读取缓存并追加新的 KV 对。

## KV cache 之上的优化 {#optimizations}

多种技术在基本 KV cache 之上进一步优化：

- **PagedAttention**（vLLM）：像操作系统页表一样以固定大小的页管理 KV cache 显存，消除显存碎片，提高吞吐量[^fn:1]
- **KV cache 量化**：用 INT8 或 FP4 而非 FP16 存储 KV 对，显存减少 2-4 倍，质量损失极小
- **滑动窗口注意力**：只缓存最近的 \\(w\\) 个 token，将缓存大小限制为 \\(O(w)\\) 而非 \\(O(n)\\)
- **KV cache 淘汰**：显存满时移除最近最少访问的缓存条目，以召回率换取容量
- **前缀缓存**：共享具有共同前缀的请求之间的 KV cache 条目（如 system prompt），摊销 prefill 成本

## 总结 {#summary}

- **key 和 value 是确定性的**：\\(k_j = W_k x_j\\) 和 \\(v_j = W_v x_j\\) 只依赖于固定权重和输入 token，一旦计算就不再变化
- **query 不被缓存**：只需要当前 token 的 query；在因果注意力中，之前的 query 不会被复用
- **KV cache 用显存换计算**：总 KV 投影工作从 \\(O(n^2)\\) 降为 \\(O(n)\\)，代价是每序列 \\(O(n)\\) 的显存
- **缓存使解码变为显存带宽受限**：每步读取不断增长的缓存，瓶颈从 FLOPs 转移到显存带宽
- **MQA 和 GQA 减小缓存体积**：跨 query head 共享 K、V，缓存缩小至 \\(G/H\\)（GQA）或 \\(1/H\\)（MQA）——LLaMA-3、Mistral、Gemma 均已采用
- **生产服务系统优化缓存管理**：PagedAttention、量化和淘汰策略都针对 KV cache 带来的显存压力

[^fn:1]: 详见 [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。
