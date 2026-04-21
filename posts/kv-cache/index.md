
## introduction {#introduction}

large language models generate text **autoregressively** — one token at a time, each new token conditioned on all previous tokens. this sequential nature creates a fundamental opportunity for optimization: most of the computation at each step is *redundant*.

the **KV cache** is the technique that exploits this redundancy. by storing the key and value vectors from previous decoding steps, we avoid re-computing them, turning an \\(O(n^2)\\) per-step cost into \\(O(n)\\) — at the price of extra memory.

## the transformer attention recap {#attention-recap}

in a transformer decoder, self-attention operates as follows. for each token, we project its embedding into three vectors:

\begin{align}
q_i &= W_q x_i \quad & \text{(query)} \\\\
k_i &= W_k x_i \quad & \text{(key)} \\\\
v_i &= W_v x_i \quad & \text{(value)}
\end{align}

the attention output for token \\(i\\) is:

\begin{equation}
\text{Attention}(q_i, K, V) = \text{softmax}\left(\frac{q_i K^T}{\sqrt{d_k}}\right) V
\end{equation}

where \\(K\\) and \\(V\\) are the stacked keys and values of **all tokens in the sequence so far**. causal masking ensures token \\(i\\) only attends to tokens \\(j \leq i\\).

## the problem: redundant computation {#redundant-computation}

consider generating a sequence of length \\(n\\). at step \\(t\\), the model needs to compute attention over tokens \\(1, \ldots, t\\):

- compute \\(q_t, k_t, v_t\\) for the new token
- compute attention scores between \\(q_t\\) and \\(k_1, \ldots, k_t\\)
- weight \\(v_1, \ldots, v_t\\) by those scores

**without caching**, at every step we would re-compute \\(k_j, v_j\\) for all previous tokens \\(j < t\\). this means:

- step 1: compute \\(k_1, v_1\\)
- step 2: re-compute \\(k_1, v_1\\), then compute \\(k_2, v_2\\)
- step 3: re-compute \\(k_1, v_1, k_2, v_2\\), then compute \\(k_3, v_3\\)
- ...
- step \\(n\\): re-compute all previous, then compute \\(k_n, v_n\\)

the total work across all steps is:

\begin{equation}
\sum_{t=1}^{n} t = \frac{n(n+1)}{2} = O(n^2)
\end{equation}

each previous token's key and value is recomputed \\(n - j\\) times for token \\(j\\). this is wasteful because \\(k_j\\) and \\(v_j\\) are **deterministic functions** of the input token \\(x_j\\) and the fixed weights \\(W_k, W_v\\).

## the solution: KV cache {#solution}

the insight is simple: **\\(k_j\\) and \\(v_j\\) do not change once computed**. they depend only on the token embedding and the model weights, neither of which changes during decoding.

with the KV cache:

- step 1: compute \\(k_1, v_1\\), store in cache
- step 2: read \\(k_1, v_1\\) from cache, compute \\(k_2, v_2\\), append to cache
- step 3: read \\(k_1, v_1, k_2, v_2\\) from cache, compute \\(k_3, v_3\\), append to cache
- ...
- step \\(t\\): read all cached KV pairs, compute only \\(k_t, v_t\\), append

the per-step work becomes \\(O(t)\\) for the attention computation (which is unavoidable), but the **key/value projection** is now \\(O(1)\\) per step instead of \\(O(t)\\). the total KV projection work across all steps drops from \\(O(n^2)\\) to \\(O(n)\\).

<img src="/images/posts/kv-cache/kv-cache-diagram.svg" alt="KV Cache: With vs Without" style="width:100%">

<span class="figure-number">Figure 1: </span>without KV cache (top), all previous tokens are re-projected at each step; with KV cache (bottom), only the new token is projected and previous KV pairs are read from cache.

### what happens at each decode step {#decode-step-detail}

concretely, at step \\(t+1\\) the four operations are:

**① compute new token's projections** — the only new KV work:

\begin{equation}
Q_{t+1} = h_{t+1} W_Q, \quad K_{t+1} = h_{t+1} W_K, \quad V_{t+1} = h_{t+1} W_V
\end{equation}

**② concat with cache**:

\begin{equation}
K_{1:t+1} = \text{concat}(K_{1:t}^{\text{cache}}, K_{t+1}), \quad V_{1:t+1} = \text{concat}(V_{1:t}^{\text{cache}}, V_{t+1})
\end{equation}

**③ compute attention** — query is a single row, so this is a \\(1 \times (t+1)\\) dot product:

\begin{equation}
\text{Output}\_{t+1} = \text{softmax}\left(\frac{Q_{t+1} K_{1:t+1}^T}{\sqrt{d_k}}\right) V_{1:t+1}
\end{equation}

**④ update cache**: append \\(K_{t+1}, V_{t+1}\\) to the cache for the next step.

## why keys and values — not queries? {#why-not-queries}

you might wonder: why cache \\(K\\) and \\(V\\) but not \\(Q\\)?

at each decoding step, we only need **one query** — the query for the newly generated token. the query vectors of previous tokens are never used again because they are never the "active" token doing the attending. in causal self-attention, only the *current* token's query attends to previous keys; previous tokens' queries are irrelevant.

in contrast, every previous token's **key** is needed for computing attention scores, and every previous token's **value** is needed for the weighted sum. so we cache those.

## memory cost of the KV cache {#memory-cost}

the KV cache is not free. for a model with:

- \\(L\\) layers
- hidden dimension \\(d\\)
- sequence length \\(n\\)
- batch size \\(b\\)

the total cache size is:

\begin{equation}
\text{KV cache size} = 2 \times L \times b \times n \times d \times \text{bytes}
\end{equation}

the factor of 2 accounts for both keys and values. for a concrete example, consider a 7B model with \\(L = 32\\), \\(d = 4096\\), batch size \\(b = 1\\), and sequence length \\(n = 2048\\):

\begin{equation}
2 \times 32 \times 1 \times 2048 \times 4096 \times 2 \approx 1\,\text{GB}
\end{equation}

(using FP16, 2 bytes per element). this grows linearly with batch size and sequence length. for \\(b = 64\\), \\(n = 8192\\), the cache would be roughly \\(32\\) GB — potentially exceeding the model's own weight memory.

{{< alert theme="warning" >}}

for long sequences and large batches, the KV cache can become the dominant memory consumer — sometimes larger than the model weights themselves. this is why techniques like PagedAttention, quantization, and eviction policies matter for production serving.

{{< /alert >}}

## reducing cache size: MQA and GQA {#mqa-gqa}

the formula above assumes standard **multi-head attention (MHA)**, where every head has its own K and V. two architectural variants reduce this substantially:

**multi-query attention (MQA)** shares a single K,V pair across all query heads. the cache shrinks by \\(n_\text{heads}\\):

\begin{equation}
\text{Memory}_\text{MQA} = B \times S \times L \times 2 \times d_k \times \text{sizeof(dtype)}
\end{equation}

**grouped-query attention (GQA)** is the middle ground: \\(G\\) groups of K,V, each shared among \\(n_\text{heads}/G\\) query heads:

\begin{equation}
\text{Memory}_\text{GQA} = B \times S \times L \times 2 \times d_k \times G \times \text{sizeof(dtype)}
\end{equation}

```text
MHA: Q [H1][H2]...[H64]   K [H1][H2]...[H64]   ← 64 KV heads
GQA: Q [H1..H8][H9..H16]..[H57..H64]  K [G1]..[G8]  ← 8 KV heads
MQA: Q [H1][H2]...[H64]   K [  shared   ]        ← 1 KV head
```

| attention type | KV heads | cache size |
|---|---|---|
| MHA | \\(H\\) | \\(1\times\\) baseline |
| GQA | \\(G\\) | \\(G/H \times\\) |
| MQA | \\(1\\) | \\(1/H \times\\) |

LLaMA-3-70B uses GQA with \\(H = 64\\) query heads and \\(G = 8\\) KV heads, giving \\(1/8\\) the KV cache of equivalent MHA. most modern LLMs (Mistral, Gemma, LLaMA-3) have adopted GQA as the default.

## compute vs memory tradeoff {#tradeoff}

the KV cache is a classic compute-memory tradeoff:

| dimension | without cache | with cache |
|---|---|---|
| **KV projection FLOPs** | \\(O(n^2)\\) total | \\(O(n)\\) total |
| **attention FLOPs** | \\(O(n^2)\\) per step | \\(O(n)\\) per step |
| **memory for KV** | \\(O(1)\\) per step | \\(O(n)\\) cumulative |
| **memory traffic** | recompute every step | read cache + write new |

in practice, autoregressive decoding is **memory-bandwidth bound**, not compute-bound. the KV cache reduces FLOPs but *increases* memory traffic because we must read the growing cache at each step. this is why serving frameworks like vLLM focus heavily on efficient KV cache memory management — the bottleneck shifts from recomputation to cache memory allocation and bandwidth.

## the two phases of LLM inference {#two-phases}

understanding KV cache also clarifies the two distinct phases of LLM inference:

**prefill (prompt processing):** the entire input prompt is processed in parallel. all KV pairs for prompt tokens are computed and cached. this phase is compute-bound because we process many tokens simultaneously.

**decoding (token generation):** one token at a time is generated. each step reads the entire KV cache, computes attention, and appends one new KV pair. this phase is memory-bandwidth-bound because each step does relatively little compute but must read the entire cache.

<img src="/images/posts/kv-cache/prefill-vs-decode.svg" alt="Prefill vs Decode: Two Phases of LLM Inference" style="width:100%">

<span class="figure-number">Figure 2: </span>prefill processes all prompt tokens in parallel and builds the initial KV cache; decoding generates one token per step, reading the cache and appending the new KV pair.

## optimizations on top of KV cache {#optimizations}

several techniques build on the basic KV cache idea:

- **PagedAttention** (vLLM): manages KV cache memory in fixed-size pages like an OS page table, eliminating memory fragmentation and enabling higher throughput[^fn:1]
- **KV cache quantization**: stores KV pairs in INT8 or FP4 instead of FP16, reducing memory by 2-4x with minimal quality loss
- **Sliding window attention**: only caches the most recent \\(w\\) tokens, bounding cache size to \\(O(w)\\) instead of \\(O(n)\\)
- **KV cache eviction**: removes least-recently-accessed cache entries when memory is full, trading recall for capacity
- **Prefix caching**: shares KV cache entries across requests with common prefixes (e.g., system prompts), amortizing prefill cost

## summary {#summary}

- **keys and values are deterministic**: \\(k_j = W_k x_j\\) and \\(v_j = W_v x_j\\) depend only on fixed weights and input tokens, so once computed, they never change
- **queries are not cached**: only the current token's query is needed; previous queries are never reused in causal attention
- **KV cache trades memory for compute**: reduces total KV projection work from \\(O(n^2)\\) to \\(O(n)\\), at the cost of \\(O(n)\\) memory per sequence
- **the cache makes decoding memory-bandwidth bound**: each step reads the growing cache, shifting the bottleneck from FLOPs to memory bandwidth
- **MQA and GQA reduce cache size**: sharing K,V across query heads cuts cache by \\(G/H\\) (GQA) or \\(1/H\\) (MQA) — used by LLaMA-3, Mistral, Gemma
- **production serving systems optimize cache management**: PagedAttention, quantization, and eviction all address the memory pressure the KV cache introduces

[^fn:1]: see [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) for details on efficient KV cache memory management.
