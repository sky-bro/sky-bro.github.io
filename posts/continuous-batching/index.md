
## the static batching problem {#static-batching}

before continuous batching existed, LLM serving systems used **static batching**: collect a batch of requests, run them all through the model together, and wait until every request in the batch finishes generating before accepting the next batch.

this sounds reasonable — batching is how you saturate a GPU — but it has a fatal flaw.

different requests produce outputs of wildly different lengths. a request asking "what is 2+2?" might finish in 5 tokens. a request asking for a short story might need 800. in a static batch, the short request finishes early and then... does nothing. the GPU keeps crunching for the long request while the short request's slot sits idle.

```text
time ──────────────────────────────────────────────────────►

Req A  [==Prefill==][D][D][D][D][D]  (Done, 5 decode steps)
Req B  [==Prefill==][D][D][D][D][D][D][D][D][D][D]  (Done, 10 steps)
Req C  [==Prefill==][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D]  (15 steps)

Batch  |←────────────────────── must wait for C ───────────────────────→|
                     ↑                        ↑
          A finished here           B finished here
          GPU slot idle             GPU slot idle
```

{{< figure src="/images/posts/continuous-batching/static-vs-continuous.svg" caption="<span class=\"figure-number\">Figure 1: </span>static batching (left) idles GPU slots after short requests finish; continuous batching (right) inserts new requests the moment a slot opens, sustaining ~96% GPU utilization." width="100%" >}}

measured on real workloads, static batching achieves **30–50% GPU utilization**. the rest is wasted waiting for the longest request in the batch.

the root cause: scheduling granularity is at the *batch* level, but the natural unit of LLM work is a single *decoding step* (one forward pass through the transformer).

## continuous batching: scheduling per iteration {#core-idea}

**continuous batching** (also called *iteration-level scheduling* or *in-flight batching*) was introduced in the Orca paper (Yu et al., OSDI 2022). the key idea:

> after every forward pass, check which requests finished and immediately insert new ones from the waiting queue.

scheduling granularity drops from *batch* to *iteration*:

```text
time ──────────────────────────────────────────────────────────────────►

Req A  [Pre][D][D][D][D][Done]
Req B  [Pre][D][D][D][D][D][D][D][Done]
Req C            [Pre][D][D][D][Done]     ← inserted when A finished
Req D                     [Pre][D][D][D][D][Done]  ← inserted when C finished

       |← iter →|← iter →|← iter →|← iter →|
       each iter: check for finished reqs, insert new ones
```

the scheduler's core loop becomes:

```python
while True:
    batch = scheduler.schedule()           # fill token budget from waiting queue
    outputs = model.forward(batch)
    for req, token in zip(batch, outputs):
        req.append(token)
        if token.is_eos or req.at_max_len:
            scheduler.finish(req)          # free KV blocks immediately
        else:
            scheduler.continue_(req)       # keep in running queue
    # next iteration starts immediately — no waiting
```

when a request finishes, its [KV cache blocks]({{< relref "paged-attention" >}}) are freed immediately and the next request from the queue can enter at the very next iteration.

## mixing prefill and decode in one forward pass {#mixing}

here is where it gets mathematically interesting. a batch under continuous batching contains two types of requests simultaneously:

- **prefill requests**: first time entering the system, need to process their entire prompt (or a [chunk]({{< relref "chunked-prefill" >}}) of it) — \\(L\\) new tokens
- **decode requests**: already in progress, each adding exactly **1** new token this iteration

they have different sequence lengths. can they share a single matrix operation?

### packing: eliminating padding {#packing}

the naive approach pads all sequences to the same length \\(L_{\max}\\). this wastes computation on pad tokens and creates huge overhead when sequence lengths vary widely.

the alternative: **pack all new tokens from all requests into a single flat sequence**, with no padding:

\begin{equation}
\mathbf{x} = [\underbrace{t_1^A, t_2^A, t_3^A}_{\text{Req A prefill, 3 tokens}},\ \underbrace{t_t^B}_{\text{Req B decode, 1 token}},\ \underbrace{t_1^C, t_2^C}_{\text{Req C prefill, 2 tokens}}]
\end{equation}

total length: 6 tokens. one matrix multiply covers all of them. no padding wasted.

{{< alert theme="info" >}}

**what "new tokens" means here**: for a prefill request, all prompt tokens are new (they haven't been processed yet, so their KV pairs don't exist in cache). for a decode request, only the *one token just generated* is new — all previous tokens are already in the KV cache and don't re-enter the packed sequence.

{{< /alert >}}

the projection step \\(X W_Q\\) (and similarly for \\(W_K, W_V\\)) runs once over the packed sequence \\(X \in \mathbb{R}^{6 \times d}\\), producing \\(Q, K, V \in \mathbb{R}^{6 \times d_k}\\).

### block-diagonal causal mask {#mask}

the problem with naively packing sequences is *cross-request attention leakage*: request B's token would attend to request A's tokens, corrupting the output.

we prevent this with a **block-diagonal causal mask**. the mask \\(M \in \mathbb{R}^{L_{\text{total}} \times L_{\text{total}}}\\) is defined as:

\begin{equation}
M_{ij} = \begin{cases}
0 & \text{if } \text{req}(i) = \text{req}(j) \text{ and } j \leq i \\
-\infty & \text{otherwise}
\end{cases}
\end{equation}

for our example (A at positions 0–2, B at position 3, C at positions 4–5), the full \\(6 \times 6\\) mask is:

\begin{equation}
M = \begin{bmatrix}
 0      & -\infty & -\infty & -\infty & -\infty & -\infty \\
 0      &  0      & -\infty & -\infty & -\infty & -\infty \\
 0      &  0      &  0      & -\infty & -\infty & -\infty \\
-\infty & -\infty & -\infty &  0      & -\infty & -\infty \\
-\infty & -\infty & -\infty & -\infty &  0      & -\infty \\
-\infty & -\infty & -\infty & -\infty &  0      &  0
\end{bmatrix}
\end{equation}

{{< figure src="/images/posts/continuous-batching/packing-mask.svg" caption="<span class=\"figure-number\">Figure 2: </span>tokens from requests A, B, C are packed into a single flat sequence (top). the block-diagonal causal mask (bottom) allows each request to attend only within its own tokens, in causal order — cross-request entries become −∞ and vanish after softmax." width="100%" >}}

three blocks appear along the diagonal: a \\(3 \times 3\\) lower-triangular block for A, a \\(1 \times 1\\) block for B, and a \\(2 \times 2\\) lower-triangular block for C. after softmax, the \\(-\infty\\) entries become exactly 0, so cross-request attention is zero.

**the result is mathematically identical to running attention separately for each request** — but using a single matrix multiply instead of three.

| approach | matmul count | effective compute |
|---|---|---|
| per-request separate | 3 small matmuls | 100% |
| padded batch \\((B, L_{\max}, d)\\) | 1 large matmul | ~30% (rest is pad waste) |
| **packing** | **1 matmul** | **100%** |

in practice, FlashAttention's `varlen` interface (`flash_attn_varlen_func`) handles this packing natively: you pass the cumulative sequence lengths (`cu_seqlens`) and it applies the block-diagonal mask internally.

## the prefill/decode split inside the KV cache {#kv-split}

packing handles the \\(QKV\\) projection step. but the attention computation differs between prefill and decode because of the KV cache:

**prefill request A** (3 new tokens, no prior cache):

the \\(K^A = [k_1^A, k_2^A, k_3^A]\\) and \\(V^A = [v_1^A, v_2^A, v_3^A]\\) computed from the packed input are written to the KV cache. attention is computed within the sequence (standard causal attention over the 3 tokens).

**decode request B** (1 new token, cache has \\(t-1\\) prior tokens):

the cache already holds \\(K^B_{\text{cache}} = [k_1^B, \ldots, k_{t-1}^B]\\). the packed input produces one new \\(q_t^B\\), which attends to the full sequence:

\begin{equation}
\text{attn}\_t^B = \text{softmax}\left(\frac{q_t^B \cdot [K^B_{\text{cache}};\ k_t^B]^T}{\sqrt{d_k}}\right) [V^B_{\text{cache}};\ v_t^B]
\end{equation}

the historical KV is read directly from the [paged KV cache]({{< relref "paged-attention" >}}) — it never re-enters the packed sequence.

this is a **vector–matrix multiply** (\\(q_t^B \in \mathbb{R}^{1 \times d_k}\\) against \\(K \in \mathbb{R}^{t \times d_k}\\)), which is why decode is fundamentally different from prefill.

## why decode is memory-bound {#memory-bound}

the compute-to-memory-access ratio (arithmetic intensity) differs dramatically between the two phases:

| | prefill | decode |
|---|---|---|
| query shape | \\((L, d_k)\\) | \\((1, d_k)\\) |
| operation type | GEMM (matrix × matrix) | GEMV (vector × matrix) |
| arithmetic intensity | high — tensor cores saturated | very low — bandwidth bound |
| bottleneck | compute | HBM bandwidth |

during decode, for each new token we must read the *entire* KV cache from HBM. the KV cache grows by one token per step, and we do only 2 \\(d_k\\)-dimensional dot products per token-step (one for scores, one for values). the FLOP/byte ratio is far below the GPU's roofline.

**how large is the KV cache per token?** for a model with \\(L\\) layers, \\(n_h\\) KV heads, and head dimension \\(d_h\\):

\begin{equation}
\text{bytes per token} = 2 \times L \times n_h \times d_h \times \text{sizeof(dtype)}
\end{equation}

for LLaMA-3 8B (\\(L = 32\\), \\(n_h = 8\\) GQA KV heads, \\(d_h = 128\\), BF16):

\begin{equation}
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KB per token}
\end{equation}

a 512-token KV cache occupies 64 MB. a 4096-token context hits 512 MB. at every decode step, the GPU must stream this entire cache from HBM — even though it only does a single token's worth of computation. an A100 has ~2 TB/s HBM bandwidth and ~312 TFLOP/s of BF16 compute; the roofline for decode GEMV is bandwidth, not compute.

## token budget and throughput {#token-budget}

continuous batching keeps GPU utilization high by maintaining a **token budget**: a cap on the total number of KV cache tokens in flight.

let GPU memory \\(M\\), model weights \\(W\\), and per-token KV size \\(k\\). the maximum concurrent tokens:

\begin{equation}
N_{\max} = \frac{M - W}{k}
\end{equation}

the scheduler's goal: keep \\(\sum_{\text{req}} L_{\text{req}} \approx N_{\max}\\) at all times.

when a request finishes and frees \\(\Delta N\\) token slots, the scheduler immediately pulls the next waiting request (which needs roughly \\(\Delta N\\) prompt tokens to start). the token budget stays near its ceiling, and the GPU stays busy.

under static batching, the expected token budget utilization is roughly \\(\bar{L} / L_{\max}\\) — for a uniform length distribution, about 50%. with continuous batching it approaches **100%**, because short requests free their slots and those slots are replenished immediately.

## latency metrics to keep in mind {#metrics}

two latency metrics matter for LLM serving:

| metric | meaning |
|---|---|
| **TTFT** — Time to First Token | time from request submission to receiving the first generated token |
| **TPOT** — Time Per Output Token | time between successive generated tokens (= 1 / decode throughput) |

continuous batching primarily improves **throughput** and **GPU utilization**. its effect on latency is nuanced:

- TTFT may *increase* slightly if the scheduler holds a new request in the queue while the token budget is full
- TPOT is mostly unchanged for a single request, but aggregate TPOT across all requests improves because the GPU is rarely idle

## the next problem {#next-problem}

continuous batching solves *when* to insert new requests. but it doesn't resolve what happens when a new request has a very long prompt.

a request with a 2048-token prompt will occupy an entire iteration for its prefill phase — 200+ ms on a typical GPU. during that time, all the decode requests in the batch are blocked. their TPOT spikes from 5 ms to 200+ ms for that one iteration.

this is **prefill-decode interference**: the compute-heavy prefill monopolizes the GPU, starving the latency-sensitive decode. it's the problem that [chunked prefill]({{< relref "chunked-prefill" >}}) addresses by slicing the prefill across multiple iterations.
