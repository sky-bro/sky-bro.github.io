
## the interference problem {#interference}

[continuous batching]({{< relref "continuous-batching" >}}) keeps the GPU busy by scheduling at iteration granularity. but one edge case breaks the latency story: **long prefills**.

when a request arrives with a 2048-token prompt, the scheduler runs it through prefill in a single iteration. on an A100, a 2048-token prefill for a 7B model takes roughly 200 ms. all the decode requests already in the batch are blocked for the entire duration.

```
time ──────────────────────────────────────────────────────────────────►

iter 1:  [Req A prefill: 2048 tokens — 200 ms                         ]
         ←────────────────────────────────────────────────────────────→
         Req B, C, D (decode) are ALL blocked for 200 ms

iter 2:  [A dec][B dec][C dec][D dec]  ← 5 ms
iter 3:  [A dec][B dec][C dec][D dec]  ← 5 ms
...
```

{{< figure src="/images/posts/chunked-prefill/prefill-blocking.svg" caption="<span class=\"figure-number\">Figure 1: </span>without chunked prefill (left), a 2048-token prefill blocks all decode requests for ~200 ms, spiking TPOT 41×. with chunked prefill at C = 512 (right), decode runs every iteration with minimal overhead." width="100%" >}}

from the perspective of Req B, C, and D, their **TPOT** (time per output token) just jumped from 5 ms to 205 ms for one step. users notice this as an irregular "stutter" in streaming output.

this is **prefill-decode interference**: the compute-bound prefill (GEMM) monopolizes the GPU, starving the latency-sensitive decode (GEMV).

the two metrics pulled in opposite directions:

| optimization | TTFT (time to first token) | TPOT (time per output token) |
|---|---|---|
| large prefill, run all at once | ↓ low — KV cache ready quickly | ↑ high — decode blocked |
| delay prefill, prioritize decode | ↑ high — new request waits | ↓ low — decode unaffected |

you can't minimize both simultaneously — unless you *slice* the prefill.

## chunked prefill: the core idea {#core-idea}

**chunked prefill** splits a long prompt into segments of size \\(C\\) (the *chunk size*) and processes one segment per iteration, interleaved with decode steps:

```
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

each iteration processes a fixed **token budget** \\(T\\):

\begin{equation}
T = C\_{\text{prefill}} + N\_{\text{decode}}
\end{equation}

where \\(C_{\text{prefill}}\\) is the number of prefill tokens processed this iteration and \\(N_{\text{decode}}\\) is the number of running decode requests. the scheduler enforces \\(C_{\text{prefill}} + N_{\text{decode}} \leq T\\).

decode requests continue running at every iteration. their TPOT becomes roughly:

\begin{equation}
\text{TPOT} \approx \frac{\text{compute}(C\_{\text{prefill}} + N\_{\text{decode}})}{\text{compute}(N\_{\text{decode}})} \times \Delta t\_{\text{decode}}
\end{equation}

with \\(C = 512\\) and \\(N_{\text{decode}} = 32\\), the TPOT overhead from a prefill chunk is small: 512 tokens of GEMM adds far less than a full 2048-token prefill would.

## why chunked prefill is mathematically exact {#correctness}

**does slicing the prefill change the result?** no — it is *exactly* equivalent to running the full prefill at once. here is why.

recall that decoder-only transformers use causal attention: position \\(i\\) only attends to positions \\(j \leq i\\).

suppose the prompt is \\([t_1, t_2, \ldots, t_L]\\), split into chunks of size \\(C\\). chunk \\(s\\) handles tokens \\([(s-1)C+1, \ldots, sC]\\).

for any token \\(t_i\\) in chunk \\(s\\):
- tokens \\(j < (s-1)C + 1\\) are from prior chunks — their \\(k_j, v_j\\) were computed in earlier iterations and stored in the KV cache
- tokens \\(j \in [(s-1)C+1, i]\\) are in the current chunk — \\(k_j, v_j\\) are computed this iteration

so the attention for \\(t_i\\) splits into two parts:

\begin{align}
\text{attn}\_{i} = \text{softmax\_merge}\Bigl(
  &\underbrace{\frac{q\_i \cdot K\_{\text{cache}}^T}{\sqrt{d\_k}}}\_{\text{attend to prior chunks}},\;
  \underbrace{\frac{q\_i \cdot K\_{\text{chunk}}^T}{\sqrt{d\_k}}}\_{\text{attend within current chunk}}
\Bigr) \cdot \begin{bmatrix} V\_{\text{cache}} \\ V\_{\text{chunk}} \end{bmatrix}
\end{align}

where `softmax_merge` is the online softmax merge (same trick as [paged attention's]({{< relref "paged-attention" >}}) block-level aggregation). FlashAttention's `flash_attn_varlen_func` handles this natively — the `cu_seqlens` parameter tells it each token's effective context length (cache history + current chunk).

after each chunk, the newly computed \\(k, v\\) vectors are written to the KV cache:

\begin{equation}
K\_{\text{cache}} \mathrel{+}= [k\_{(s-1)C+1}, \ldots, k\_{sC}]
\end{equation}

the next chunk sees this extended cache. by induction, after all \\(\lceil L/C \rceil\\) chunks, the KV cache contains exactly what a full-prompt prefill would have produced — the final decode step is indistinguishable from the non-chunked case.

## TTFT/TPOT tradeoff and chunk size selection {#tradeoff}

chunk size \\(C\\) is the key tuning knob:

\begin{equation}
\text{TTFT} \approx \left\lceil \frac{L\_{\text{prompt}}}{C} \right\rceil \times \Delta t\_{\text{iter}}
\end{equation}

\begin{equation}
\text{TPOT jitter} \propto \frac{C}{N\_{\text{decode}}} \times \frac{\text{FLOP}\_{\text{GEMM}}}{\text{FLOP}\_{\text{GEMV}}}
\end{equation}

- **larger \\(C\\)**: fewer iterations to complete prefill → lower TTFT. but each prefill chunk is larger → more decode interference per iteration → higher TPOT jitter.
- **smaller \\(C\\)**: decode runs with minimal interference → stable TPOT. but prefill needs more iterations → higher TTFT.

the sweet spot depends on the ratio of active decode requests to prefill tokens. typical production defaults:

| engine | default chunk size |
|---|---|
| vLLM (v0.4+) | 512 tokens |
| SGLang | 512 tokens |
| TensorRT-LLM | 1024 tokens |

with \\(C = 512\\) and a 2048-token prompt: 4 iterations to complete prefill, each adding just 512 tokens of GEMM overhead to the decode step. total TTFT increase versus full-prefill: \\(3 \times 5\text{ ms} = 15\text{ ms}\\) — negligible for most use cases.

## FLOPs analysis: no overhead from chunking {#flops}

an important sanity check: does chunking *add* FLOPs? the answer is no.

for a single transformer layer, attention FLOPs processing \\(L\\) tokens is:

\begin{equation}
\text{FLOP}\_{\text{attn}}(L) = 4L^2 d + 4Ld^2
\end{equation}

the \\(4L^2 d\\) term comes from \\(QK^T\\) and \\(\text{attn} \cdot V\\); \\(4Ld^2\\) from the four projection matrices.

**without chunking**: one call with \\(L\\) tokens.

**with chunking**: \\(\lceil L/C \rceil\\) calls, each processing \\(C\\) prompt tokens against an expanding KV cache. the total attention FLOPs across all chunks:

\begin{align}
\text{FLOP}\_{\text{chunk-attn}} &= 4d^2 \sum\_{s=1}^{L/C} C + 4d \sum\_{s=1}^{L/C} C \cdot (sC) \\\\
&= 4Ld^2 + 4d \cdot C^2 \cdot \frac{(L/C)(L/C + 1)}{2} \\\\
&\approx 4Ld^2 + 4d \cdot \frac{L^2}{2} = 4L^2 d + 4Ld^2
\end{align}

exactly the same as the non-chunked case. **chunking distributes the same FLOPs across more iterations — it does not add computation.**

## IO overhead: negligible in practice {#io-overhead}

the one real cost is writing new KV vectors to HBM at the end of each chunk. for chunk size \\(C\\), model with \\(n_h\\) KV heads, head dim \\(d_h\\), \\(L_{\text{layers}}\\) layers, and BF16:

\begin{equation}
\text{write per chunk} = C \times 2 \times L\_{\text{layers}} \times n\_h \times d\_h \times 2 \text{ bytes}
\end{equation}

for LLaMA-3 8B (\\(L = 32, n_h = 8, d_h = 128\\)) and \\(C = 512\\):

\begin{equation}
512 \times 2 \times 32 \times 8 \times 128 \times 2 = 67{,}108{,}864 \text{ bytes} \approx 64 \text{ MB}
\end{equation}

at A100's HBM bandwidth of ~2 TB/s:

\begin{equation}
\frac{64 \times 10^6}{2 \times 10^{12}} = 32 \text{ μs}
\end{equation}

32 microseconds, compared to an iteration time of ~5 ms. the IO overhead is **<1% of iteration cost** — genuinely negligible.

## interaction with prefix caching {#prefix-cache-interaction}

chunked prefill and [prefix caching]({{< relref "prefix-caching" >}}) compose nicely. if the first \\(k\\) blocks of a prompt are already in the cache, those blocks are skipped entirely:

```
prompt: [system prompt — 1024 tokens][user query — 1024 tokens]
             (cached — skip)              (must compute)

with prefix cache + chunked prefill (C = 512):
  iter 1: [user query tokens   0–511]   ← only 2 chunks instead of 4
  iter 2: [user query tokens 512–1023]
  iter 3: [decode]
```

the effective prefill length after a cache hit is only the *uncached suffix*. TTFT drops further, and fewer iterations are consumed for prefill.

## scheduler implementation {#implementation}

the scheduling logic in SGLang looks roughly like:

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

the key property: `prefill_this_iter` can be less than `remaining_prefill`, capturing the "partial" prefill. the next iteration the scheduler picks up where it left off.

## comparison with disaggregated prefill {#vs-disaggregated}

chunked prefill is the **in-place** solution to prefill-decode interference: both still share the same GPU, just interleaved more carefully.

**disaggregated prefill** takes the more radical approach: route prefill to separate machines entirely, so decode GPUs never see prefill traffic at all.

| dimension | chunked prefill | disaggregated prefill |
|---|---|---|
| hardware requirement | single GPU / node | separate prefill + decode pools |
| TPOT | significantly improved | optimal (zero interference) |
| TTFT | slightly increased (chunking adds iterations) | significantly better (dedicated prefill resources) |
| network overhead | none | KV cache migration across nodes |
| implementation complexity | low (scheduler-only change) | high (cluster coordination) |
| when to use | general production serving | large-scale, SLO-critical deployments |

a full treatment of disaggregated prefill is in the [next post]({{< relref "disaggregated-prefill" >}}).

## summary {#summary}

chunked prefill is arguably the best-value optimization in LLM serving:

- **zero FLOPs overhead** — chunking distributes the same work, not more work
- **negligible IO overhead** — ~32 μs of KV writes per chunk vs 5 ms iteration time
- **straightforward implementation** — only the scheduler changes; the attention kernel is unmodified
- **significant TPOT improvement** — decode requests no longer stall for long prefills
- **composable** — works naturally with prefix caching (skip cached chunks), paged attention (KV written block by block), and continuous batching (same iteration-level loop)

the only cost is a modest increase in TTFT proportional to \\(C_{\text{chunk}} \times \lceil L/C \rceil\\), which in practice is well within acceptable bounds.
