
## the batching problem {#batching-problem}

batching is how an LLM serving system keeps a GPU busy. one request rarely has enough work to saturate the device; many requests together turn small matrix operations into larger ones. the catch is that requests do not finish at the same time.

before continuous batching, serving systems often used **static batching**: collect a group of requests, run them together, and wait until every request in the group finishes before admitting the next group.

```text
time ------------------------------------------------------------>

Req A  [==Prefill==][D][D][D][D][D]  done after 5 decode steps
Req B  [==Prefill==][D][D][D][D][D][D][D][D][D][D]
Req C  [==Prefill==][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D]

Batch  |------------------ must wait for Req C ------------------|
                     A slot idle              B slot idle
```

{{< figure src="/images/posts/continuous-batching/static-vs-continuous.svg" caption="<span class=\"figure-number\">Figure 1: </span>static batching (left) leaves GPU slots idle after short requests finish; continuous batching (right) inserts new requests as soon as slots open." width="100%" >}}

the short requests release useful capacity early, but static batching cannot reuse that capacity until the longest request ends. the root cause is a granularity mismatch:

- static batching schedules at the **batch** level
- autoregressive inference naturally advances at the **iteration** level
- each decode iteration produces one new token per active request

the scheduler should therefore make a decision after each iteration, not after each whole response.

## iteration-level scheduling {#iteration-level-scheduling}

**continuous batching** is also called *iteration-level scheduling* or *in-flight batching*. the idea, popularized by Orca, is simple: after every forward pass, remove finished requests and immediately admit new requests into freed slots.

the core loop becomes:

```python
while True:
    batch = scheduler.schedule()       # choose active requests for this iteration
    outputs = model.forward(batch)

    for req, token in zip(batch, outputs):
        req.append(token)
        if token.is_eos or req.at_max_len:
            scheduler.finish(req)      # free KV blocks
        else:
            scheduler.continue_(req)   # keep it active for the next iteration
```

a request returns its [paged KV cache]({{< relref "paged-attention" >}}) blocks to the free pool only when it finishes, is cancelled, or hits its maximum length. active decode requests keep their historical KV because every decode step still needs that cache.

```text
time ------------------------------------------------------------>

Req A  [Pre][D][D][D][D][Done]
Req B  [Pre][D][D][D][D][D][D][D][Done]
Req C            [Pre][D][D][D][Done]       inserted when A finishes
Req D                     [Pre][D][D][D]    inserted when C finishes

       | iter | iter | iter | iter |
       every iteration: finish old work, admit new work
```

static batching asks, "is the current batch done?" continuous batching asks, "given the current memory and token budget, what work should fill the next iteration?" that is why utilization improves: the scheduler keeps replenishing the active set instead of leaving completed slots empty.

## packing prefill and decode together {#packing-prefill-decode}

continuous batching means **rescheduling the active set after every forward pass**. one iteration can contain:

- **prefill**: a prompt segment from a new request, either the whole prompt or one [chunk]({{< relref "chunked-prefill" >}}) of it
- **decode**: one generated token from an existing request, used to predict the next token

with prefill/decode disaggregation, prefill workers and decode workers may maintain separate continuous batching queues. here we focus on the case that exposes the mechanism most clearly: one forward pass contains both prefill rows and decode rows.

suppose one iteration contains:

- request A: 3 prefill tokens
- request B: 1 decode token
- request C: 2 prefill tokens

the system packs the new tokens for this iteration into one flat input:

$$X = [t_1^A, t_2^A, t_3^A, t_t^B, t_1^C, t_2^C]$$

then it runs the projections once:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

"flat" describes the physical layout only. the rows do not become one long text sequence. each row still carries its request id, position id, and sequence boundary:

| packed index | token | request | position id |
|---:|---|---|---:|
| 0 | `t1(A)` | A | 1 |
| 1 | `t2(A)` | A | 2 |
| 2 | `t3(A)` | A | 3 |
| 3 | `t(B,t)` | B | `t` |
| 4 | `t1(C)` | C | 1 |
| 5 | `t2(C)` | C | 2 |

RoPE or learned positional embeddings see each request's own position ids. A's third token may sit next to B's token in memory, but they are still different sequences semantically.

{{< figure src="/images/posts/continuous-batching/mixed-forward-flow.svg" caption="<span class=\"figure-number\">Figure 2: </span>the core path of one mixed iteration: the scheduler chooses new tokens, linear layers process them as one packed matrix, and attention restores per-request context using boundaries, position ids, masks, and KV cache lookups." width="100%" >}}

### preventing cross-request attention {#block-diagonal-mask}

packing alone would be wrong. request B must not attend to request A's prompt. the fix is a **block-diagonal causal mask**: causal attention is allowed inside each request, and all cross-request positions are blocked.

for positions `i` and `j` in the packed sequence:

$$M_{ij}=0\ \text{when req}(i)=\text{req}(j)\ \text{and}\ j\le i,\quad M_{ij}=-\infty\ \text{otherwise}$$

{{< figure src="/images/posts/continuous-batching/packing-mask.svg" caption="<span class=\"figure-number\">Figure 3: </span>tokens from requests A, B, and C are packed into one flat sequence. the block-diagonal causal mask lets each request attend only to its own prefix." width="100%" >}}

the result is equivalent to running attention separately per request, but it uses one larger kernel launch and one packed representation. in practice, FlashAttention-style variable-length interfaces take cumulative sequence lengths (`cu_seqlens`) and enforce these boundaries inside the attention kernel.

### how this interacts with the KV cache {#kv-cache-interaction}

packing describes the new tokens computed in this iteration; historical context comes from the KV cache. the important split is:

- **linear layers** only see the packed rows for this iteration, so prefill rows and decode rows can share one `X`.
- **attention** uses request id, position id, sequence boundary, causal mask, and the KV block table to select the visible keys for each query row.

{{< figure src="/images/posts/continuous-batching/packed-forward-kv-lifecycle.svg" caption="<span class=\"figure-number\">Figure 4: </span>one packed forward pass: the scheduler picks new tokens, linear layers run a shared packed matmul, and attention reads/writes the right KV cache entries for each request before sampling selected logits." width="100%" >}}

for a prefill chunk, the prompt tokens in that chunk are new because their KV entries have not been written yet. with chunked prefill, later chunks read KV written by earlier chunks and append the new KV from the current chunk.

for decode, there is usually one query row per active request. for request B:

$$\operatorname{Attn}^{B,t}=\operatorname{softmax}\left(q^{B,t}\left[K^{B,\mathrm{cache}};k^{B,t}\right]^{T}/\sqrt{d^{\mathrm{k}}}\right)\left[V^{B,\mathrm{cache}};v^{B,t}\right]$$

this formula describes B from the perspective of one query row. it does not mean the system runs a separate tiny kernel for B. many decode rows and prefill rows still live in the same packed batch; the attention kernel looks up each row's own KV block table.

prefill also produces the first output token. a decoder-only transformer computes hidden states and logits for every prompt position, but serving systems usually sample from only the final prompt token:

$$\text{first output token}=\operatorname{sample}\left(\operatorname{logits}(h^{\mathrm{last\ prompt}})\right)$$

if the prompt is split into chunks, only the final chunk samples the first output token; earlier chunks only fill the KV cache.

### one complete mixed iteration {#mixed-iteration-example}

take a tiny example with hidden size `d`. this iteration contains:

- request A: a new prompt chunk with 3 tokens and no KV cache yet
- request B: a prompt of length 4; prefill has already sampled the first output token `B5`, and this iteration decodes from `B5` to predict `B6`
- request C: a new prompt chunk with 2 tokens and no KV cache yet

there are `3 + 1 + 2 = 6` new tokens, so after embedding:

$$X\in\mathbb{R}^{6\times d},\quad Q,K,V\in\mathbb{R}^{6\times d}$$

attention is split by request:

| request | new tokens this iteration | position id | visible context |
|---|---:|---|---|
| A | 3 | 1, 2, 3 | A's 3 prompt tokens, causal |
| B | 1 | 5 | B's prompt cache `B1..B4` + current token `B5`, used to predict `B6` |
| C | 2 | 1, 2 | C's 2 prompt tokens, causal |

the forward pass is:

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

B's query never sees A or C. position id `5` means `B1..B4` are already cached and `B5` is the current decode input; attention over `B1..B5` produces the hidden state whose logits predict `B6`.

### why packed forward is equivalent {#decoder-only-correctness}

correctness depends on one condition:

> packed row `i` must see exactly the same context it would see if its request ran alone.

linear layers are row-wise:

$$q^{i}=x^{i}W^{Q},\quad k^{i}=x^{i}W^{K},\quad v^{i}=x^{i}W^{V}$$

the visible set for row `i` is:

$$S(i)=\lbrace\text{same request, earlier-or-current positions}\rbrace\cup\lbrace\text{that request's KV cache}\rbrace$$

rows from other requests are not in `S(i)`; implementation-wise, the mask gives them negative infinity. MLP, residual connections, and layer norm are also row-wise, so they do not mix requests.

therefore, as long as position ids, sequence boundaries, causal masks, and KV-cache indices are correct, packed forward is equivalent to running each request separately and concatenating the results. the system shares physical computation, not semantic context.

## token budgets and latency {#token-budgets-latency}

continuous batching keeps the GPU busy by maintaining a **token budget**: a cap on active KV cache usage. if GPU memory is `M`, model weights occupy `W`, and each cached token costs `k` bytes, then:

$$N_{\max} = \frac{M-W}{k}$$

the scheduler tries to keep:

$$\sum_{\text{active req}} L_{\text{req}} \approx N_{\max}$$

when a request frees `ΔN` token slots, the scheduler admits new work that fits into those slots. this is why continuous batching pairs naturally with [paged attention]({{< relref "paged-attention" >}}): memory can be freed and reused at block granularity.

two granularities matter:

- **prompt chunk** is a scheduling unit: how many prompt tokens a prefill request may bring into this iteration.
- **paged KV block** is a memory allocation unit: token KV entries are stored in fixed-size blocks until the request finishes, is cancelled, or reaches its limit.

so, when prefill and decode are packed together, only the **new tokens for this iteration** are packed. this does not imply the full prompt is processed at once, and it does not allocate KV for all future output upfront.

### why decode is bandwidth-bound {#decode-bandwidth}

decode produces one token at a time, but each step must read the request's full KV history. for a model with `L` layers, `n_h` KV heads, head dimension `d_h`, and a given dtype:

$$\text{bytes per token}=2 \times L \times n_h \times d_h \times \text{sizeof(dtype)}$$

for LLaMA-3 8B with `L=32`, `n_h=8` GQA KV heads, `d_h=128`, and BF16:

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB per token}$$

a 4096-token context therefore needs about 512 MB of KV data. each decode step streams that cache from HBM, so the bottleneck is often memory bandwidth rather than tensor-core compute.

### TTFT and TPOT {#ttft-tpot}

two latency metrics matter:

| metric | meaning |
|---|---|
| **TTFT** | time to first token |
| **TPOT** | time per output token |

continuous batching mostly improves throughput and utilization. TTFT may increase when a new request waits behind a full token budget. TPOT for one isolated request may not change much, but aggregate TPOT improves because GPU slots are rarely idle.

## the next bottleneck {#next-bottleneck}

continuous batching decides **when** to admit new requests. it does not decide **how much work** a newly admitted request may bring into one iteration.

if a new request has a long prompt, a 2048-token prefill can monopolize an iteration for hundreds of milliseconds. active decode requests wait during that iteration, so their TPOT spikes.

this is **prefill-decode interference**:

- prefill is compute-heavy and likes large chunks
- decode is latency-sensitive and wants short, frequent iterations
- continuous batching places both in the same scheduling loop

the next step is [chunked prefill]({{< relref "chunked-prefill" >}}): split long prefill work across iterations so decode requests keep making progress.
