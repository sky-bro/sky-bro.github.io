
## the batching problem {#batching-problem}

batching is how an LLM serving system keeps a GPU busy. one request rarely has enough work to saturate the device, but many requests together can turn small matrix operations into large ones. the catch is that requests do not finish at the same time.

before continuous batching, serving systems often used **static batching**: collect a group of requests, run them together, and wait until every request in the group finishes before admitting the next group.

```text
time ------------------------------------------------------------>

Req A  [==Prefill==][D][D][D][D][D]  done after 5 decode steps
Req B  [==Prefill==][D][D][D][D][D][D][D][D][D][D]
Req C  [==Prefill==][D][D][D][D][D][D][D][D][D][D][D][D][D][D][D]

Batch  |------------------ must wait for Req C ------------------|
                     A slot idle              B slot idle
```

{{< figure src="/images/posts/continuous-batching/static-vs-continuous.svg" caption="<span class=\"figure-number\">Figure 1: </span>static batching (left) idles GPU slots after short requests finish; continuous batching (right) inserts new requests the moment a slot opens, sustaining high GPU utilization." width="100%" >}}

the short requests release useful capacity early, but static batching cannot reuse that capacity until the longest request ends. measured on real workloads, this can leave utilization around **30-50%** even though there is a queue of waiting work.

the root cause is a mismatch of granularity:

- static batching schedules at the **batch** level
- autoregressive inference naturally advances at the **iteration** level
- each decode iteration produces exactly one new token per active request

the scheduler should therefore make a scheduling decision after each iteration, not after each whole response.

## iteration-level scheduling {#iteration-level-scheduling}

**continuous batching** is also called *iteration-level scheduling* or *in-flight batching*. the idea, popularized by Orca, is simple:

> after every forward pass, remove finished requests and immediately admit new requests into the freed slots.

the core loop becomes:

```python
while True:
    batch = scheduler.schedule()       # choose active requests for this iteration
    outputs = model.forward(batch)

    for req, token in zip(batch, outputs):
        req.append(token)
        if token.is_eos or req.at_max_len:
            scheduler.finish(req)      # free KV blocks immediately
        else:
            scheduler.continue_(req)   # keep it active for the next iteration
```

when a request finishes, its [paged KV cache]({{< relref "paged-attention" >}}) blocks are returned to the free pool. a waiting request can enter on the very next forward pass.

```text
time ------------------------------------------------------------>

Req A  [Pre][D][D][D][D][Done]
Req B  [Pre][D][D][D][D][D][D][D][Done]
Req C            [Pre][D][D][D][Done]       inserted when A finishes
Req D                     [Pre][D][D][D]    inserted when C finishes

       | iter | iter | iter | iter |
       every iteration: finish old work, admit new work
```

this changes the invariant the system tries to maintain. static batching asks, "is the current batch done?" continuous batching asks, "given the current memory and token budget, what work should fill the next iteration?"

that small change is why utilization improves: the scheduler keeps replenishing the active set instead of letting completed slots sit empty.

## packing prefill and decode together {#packing-prefill-decode}

continuous batching creates a less obvious question. a single iteration can contain different kinds of work:

- **prefill**: a new request needs to process many prompt tokens, or one [chunk]({{< relref "chunked-prefill" >}}) of a long prompt
- **decode**: an existing request contributes exactly one newly generated token

these sequence lengths differ. padding every request to the longest sequence length would waste most of the computation, so serving systems instead **pack** the new tokens into one flat sequence.

for example, suppose one iteration contains:

- request A: 3 prefill tokens
- request B: 1 decode token
- request C: 2 prefill tokens

the model can form one packed input:

$$X = [t_1^A, t_2^A, t_3^A, t_t^B, t_1^C, t_2^C]$$

the projection matrices run once over this packed sequence:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

no padding tokens enter the matrix multiply.

{{< alert theme="info" >}}

for a prefill request, the prompt tokens are "new" because their KV entries do not exist yet. for a decode request, only the latest generated token is new; the historical tokens already live in the KV cache.

{{< /alert >}}

### preventing cross-request attention {#block-diagonal-mask}

packing alone would be wrong: request B must not attend to request A's prompt. the fix is a **block-diagonal causal mask**. inside each request block, causal attention is allowed. across different request blocks, attention is blocked with negative infinity.

for positions `i` and `j` in the packed sequence:

$$M_{ij}=0\ \text{when req}(i)=\text{req}(j)\ \text{and}\ j\le i,\quad M_{ij}=-\infty\ \text{otherwise}$$

{{< figure src="/images/posts/continuous-batching/packing-mask.svg" caption="<span class=\"figure-number\">Figure 2: </span>tokens from requests A, B, and C are packed into one flat sequence. the block-diagonal causal mask lets each request attend only to its own prefix, so cross-request entries vanish after softmax." width="100%" >}}

the result is mathematically equivalent to running attention separately per request, but it uses one larger kernel launch and one packed representation.

| approach | matmul shape | waste |
|---|---|---|
| separate requests | many small operations | poor GPU occupancy |
| padded batch | one large padded operation | pad-token compute |
| packed batch | one large useful operation | minimal |

in practice, FlashAttention-style variable-length interfaces take cumulative sequence lengths (`cu_seqlens`) and apply this masking pattern inside the attention kernel.

### how this interacts with the KV cache {#kv-cache-interaction}

packing describes the new tokens processed in the current iteration. the historical context comes from the KV cache.

for a prefill request, the new keys and values are written into the cache and attention is computed causally over the prompt chunk. for a decode request, the new query attends over both the cached history and the newly appended key:

$$\operatorname{attn}_t^B=\operatorname{softmax}\left(q_t^B [K_{\text{cache}}^B; k_t^B]^T / \sqrt{d_k}\right)[V_{\text{cache}}^B; v_t^B]$$

the cached keys and values do not re-enter the packed input. they are read from the paged KV cache by the attention kernel.

this is the key asymmetry: prefill is mostly matrix-matrix work; decode is mostly one query reading a long KV history.

## token budgets and latency {#token-budgets-latency}

continuous batching keeps the GPU busy by maintaining a **token budget**: a cap on the amount of KV cache the active set may occupy.

if GPU memory is \(M\), model weights occupy \(W\), and each cached token costs \(k\) bytes, then the rough upper bound is:

$$N_{\max} = \frac{M-W}{k}$$

the scheduler tries to keep:

$$\sum_{\text{active req}} L_{\text{req}} \approx N_{\max}$$

when a request finishes and frees \(\Delta N\) token slots, the scheduler admits new work that fits into those slots. this is why continuous batching pairs naturally with [paged attention]({{< relref "paged-attention" >}}): memory can be freed and reused at block granularity instead of being tied to a fixed contiguous allocation.

### why decode is bandwidth-bound {#decode-bandwidth}

decode looks cheap because it produces one token, but each decode step must read the request's entire KV history. for a model with \(L\) layers, \(n_h\) KV heads, head dimension \(d_h\), and a given dtype:

$$\text{bytes per token}=2 \times L \times n_h \times d_h \times \text{sizeof(dtype)}$$

for LLaMA-3 8B with \(L=32\), \(n_h=8\) GQA KV heads, \(d_h=128\), and BF16:

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB per token}$$

a 4096-token context therefore needs about 512 MB of KV data. each decode step streams that cache from HBM, so the bottleneck is often memory bandwidth rather than tensor-core compute.

### TTFT and TPOT {#ttft-tpot}

two latency metrics matter:

| metric | meaning |
|---|---|
| **TTFT** | time to first token |
| **TPOT** | time per output token |

continuous batching mostly improves throughput and utilization. TTFT may increase when the scheduler queues a new request behind a full token budget. TPOT for one isolated request may not change much, but aggregate TPOT improves because GPU slots are rarely idle.

## the next bottleneck {#next-bottleneck}

continuous batching solves **when** to admit new requests. it does not by itself solve **how much work** a newly admitted request is allowed to bring into one iteration.

this matters when a new request has a long prompt. a 2048-token prefill can monopolize one iteration for hundreds of milliseconds. during that iteration, active decode requests wait, so their TPOT spikes.

this is **prefill-decode interference**:

- prefill is compute-heavy and likes large chunks
- decode is latency-sensitive and wants frequent short iterations
- continuous batching places them in the same scheduling loop

the next step is [chunked prefill]({{< relref "chunked-prefill" >}}): slice a long prefill across several iterations so decode requests keep making progress.
