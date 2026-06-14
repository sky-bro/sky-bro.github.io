
LLM inference looks like one operation: send a prompt, get tokens back. under the hood it is two workloads sharing the same model weights.

**prefill** processes the input prompt and builds the initial KV cache. **decode** generates new tokens one step at a time while reading that cache. the weights are the same, but the hardware bottleneck is not. prefill behaves like a large batched matrix multiplication problem; decode behaves like a stream of small queries repeatedly reading a growing memory table.

that split explains why serving engines talk about **TTFT** (time to first token) and **TPOT** (time per output token) separately, why [continuous batching]({{< relref "continuous-batching" >}}) mixes prefill and decode in one iteration, why [chunked prefill]({{< relref "chunked-prefill" >}}) exists, and why [disaggregated prefill]({{< relref "disaggregated-prefill" >}}) is a natural production architecture.

{{< figure src="/images/posts/prefill-vs-decode/two-bottlenecks.svg" caption="<span class=\"figure-number\">Figure 1: </span>prefill runs many prompt tokens through dense GEMM and mainly controls TTFT. decode advances one new token per request, repeatedly reads the KV cache, and mainly controls TPOT." width="100%" >}}

## the smallest useful example {#small-example}

use a tiny decoder-only Transformer request:

- prompt length: 4 tokens, `A B C D`
- output length: 3 tokens, `x y z`
- one attention layer, one KV head, tiny hidden size

the request has two phases:

```text
prefill:
  input:  A B C D
  output: logits for D, plus KV cache for A B C D

decode step 1:
  input:  x
  read:   KV(A B C D)
  output: logits for x, append KV(x)

decode step 2:
  input:  y
  read:   KV(A B C D x)
  output: logits for y, append KV(y)

decode step 3:
  input:  z
  read:   KV(A B C D x y)
  output: logits for z, append KV(z)
```

the important asymmetry is not that prefill uses the prompt and decode uses generated text. the important asymmetry is **how many new tokens enter the model in one forward pass**.

in prefill, all prompt tokens are new. if the prompt length is \\(S\\), the hidden state matrix is shaped like:

$$
X_{\text{prefill}} \in \mathbb{R}^{S \times d}
$$

linear layers therefore look like matrix-matrix multiplication:

$$
Y = X_{\text{prefill}} W,\quad
X_{\text{prefill}} \in \mathbb{R}^{S \times d},\ W \in \mathbb{R}^{d \times d}
$$

in decode, each request contributes only one new token per step:

$$
x_{\text{decode}} \in \mathbb{R}^{1 \times d}
$$

the same linear layer becomes a matrix-vector-like operation:

$$
y = x_{\text{decode}} W
$$

with batching, decode is not literally a single vector; it is \\(B\\) one-token rows from \\(B\\) active requests. but it is still much smaller along the sequence dimension than prefill. that shape difference changes the GPU bottleneck.

## why prefill likes compute {#prefill-compute}

prefill has many tokens available at once. that gives the GPU large dense matrices:

```text
prompt tokens S = 2048
hidden size   d = 4096

X: 2048 x 4096
W: 4096 x 4096
Y: 2048 x 4096
```

the weight matrix \\(W\\) is loaded, then reused across many prompt-token rows. tensor cores can run large GEMMs efficiently because there is enough work to amortize memory movement and scheduling overhead.

the same is true for the attention score matrix during prefill. with causal masking, token \\(i\\) can only attend to tokens \\(j \leq i\\), but the computation still has many queries and many keys available in one pass:

$$
Q_{\text{prefill}} K_{\text{prefill}}^\top
\in \mathbb{R}^{S \times S}
$$

that is why prefill is often described as **compute-bound**. the bottleneck is close to how fast the GPU can multiply large matrices. if the prompt is long, prefill can be expensive, but it is also the phase where batching and tensor-core utilization are easiest.

operationally, prefill mainly controls **TTFT**:

```text
request arrives
  -> wait in queue
  -> run prefill
  -> sample first token
  -> user sees first token
```

if prefill is slow, the user waits longer before seeing anything. prefix caching, prompt compression, chunked prefill tuning, and disaggregated prefill all affect this path.

## why decode likes memory bandwidth {#decode-bandwidth}

decode has the opposite shape. each active request adds one new query, but that query must attend to the entire history.

for a single request at context length \\(T\\):

$$
q_{\text{new}} K_{\leq T}^{\top}
\in \mathbb{R}^{1 \times T}
$$

the query is tiny. the KV cache is not. every layer needs to read the historical keys and values for that request. with \\(L\\) layers, \\(n_{kv}\\) KV heads, head dimension \\(d_h\\), dtype size \\(b\\) bytes, and context length \\(T\\), the KV bytes for one request are approximately:

$$
\text{KV bytes} = 2 \times L \times n_{kv} \times d_h \times T \times b
$$

the factor 2 is for keys and values. for a model with 32 layers, 32 KV heads, head dimension 128, FP16 cache, and a 4096-token context:

$$
2 \times 32 \times 32 \times 128 \times 4096 \times 2
\approx 2\ \text{GB}
$$

each decode step does not necessarily move all of that from HBM in the most naive way because kernels and caches matter, but the order of magnitude shows the problem: decode spends a lot of time reading historical KV data. the amount read grows with context length and concurrency.

that is why decode is often **memory-bandwidth-bound**. the GPU may have plenty of FLOPs left, but it cannot generate the next token until the relevant KV cache data is available.

operationally, decode mainly controls **TPOT**:

```text
after first token:
  decode step
  sample token
  decode step
  sample token
  ...
```

if decode is slow, streaming output feels sluggish. if decode has jitter, the stream pauses irregularly.

## the same model wants two different batch shapes {#batch-shapes}

prefill and decode both benefit from batching, but not in the same way.

| dimension | prefill | decode |
|---|---|---|
| new tokens per request | many prompt tokens | one generated token |
| main matrix shape | large GEMM | small GEMM/GEMV-like work |
| attention input | many queries over a prompt | one query over long history |
| common bottleneck | tensor-core compute | HBM bandwidth and KV cache layout |
| user metric | TTFT | TPOT |
| serving pressure | admit new work quickly | keep active streams smooth |

this creates a scheduler conflict. a large prefill wants to run as a big chunk because that maximizes compute efficiency and lowers TTFT for the new request. active decode requests want short, frequent iterations because each user is waiting for the next streamed token.

if the scheduler runs a 2048-token prefill in one large forward pass, it may be efficient for that request, but all existing decode requests wait. their TPOT spikes. if the scheduler always prioritizes decode, existing streams stay smooth, but new requests wait longer before their first token.

this is the basic **prefill-decode interference** problem:

```text
large prefill:
  good tensor-core utilization
  good TTFT for the new request
  bad TPOT jitter for running decode requests

decode-first scheduling:
  good stream smoothness
  worse queueing and TTFT for new requests
```

there is no single "best batch size" without a workload. a chat workload with short prompts and long outputs is decode-heavy. a retrieval or agent workload with long prompts and short answers is prefill-heavy. a code assistant with a long shared system prompt may become prefill-light only if prefix caching works.

## what this means for serving engines {#serving-engines}

once the two bottlenecks are separate, many serving-engine features become easier to place.

**KV cache** is the boundary between phases. prefill writes the initial cache; decode repeatedly reads and appends to it. this is why cache layout, block allocation, and eviction are serving-engine concerns rather than just model concerns.

**PagedAttention** makes decode feasible under high concurrency. decode is memory intensive, so wasting memory through fragmentation directly reduces the number of active streams the system can serve.

**Continuous batching** fills every iteration with a mixture of work. the scheduler does not batch whole requests; it batches the next unit of model work: prompt tokens from prefill requests and one-token steps from decode requests.

**Chunked prefill** slices long prefill work so that decode can keep making progress. it intentionally trades a small TTFT increase for lower TPOT jitter.

**Prefix caching** removes repeated prefill work for shared prefixes. it improves TTFT, but it does not make decode cheaper after the request enters the generation phase.

**Disaggregated prefill** is the large-scale version of the same idea: if prefill and decode want different hardware operating points, put them on different workers and transfer KV cache between them.

## a profiling checklist {#profiling-checklist}

when a serving benchmark is slow, first ask which phase is slow.

| symptom | likely phase | first things to inspect |
|---|---|---|
| high TTFT, normal TPOT | prefill or queueing | prompt length, prefix-cache hit rate, prefill batch size, queue delay |
| normal TTFT, high TPOT | decode | KV cache size, memory bandwidth, active batch size, attention kernel |
| smooth early tokens, then slowdown | decode with growing context | context length, KV cache layout, KV quantization, eviction pressure |
| periodic stream stalls | prefill-decode interference | long prompt admissions, chunked prefill settings, token budget |
| high GPU memory, low throughput | decode capacity limit | max concurrent sequences, block fragmentation, KV dtype |

the key is not to say "the model is slow." the serving system is a pipeline. prefill and decode stress different parts of the pipeline, so the fix has to match the phase.

## conclusion {#conclusion}

prefill and decode use the same Transformer weights, but they are not the same workload.

- prefill has many new tokens, large GEMMs, high compute utilization, and mostly affects TTFT
- decode has one new token per request, repeated KV cache reads, memory-bandwidth pressure, and mostly affects TPOT
- scheduling is hard because prefill wants large chunks while decode wants frequent short iterations

this is the mental model behind the rest of the inference-engine stack. once you see prefill and decode as two bottlenecks, the designs of continuous batching, chunked prefill, prefix caching, PagedAttention, and disaggregated prefill stop looking like separate tricks. they are different ways of controlling where work sits between compute, memory, and latency.
