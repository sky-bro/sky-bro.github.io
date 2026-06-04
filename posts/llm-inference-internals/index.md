
This series answers why inference engines are shaped the way they are. The focus is not framework APIs, but the core mechanisms behind vLLM / SGLang-style serving engines: prefill/decode, KV cache, PagedAttention, continuous batching, prefix caching, chunked prefill, and disaggregated prefill.

## Existing Posts {#existing-posts}

Read the existing posts in this order:

1. [Estimating Compute and Memory Requirements for LLM Training and Inference]({{< relref "/posts/llm-flops-memory-estimation" >}})
2. [From Absolute Positional Encoding to RoPE: Why Position Can Be a Rotation]({{< relref "/posts/positional-encoding-to-rope" >}})
3. [Why KV Cache Works in LLM Inference]({{< relref "/posts/kv-cache" >}})
4. [Paged Attention: Virtual Memory for the GPU]({{< relref "/posts/paged-attention" >}})
5. [Continuous Batching: Scheduling at Iteration Granularity]({{< relref "/posts/continuous-batching" >}})
6. [Chunked Prefill: Slicing the Prefill to Protect Decode Latency]({{< relref "/posts/chunked-prefill" >}})
7. [Prefix Caching: Reusing KV Cache Across Requests]({{< relref "/posts/prefix-caching" >}})
8. [Disaggregated Prefill: Splitting Compute Across Machines]({{< relref "/posts/disaggregated-prefill" >}})

## Planned Posts {#planned-posts}

- Prefill vs decode: why one model has two very different bottlenecks
- The scheduler's real objective: bigger batches are not always better
- KV cache eviction: LRU, prefix trees, reference counts, and cache pollution

## Questions Each Post Should Answer {#questions}

- What production problem does this mechanism solve?
- Does it mainly affect TTFT, TPOT, throughput, or memory capacity?
- How does it change KV cache, scheduler, attention kernels, or GPU workload?
- Which vLLM / SGLang design or parameter does it map to?
