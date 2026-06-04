
这个系列回答“推理引擎为什么长这样”。重点不是框架 API，而是 vLLM / SGLang 这类 serving engine 背后的核心机制：prefill/decode 分离、KV cache、PagedAttention、continuous batching、prefix caching、chunked prefill 和 disaggregated prefill。

## 已有文章 {#existing-posts}

建议按这个顺序读：

1. [如何估算 LLM 训练和推理需要多少算力与显存]({{< relref "/posts/llm-flops-memory-estimation" >}})
2. [从绝对位置编码到 RoPE：位置为什么可以被旋转表示]({{< relref "/posts/positional-encoding-to-rope" >}})
3. [LLM 推理中为什么 K、V 可以被缓存]({{< relref "/posts/kv-cache" >}})
4. [Paged Attention：GPU 上的虚拟内存]({{< relref "/posts/paged-attention" >}})
5. [Continuous Batching：按迭代粒度调度]({{< relref "/posts/continuous-batching" >}})
6. [Chunked Prefill：把 Prefill 切片，保护 Decode 延迟]({{< relref "/posts/chunked-prefill" >}})
7. [Prefix Caching：跨请求复用 KV Cache]({{< relref "/posts/prefix-caching" >}})
8. [Disaggregated Prefill：把计算拆到不同机器上]({{< relref "/posts/disaggregated-prefill" >}})

## 后续文章 {#planned-posts}

- Prefill vs Decode：为什么同一个模型有两个完全不同的瓶颈
- Scheduler 的真实目标函数：不是 batch 越大越好
- KV Cache Eviction：LRU、prefix tree、引用计数和缓存污染

## 每篇文章要回答的问题 {#questions}

- 这个机制解决什么生产问题？
- 它主要影响 TTFT、TPOT、throughput，还是显存容量？
- 它如何改变 KV cache、scheduler、attention kernel 或 GPU workload？
- 它和 vLLM / SGLang 中的哪个设计或参数对应？
