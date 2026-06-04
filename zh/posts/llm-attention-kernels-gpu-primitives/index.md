
这个系列专门放 kernel 和 GPU 基元。它和推理引擎机制系列的区别是：机制系列解释“系统为什么需要这个优化”，这里解释“这个优化在 kernel 和内存访问层面如何实现”。

## 已有文章 {#existing-posts}

1. [Triton 中的融合 Softmax]({{< relref "/posts/fused-softmax" >}})
2. [Online Softmax：为任意大行设计的分块算法]({{< relref "/posts/online-softmax" >}})

## 后续文章 {#planned-posts}

- FlashAttention：online softmax 如何变成 IO-aware attention
- FlashAttention 到 PagedAttention：attention kernel 和 cache layout 如何互相限制
- PagedAttention kernel：block table 如何进入 attention 访存路径
- Triton profiling：用 roofline 看 bandwidth-bound 和 compute-bound
- Decode kernel 为什么更容易被 HBM 带宽限制

## 每篇文章要回答的问题 {#questions}

- 这个 kernel 主要省了哪类内存访问？
- 数据在 HBM、L2、shared memory、register 之间如何移动？
- 它改善的是 prefill、decode，还是两者都改善？
- 它和 vLLM / SGLang 的 serving 参数或 cache layout 有什么耦合？
