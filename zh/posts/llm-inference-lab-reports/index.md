
这条系列专门放实验报告。它和源码阅读、机制解释的区别是：每篇文章都要有可复现环境、命令、指标、图表或表格，以及明确的调参结论。

推理引擎岗位面试里，“知道 PagedAttention / prefix cache / chunked prefill”只是第一层。更有说服力的是能回答：在什么 workload 下它有效？指标改善多少？瓶颈从哪里转移到了哪里？如果线上指标变差，应该先看什么？

## 实验顺序 {#experiment-order}

建议按这个顺序做：

1. 搭一个 vLLM / SGLang 基准测试环境
2. 实验：batch size 和 `max_num_batched_tokens` 如何影响吞吐与延迟
3. 实验：prefix cache 命中率如何改变 TTFT
4. 实验：chunked prefill 的 chunk size 调参
5. 实验：PagedAttention 显存碎片对比
6. 实验：量化模型的显存、速度、质量三角
7. 最终项目：推理服务 profiler dashboard，展示 TTFT、TPOT、cache hit rate、显存水位和调参建议

## 每篇实验报告的固定格式 {#format}

每篇实验报告都应该包含：

- **问题**：这次实验验证什么假设？
- **环境**：GPU、driver、CUDA、模型、框架版本、启动参数。
- **负载**：prompt 长度、输出长度、并发、请求分布、是否共享前缀。
- **指标**：TTFT、TPOT、throughput、显存水位、cache hit rate、GPU 利用率。
- **结果**：用表格或图说明关键变化。
- **解释**：把结果反推回 prefill、decode、KV cache、scheduler 或 kernel。
- **结论**：下一次部署或调参时应该怎么做。

如果一篇文章没有这些信息，它更像学习笔记；有了这些信息，才是可以拿来证明工程能力的实验报告。
