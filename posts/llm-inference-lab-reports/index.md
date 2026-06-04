
This series is for experiment reports. Unlike mechanism explainers or source-reading notes, each post should include a reproducible environment, commands, metrics, tables or figures, and concrete tuning conclusions.

For inference-engine interviews, knowing the names PagedAttention, prefix cache, and chunked prefill is only the first layer. The stronger signal is being able to answer: which workload benefits, how much did the metric improve, where did the bottleneck move, and what should we inspect first if production metrics regress?

## Experiment Order {#experiment-order}

The planned order is:

1. Build a vLLM / SGLang benchmark environment
2. Experiment: how batch size and `max_num_batched_tokens` change throughput and latency
3. Experiment: how prefix cache hit rate changes TTFT
4. Experiment: tuning chunk size for chunked prefill
5. Experiment: PagedAttention and memory fragmentation
6. Experiment: the memory, speed, and quality triangle for quantized models
7. Final project: an inference-service profiler dashboard for TTFT, TPOT, cache hit rate, memory watermark, and tuning suggestions

## Standard Report Format {#format}

Each lab report should include:

- **Question**: what hypothesis is this experiment testing?
- **Environment**: GPU, driver, CUDA, model, framework version, and launch parameters.
- **Workload**: prompt length, output length, concurrency, request distribution, and whether prefixes are shared.
- **Metrics**: TTFT, TPOT, throughput, memory watermark, cache hit rate, and GPU utilization.
- **Results**: tables or figures that show the key changes.
- **Explanation**: connect the result back to prefill, decode, KV cache, scheduler, or kernels.
- **Conclusion**: what should change in the next deployment or tuning pass?

Without these details, a post is mostly a learning note. With them, it becomes evidence of engineering judgment.
