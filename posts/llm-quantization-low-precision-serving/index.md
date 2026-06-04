
This series is for quantization and low-precision serving. It deserves its own track because quantization touches representation, error control, calibration data, kernel support, KV cache, memory bandwidth, and quality regression.

## Existing Posts {#existing-posts}

1. [A Survey of LLM Quantization: From Linear Quantization to Codebooks]({{< relref "/posts/llm-quantization" >}})

## Planned Posts {#planned-posts}

- KV cache quantization: beyond weight memory, the cache is often the real footprint
- FP8 serving: E4M3 / E5M2, activation scales, and Tensor Core paths
- INT4 weight-only serving: why saving memory does not always mean going faster
- GPTQ / AWQ / SmoothQuant engineering boundaries
- NF4 / AQLM: why lower bit widths need codebooks
- Quantization benchmarks: measuring the quality, speed, and memory triangle

## Questions Each Post Should Answer {#questions}

- Are we quantizing weights, activations, KV cache, or a communication/storage format?
- Does the benefit come from capacity, HBM bandwidth, Tensor Core throughput, or disk size?
- Does the error mainly come from outliers, scale granularity, rounding, or clipping?
- How do vLLM / SGLang load, observe, and roll back this choice?
