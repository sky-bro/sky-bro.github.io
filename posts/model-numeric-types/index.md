
## The Short Answer {#quick-answer}

Models do not use only floating-point types. Integers appear too. The useful distinction is not simply "float versus int", but where the type is used.

| Location | Common types | Purpose |
| --- | --- | --- |
| Training compute | FP32, TF32, FP16, BF16 | Keep gradients and activations stable while using Tensor Cores |
| Inference compute | BF16, FP16, FP8, INT8 | Reduce bandwidth and compute cost |
| Weight storage | BF16, FP16, FP8, INT8, INT4, NF4 | Shrink model files and GPU memory |
| KV cache / activation | BF16, FP16, FP8, INT8 | Save memory for long context and high concurrency |
| token ids / masks / indices | INT32, INT64, bool | Represent discrete structure, not quantized parameters |

One sentence is enough for the main idea: **training is usually dominated by floating-point compute; inference and storage often use low-precision floating point and integers; when integers represent model values, they usually need a scale, zero point, or codebook to become approximate real numbers again.**

For example, if a weight is `w = 0.15625`, INT8 storage might look like this:

```text
scale = 0.01
q = round(w / scale) = 16
dequantized w ~= q * scale = 0.16
```

The integer is a compact code. The model still cares about the decoded approximate real value.

## Floating Point: Range Versus Precision {#floating-types}

A floating-point value has three parts: sign, exponent, and mantissa. The sign stores positive or negative. The exponent decides how far the number line reaches, or range. The mantissa decides how dense nearby tick marks are, or precision.

{{< figure src="/images/posts/model-numeric-types/range-vs-precision.svg" caption="<span class=\"figure-number\">Figure 1: </span>Floating-point formats split bits into sign, exponent, and mantissa. Exponent decides how far the number line reaches; mantissa decides how dense nearby tick marks are. FP8 E4M3 / E5M2 is the same tradeoff under a tighter budget." width="96%" >}}

BF16 and FP16 are both 16-bit formats, but they spend those bits differently:

- BF16: 1 sign + 8 exponent + 7 mantissa. It keeps an FP32-like range but has coarser local precision.
- FP16: 1 sign + 5 exponent + 10 mantissa. It has less range but denser spacing near the same magnitude.

Near 1.0, BF16's next representable value is about `2^-7 = 0.0078125` away. FP16's step is about `2^-10 = 0.0009765625`. FP16 is therefore roughly 8x denser near 1.0, while BF16 is less likely to overflow or underflow when magnitudes change.

The common floating-point formats are:

| Type | Core intuition | Common use |
| --- | --- | --- |
| FP32 | Large range and high precision, but expensive in memory and bandwidth | optimizer states, master weights, some accumulation paths |
| TF32 | FP32 range with a shorter mantissa, NVIDIA Tensor Core friendly | accelerated FP32 matmul on Ampere and newer GPUs |
| FP16 | finer local precision than BF16, smaller range | mixed precision training and inference |
| BF16 | FP32-like range, coarser local precision | common baseline for modern LLM training and inference |
| FP8 E4M3 | more mantissa, better local precision | weight / activation paths |
| FP8 E5M2 | more exponent, wider range | gradients or tensors with larger dynamic range |

## Integers: Indices Are Not Quantization {#integer-types}

There are two kinds of integers in model systems.

The first kind is ordinary discrete data: token ids, position ids, attention masks, MoE routing indices, and embedding lookup indices. These values are labels or control data. `token id = 42` means the 42nd vocabulary entry, not a model parameter approximately equal to 42.0.

The second kind is quantized numeric data:

| Type | Common target | Interpretation | Main benefit |
| --- | --- | --- | --- |
| INT8 | weights, activations, KV cache | scale / zero point | saves memory and bandwidth; some hardware has high throughput |
| INT4 | mostly weights | group-wise scale | sharply reduces weight memory |
| NF4 | QLoRA weight codes | non-uniform codebook | better 4-bit fit for roughly normal weight distributions |

Be careful with phrases like "INT4 model". They often mean weights are stored in 4 bits. The kernel may still decode those weights into FP16/BF16 and accumulate in higher precision. It does not mean every operation is performed as 4-bit integer arithmetic.

## Three Questions for Any Dtype {#reading-dtype}

Do not treat dtype as one label. A matmul has at least three layers:

| Question | Example |
| --- | --- |
| storage dtype: how is the tensor stored? | weights stored as INT4, scales stored as FP16 |
| compute dtype: what enters the multiply path? | activations are BF16; weights are decoded inside the kernel |
| accumulation dtype: how are products summed? | BF16 multiply with FP32 accumulation |

Accumulation matters because matrix multiplication sums many products. Inputs can be low precision, but the intermediate sum often needs higher precision; otherwise error accumulates across the hidden dimension.

When reading a paper, model card, or serving config, ask three questions:

1. Does this dtype describe weights, activations, KV cache, gradients, optimizer states, or token/index data?
2. Is it storage, compute, or accumulation?
3. If it is integer numeric data, where are the scale, zero point, or codebook? Is the granularity per-tensor, per-channel, or per-group?

This keeps the vocabulary from blending together. BF16, FP16, and FP8 are mainly about allocating exponent and mantissa bits. INT8, INT4, and NF4 are mainly about approximating a real-valued distribution with fewer bits. INT32 and INT64 token ids are ordinary indices.

For follow-up reading, use this post as the entry point: resource estimation is in [Estimating LLM Training and Inference Compute and Memory]({{< relref "/posts/llm-flops-memory-estimation" >}}), while quantization error and codebooks are in [A Survey of LLM Quantization: From Linear Quantization to Codebooks]({{< relref "/posts/llm-quantization" >}}).
