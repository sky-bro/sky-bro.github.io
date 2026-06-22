
## 先给结论 {#quick-answer}

模型里不是只有浮点类型，也会用整数类型。最重要的区别不是“浮点 vs 整数”，而是这个类型用在什么位置：

| 位置 | 常见类型 | 作用 |
| --- | --- | --- |
| 训练计算 | FP32、TF32、FP16、BF16 | 保持梯度和激活稳定，同时利用 Tensor Core |
| 推理计算 | BF16、FP16、FP8、INT8 | 降低带宽和计算成本 |
| 权重存储 | BF16、FP16、FP8、INT8、INT4、NF4 | 减少模型文件和显存占用 |
| KV cache / activation | BF16、FP16、FP8、INT8 | 长上下文和高并发时省显存 |
| token id / mask / index | INT32、INT64、bool | 表示离散索引，不是量化参数 |

一句话记住：**训练主路径通常是浮点；推理和存储会大量使用低精度浮点和整数；整数如果表示模型数值，通常需要 scale、zero point 或 codebook 才能还原成近似实数。**

比如一个权重 `w = 0.15625`，用 INT8 存时并不是直接存“0.15625 这个整数”，而是类似这样：

```text
scale = 0.01
q = round(w / scale) = 16
dequantized w ~= q * scale = 0.16
```

也就是说，整数只是压缩后的编码；模型计算关心的仍然是解码后的近似实数。

## 浮点：range 和 precision 的交换 {#floating-types}

浮点数可以拆成三块：sign、exponent、mantissa。sign 管正负；exponent 决定数轴能伸多远，也就是 range；mantissa 决定同一数量级附近的刻度有多密，也就是 precision。

{{< figure src="/images/posts/model-numeric-types/range-vs-precision.svg" caption="<span class=\"figure-number\">Figure 1: </span>浮点格式把 bit 分给 sign、exponent 和 mantissa。exponent 决定数轴能伸多远，mantissa 决定同一数量级附近的刻度有多密；FP8 的 E4M3 / E5M2 也是同一个交换。" width="96%" >}}

BF16 和 FP16 都是 16 bit，但分配不同：

- BF16：1 sign + 8 exponent + 7 mantissa。范围接近 FP32，局部精度较粗。
- FP16：1 sign + 5 exponent + 10 mantissa。范围小一些，但同一数量级附近刻度更密。

在 1.0 附近，BF16 的相邻可表示数间距约是 `2^-7 = 0.0078125`；FP16 约是 `2^-10 = 0.0009765625`。所以 FP16 在 1 附近大约比 BF16 密 8 倍，但 BF16 更不容易因为数量级变化而 overflow / underflow。

常见浮点类型可以这样记：

| 类型 | 核心直觉 | 常见用途 |
| --- | --- | --- |
| FP32 | 范围大、精度高，但显存和带宽贵 | optimizer state、master weight、部分 accumulation |
| TF32 | FP32 的范围，较短 mantissa，NVIDIA Tensor Core 友好 | Ampere 之后的 FP32 matmul 加速 |
| FP16 | 局部精度比 BF16 细，范围较小 | mixed precision 训练和推理 |
| BF16 | 范围接近 FP32，局部精度较粗 | 现代 LLM 训练和推理常用 baseline |
| FP8 E4M3 | mantissa 多一点，局部精度好一点 | weight / activation 路径 |
| FP8 E5M2 | exponent 多一点，范围大一点 | gradient 或动态范围更大的张量 |

## 整数：索引和量化不是一回事 {#integer-types}

模型系统里会出现两类整数。

第一类是普通离散数据：token id、position id、attention mask、MoE routing index、embedding lookup index。这些值本来就是索引或控制信息，`token id = 42` 表示词表里的第 42 个 token，不表示一个近似为 42.0 的参数。

第二类才是量化数值：

| 类型 | 常见对象 | 解释方式 | 主要收益 |
| --- | --- | --- | --- |
| INT8 | weight、activation、KV cache | scale / zero point | 省显存和带宽，部分硬件吞吐高 |
| INT4 | weight 为主 | group-wise scale | 大幅降低权重显存 |
| NF4 | QLoRA weight code | 非均匀 codebook | 4 bit 下更适配近似正态分布的权重 |

看到“INT4 模型”时要小心：它通常表示权重以 4 bit 存储，计算时可能在 kernel 内部解码到 FP16/BF16，再用更高精度累加；不等于所有运算都以 4-bit integer 完成。

## 读 dtype 时问三个问题 {#reading-dtype}

不要把 dtype 当成一个单一标签。一个 matmul 至少有三层类型：

| 问题 | 例子 |
| --- | --- |
| storage dtype：张量怎么存？ | weight 存成 INT4，scale 存成 FP16 |
| compute dtype：乘法输入怎么进 kernel？ | activation 是 BF16，weight 在 kernel 里解码 |
| accumulation dtype：乘积怎么累加？ | BF16 multiply + FP32 accumulation |

accumulation 很重要，因为矩阵乘法会把很多乘积加起来。输入可以是低精度，但中间求和常常需要更高精度，否则误差会沿 hidden dimension 累积。

实际读论文、模型卡或 serving 配置时，按这三个问题拆：

1. 这个 dtype 描述的是 weight、activation、KV cache、gradient、optimizer state，还是 token/index？
2. 它是 storage、compute，还是 accumulation？
3. 如果是整数数值，scale / zero point / codebook 在哪里，粒度是 per-tensor、per-channel，还是 per-group？

这样就不容易把 BF16、FP8、INT8、INT4 混成一团。BF16/FP16/FP8 主要是在分配 exponent 和 mantissa；INT8/INT4/NF4 主要是在用更少 bit 近似原来的实数分布；INT32/INT64 token id 只是普通索引。

继续读的话，可以把这篇作为入口：资源估算看[如何估算 LLM 训练和推理需要多少算力与显存]({{< relref "/posts/llm-flops-memory-estimation" >}})，量化误差和 codebook 看[大模型量化综述：从线性量化到码本量化]({{< relref "/posts/llm-quantization" >}})。
