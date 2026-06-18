
同一个 prompt，为什么把 `temperature` 调低会更稳定，把 `top_p` 调低会更保守，把 `top_k` 调小会更像“只从前几个答案里挑”？这些参数不是三种魔法风格，而是在**下一 token 的概率分布**上做了三类很具体的操作。

本文先用一个 5-token 的小例子建立直觉，再对照 vLLM V1 的 sampler 路径看它们在源码中的位置。

## 先抓住一个 token 的采样过程 {#one-step}

模型每一步 forward 后，不是直接输出文字，而是给词表里每个 token 一个 logit。logit 可以理解成“还没归一化的分数”。假设下一步有 5 个候选 token：

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| logit | 4 | 3 | 2 | 1 | 0 |

如果不做任何过滤，`temperature = 1` 时 softmax 后大约是：

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| probability | 0.636 | 0.234 | 0.086 | 0.032 | 0.012 |

也就是说，A 最可能，但 B/C/D/E 仍然可能被抽到。采样参数控制的就是这张表：先改变概率曲线，再决定哪些 token 可以进入候选池，最后从剩下的概率里抽一个 token。

{{< figure src="/images/posts/llm-sampling-temperature-top-p-top-k/sampling-filter-flow.svg" caption="<span class=\"figure-number\">Figure 1: </span>temperature 改变概率曲线；top-k 按排名截断；top-p 按累计概率质量截断；剩下的 token 重新归一化后再抽样。" width="96%" >}}

## temperature：改变分布的尖锐程度 {#temperature}

temperature 作用在 softmax 之前：\\(p\_i = \exp(z\_i / T) / \sum\_j \exp(z\_j / T)\\)。这里 \\(z\_i\\) 是第 \\(i\\) 个 token 的 logit，\\(T\\) 是 temperature。

- \\(T < 1\\)：高分 token 更突出，分布更尖，输出更稳定。
- \\(T = 1\\)：不额外缩放 logits。
- \\(T > 1\\)：低分 token 被抬起来，分布更平，输出更多样。
- \\(T \approx 0\\)：接近 greedy decoding，基本总选 logit 最大的 token。

关键不是 softmax 变了，而是 **logit 间隔被 temperature 缩放了**。两个 token 的相对差距原本是 \\(z\_a - z\_b\\)，调温后变成 \\((z\_a - z\_b) / T\\)。所以 \\(T > 1\\) 会把差距压小，强 token 没那么强；\\(T < 1\\) 会把差距放大，强 token 更强。

{{< figure src="/images/posts/llm-sampling-temperature-top-p-top-k/temperature-softmax-curve.svg" caption="<span class=\"figure-number\">Figure 2: </span>固定 logits 时，temperature 越大，最高 logit token 的概率会下降，低 logit token 的概率会上升；极限上分布会趋向均匀。" width="92%" >}}

还是刚才的 logits `[4, 3, 2, 1, 0]`：

| temperature | A | B | C | D | E | 直觉 |
|---:|---:|---:|---:|---:|---:|---|
| 0.7 | 0.761 | 0.182 | 0.044 | 0.010 | 0.003 | A 几乎统治候选池 |
| 1.0 | 0.636 | 0.234 | 0.086 | 0.032 | 0.012 | 原始 softmax |
| 1.5 | 0.505 | 0.259 | 0.133 | 0.068 | 0.035 | 尾部 token 更有机会 |

temperature 不直接删除 token。它改变的是**相对概率斜率**：A 和 B 的 logit 差原本是 1，除以 0.7 后有效差距变成 1.43，除以 1.5 后有效差距变成 0.67。softmax 对差距很敏感，所以前者让 A 更容易被抽到，后者把概率让给 B/C/D/E。

这也解释了一个常见误解：`temperature` 高不等于“更聪明”，低也不等于“更正确”。它只是在给采样分布加热或降温。问答、代码、信息抽取通常希望低温；创意写作、改写、头脑风暴可以适当升温。

## top-k：只留下排名前 k 的 token {#top-k}

top-k 是最直观的截断规则：排序后，只允许概率最高的 \\(k\\) 个 token 参与抽样，其余 token 的概率设为 0，然后对剩下的 token 重新归一化。

如果 `top_k = 3`，候选池只剩 A/B/C：

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| 原始概率 | 0.636 | 0.234 | 0.086 | 0.032 | 0.012 |
| top-k mask | keep | keep | keep | drop | drop |
| 重新归一化后 | 0.666 | 0.245 | 0.090 | 0 | 0 |

它的优点是边界清晰：每一步最多看 \\(k\\) 个候选。缺点也来自这里：\\(k\\) 是固定个数，不关心分布形状。

举两个极端：

- 如果模型非常确定，A 的概率是 0.95，`top_k = 50` 仍然会放进很多几乎不该出现的尾部 token。
- 如果模型很不确定，前 100 个 token 的概率都差不多，`top_k = 10` 又可能砍掉合理候选。

所以 top-k 控制的是**候选池大小上限**，不是概率质量。

## top-p：留下累计概率够高的一小团 token {#top-p}

top-p 又叫 nucleus sampling。它不是固定留下几个 token，而是按概率从高到低累加，直到累计概率超过阈值 \\(p\\)。留下这组 token，其余 token 丢掉，再归一化抽样。

用同一组概率，`top_p = 0.8`：

| rank | token | probability | cumulative |
|---:|---|---:|---:|
| 1 | A | 0.636 | 0.636 |
| 2 | B | 0.234 | 0.871 |
| 3 | C | 0.086 | 0.957 |
| 4 | D | 0.032 | 0.988 |
| 5 | E | 0.012 | 1.000 |

累计到 A 还不够 0.8，加上 B 后超过 0.8，所以候选池是 A/B。重新归一化：

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| top-p 后 | 0.731 | 0.269 | 0 | 0 | 0 |

top-p 的关键是候选池大小会随分布变化：

- 模型很确定时，可能只留下 1-2 个 token。
- 模型犹豫时，可能留下几十甚至上百个 token。

所以 top-p 控制的是**保留多少概率质量**，不是固定 token 个数。这通常比 top-k 更贴近“别采太离谱，但也别死板”的需求。

## 三者怎么组合 {#composition}

可以把一次采样想成四步：

```text
logits
  -> apply penalties / logits processors
  -> divide by temperature
  -> apply top-k / top-p masks
  -> softmax + random draw
```

如果三者同时设置，它们不是互相替代，而是叠加约束：

- temperature 先改变分布形状。
- top-k 限制最多多少个 token 能进候选池。
- top-p 再按累计概率质量收紧候选池。

一个实用心智模型：

| 目标 | 常见设置方向 | 原因 |
|---|---|---|
| 稳定问答、摘要、抽取 | 较低 temperature，较高 top-p，top-k 可关 | 让高概率答案更稳定，少做硬截断 |
| 代码生成 | 低 temperature，必要时关掉随机性 | 错一个 token 可能破坏语法或 API |
| 创意写作 | 中等或偏高 temperature，top-p 0.8-0.95 | 保留多样性，同时切掉长尾噪声 |
| 想严格复现 | temperature 接近 0，加 seed 仍要注意实现细节 | greedy 路径最稳定，随机路径依赖采样实现 |

不要把 `temperature=2, top_p=0.1, top_k=1` 这种组合当成“更强控制”。`top_k=1` 已经把候选池压成一个 token，temperature 和 top-p 基本没有发挥空间。

## vLLM 源码里在哪里看 {#vllm-source}

vLLM 的外部 API 参数最终会落到 `SamplingParams`。当前 checkout 中，核心定义在：

- `vllm/sampling_params.py`
- `vllm/v1/sample/sampler.py`
- `vllm/v1/sample/metadata.py`
- `vllm/v1/sample/ops/topk_topp_sampler.py`

`SamplingParams` 里默认值很能说明语义：

```python
temperature: float = 1.0
top_p: float = 1.0
top_k: int = 0
```

也就是默认不调温、不做 top-p 截断、不做 top-k 截断。校验规则也对应上面的定义：`top_p` 必须在 `(0, 1]`，`top_k` 为 0 或 -1 表示禁用，temperature 不能为负。

温度接近 0 时，vLLM 会走 greedy 语义，并把 top-p/top-k/min-p 重置成不生效的值。原因很简单：如果已经是 greedy，就不需要再构造随机候选池。

真正执行采样的顺序在 `vllm/v1/sample/sampler.py` 的 `Sampler.sample()`：

```python
greedy_sampled = self.greedy_sample(logits)
...
logits = self.apply_temperature(logits, sampling_metadata.temperature, ...)
...
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits,
    sampling_metadata.generators,
    sampling_metadata.top_k,
    sampling_metadata.top_p,
)
```

`TopKTopPSampler` 再根据平台选择 native、Triton、FlashInfer、XPU 或 ROCm 路径。但从语义上看，它们都在做同一件事：对 logits/probabilities 应用 top-k/top-p 过滤，然后按过滤后的分布抽样。优化实现可以不同，采样分布的数学含义不能变。

## 小结 {#summary}

temperature、top-k、top-p 都在控制“下一 token 从哪里抽”，但控制维度不同：

- temperature 改变概率分布的尖锐程度。
- top-k 按排名限制候选 token 个数。
- top-p 按累计概率质量动态决定候选集合。

理解这三者后，很多推理参数就不再像玄学旋钮。它们只是对同一张 next-token 概率表做缩放、截断和归一化。源码里的 sampler 也正是围绕这张表组织起来的。
