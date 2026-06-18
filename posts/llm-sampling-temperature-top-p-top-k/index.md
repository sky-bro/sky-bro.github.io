
Why does lowering `temperature` make an answer more stable? Why does lowering `top_p` make it more conservative? Why does a small `top_k` feel like asking the model to choose only from the obvious candidates? These are not three separate magic styles. They are three concrete operations on the **next-token probability distribution**.

This post builds the intuition with a 5-token example, then maps the same idea to the vLLM V1 sampler.

## Start With One Sampling Step {#one-step}

After one model forward pass, the model does not directly emit text. It assigns one logit to every token in the vocabulary. A logit is an unnormalized score. Suppose the next-token candidates are:

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| logit | 4 | 3 | 2 | 1 | 0 |

With `temperature = 1` and no filtering, softmax gives approximately:

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| probability | 0.636 | 0.234 | 0.086 | 0.032 | 0.012 |

A is most likely, but B/C/D/E can still be sampled. The common sampling parameters control this table: first reshape the probability curve, then decide which tokens are allowed into the candidate pool, then draw one token from the remaining probabilities.

{{< figure src="/images/posts/llm-sampling-temperature-top-p-top-k/sampling-filter-flow.svg" caption="<span class=\"figure-number\">Figure 1: </span>temperature reshapes the probability curve; top-k cuts by rank; top-p cuts by cumulative probability mass; the remaining tokens are renormalized before sampling." width="96%" >}}

## Temperature: Sharpen Or Flatten The Distribution {#temperature}

Temperature acts before softmax: \\(p\_i = \exp(z\_i / T) / \sum\_j \exp(z\_j / T)\\). Here \\(z\_i\\) is the logit for token \\(i\\), and \\(T\\) is temperature.

- \\(T < 1\\): high-score tokens become more dominant, so output is more stable.
- \\(T = 1\\): logits are not additionally scaled.
- \\(T > 1\\): low-score tokens get more probability mass, so output is more diverse.
- \\(T \approx 0\\): the path approaches greedy decoding, usually selecting the maximum-logit token.

The key is not that softmax changes. The key is that **temperature rescales logit gaps**. The relative gap between two tokens is \\(z\_a - z\_b\\). After temperature, it becomes \\((z\_a - z\_b) / T\\). So \\(T > 1\\) shrinks the gap and makes the strong token less dominant; \\(T < 1\\) expands the gap and makes the strong token more dominant.

{{< figure src="/images/posts/llm-sampling-temperature-top-p-top-k/temperature-softmax-curve.svg" caption="<span class=\"figure-number\">Figure 2: </span>With fixed logits, increasing temperature lowers the probability of the highest-logit token and raises the probability of low-logit tokens; in the limit, the distribution moves toward uniform." width="92%" >}}

For the same logits `[4, 3, 2, 1, 0]`:

| temperature | A | B | C | D | E | intuition |
|---:|---:|---:|---:|---:|---:|---|
| 0.7 | 0.761 | 0.182 | 0.044 | 0.010 | 0.003 | A almost dominates |
| 1.0 | 0.636 | 0.234 | 0.086 | 0.032 | 0.012 | original softmax |
| 1.5 | 0.505 | 0.259 | 0.133 | 0.068 | 0.035 | tail tokens get more chance |

Temperature does not directly delete tokens. It changes the **slope of relative probabilities**. The logit gap between A and B is 1. Dividing by 0.7 turns it into an effective gap of 1.43; dividing by 1.5 turns it into 0.67. Softmax is sensitive to that gap, so the first case makes A easier to sample, while the second gives probability mass back to B/C/D/E.

That explains a common misconception: high temperature does not mean "smarter", and low temperature does not mean "more correct". Temperature only heats up or cools down the sampling distribution. QA, code, and extraction often want lower temperature. Creative writing, rewriting, and brainstorming can tolerate more heat.

## Top-k: Keep The Best k Tokens By Rank {#top-k}

Top-k is the most direct truncation rule: sort the tokens, keep only the \\(k\\) highest-probability tokens, set all other probabilities to 0, then renormalize the survivors.

If `top_k = 3`, the candidate pool is A/B/C:

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| original probability | 0.636 | 0.234 | 0.086 | 0.032 | 0.012 |
| top-k mask | keep | keep | keep | drop | drop |
| after renormalization | 0.666 | 0.245 | 0.090 | 0 | 0 |

The advantage is a clear boundary: at each step, at most \\(k\\) candidates can be sampled. The weakness is the same thing: \\(k\\) is a fixed count and does not care about the shape of the distribution.

Two extremes show the issue:

- If the model is very certain and A has probability 0.95, `top_k = 50` still admits many tail tokens that probably should not appear.
- If the model is uncertain and the top 100 tokens are similarly plausible, `top_k = 10` may remove reasonable candidates.

So top-k controls the **maximum candidate-pool size**, not probability mass.

## Top-p: Keep Enough Probability Mass {#top-p}

Top-p is also called nucleus sampling. It does not keep a fixed number of tokens. It sorts tokens by probability, accumulates probability mass from high to low, stops when the cumulative mass exceeds \\(p\\), drops the rest, and samples after renormalization.

Using the same probabilities, with `top_p = 0.8`:

| rank | token | probability | cumulative |
|---:|---|---:|---:|
| 1 | A | 0.636 | 0.636 |
| 2 | B | 0.234 | 0.871 |
| 3 | C | 0.086 | 0.957 |
| 4 | D | 0.032 | 0.988 |
| 5 | E | 0.012 | 1.000 |

A alone is not enough. A+B exceeds 0.8, so the candidate pool becomes A/B. After renormalization:

| token | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| after top-p | 0.731 | 0.269 | 0 | 0 | 0 |

The candidate-pool size changes with the distribution:

- When the model is confident, top-p may keep only one or two tokens.
- When the model is uncertain, top-p may keep dozens or hundreds.

So top-p controls **how much probability mass to preserve**, not a fixed token count. That often matches the practical goal: avoid bizarre tail samples without making the model completely rigid.

## How The Three Combine {#composition}

One sampling step can be read as:

```text
logits
  -> apply penalties / logits processors
  -> divide by temperature
  -> apply top-k / top-p masks
  -> softmax + random draw
```

When all three are set, they are constraints on the same distribution:

- temperature reshapes the distribution first.
- top-k limits how many tokens can enter the candidate pool.
- top-p further tightens the pool by cumulative probability mass.

A useful mental model:

| goal | common direction | why |
|---|---|---|
| stable QA, summarization, extraction | lower temperature, high top-p, often no top-k | stabilize high-probability answers without hard rank cuts |
| code generation | low temperature, sometimes no randomness | one wrong token can break syntax or APIs |
| creative writing | medium or higher temperature, top-p around 0.8-0.95 | keep diversity while trimming long-tail noise |
| strict reproducibility | near-zero temperature, seed if sampling remains | greedy is the most stable path; random paths depend on sampler details |

Combinations like `temperature=2, top_p=0.1, top_k=1` are not "stronger control". `top_k=1` already collapses the candidate pool to one token, leaving little room for temperature or top-p to matter.

## Where This Lives In vLLM {#vllm-source}

External API fields eventually become `SamplingParams` in vLLM. In the current checkout, the relevant files are:

- `vllm/sampling_params.py`
- `vllm/v1/sample/sampler.py`
- `vllm/v1/sample/metadata.py`
- `vllm/v1/sample/ops/topk_topp_sampler.py`

The defaults in `SamplingParams` are a compact statement of the semantics:

```python
temperature: float = 1.0
top_p: float = 1.0
top_k: int = 0
```

By default, vLLM does not rescale logits, does not apply top-p truncation, and does not apply top-k truncation. The validation rules match the definitions: `top_p` must be in `(0, 1]`, `top_k` uses 0 or -1 as disabled, and temperature cannot be negative.

When temperature is close to 0, vLLM switches to greedy semantics and resets top-p/top-k/min-p to inactive values. If the request is greedy, there is no need to build a random candidate pool.

The sampling order appears in `Sampler.sample()` in `vllm/v1/sample/sampler.py`:

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

`TopKTopPSampler` then chooses a native, Triton, FlashInfer, XPU, or ROCm path depending on platform and configuration. The implementation may change for performance, but the semantics are the same: apply top-k/top-p filtering, then draw from the filtered distribution.

## Summary {#summary}

Temperature, top-k, and top-p all control where the next token is sampled from, but they control different dimensions:

- temperature changes how sharp or flat the probability distribution is.
- top-k limits the candidate count by rank.
- top-p dynamically chooses the candidate set by cumulative probability mass.

Once you see them as operations on one next-token probability table, these parameters stop feeling like folklore. The sampler code is organized around that same table: scale, filter, renormalize, sample.
