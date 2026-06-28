
In the previous training posts, we separated the loop into a few pieces:

- the [loss function]({{< relref "/posts/loss-functions-cross-entropy/" >}}) defines what counts as wrong;
- [forward and backward propagation]({{< relref "/posts/forward-and-backward-propagation/" >}}) computes gradients for each parameter;
- [gradient descent]({{< relref "/posts/batch-vs-stochastic-gradient-descent/" >}}) updates parameters using those gradients.

But in real training code, we usually do not write:

```python
param = param - lr * param.grad
```

We write:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
optimizer.step()
```

So what does the optimizer actually do? Why is AdamW so common? Is modern neural-network training basically "just use AdamW", or are there still meaningful alternatives?

The short answer:

> AdamW is a very strong default in modern deep learning, especially for Transformers, LLMs, diffusion models, and vision-language models. But it is not the only useful optimizer. SGD with momentum, Adafactor, LAMB/LARS, Lion, Sophia, and other methods still matter under different constraints.

This post starts with a tiny update example, then follows the historical line from SGD to Momentum, RMSProp, Adam, and AdamW.

{{< figure src="/images/posts/optimizers-adamw/optimizer-evolution-map.svg" caption="<span class=\"figure-number\">Figure 1: </span>Optimizer evolution map: the main line runs from SGD to AdamW, while side branches usually serve memory, large-batch, or research-specific constraints." width="100%" >}}

## Start With The Selection Table {#cheat-sheet}

| Optimizer | Core idea | Common role | Main cost |
| --- | --- | --- | --- |
| SGD | take one step along the negative gradient | teaching, small models, some CNN/vision training | sensitive to learning rate, can converge slowly |
| SGD + Momentum | accumulate historical direction, reduce oscillation | CNNs and supervised-learning baselines with strong generalization | still needs careful learning-rate tuning |
| RMSProp / Adagrad | use different step sizes per parameter | RNNs, sparse features, earlier deep learning | less common as the default for large modern models |
| Adam | momentum plus adaptive per-parameter scaling | general deep-learning baseline | L2 weight decay is coupled with adaptive scaling |
| AdamW | Adam plus decoupled weight decay | strong default for Transformers, LLMs, diffusion, modern large models | stores first and second moment states, so optimizer memory is large |
| Adafactor | factorize second-moment state to save memory | very large models and memory-constrained training | behavior and hyperparameters are less direct than AdamW |
| LAMB / LARS | layer-wise trust ratio | very large batch training | more complex, mainly useful for large-batch scaling |
| Lion | momentum with sign updates | some vision and generative-model experiments | not consistently better than AdamW across tasks |
| Sophia / second-order approximations | use curvature information to scale updates | language-model pretraining research | more complex, not as default in the ecosystem |

Practical conclusion:

- If you do not know what to use, **AdamW is a good starting point**.
- For traditional CNNs or supervised-learning generalization baselines, **SGD with momentum** is still worth comparing.
- If optimizer state is too expensive, consider **Adafactor, 8-bit optimizers, or ZeRO/FSDP optimizer sharding**.
- For very large batches or specific pretraining setups, LAMB, LARS, Sophia, or Lion may be useful, but they are usually not the first move.

## From One Update To SGD {#from-update-to-sgd}

### What An Optimizer Updates {#what-optimizer-updates}

Suppose there is only one parameter \\(w\\), and the loss is:

$$L(w) = (w - 3)^2$$

When \\(w=0\\):

$$\frac{dL}{dw} = 2(w-3) = -6$$

The gradient is -6, meaning the loss decreases if \\(w\\) increases. The most basic SGD update is:

$$w_{t+1} = w_t - \eta g_t$$

where \\(g_t\\) is the current gradient and \\(\eta\\) is the learning rate. If \\(\eta=0.1\\):

$$w_1 = 0 - 0.1\times(-6) = 0.6$$

The optimizer's minimal job is: take each parameter's gradient and decide how much to change the parameter this step.

Real neural networks do not have one parameter. They have millions to trillions. Different parameters can have very different gradient scales; mini-batches introduce gradient noise; the loss surface can have narrow valleys, flat regions, and steep regions. Most optimizer improvements revolve around three questions: is the direction stable, is the step size scaled appropriately, and is regularization applied cleanly?

{{< figure src="/images/posts/optimizers-adamw/sgd-momentum-adam-intuition.svg" caption="<span class=\"figure-number\">Figure 2: </span>The core intuition behind SGD, Momentum, and Adam: SGD follows the current gradient, Momentum smooths historical direction, and Adam uses first moments for direction plus second moments for scale normalization." width="100%" >}}

### SGD: The Plain Baseline {#sgd}

SGD updates parameters as:

$$\theta_{t+1} = \theta_t - \eta g_t$$

where \\(\theta\\) denotes all parameters and \\(g_t=\nabla_\theta L_t(\theta_t)\\) is the current mini-batch gradient.

SGD is simple, low-state, and easy to reason about. It stores almost no extra state beyond parameters and gradients. Its weakness is also clear: as the left panel of Figure 2 shows, if the loss surface looks like a long narrow valley, the gradient may oscillate across the steep direction while moving slowly along the direction that actually matters. That motivates momentum.

## From Momentum To Adam {#from-momentum-to-adam}

### Momentum: Do Not Trust Only The Current Step {#momentum}

Momentum adds a velocity variable \\(v\\):

$$v_t = \beta v_{t-1} + (1-\beta)g_t$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

Intuitively, \\(v_t\\) is an exponential moving average of gradients. Directions that remain consistent accumulate; directions that keep flipping sign cancel out.

As the middle panel of Figure 2 shows, Momentum lets sign-flipping components cancel while persistent directions accumulate. The update becomes aligned with the long-term direction instead of being dominated by one noisy mini-batch.

This is why SGD with momentum is still relevant: it is cheap, stable, and only stores one extra velocity tensor. In some CNN/vision tasks, it remains a strong baseline and is often associated with good generalization.

### Adaptive Learning Rates: Parameters Need Different Step Sizes {#adaptive-learning-rate}

Momentum addresses noisy direction. Another issue is that different parameters can have very different gradient scales. With one shared effective learning rate, small-gradient parameters may barely move while large-gradient parameters may take steps that are too large.

Adagrad and RMSProp record historical squared gradients and use them to scale updates. A typical RMSProp form is:

$$s_t = \rho s_{t-1} + (1-\rho)g_t^2$$

$$\theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{s_t}+\epsilon}$$

If a parameter has large gradients for a long time, \\(s_t\\) grows, the denominator grows, and the effective step size becomes smaller. If a parameter has small gradients, the denominator stays smaller and the effective step is relatively amplified.

The core idea:

> parameters have different historical gradient scales, so they should not all share exactly the same effective learning rate.

Adam combines this adaptive scaling with momentum.

### Adam: First Moment Plus Second Moment {#adam}

Adam keeps two moving averages:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

where:

- \\(m_t\\): first moment, similar to momentum, recording the average gradient direction;
- \\(v_t\\): second moment, recording squared-gradient scale.

Since \\(m_0=0\\) and \\(v_0=0\\), the moving averages are biased toward zero at the beginning. Adam uses bias correction:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

The update is:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

In words:

> move along the historical average gradient direction, but normalize each parameter's step by its historical squared-gradient scale.

The right panel of Figure 2 shows the key point: a dimension can have a large average gradient, but if its historical squared gradient is also large, Adam shrinks it through \\(\sqrt{\hat{v}_t}\\). Small but stable directions are not drowned out. This is why Adam is useful: it is usually less sensitive to learning-rate tuning than SGD, often converges quickly early in training, and adapts to different parameter scales.

But Adam has an important issue: weight decay is not handled cleanly.

## AdamW's Position: Strong Default, Not The Only Answer {#adamw-position}

### AdamW: Decoupled Weight Decay Is The Key Fix {#adamw}

Weight decay pulls weights toward zero:

$$\theta \leftarrow \theta - \eta \lambda \theta$$

For plain SGD, adding L2 regularization to the loss and applying weight decay during the update are equivalent in form:

$$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t)$$

But Adam does not use the raw gradient directly. It puts the gradient into \\(m_t\\) and \\(v_t\\), then performs adaptive scaling. If \\(\lambda\theta\\) is mixed into Adam's gradient, it is also scaled by the second-moment machinery:

$$\theta_{t+1} = \theta_t - \eta \frac{\text{Adam moments of }(g_t+\lambda\theta_t)}{\sqrt{\text{second moment}}+\epsilon}$$

Now "pull weights toward zero" is no longer controlled only by \\(\lambda\\). It is entangled with each parameter's historical gradient scale.

AdamW changes this: **do not treat weight decay as part of the gradient; apply it separately from Adam's adaptive gradient update**.

{{< figure src="/images/posts/optimizers-adamw/adam-vs-adamw-weight-decay.svg" caption="<span class=\"figure-number\">Figure 3: </span>Adam vs AdamW: Adam mixes weight decay into the adaptive moment path; AdamW applies weight decay through a separate path, making regularization strength cleaner." width="100%" >}}

One way to write it is:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_t$$

Or as two conceptual steps:

1. use Adam's adaptive rule to update parameters from gradients;
2. separately apply decoupled weight decay to shrink weights.

That is the W in AdamW: Adam with decoupled Weight decay.

#### Why This Matters {#why-adamw-matters}

AdamW is not merely Adam with a new name. It fixes a boundary problem in Adam's regularization:

- Adam decides how to move based on gradient history;
- weight decay controls weight magnitude;
- after decoupling, \\(\lambda\\) behaves more like an independent regularization strength.

That is why modern Transformer training often starts with a recipe like:

```python
optimizer = AdamW(
    params,
    lr=...,
    betas=(0.9, 0.95),  # or (0.9, 0.999), depending on setup
    weight_decay=...
)
```

Real recipes also combine AdamW with learning-rate warmup, cosine decay, gradient clipping, and parameter groups. For example, bias terms and LayerNorm/RMSNorm scale parameters are often excluded from weight decay.

### Is AdamW The Only Default Answer {#is-adamw-the-only-answer}

AdamW's position is best described this way:

> it is one of the most common and reliable strong defaults in modern deep learning, but it is not the only theoretically or practically valid optimizer.

AdamW is natural when:

- training Transformers, LLMs, diffusion models, or multimodal models;
- you want a strong baseline before studying optimizer variants;
- training budget is limited and you want fewer tuning surprises;
- there is no strong task-specific evidence favoring another optimizer.

AdamW is not always the first choice when:

- **traditional supervised CNN training**: SGD with momentum is still commonly compared, especially for vision benchmarks and generalization baselines.
- **memory is tight**: AdamW stores \\(m\\) and \\(v\\), two optimizer-state tensors. Mixed precision may also keep fp32 master weights. Adafactor, 8-bit Adam, or optimizer sharding may be better.
- **batch size is extremely large**: LAMB and LARS can help with large-batch scaling.
- **research pretraining setups**: Lion, Sophia, Shampoo, and related methods can be faster or cheaper in specific settings, but need validation.
- **sparse features or online learning**: Adagrad/FTRL-style methods still have a place in recommendation and advertising systems.

So the practical statement is not "everything is AdamW now". It is:

> AdamW is the default starting point for many modern neural networks; other optimizers are deliberate choices under specific model, data, batch-size, memory, or generalization constraints.

## Training Recipe And Historical Map {#recipe-and-history}

### The Optimizer Is Only Part Of The Training Recipe {#training-recipe}

Training behavior is rarely determined by the optimizer alone. It is the combination of optimizer, schedule, regularization, and batch size.

For AdamW, the actual recipe usually includes:

- learning rate: one of the most important hyperparameters;
- betas: how long the first and second moments remember history;
- weight decay: regularization strength;
- warmup: gradually increase learning rate at the beginning to avoid unstable early updates;
- decay schedule: cosine decay, linear decay, constant schedule, and so on;
- gradient clipping: limit unusually large gradients;
- parameter groups: use different weight decay or learning rates for norm, bias, embedding, or other parameter groups.

A common mistake is assuming that switching optimizers will fix all training instability. More often, instability comes from learning rate, warmup, batch size, initialization, normalization, loss scaling, or data issues.

The optimizer decides "how to step after gradients are available". It cannot replace a good loss, clean data, a stable model architecture, or a sensible training schedule.

### A Historical Map {#historical-map}

You can read optimizer history as a sequence of patches to SGD's weaknesses: noisy direction, different gradient scales across parameters, entanglement between regularization and adaptive updates, and optimizer-state cost in large-model training. Figure 1 is the compressed version of that history.

If you remember one line:

> SGD follows the gradient; Momentum stabilizes direction; RMSProp and Adagrad give parameters their own step sizes; Adam combines both ideas; AdamW decouples weight decay from Adam's adaptive gradient machinery, which is why it became a strong default for modern deep learning.

### Further Reading {#further-reading}

- Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.
- Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017.
- Noam Shazeer and Mitchell Stern, [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235), 2018.
- Xiangning Chen et al., [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675), 2023.
- Hong Liu et al., [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342), 2023.
