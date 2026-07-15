
KL divergence is often described as a distance between two distributions. That is half useful and half dangerous. It compares two distributions, but **it is not a distance because direction matters**. More precisely, KL is nonnegative and vanishes only when \\(P=Q\\), but it fails the symmetry and triangle-inequality requirements of a metric.

The formula is:

$$D_{\mathrm{KL}}(P\Vert Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$$

Do not read this as "the distance between P and Q." A better reading is:

**If the true or target distribution is \\(P\\), but I use \\(Q\\) to approximate it, how much extra coding cost do I pay?**

The two arguments have different roles:

- \\(P\\): decides which locations matter, because the sum is weighted by \\(P(x)\\);
- \\(Q\\): approximates \\(P\\), and gets punished if it assigns too little probability where \\(P\\) has mass.

{{< figure src="/images/posts/kl-divergence-not-a-distance/kl-direction.svg" caption="<span class=\"figure-number\">Figure 1: </span>KL direction is not a notation detail. It asks two different questions: can Q cover what P cares about, or can P explain what Q actually uses?" width="100%" >}}

This post fits after [Entropy, Cross Entropy, and KL Divergence]({{< relref "/posts/entropy-cross-entropy-kl/" >}}) and [Loss Functions: What a Model Is Really Optimizing]({{< relref "/posts/loss-functions-cross-entropy/" >}}). The first establishes the coding-cost view; the second places KL in the loss-function family. Here we carry that intuition into SFT and RLHF KL penalties.

## Start From Coding Cost {#coding-cost}

Suppose an event has two outcomes, A and B. The true distribution is:

$$P=[0.5,\ 0.5]$$

Both A and B matter. Now approximate it with:

$$Q=[0.99,\ 0.01]$$

\\(Q\\) believes A is almost certain and B is rare. If the real world is actually \\(P\\), this is a bad code: B happens half the time, but \\(Q\\) assigns it only 1% probability.

Using natural logs:

$$D_{\mathrm{KL}}(P\Vert Q)=0.5\log\frac{0.5}{0.99}+0.5\log\frac{0.5}{0.01}\approx 1.61$$

Reverse the direction:

$$D_{\mathrm{KL}}(Q\Vert P)=0.99\log\frac{0.99}{0.5}+0.01\log\frac{0.01}{0.5}\approx 0.64$$

Same distributions, different values. The reason is that the questions changed:

- \\(D_{\mathrm{KL}}(P\Vert Q)\\): if \\(P\\) is the target, \\(Q\\) missed B, which \\(P\\) considers common.
- \\(D_{\mathrm{KL}}(Q\Vert P)\\): if \\(Q\\) is the target, \\(Q\\) mostly cares about A, and \\(P\\) still gives A nontrivial probability.

The first argument is not arbitrary. It is the distribution that weights where mistakes matter.

## Zero Probability Makes The Direction Obvious {#zero-probability}

Take a sharper example:

$$P=[0.5,\ 0.5],\quad Q=[1.0,\ 0.0]$$

\\(Q\\) says B is impossible, while \\(P\\) says B happens half the time. Then:

$$D_{\mathrm{KL}}(P\Vert Q)=\infty$$

The term for B contains \\(\log(P(B)/Q(B))\\), and \\(Q(B)=0\\). This is not a numerical nuisance. It is a semantic failure: you are using a code that says B cannot happen to encode a world where B happens often.

But the reverse direction is finite:

$$D_{\mathrm{KL}}(Q\Vert P)=1.0\log\frac{1.0}{0.5}+0.0\log\frac{0.0}{0.5}\approx 0.69$$

By the limiting convention, \\(0\log(0/q)\\) is defined as 0. \\(Q\\) only produces A, and \\(P\\) assigns A nonzero probability.

The practical rule:

**Where the left distribution has mass, the right distribution must also assign enough probability. If the right side misses the left side's support, KL becomes severe.**

## Relationship To Cross Entropy And SFT {#cross-entropy}

Entropy, cross entropy, and KL are connected by:

$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$$

where:

- \\(H(P)\\): unavoidable coding cost under the true distribution \\(P\\);
- \\(H(P,Q)\\): average cost when the true distribution is \\(P\\), but we code using \\(Q\\);
- \\(D_{\mathrm{KL}}(P\Vert Q)\\): extra cost from using the wrong distribution \\(Q\\).

In classification, the data distribution \\(P_{\text{data}}\\) is fixed and the model distribution \\(Q_\theta\\) changes. Minimizing cross entropy:

$$H(P_{\text{data}}, Q_\theta)$$

is equivalent to minimizing:

$$D_{\mathrm{KL}}(P_{\text{data}}\Vert Q_\theta)$$

because \\(H(P_{\text{data}})\\) does not depend on model parameters. This direction matters: the model must assign probability to what the data actually produces.

Language-model next-token prediction has the same structure. Cross entropy increases the probability of real tokens from the dataset. It pushes the model distribution to cover the data distribution, rather than only picking a narrow favorite subset.

SFT uses the same direction. Given a prompt \\(x\\) and a human-written answer \\(y_{1:T}\\), the empirical data distribution can be viewed as a one-hot distribution that places all probability mass on the observed token at each position. At each position, SFT minimizes \\(-\log \pi^{\theta}(y^{(t)}\mid x,y^{(1:t-1)})\\), then sums this over token positions.

This is cross entropy from the empirical data distribution to the model distribution, equivalently a data-to-model KL: \\(D_{\mathrm{KL}}(P^{\mathrm{data}}\Vert \pi^{\theta})\\).

If you have heard that "SFT and RL use KL in opposite directions," this is usually the contrast: **in SFT, the data or teacher answer is on the left and the current model is on the right; in RLHF KL penalty, the current policy is often on the left and the reference policy is on the right.** SFT asks whether the model gives enough probability to the data answer. RLHF asks whether responses sampled from the current model have moved into regions the reference model would consider unnatural.

## Two Directions: Mode-Covering And Mode-Seeking {#mode-covering}

A common machine-learning shorthand is:

- \\(D_{\mathrm{KL}}(P\Vert Q)\\): more **mode-covering**; \\(Q\\) is pressured to cover the main modes of \\(P\\);
- \\(D_{\mathrm{KL}}(Q\Vert P)\\): more **mode-seeking**; \\(Q\\) is pressured to concentrate where \\(P\\) is high.

Think of modes as peaks in a distribution. If target \\(P\\) has two peaks and we optimize \\(D_{\mathrm{KL}}(P\Vert Q)\\), missing either peak is punished because \\(P\\) weights both. \\(Q\\) tends to cover both peaks.

If we optimize \\(D_{\mathrm{KL}}(Q\Vert P)\\), the expectation is weighted by \\(Q\\). As long as \\(Q\\) avoids regions where \\(P\\) is low, it may concentrate on one peak and ignore another.

{{< figure src="/images/posts/kl-divergence-not-a-distance/kl-continuous-modes.svg" caption="<span class=\"figure-number\">Figure 2: </span>With continuous distributions, the contrast is easier to see: forward KL fears missing P's peaks, while reverse KL fears putting Q's mass into P's low-density valley." width="100%" >}}

The shorthand is not the whole theory, but it is a useful engineering intuition:

| Direction | Error it fears | Typical behavior |
|---|---|---|
| \\(D_{\mathrm{KL}}(P\Vert Q)\\) | \\(Q\\) misses what \\(P\\) often produces | cover the target distribution |
| \\(D_{\mathrm{KL}}(Q\Vert P)\\) | \\(Q\\) puts mass where \\(P\\) is low | concentrate on high-probability modes |

## Back To RLHF: What KL Penalty Constrains {#rlhf-kl}

In RLHF or PPO-style language-model training, a common objective is:

$$\underset{\theta}{\max}\ \mathbb{E}^{y\sim\pi^{\theta}(\cdot\mid x)}[r(x,y)]-\beta D_{\mathrm{KL}}(\pi^{\theta}(\cdot\mid x)\Vert\pi^{\mathrm{ref}}(\cdot\mid x))$$

Here:

- \\(\pi_\theta\\): the current trainable policy, i.e. the language model being updated;
- \\(\pi_{\mathrm{ref}}\\): the frozen reference policy, often the SFT model;
- \\(r(x,y)\\): reward from a reward model or verifier;
- \\(\beta\\): strength of the KL penalty.

This direction says:

**Responses that the current model actually samples should not move too far from what the reference model considers natural.**

Why this direction? RL rollouts are sampled from the current policy \\(\pi_\theta\\). Training observes the token trajectories the current model actually emits, so it can estimate:

$$\log \pi_\theta(y\mid x)-\log \pi_{\mathrm{ref}}(y\mid x)$$

If the current model starts emitting text that the reference model considers extremely unnatural, this term grows and the KL penalty pulls it back.

If we used \\(D_{\mathrm{KL}}(\pi_{\mathrm{ref}}\Vert\pi_\theta)\\), the meaning would change:

**Anything the reference model would output should still be covered by the current model.**

That is not meaningless, but it constrains a different problem. It says "do not lose the reference model's coverage" rather than "do not let the current model run into strange regions the reference dislikes." In PPO/RLHF, because rollouts come from the current policy, \\(D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\\) is more natural to estimate from samples.

## Why The KL Penalty Cannot Be Too Small Or Too Large {#kl-strength}

The KL penalty acts like a leash:

- too loose: the model can exploit reward-model loopholes, becoming verbose, hacky, stylistically strange, or generally worse;
- too tight: the model cannot move far enough from the SFT reference, so RL cannot learn much from preference or verifier rewards.

That is why RLHF systems tune \\(\beta\\), sometimes with an adaptive KL controller that keeps observed KL near a target range.

This also explains why SFT and RLHF are partners. SFT gives a reasonable reference policy. RLHF then improves behavior near that reference. If the reference is bad, the KL penalty ties the model to a bad region. Without the KL penalty, RL can push the model into the reward model's blind spots.

## Summary {#summary}

The core idea of KL divergence is not "how far apart are two distributions?" It is:

**If I treat the left distribution as the true or target distribution and approximate it with the right distribution, how much error do I pay in places the left distribution considers important?**

So:

- \\(D_{\mathrm{KL}}(P\Vert Q)\\) punishes \\(Q\\) for missing what \\(P\\) cares about;
- \\(D_{\mathrm{KL}}(Q\Vert P)\\) punishes \\(Q\\) for putting mass where \\(P\\) has little support;
- cross entropy training often corresponds to minimizing \\(D_{\mathrm{KL}}(P_{\text{data}}\Vert Q_\theta)\\);
- RLHF KL penalty often uses \\(D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\\), constraining the current policy's actual outputs from drifting too far from the reference.

Once you remember "the left side weights the mistakes," KL direction stops being notation trivia and becomes the training objective itself.

## Further Reading {#further-reading}

- Cover and Thomas, *Elements of Information Theory*, Chapter 2.
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.
