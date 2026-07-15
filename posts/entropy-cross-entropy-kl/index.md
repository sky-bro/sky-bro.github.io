
Entropy, cross entropy, and KL divergence often look like three separate formulas:

- entropy: \\(H(P)\\)
- cross entropy: \\(H(P,Q)\\)
- KL divergence: \\(D_{\mathrm{KL}}(P\Vert Q)\\)

These letters are conventions rather than variables derived from the formulas. Shannon used \\(H\\) for entropy in his foundational information-theory paper;[^fn:shannon] there is no single agreed explanation for why he chose that letter, so it is safest to treat it as historical notation. Cross entropy still measures an average coding cost and generalizes entropy, hence \\(H(P,Q)\\): the first argument is the true distribution and the second is the coding distribution. KL uses \\(D\\) to emphasize **divergence** between distributions. A divergence is not necessarily a metric: KL is asymmetric and does not satisfy the triangle inequality.

But they are really one story: **if data is produced by distribution \\(P\\), and I use some code based on distribution \\(Q\\), how much average coding cost do I pay?**

This post is a foundation for [Loss Functions: What a Model Is Really Optimizing]({{< relref "/posts/loss-functions-cross-entropy/" >}}) and the planned follow-up *Why KL Divergence Is Not a Distance*. The loss-function post explains when to use each loss; the follow-up will explain why KL direction matters. This post focuses on the basic relationship among entropy, cross entropy, and KL.

{{< figure src="/images/posts/entropy-cross-entropy-kl/coding-cost-decomposition.svg" caption="<span class=\"figure-number\">Figure 1: </span>Cross entropy is the total coding cost; entropy is the unavoidable cost of the true distribution; KL is the extra cost paid for using the wrong distribution." width="100%" >}}

## A Two-Outcome World {#toy-world}

Suppose weather has only two outcomes: sunny and rainy. The true distribution is:

$$P=[0.75,\ 0.25]$$

Sunny happens 75% of the time, and rainy happens 25% of the time. If we know the true distribution and use the best code for \\(P\\), the average coding cost is entropy:

$$H(P)=-\sum_x P(x)\log P(x)$$

Plugging in the numbers:

$$H(P)=-(0.75\log 0.75+0.25\log 0.25)\approx 0.562$$

We are using natural logs, so the unit is nats. Intuitively, \\(H(P)\\) measures how uncertain the true world is:

- if every day is sunny, \\(P=[1,0]\\), almost no extra information is needed, so entropy is 0;
- if sunny and rainy are equally likely, \\(P=[0.5,0.5]\\), the world is harder to predict, so entropy is higher;
- \\(P=[0.75,0.25]\\) sits between those two cases.

“More uniform means more entropy” needs one boundary condition: **for a fixed number of outcomes, the uniform distribution maximizes entropy**. For a binary distribution \\(P=[p,1-p]\\),

$$H(p)=-p\log p-(1-p)\log(1-p)$$

Its first and second derivatives are:

$$H'(p)=\log\frac{1-p}{p},\qquad H''(p)=-\frac{1}{p}-\frac{1}{1-p}<0$$

Solving \\(H'(p)=0\\) gives only \\(p=0.5\\). Because the second derivative is always negative, the curve is strictly concave and this point is the unique maximum, with \\(H(P)=\log 2\\). More generally, for a fixed set of \\(n\\) outcomes, let \\(U(x)=1/n\\). Then

$$D_{\mathrm{KL}}(P\Vert U)=\sum_x P(x)\log\frac{P(x)}{1/n}=\log n-H(P)\ge 0$$

Therefore \\(H(P)\le\log n\\), with equality only when \\(P=U\\). The intuition agrees: concentrated probability makes the next outcome easier to guess, while a uniform distribution prevents us from ruling out any outcome in advance.

Entropy is not model error. It is the **unavoidable cost of the true distribution itself**.

## Cross Entropy: Coding P With Q {#cross-entropy}

Now suppose we do not know the true distribution. We use a model distribution \\(Q\\) as the coding scheme. For example, the model says:

$$Q=[0.5,\ 0.5]$$

It treats sunny and rainy as equally common. The real world still produces data according to \\(P=[0.75,0.25]\\), but the code is built from \\(Q\\). The average cost is cross entropy:

$$H(P,Q)=-\sum_x P(x)\log Q(x)$$

With the numbers above:

$$H(P,Q)=-(0.75\log 0.5+0.25\log 0.5)\approx 0.693$$

This is larger than \\(H(P)\approx 0.562\\). The reason is simple: the real world is sunny more often, but \\(Q\\) fails to use that structure.

If the model is closer to the truth, say \\(Q=[0.7,0.3]\\), cross entropy decreases:

$$H(P,Q)=-(0.75\log 0.7+0.25\log 0.3)\approx 0.568$$

That is already close to the true entropy 0.562. The key intuition is:

**Cross entropy measures the average cost of explaining data from \\(P\\) using a model distribution \\(Q\\).**

## KL: The Extra Cost {#kl}

The central identity is:

$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$$

Equivalently:

$$D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$$

So KL is not an unrelated new object. It is:

**the extra cost paid when we use \\(Q\\) to code data from \\(P\\), compared with the best possible code for \\(P\\).**

For the example above:

| Quantity | Value | Meaning |
|---|---:|---|
| \\(H(P)\\) | 0.562 | unavoidable uncertainty of the true world |
| \\(H(P,Q)\\), with \\(Q=[0.5,0.5]\\) | 0.693 | total cost when using the wrong model |
| \\(D_{\mathrm{KL}}(P\Vert Q)\\) | 0.131 | extra cost caused by using \\(Q\\) |

If \\(Q=P\\), cross entropy equals entropy and KL is 0. No extra cost is wasted.

If \\(Q\\) assigns too little probability where \\(P\\) often produces data, cross entropy grows and KL grows. This is why training a model by giving higher probability to real examples directly lowers the loss.

## Why Training Uses Cross Entropy {#training}

In classification, we do not know the full true distribution \\(P\\). We only observe samples. If one sample's true class is rainy, the one-hot label can be viewed as an empirical distribution:

$$P^{\mathrm{sample}}=[0,\ 1]$$

The model predicts:

$$Q_\theta=[0.8,\ 0.2]$$

Cross entropy becomes:

$$H(P^{\mathrm{sample}},Q_\theta)=-\log 0.2\approx 1.61$$

This is the familiar negative log likelihood: look at the probability assigned to the true class. The lower that probability is, the larger the loss; the higher it is, the smaller the loss.

Language-model next-token prediction is the same structure. Given a context, the dataset tells us the next real token; cross entropy asks the model to assign that token higher probability. SFT has the same mathematical form, but the data changes from raw text to prompt-answer examples.

From the KL view, cross entropy training pushes the model distribution \\(Q_\theta\\) toward the data distribution \\(P\\). Since:

$$H(P,Q_\theta)=H(P)+D_{\mathrm{KL}}(P\Vert Q_\theta)$$

and \\(P\\) is fixed during training, \\(H(P)\\) is constant. Minimizing cross entropy is therefore equivalent to minimizing \\(D_{\mathrm{KL}}(P\Vert Q_\theta)\\).

## Summary {#summary}

The three concepts can be remembered this way:

| Concept | Question | Formula |
|---|---|---|
| Entropy | How hard is the true distribution \\(P\\) to code? | \\(H(P)\\) |
| Cross entropy | If data comes from \\(P\\), but I code using \\(Q\\), what is the total cost? | \\(H(P,Q)\\) |
| KL divergence | How much extra cost do I pay because I used \\(Q\\) instead of the best code? | \\(D_{\mathrm{KL}}(P\Vert Q)\\) |

The core identity is:

$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$$

Cross entropy is not an isolated loss. It asks: **can the model distribution \\(Q\\) explain real data from \\(P\\) with low coding cost?** KL then decomposes that cost into unavoidable uncertainty from the data and extra waste caused by the wrong model distribution.

[^fn:shannon]: Claude E. Shannon, [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), 1948.
