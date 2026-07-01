
Probability distributions can easily turn into a formula list: Bernoulli, Binomial, Poisson, Normal, Exponential, Gamma, Beta, and so on. If we only memorize probability mass functions or density functions, the names blur together quickly.

A more durable way to remember them is to ask two questions first:

- what is the random variable counting or measuring?
- how much does it vary around its center?

The mean answers "where is the center?" Variance and standard deviation answer "how spread out is it around that center?" This post puts common distributions on one map, focusing on their mean, variance, standard deviation, and the intuition behind the formulas.

{{< figure src="/images/posts/common-distributions-variance-std/distribution-shapes-overview.svg" caption="<span class=\"figure-number\">Figure 1: </span>A shape-first overview of common distributions: discrete distributions use probability-mass bars, while continuous distributions use density curves. The figure gives intuition; the tables below carry the formulas." width="100%" >}}

The shapes in the figure come from probability mass functions (PMFs) or probability density functions (PDFs):

| Distribution | Formula behind the plotted shape |
| --- | --- |
| Bernoulli | \\(\Pr(X=1)=p,\ \Pr(X=0)=1-p\\) |
| Binomial | \\(\Pr(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\\) |
| Poisson | \\(\Pr(X=k)=e^{-\lambda}\lambda^k/k!\\) |
| Geometric | \\(\Pr(X=k)=(1-p)^{k-1}p,\ k=1,2,\ldots\\) |
| Uniform | \\(f(x)=1/(b-a),\ a\le x\le b\\) |
| Normal | \\(f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-(x-\mu)^2/(2\sigma^2)\right]\\) |
| Exponential | \\(f(x)=\lambda e^{-\lambda x},\ x\ge 0\\) |
| Gamma / Beta | \\(f\_{\text{Gamma}}(x)=x^{k-1}e^{-x/\theta}/(\Gamma(k)\theta^k)\\); \\(f\_{\text{Beta}}(x)=x^{\alpha-1}(1-x)^{\beta-1}/B(\alpha,\beta)\\) |

## Start With Variance And Standard Deviation {#variance-first}

For a random variable \\(X\\), the variance is:

$$\operatorname{Var}(X) = \mathbb{E}\left[(X-\mathbb{E}[X])^2\right]$$

It measures the average squared distance from the mean. The usual computational identity is:

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

The standard deviation is the square root of variance:

$$\sigma_X = \sqrt{\operatorname{Var}(X)}$$

Why do we need standard deviation as well? Because variance squares the unit. If \\(X\\) is measured in seconds, variance is measured in squared seconds. Standard deviation returns to seconds, so it is easier to compare directly with the original variable.

Three rules are worth keeping close:

| Operation | Mean | Variance | Standard deviation |
| --- | --- | --- | --- |
| Shift \\(X+c\\) | \\(\mathbb{E}[X]+c\\) | \\(\operatorname{Var}(X)\\) | \\(\sigma_X\\) |
| Scale \\(aX\\) | \\(a\mathbb{E}[X]\\) | \\(a^2\operatorname{Var}(X)\\) | \\(\lvert a\rvert\sigma_X\\) |
| Add independent variables \\(X+Y\\) | \\(\mathbb{E}[X]+\mathbb{E}[Y]\\) | \\(\operatorname{Var}(X)+\operatorname{Var}(Y)\\) | not directly additive |

The last row is the one to remember: when independent random variables are added, **variances add, not standard deviations**. Many distribution formulas are just this rule in disguise.

## Discrete Distributions: From One Trial To Counts {#discrete-distributions}

Discrete distributions usually count trials, successes, events, or the waiting time measured in number of attempts.

| Distribution | What the random variable measures | Parameters | Mean | Variance | Standard deviation |
| --- | --- | --- | --- | --- | --- |
| Bernoulli | whether one 0/1 trial succeeds | \\(p\\) | \\(p\\) | \\(p(1-p)\\) | \\(\sqrt{p(1-p)}\\) |
| Binomial | number of successes in \\(n\\) independent trials | \\(n,p\\) | \\(np\\) | \\(np(1-p)\\) | \\(\sqrt{np(1-p)}\\) |
| Geometric | trial number of the first success | \\(p\\) | \\(1/p\\) | \\((1-p)/p^2\\) | \\(\sqrt{1-p}/p\\) |
| Negative binomial | number of trials needed to get the \\(r\\)-th success | \\(r,p\\) | \\(r/p\\) | \\(r(1-p)/p^2\\) | \\(\sqrt{r(1-p)}/p\\) |
| Poisson | number of rare events in a fixed window | \\(\lambda\\) | \\(\lambda\\) | \\(\lambda\\) | \\(\sqrt{\lambda}\\) |
| Discrete uniform | one value chosen uniformly from \\(1,2,\ldots,n\\) | \\(n\\) | \\((n+1)/2\\) | \\((n^2-1)/12\\) | \\(\sqrt{(n^2-1)/12}\\) |

### Bernoulli And Binomial: One Success Versus Many Successes {#bernoulli-binomial}

Bernoulli is the smallest random experiment: success is 1, failure is 0.

If \\(X\sim\operatorname{Bernoulli}(p)\\), then:

$$\mathbb{E}[X]=p,\qquad \operatorname{Var}(X)=p(1-p)$$

The variance is largest when \\(p=0.5\\). The intuition is simple: if the success probability is near 0 or near 1, the result is almost determined; if success and failure are equally likely, uncertainty is highest.

Binomial is a sum of \\(n\\) independent Bernoulli variables:

$$Y=X_1+X_2+\cdots+X_n,\qquad X_i\sim\operatorname{Bernoulli}(p)$$

Therefore:

$$\mathbb{E}[Y]=np,\qquad \operatorname{Var}(Y)=np(1-p)$$

This is not a formula that has to be memorized in isolation. It is a direct consequence of "variance adds under independent sums."

For example, flip a fair coin 100 times. The number of heads has mean 50, variance 25, and standard deviation 5. Seeing 45 to 55 heads is not surprising; seeing 30 or 70 heads would be extreme.

### Geometric: Waiting For The First Success {#geometric}

The geometric distribution answers a waiting question: if each trial succeeds with probability \\(p\\), on which trial does the first success occur?

If \\(X\sim\operatorname{Geometric}(p)\\), using the "number of trials" convention \\(X=1,2,3,\ldots\\), then:

$$\mathbb{E}[X]=\frac{1}{p},\qquad \operatorname{Var}(X)=\frac{1-p}{p^2}$$

If \\(p=0.2\\), the average wait is 5 trials. But the variance is 20 and the standard deviation is about 4.47, so the waiting time is widely spread: sometimes the first trial succeeds, and sometimes the wait is long.

{{< alert theme="info" >}}

Some books define the geometric distribution as "the number of failures before the first success," starting at 0. Under that convention, the mean is \\((1-p)/p\\), while the variance is still \\((1-p)/p^2\\). Always check which convention is being used.

{{< /alert >}}

### Poisson: Counting Rare Events {#poisson}

The Poisson distribution models event counts in a fixed window: requests arriving at a server per minute, clicks per day, mutations in a DNA segment, and similar count processes.

If \\(X\sim\operatorname{Poisson}(\lambda)\\), then:

$$\mathbb{E}[X]=\lambda,\qquad \operatorname{Var}(X)=\lambda$$

The special feature of the Poisson distribution is that its mean equals its variance. If \\(\lambda=100\\), the standard deviation is 10; if \\(\lambda=4\\), the standard deviation is 2. The relative spread is:

$$\frac{\sigma}{\mu}=\frac{\sqrt{\lambda}}{\lambda}=\frac{1}{\sqrt{\lambda}}$$

So larger counts have smaller relative noise. This is why high-volume systems often look smoother: absolute variation is larger, but variation as a fraction of the mean is smaller.

Poisson can also be seen as a Binomial limit: the number of trials \\(n\\) is large, the single-trial success probability \\(p\\) is small, and \\(np=\lambda\\) stays fixed. It is the mathematical model for "many opportunities, each individually rare."

## Continuous Distributions: Uniform, Normal, Waiting Times, And Proportions {#continuous-distributions}

Continuous distributions usually describe measurements, errors, proportions, waiting times, or positive scales.

| Distribution | What the random variable describes | Parameters | Mean | Variance | Standard deviation |
| --- | --- | --- | --- | --- | --- |
| Uniform | a value chosen with no preference inside an interval | \\(a,b\\) | \\((a+b)/2\\) | \\((b-a)^2/12\\) | \\((b-a)/\sqrt{12}\\) |
| Normal | error or measurement from many small independent perturbations | \\(\mu,\sigma^2\\) | \\(\mu\\) | \\(\sigma^2\\) | \\(\sigma\\) |
| Exponential | waiting time until the next event | \\(\lambda\\) | \\(1/\lambda\\) | \\(1/\lambda^2\\) | \\(1/\lambda\\) |
| Gamma | waiting time until the \\(k\\)-th event | \\(k,\theta\\) | \\(k\theta\\) | \\(k\theta^2\\) | \\(\sqrt{k}\theta\\) |
| Beta | proportion or probability on \\([0,1]\\) | \\(\alpha,\beta\\) | \\(\alpha/(\alpha+\beta)\\) | \\(\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]\\) | square root of variance |
| Chi-square | sum of squared standard normal variables | \\(\nu\\) | \\(\nu\\) | \\(2\nu\\) | \\(\sqrt{2\nu}\\) |
| Student's t | standardized uncertainty of a small-sample mean | \\(\nu\\) | 0 for \\(\nu>1\\) | \\(\nu/(\nu-2)\\) for \\(\nu>2\\) | \\(\sqrt{\nu/(\nu-2)}\\) |
| F | ratio of two independent sample variances or scaled Chi-square variables | \\(d\_1,d\_2\\) | \\(d\_2/(d\_2-2)\\) for \\(d\_2>2\\) | \\(\frac{2d\_2^2(d\_1+d\_2-2)}{d\_1(d\_2-2)^2(d\_2-4)}\\) for \\(d\_2>4\\) | square root of variance |

### Uniform: Range Without Preference {#uniform}

If \\(X\sim\operatorname{Uniform}(a,b)\\), every position in the interval is equally likely. The mean is the midpoint:

$$\mathbb{E}[X]=\frac{a+b}{2}$$

The variance depends only on interval width:

$$\operatorname{Var}(X)=\frac{(b-a)^2}{12}$$

This matches the transformation rules: shifting the whole interval does not change spread; doubling the interval width doubles the standard deviation and quadruples the variance.

### Normal: The Shape Of Added Error {#normal}

A normal distribution is written as:

$$X\sim\mathcal{N}(\mu,\sigma^2)$$

Its mean is \\(\mu\\), its variance is \\(\sigma^2\\), and its standard deviation is \\(\sigma\\). The parameters directly encode center and scale.

Normal distributions are common not because everything is naturally normal, but because sums of many small independent perturbations tend to become approximately normal. That is the core intuition behind the central limit theorem.

The usual empirical rule:

- about 68% of values lie in \\(\mu\pm 1\sigma\\);
- about 95% lie in \\(\mu\pm 2\sigma\\);
- about 99.7% lie in \\(\mu\pm 3\sigma\\).

So for a normal distribution, the standard deviation is especially concrete: it gives a typical scale of deviation.

### Exponential And Gamma: Waiting For One Event Versus Many Events {#exponential-gamma}

If events occur at average rate \\(\lambda\\), the waiting time until the next event is often modeled with an exponential distribution:

$$X\sim\operatorname{Exponential}(\lambda),\qquad \mathbb{E}[X]=\frac{1}{\lambda},\qquad \operatorname{Var}(X)=\frac{1}{\lambda^2}$$

Its standard deviation also equals \\(1/\lambda\\), the same as the mean. This means waiting times are very spread out: an average wait of 10 seconds does not mean most waits are close to 10 seconds.

The waiting time until the \\(k\\)-th event is a sum of \\(k\\) independent exponential waiting times, which gives a Gamma distribution. Under the shape-scale parameterization:

$$X\sim\operatorname{Gamma}(k,\theta),\qquad \mathbb{E}[X]=k\theta,\qquad \operatorname{Var}(X)=k\theta^2$$

Again, "variance adds under independent sums" appears: waiting for \\(k\\) events multiplies the mean by \\(k\\), multiplies the variance by \\(k\\), but only multiplies the standard deviation by \\(\sqrt{k}\\).

{{< alert theme="info" >}}

Gamma has two common parameterizations: shape-scale \\((k,\theta)\\) and shape-rate \\((\alpha,\beta)\\). If rate \\(\beta=1/\theta\\) is used, the mean is \\(\alpha/\beta\\) and the variance is \\(\alpha/\beta^2\\).

{{< /alert >}}

### Beta: Uncertainty Over A Proportion {#beta}

The Beta distribution lives on \\([0,1]\\), so it is useful for modeling uncertainty over a proportion or probability. For example: what is the true click-through rate of a button?

If \\(X\sim\operatorname{Beta}(\alpha,\beta)\\), then:

$$\mathbb{E}[X]=\frac{\alpha}{\alpha+\beta}$$

$$\operatorname{Var}(X)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

One useful mental model is to treat \\(\alpha\\) and \\(\beta\\) as pseudo-counts of successes and failures. As \\(\alpha+\beta\\) grows, the distribution becomes more concentrated and the variance shrinks. That matches the idea that more evidence makes a proportion estimate more certain.

For example, \\(\operatorname{Beta}(2,2)\\) and \\(\operatorname{Beta}(20,20)\\) both have mean 0.5, but the latter has much smaller variance because it represents stronger evidence.

## Common Relationships: Distributions Are Not Isolated {#relationships}

The relationships between distributions are often easier to remember than isolated formulas.

| Relationship | Intuition |
| --- | --- |
| Binomial = sum of Bernoulli variables | total successes across many 0/1 trials |
| Poisson ≈ Binomial under rare events | \\(n\\) large, \\(p\\) small, \\(np=\lambda\\) |
| Gamma = sum of Exponential variables | waiting time until the \\(k\\)-th event |
| Chi-square = sum of squared standard Normals | foundation for variance estimates and quadratic forms |
| Normal ≈ sum of many small independent perturbations | central limit theorem intuition |
| Beta is conjugate to Binomial | use Beta for an unknown success probability, then update with Binomial evidence |

A unifying view:

> Means usually grow linearly with total amount; variances also grow linearly under independent sums; standard deviations grow only with the square root.

This explains many formulas:

- the variance of \\(n\\) Bernoulli trials is \\(np(1-p)\\);
- the variance of \\(k\\) exponential waiting times is \\(k\theta^2\\);
- the variance of a Chi-square variable with \\(\nu\\) degrees of freedom is \\(2\nu\\);
- the relative spread of a Poisson count is \\(1/\sqrt{\lambda}\\).

## How To Choose A Distribution {#how-to-choose}

In modeling, first choose by the value range and meaning of the random variable:

| What you are modeling | Common candidates |
| --- | --- |
| one success/failure event | Bernoulli |
| number of successes in a fixed number of trials | Binomial |
| number of trials until the first success | Geometric |
| number of events in a fixed window | Poisson |
| continuous value with no preference inside an interval | Uniform |
| measurement error or sum of many small noises | Normal |
| waiting time until the next event | Exponential |
| total waiting time for multiple events | Gamma |
| proportion or probability on \\([0,1]\\) | Beta |
| sample variance and standardized test statistics | Chi-square, Student's t, F |

Then use variance as a sanity check. If count data has mean around 10 but sample variance around 200, a simple Poisson model may be too narrow because Poisson requires mean and variance to match. A Negative Binomial or mixture model may be more appropriate. Conversely, if the data is constrained to \\([0,1]\\), using an unbounded Normal model requires care because it assigns probability outside the valid range.

## Summary {#summary}

Common distributions are not just formula tables. They are a language for what a random variable is doing:

- Bernoulli counts whether one trial succeeds, and Binomial counts total successes across many trials;
- Geometric counts how many trials are needed until the first success;
- Poisson counts rare events inside a fixed window;
- Uniform represents no preference inside a range;
- Normal represents the shape produced by many small independent perturbations;
- Exponential and Gamma describe waiting times;
- Beta describes uncertainty over proportions or probabilities;
- Chi-square, t, and F appear in variance estimation and hypothesis testing.

Variance and standard deviation are the scale language for these distributions. The mean gives the center, variance gives squared-scale spread, and standard deviation brings that spread back to the original unit. The main structure to remember is not every formula by force, but the rules that keep reappearing: shifts do not change variance, scaling changes variance quadratically, and independent sums add variances.
