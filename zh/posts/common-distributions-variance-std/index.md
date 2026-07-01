
学概率时，很多分布看起来像一张公式清单：Bernoulli、Binomial、Poisson、Normal、Exponential、Gamma、Beta……如果只背概率质量函数或密度函数，很容易忘记它们各自回答什么问题。

更稳定的记法是先问两个问题：

- 这个随机变量在数什么？
- 它的波动有多大？

均值回答“中心在哪里”；方差和标准差回答“围绕中心散得有多开”。这篇文章把常见分布放在同一张地图里，重点总结它们的均值、方差、标准差，以及这些公式背后的直觉。

{{< figure src="/images/posts/common-distributions-variance-std/distribution-shapes-overview.svg" caption="<span class=\"figure-number\">图 1： </span>常见分布的形状速览：离散分布用柱状/针状图表示概率质量，连续分布用曲线表示密度；图只负责直觉，公式放在后面的表格里。" width="100%" >}}

图中的“形状”来自概率质量函数（PMF）或概率密度函数（PDF）：

| 分布 | 图对应的公式 |
| --- | --- |
| Bernoulli | \\(\Pr(X=1)=p,\ \Pr(X=0)=1-p\\) |
| Binomial | \\(\Pr(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\\) |
| Poisson | \\(\Pr(X=k)=e^{-\lambda}\lambda^k/k!\\) |
| Geometric | \\(\Pr(X=k)=(1-p)^{k-1}p,\ k=1,2,\ldots\\) |
| Uniform | \\(f(x)=1/(b-a),\ a\le x\le b\\) |
| Normal | \\(f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-(x-\mu)^2/(2\sigma^2)\right]\\) |
| Exponential | \\(f(x)=\lambda e^{-\lambda x},\ x\ge 0\\) |
| Gamma / Beta | \\(f\_{\text{Gamma}}(x)=x^{k-1}e^{-x/\theta}/(\Gamma(k)\theta^k)\\)；\\(f\_{\text{Beta}}(x)=x^{\alpha-1}(1-x)^{\beta-1}/B(\alpha,\beta)\\) |

## 先把方差和标准差讲清楚 {#variance-first}

随机变量 \\(X\\) 的方差定义为：

$$\operatorname{Var}(X) = \mathbb{E}\left[(X-\mathbb{E}[X])^2\right]$$

它衡量的是：样本值离均值的平方距离，平均起来有多大。常用的等价计算式是：

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

标准差是方差的平方根：

$$\sigma_X = \sqrt{\operatorname{Var}(X)}$$

为什么还要标准差？因为方差的单位会被平方。例如 \\(X\\) 的单位是“秒”，方差单位就是“秒平方”；标准差重新回到“秒”，更适合和原变量直接比较。

三个规则很有用：

| 操作 | 均值 | 方差 | 标准差 |
| --- | --- | --- | --- |
| 平移 \\(X+c\\) | \\(\mathbb{E}[X]+c\\) | \\(\operatorname{Var}(X)\\) | \\(\sigma_X\\) |
| 缩放 \\(aX\\) | \\(a\mathbb{E}[X]\\) | \\(a^2\operatorname{Var}(X)\\) | \\(\lvert a\rvert\sigma_X\\) |
| 独立相加 \\(X+Y\\) | \\(\mathbb{E}[X]+\mathbb{E}[Y]\\) | \\(\operatorname{Var}(X)+\operatorname{Var}(Y)\\) | 不能直接相加 |

注意最后一行：独立随机变量相加时，**方差相加，不是标准差相加**。这也是很多分布方差公式的来源。

## 离散分布：从一次试验到计数 {#discrete-distributions}

离散分布通常在“数次数、数个数、数第几次成功”。

| 分布 | 随机变量在数什么 | 参数 | 均值 | 方差 | 标准差 |
| --- | --- | --- | --- | --- | --- |
| Bernoulli | 一次 0/1 试验是否成功 | \\(p\\) | \\(p\\) | \\(p(1-p)\\) | \\(\sqrt{p(1-p)}\\) |
| Binomial | \\(n\\) 次独立试验中成功几次 | \\(n,p\\) | \\(np\\) | \\(np(1-p)\\) | \\(\sqrt{np(1-p)}\\) |
| Geometric | 第一次成功发生在第几次试验 | \\(p\\) | \\(1/p\\) | \\((1-p)/p^2\\) | \\(\sqrt{1-p}/p\\) |
| Negative binomial | 得到第 \\(r\\) 次成功需要几次试验 | \\(r,p\\) | \\(r/p\\) | \\(r(1-p)/p^2\\) | \\(\sqrt{r(1-p)}/p\\) |
| Poisson | 固定时间/空间窗口内发生几次稀有事件 | \\(\lambda\\) | \\(\lambda\\) | \\(\lambda\\) | \\(\sqrt{\lambda}\\) |
| Discrete uniform | \\(1,2,\ldots,n\\) 中等概率取一个 | \\(n\\) | \\((n+1)/2\\) | \\((n^2-1)/12\\) | \\(\sqrt{(n^2-1)/12}\\) |

### Bernoulli 和 Binomial：一次成功与多次成功 {#bernoulli-binomial}

Bernoulli 是最小的随机试验：成功记为 1，失败记为 0。

若 \\(X\sim\operatorname{Bernoulli}(p)\\)，则：

$$\mathbb{E}[X]=p,\qquad \operatorname{Var}(X)=p(1-p)$$

这个方差在 \\(p=0.5\\) 时最大。直觉很简单：如果成功概率接近 0 或 1，结果几乎确定，波动小；如果成功和失败各半，最不确定，波动最大。

Binomial 是 \\(n\\) 个独立 Bernoulli 的和：

$$Y=X_1+X_2+\cdots+X_n,\qquad X_i\sim\operatorname{Bernoulli}(p)$$

所以：

$$\mathbb{E}[Y]=np,\qquad \operatorname{Var}(Y)=np(1-p)$$

这不是一个需要死背的公式，而是“独立相加时方差相加”的直接结果。

举个例子：一枚硬币抛 100 次，\\(p=0.5\\)。正面次数的均值是 50，方差是 25，标准差是 5。也就是说，正面次数落在 45 到 55 附近并不奇怪；落在 30 或 70 就非常极端。

### Geometric：等待第一次成功 {#geometric}

Geometric 分布回答的是等待问题：每次试验成功概率为 \\(p\\)，第一次成功在第几次出现？

若 \\(X\\sim\operatorname{Geometric}(p)\\)，这里采用“试验次数”版本，即 \\(X=1,2,3,\ldots\\)，则：

$$\mathbb{E}[X]=\frac{1}{p},\qquad \operatorname{Var}(X)=\frac{1-p}{p^2}$$

如果成功概率 \\(p=0.2\\)，平均要等 5 次。但方差是 20，标准差约 4.47，说明等待时间很分散：有时第 1 次就成功，有时要等很久。

{{< alert theme="info" >}}

有些教材把 Geometric 定义为“第一次成功前失败了几次”，取值从 0 开始。这时均值是 \\((1-p)/p\\)，方差仍是 \\((1-p)/p^2\\)。读公式时一定先确认采用哪种约定。

{{< /alert >}}

### Poisson：稀有事件计数 {#poisson}

Poisson 分布适合描述固定窗口中的事件个数，例如一分钟内服务器收到的请求数、一个网页一天内收到的点击数、某段 DNA 上突变的数量。

若 \\(X\sim\operatorname{Poisson}(\lambda)\\)，则：

$$\mathbb{E}[X]=\lambda,\qquad \operatorname{Var}(X)=\lambda$$

Poisson 最特别的地方是均值等于方差。\\(\lambda=100\\) 时，标准差是 10；\\(\lambda=4\\) 时，标准差是 2。相对波动大约是：

$$\frac{\sigma}{\mu}=\frac{\sqrt{\lambda}}{\lambda}=\frac{1}{\sqrt{\lambda}}$$

所以计数越大，相对波动越小。这也是为什么大流量系统看起来更“平滑”：绝对波动变大了，但相对均值的比例变小了。

Poisson 还可以看成 Binomial 的极限：试验次数 \\(n\\) 很大、单次成功概率 \\(p\\) 很小，但 \\(np=\lambda\\) 保持固定。这就是“很多机会，每个机会都很罕见”的数学模型。

## 连续分布：从均匀、正态到等待时间 {#continuous-distributions}

连续分布通常描述测量值、误差、比例、等待时间或正数尺度。

| 分布 | 随机变量在描述什么 | 参数 | 均值 | 方差 | 标准差 |
| --- | --- | --- | --- | --- | --- |
| Uniform | 区间内等概率取值 | \\(a,b\\) | \\((a+b)/2\\) | \\((b-a)^2/12\\) | \\((b-a)/\sqrt{12}\\) |
| Normal | 多个小独立扰动叠加后的误差/测量值 | \\(\mu,\sigma^2\\) | \\(\mu\\) | \\(\sigma^2\\) | \\(\sigma\\) |
| Exponential | 等待下一次事件的时间 | \\(\lambda\\) | \\(1/\lambda\\) | \\(1/\lambda^2\\) | \\(1/\lambda\\) |
| Gamma | 等待第 \\(k\\) 次事件的时间 | \\(k,\theta\\) | \\(k\theta\\) | \\(k\theta^2\\) | \\(\sqrt{k}\theta\\) |
| Beta | \\([0,1]\\) 上的比例/概率 | \\(\alpha,\beta\\) | \\(\alpha/(\alpha+\beta)\\) | \\(\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]\\) | 方差开根号 |
| Chi-square | 标准正态平方和 | \\(\nu\\) | \\(\nu\\) | \\(2\nu\\) | \\(\sqrt{2\nu}\\) |
| Student's t | 小样本均值标准化后的不确定性 | \\(\nu\\) | 0（\\(\nu>1\\)） | \\(\nu/(\nu-2)\\)（\\(\nu>2\\)） | \\(\sqrt{\nu/(\nu-2)}\\) |
| F | 两个独立样本方差比或两个缩放 Chi-square 的比 | \\(d\_1,d\_2\\) | \\(d\_2/(d\_2-2)\\)（\\(d\_2>2\\)） | \\(\frac{2d\_2^2(d\_1+d\_2-2)}{d\_1(d\_2-2)^2(d\_2-4)}\\)（\\(d\_2>4\\)） | 方差开根号 |

### Uniform：只有范围，没有偏好 {#uniform}

若 \\(X\sim\operatorname{Uniform}(a,b)\\)，每个区间位置同样可能。均值在中点：

$$\mathbb{E}[X]=\frac{a+b}{2}$$

方差只取决于区间长度：

$$\operatorname{Var}(X)=\frac{(b-a)^2}{12}$$

这很好理解：把整个区间平移不会改变离散程度；把区间宽度放大 2 倍，标准差也放大 2 倍，方差放大 4 倍。

### Normal：误差叠加后的形状 {#normal}

正态分布写作：

$$X\sim\mathcal{N}(\mu,\sigma^2)$$

它的均值就是 \\(\mu\\)，方差就是 \\(\sigma^2\\)，标准差就是 \\(\sigma\\)。这里参数直接把中心和尺度写进了分布名。

正态分布常见不是因为所有东西天然正态，而是因为很多独立小扰动相加后会趋近正态。这是中心极限定理的核心直觉。

经验规则：

- 约 68% 的值落在 \\(\mu\pm 1\sigma\\)；
- 约 95% 的值落在 \\(\mu\pm 2\sigma\\)；
- 约 99.7% 的值落在 \\(\mu\pm 3\sigma\\)。

所以标准差在正态分布里特别直观：它给了一个“典型偏离量”的尺度。

### Exponential 和 Gamma：等待一个事件与等待多个事件 {#exponential-gamma}

如果事件以平均速率 \\(\lambda\\) 发生，等待下一个事件的时间常用 Exponential 分布：

$$X\sim\operatorname{Exponential}(\lambda),\qquad \mathbb{E}[X]=\frac{1}{\lambda},\qquad \operatorname{Var}(X)=\frac{1}{\lambda^2}$$

它的标准差也等于 \\(1/\lambda\\)，和均值相同。这意味着等待时间的波动非常大：平均等 10 秒，不代表大多数时候都接近 10 秒。

等待第 \\(k\\) 次事件的时间是 \\(k\\) 个独立 Exponential 的和，也就是 Gamma 分布。若使用 shape-scale 参数化：

$$X\sim\operatorname{Gamma}(k,\theta),\qquad \mathbb{E}[X]=k\theta,\qquad \operatorname{Var}(X)=k\theta^2$$

这里再次出现“独立相加时方差相加”：等待 \\(k\\) 个事件，均值放大 \\(k\\) 倍，方差也放大 \\(k\\) 倍，但标准差只放大 \\(\sqrt{k}\\) 倍。

{{< alert theme="info" >}}

Gamma 有两种常见参数化：shape-scale \\((k,\theta)\\) 和 shape-rate \\((\alpha,\beta)\\)。如果使用 rate \\(\beta=1/\theta\\)，则均值是 \\(\alpha/\beta\\)，方差是 \\(\alpha/\beta^2\\)。

{{< /alert >}}

### Beta：比例的不确定性 {#beta}

Beta 分布定义在 \\([0,1]\\)，适合描述比例或概率本身的不确定性。例如“某个按钮的真实点击率是多少”。

若 \\(X\sim\operatorname{Beta}(\alpha,\beta)\\)，则：

$$\mathbb{E}[X]=\frac{\alpha}{\alpha+\beta}$$

$$\operatorname{Var}(X)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

可以把 \\(\alpha\\) 和 \\(\beta\\) 粗略理解为成功和失败的伪计数。\\(\alpha+\beta\\) 越大，分布越集中，方差越小；这对应“样本越多，对比例估计越有把握”。

例如 \\(\operatorname{Beta}(2,2)\\) 和 \\(\operatorname{Beta}(20,20)\\) 的均值都是 0.5，但后者方差小得多，因为它表示更强的证据。

## 常见关系：很多分布不是孤立的 {#relationships}

把分布之间的关系记住，比单独背公式更可靠。

| 关系 | 直觉 |
| --- | --- |
| Binomial = 多个 Bernoulli 相加 | 多次 0/1 试验的成功总数 |
| Poisson ≈ 稀有事件下的 Binomial | \\(n\\) 很大、\\(p\\) 很小、\\(np=\lambda\\) |
| Gamma = 多个 Exponential 相加 | 等待第 \\(k\\) 次事件 |
| Chi-square = 多个标准 Normal 平方相加 | 方差估计和二次型的基础 |
| Normal ≈ 很多小独立扰动相加 | 中心极限定理的主要直觉 |
| Beta 和 Binomial 共轭 | 用 Beta 表示未知成功率，用 Binomial 更新证据 |

一个统一视角是：

> 均值通常跟“总量”线性增长，方差在独立相加时也线性增长，但标准差只按平方根增长。

这解释了很多公式：

- \\(n\\) 次 Bernoulli 的方差是 \\(np(1-p)\\)；
- \\(k\\) 个 Exponential 的方差是 \\(k\theta^2\\)；
- \\(\nu\\) 个标准正态平方的 Chi-square 方差是 \\(2\nu\\)；
- Poisson 的相对波动是 \\(1/\sqrt{\lambda}\\)。

## 怎么选择分布 {#how-to-choose}

实际建模时，可以先按随机变量的取值范围和语义来选：

| 你在建模什么 | 常见候选 |
| --- | --- |
| 一次成功/失败 | Bernoulli |
| 固定次数试验中的成功数 | Binomial |
| 等到第一次成功要几次 | Geometric |
| 固定窗口里的事件数 | Poisson |
| 区间内没有偏好的连续值 | Uniform |
| 测量误差或许多小噪声之和 | Normal |
| 等待下一个事件的时间 | Exponential |
| 等待多个事件的总时间 | Gamma |
| \\([0,1]\\) 上的比例或概率 | Beta |
| 样本方差、标准化统计量 | Chi-square、Student's t、F |

最后再用方差检查模型是否合理。比如数据的计数均值约为 10，但样本方差约为 200，那么简单 Poisson 可能不够，因为 Poisson 要求均值等于方差；这时可能要考虑 Negative Binomial 或混合模型。反过来，如果数据被固定在 \\([0,1]\\)，却用无限支撑的 Normal 去建模，也要小心边界外概率带来的问题。

## 总结 {#summary}

常见分布不只是公式表，而是一组关于“随机变量在数什么”的语言：

- Bernoulli 数一次是否成功，Binomial 数多次成功总数；
- Geometric 数等到第一次成功的试验次数；
- Poisson 数固定窗口里的稀有事件；
- Uniform 表示范围内没有偏好；
- Normal 表示许多小扰动相加后的误差形状；
- Exponential 和 Gamma 描述等待时间；
- Beta 描述比例或概率的不确定性；
- Chi-square、t、F 常出现在方差估计和假设检验里。

方差和标准差则是这些分布的尺度语言。均值告诉我们中心，方差告诉我们平方尺度上的波动，标准差把波动带回原单位。真正要记住的不是每一个公式，而是这些公式反复体现的结构：平移不改变方差，缩放会平方地改变方差，独立相加时方差相加。
