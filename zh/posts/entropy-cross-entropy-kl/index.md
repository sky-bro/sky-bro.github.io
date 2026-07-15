
很多机器学习概念第一次看起来像三套公式：

- entropy：\\(H(P)\\)
- cross entropy：\\(H(P,Q)\\)
- KL divergence：\\(D_{\mathrm{KL}}(P\Vert Q)\\)

这里的字母首先是一种约定，而不是公式推导出来的变量名。Shannon 在信息论奠基论文中用 \\(H\\) 表示 entropy；[^fn:shannon]为什么选 `H` 没有公认的唯一解释，因此把它记作历史符号最准确。cross entropy 仍然衡量平均编码代价，是 entropy 的推广，所以沿用 \\(H(P,Q)\\)：第一个参数是真实分布，第二个是用于编码的分布。KL 使用 \\(D\\)，是为了强调它衡量两个分布之间的 **divergence（偏离程度）**；但 divergence 不等于 metric，KL 不对称，也不满足三角不等式。

但它们其实在讲同一件事：**如果世界按 \\(P\\) 产生数据，我用什么编码方案去描述这些数据，平均要付多少代价？**

这篇文章可以作为 [Loss Function：模型到底在优化什么]({{< relref "/posts/loss-functions-cross-entropy/" >}}) 的前置基础，也是后续文章《KL Divergence 为什么不是距离》的概念起点。前者讲 loss 该怎么选，后者会讲 KL 的方向为什么不能随便换；本文只回答一个更基础的问题：entropy、cross entropy 和 KL 到底是什么关系。

{{< figure src="/images/posts/entropy-cross-entropy-kl/coding-cost-decomposition.svg" caption="<span class=\"figure-number\">Figure 1: </span>Cross entropy 是总编码代价；entropy 是真实分布本身无法避免的代价；KL 是因为使用错误分布而额外付出的代价。" width="100%" >}}

## 一个两结果世界 {#toy-world}

假设天气只有两种结果：sunny 和 rainy。真实世界的分布是：

$$P=[0.75,\ 0.25]$$

也就是说，sunny 出现 75%，rainy 出现 25%。如果我们知道真实分布，并使用最适合 \\(P\\) 的编码方式，平均编码代价就是 entropy：

$$H(P)=-\sum_x P(x)\log P(x)$$

代入数字：

$$H(P)=-(0.75\log 0.75+0.25\log 0.25)\approx 0.562$$

这里使用自然对数，所以单位是 nats。直觉上，\\(H(P)\\) 衡量的是真实世界本身有多不确定：

- 如果每天都 sunny，\\(P=[1,0]\\)，几乎不用额外信息，entropy 是 0；
- 如果 sunny/rainy 各 50%，\\(P=[0.5,0.5]\\)，更难猜，entropy 更大；
- \\(P=[0.75,0.25]\\) 介于两者之间。

“越均匀，entropy 越大”需要加一个边界：**在结果数量固定时，均匀分布使 entropy 达到最大值**。对二元分布 \\(P=[p,1-p]\\)，有：

$$H(p)=-p\log p-(1-p)\log(1-p)$$

它的一阶和二阶导数是：

$$H'(p)=\log\frac{1-p}{p},\qquad H''(p)=-\frac{1}{p}-\frac{1}{1-p}<0$$

令 \\(H'(p)=0\\) 只得到 \\(p=0.5\\)；又因为二阶导数始终小于 0，曲线严格凹，所以这是唯一最大值。此时 \\(H(P)=\log 2\\)。更一般地，对固定的 \\(n\\) 个结果，均匀分布 \\(U(x)=1/n\\) 满足：

$$D_{\mathrm{KL}}(P\Vert U)=\sum_x P(x)\log\frac{P(x)}{1/n}=\log n-H(P)\ge 0$$

因此 \\(H(P)\le\log n\\)，并且只有 \\(P=U\\) 时取等号。直觉也一致：概率越集中，观察结果前越容易猜；概率越均匀，每个结果越难提前排除。

所以 entropy 不是模型犯错的代价，而是**真实分布本身的不可避免代价**。

## Cross entropy：用 Q 编码 P {#cross-entropy}

现在假设我们不知道真实分布，只能用模型给出的分布 \\(Q\\) 来编码。比如模型认为：

$$Q=[0.5,\ 0.5]$$

它把 sunny 和 rainy 看成一样常见。真实世界仍然按 \\(P=[0.75,0.25]\\) 出数据，但编码系统用的是 \\(Q\\)。平均代价就是 cross entropy：

$$H(P,Q)=-\sum_x P(x)\log Q(x)$$

代入数字：

$$H(P,Q)=-(0.75\log 0.5+0.25\log 0.5)\approx 0.693$$

这比 \\(H(P)\approx 0.562\\) 更大。原因很直接：真实世界 sunny 更多，但 \\(Q\\) 没有利用这个结构，还在用“sunny/rainy 一样多”的编码方案。

如果模型更接近真实分布，比如 \\(Q=[0.7,0.3]\\)，cross entropy 会下降：

$$H(P,Q)=-(0.75\log 0.7+0.25\log 0.3)\approx 0.568$$

这已经接近真实 entropy 0.562。关键直觉是：

**Cross entropy 衡量的是：真实数据按 \\(P\\) 来，但模型用 \\(Q\\) 解释它，平均要付多少代价。**

## KL：多出来的那部分 {#kl}

现在看最重要的关系：

$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$$

换句话说：

$$D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$$

这句话非常有用。它说 KL 不是凭空来的新概念，而是：

**用 \\(Q\\) 编码 \\(P\\) 时，比使用最优编码多付出的额外代价。**

在前面的例子里：

| 量 | 数值 | 含义 |
|---|---:|---|
| \\(H(P)\\) | 0.562 | 真实世界本身的不确定性 |
| \\(H(P,Q)\\), \\(Q=[0.5,0.5]\\) | 0.693 | 用错误模型编码真实世界的总代价 |
| \\(D_{\mathrm{KL}}(P\Vert Q)\\) | 0.131 | 错用 \\(Q\\) 多付出的代价 |

如果 \\(Q=P\\)，cross entropy 就等于 entropy，KL 等于 0。模型没有额外浪费。

如果 \\(Q\\) 在 \\(P\\) 经常出现的位置给太低概率，cross entropy 会变大，KL 也会变大。这就是为什么训练模型时“给真实样本更高概率”会直接降低 loss。

## 为什么训练会用 cross entropy {#training}

训练分类模型时，我们不知道完整的真实分布 \\(P\\)，只拿到样本。比如某个样本真实类别是 rainy，那么 one-hot 标签可以看作一个经验分布：

$$P^{\mathrm{sample}}=[0,\ 1]$$

模型输出：

$$Q_\theta=[0.8,\ 0.2]$$

cross entropy 是：

$$H(P^{\mathrm{sample}},Q_\theta)=-\log 0.2\approx 1.61$$

这就是常见的 negative log likelihood：只看真实类别被模型分到了多少概率。真实类别概率越低，loss 越大；真实类别概率越高，loss 越小。

语言模型 next-token prediction 也是同一件事。给定上下文，训练数据告诉你下一个真实 token 是什么；cross entropy 要求模型给这个 token 更高概率。SFT 也是同样的形式，只是数据从原始文本变成了 prompt-answer 数据。

从 KL 的角度看，训练 cross entropy 等价于让模型分布 \\(Q_\theta\\) 靠近数据分布 \\(P\\)。因为：

$$H(P,Q_\theta)=H(P)+D_{\mathrm{KL}}(P\Vert Q_\theta)$$

训练时 \\(P\\) 固定，所以 \\(H(P)\\) 是常数。最小化 cross entropy，就等价于最小化 \\(D_{\mathrm{KL}}(P\Vert Q_\theta)\\)。

## 一句话总结 {#summary}

这三个量可以这样记：

| 概念 | 问的问题 | 公式 |
|---|---|---|
| Entropy | 真实分布 \\(P\\) 本身有多难编码？ | \\(H(P)\\) |
| Cross entropy | 真实分布是 \\(P\\)，但我用 \\(Q\\) 编码，总代价是多少？ | \\(H(P,Q)\\) |
| KL divergence | 用 \\(Q\\) 比用最优编码多浪费多少？ | \\(D_{\mathrm{KL}}(P\Vert Q)\\) |

最核心的关系是：

$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$$

所以 cross entropy 不是一个孤立的 loss。它是在问：**模型分布 \\(Q\\) 能不能用较低编码代价解释真实数据 \\(P\\)**。KL 则把这个代价拆开，告诉你其中有多少是数据本身的不确定性，有多少是模型分布不对造成的额外浪费。

[^fn:shannon]: Claude E. Shannon, [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), 1948.
