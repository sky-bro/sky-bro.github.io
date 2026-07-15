
KL divergence 最容易被误读成“两个分布之间的距离”。这个说法有一半对：它确实在比较两个分布；但另一半很危险：**KL 不是距离，因为方向有意义。** 更准确地说，KL 虽然满足非负性，而且只有 \\(P=Q\\) 时才为 0，却不满足 metric 要求的对称性和三角不等式。

如果你只记一个公式：

$$D_{\mathrm{KL}}(P\Vert Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$$

不要把它读成“P 和 Q 的距离”。更好的读法是：

**如果真实/目标分布是 \\(P\\)，但我用 \\(Q\\) 去近似它，会多付出多少编码代价。**

这句话里，\\(P\\) 和 \\(Q\\) 的角色不对称：

- \\(P\\)：决定哪些位置重要，因为求和时用 \\(P(x)\\) 加权；
- \\(Q\\)：被拿来近似 \\(P\\)，如果它在 \\(P\\) 重要的位置给太低概率，就会被惩罚。

{{< figure src="/images/posts/kl-divergence-not-a-distance/kl-direction.svg" caption="<span class=\"figure-number\">Figure 1: </span>KL 的方向不是符号细节，而是在问两个不同问题：Q 能不能覆盖 P 关心的地方，或者 P 能不能解释 Q 实际使用的地方。" width="100%" >}}

这篇文章接在 [Entropy、Cross Entropy 和 KL Divergence]({{< relref "/posts/entropy-cross-entropy-kl/" >}}) 与 [Loss Function：模型到底在优化什么]({{< relref "/posts/loss-functions-cross-entropy/" >}}) 后面看最自然：前者建立编码代价视角，后者解释 KL 在 loss 家族中的位置。本文再把这套直觉带到 SFT 和 RLHF 的 KL penalty。

## 先从编码代价看 KL {#coding-cost}

假设一个事件只有两个结果：A 和 B。真实分布是：

$$P=[0.5,\ 0.5]$$

这表示 A、B 都常见。现在你用另一个分布 \\(Q\\) 来编码它：

$$Q=[0.99,\ 0.01]$$

\\(Q\\) 几乎认定 A 会发生，B 很少发生。问题来了：如果真实世界其实是 \\(P\\)，你用 \\(Q\\) 的信念去编码，会发生什么？

B 在真实世界里有 50% 概率，但 \\(Q\\) 只给它 1% 概率。也就是说，\\(Q\\) 严重低估了一个经常出现的事件。KL 会重罚这种错误。

手算一下，使用自然对数：

$$D_{\mathrm{KL}}(P\Vert Q)=0.5\log\frac{0.5}{0.99}+0.5\log\frac{0.5}{0.01}\approx 1.61$$

反过来：

$$D_{\mathrm{KL}}(Q\Vert P)=0.99\log\frac{0.99}{0.5}+0.01\log\frac{0.01}{0.5}\approx 0.64$$

同一对分布，两个方向不一样。原因不是公式怪，而是两个方向在问不同问题：

- \\(D_{\mathrm{KL}}(P\Vert Q)\\)：如果 \\(P\\) 是目标，\\(Q\\) 漏掉了 \\(P\\) 认为常见的 B，代价很大。
- \\(D_{\mathrm{KL}}(Q\Vert P)\\)：如果 \\(Q\\) 是目标，\\(Q\\) 自己几乎只关心 A，而 \\(P\\) 至少也给 A 50% 概率，代价没那么大。

所以 KL 的第一个参数不是“随便放左边的分布”，而是**拿来加权错误的位置**。

## 零概率会让方向差异更明显 {#zero-probability}

再看一个极端例子：

$$P=[0.5,\ 0.5],\quad Q=[1.0,\ 0.0]$$

\\(Q\\) 认为 B 完全不可能发生。但 \\(P\\) 认为 B 有 50% 概率。于是：

$$D_{\mathrm{KL}}(P\Vert Q)=\infty$$

因为公式里有 \\(\log(P(B)/Q(B))\\)，而 \\(Q(B)=0\\)。这不是数值小问题，而是语义问题：你用一个认为 B 不可能的编码系统，去编码一个经常出现 B 的真实世界，代价无限大。

但反过来：

$$D_{\mathrm{KL}}(Q\Vert P)=1.0\log\frac{1.0}{0.5}+0.0\log\frac{0.0}{0.5}\approx 0.69$$

这里按极限约定把 \\(0\log(0/q)\\) 记为 0。\\(Q\\) 自己只会产生 A，而 \\(P\\) 给 A 的概率不是 0，所以没有无限大问题。

这个例子给出一个实用判断：

**KL 的左边分布在哪里有质量，右边分布就必须在那里也给足够概率。右边漏掉左边的支持集，会非常严重。**

## 和 Cross Entropy、SFT 是什么关系 {#cross-entropy}

KL、entropy、cross entropy 的关系是：

$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$$

其中：

- \\(H(P)\\)：真实分布 \\(P\\) 自己的不可避免编码代价；
- \\(H(P,Q)\\)：真实分布是 \\(P\\)，但使用 \\(Q\\) 编码的平均代价；
- \\(D_{\mathrm{KL}}(P\Vert Q)\\)：因为用了错误的 \\(Q\\)，额外多付出的代价。

训练分类模型时，数据分布 \\(P_{\text{data}}\\) 是固定的，模型分布 \\(Q_\theta\\) 会变。最小化 cross entropy：

$$H(P_{\text{data}}, Q_\theta)$$

等价于最小化：

$$D_{\mathrm{KL}}(P_{\text{data}}\Vert Q_\theta)$$

因为 \\(H(P_{\text{data}})\\) 不依赖模型参数。这个方向很重要：模型必须在数据真实出现的位置给足概率。比如训练集中某个类别经常出现，模型就不能把它概率压到接近 0。

语言模型 next-token prediction 也是同一个结构。训练数据给出真实 token，cross entropy 让模型在这些真实 token 上提高概率。它本质上是让模型分布覆盖数据分布，而不是让模型只挑自己最喜欢的一小块模式。

SFT 也是这个方向。给定 prompt \\(x\\) 和人工答案 token 序列 \\(y_{1:T}\\)，经验数据分布可以看成在每个位置把概率质量放在真实 token 上的 one-hot 分布。对每个位置来说，SFT 最小化的是 \\(-\log \pi^{\theta}(y^{(t)}\mid x,y^{(1:t-1)})\\)，再把所有 token 位置加起来。

也就是最小化经验分布到模型分布的 cross entropy，等价于一个 data-to-model 方向的 KL：\\(D_{\mathrm{KL}}(P^{\mathrm{data}}\Vert \pi^{\theta})\\)。

如果你听说“SFT 和 RL 里的 KL 方向是反的”，通常指的是这个对比：**SFT 里数据/教师答案在左边，当前模型在右边；而 RLHF 的 KL penalty 常把当前 policy 放左边，reference policy 放右边。** 前者问“模型有没有给数据答案足够概率”，后者问“当前模型实际采样出来的回答有没有跑到 reference 不认可的地方”。

## 两个方向的直觉：mode-covering 和 mode-seeking {#mode-covering}

在机器学习里，经常会粗略说：

- \\(D_{\mathrm{KL}}(P\Vert Q)\\)：更 **mode-covering**，希望 \\(Q\\) 覆盖 \\(P\\) 的主要模式；
- \\(D_{\mathrm{KL}}(Q\Vert P)\\)：更 **mode-seeking**，希望 \\(Q\\) 集中在 \\(P\\) 的高概率区域。

这里的 mode 可以理解成“概率分布的峰”。比如目标分布 \\(P\\) 有两个高峰：一个在左边，一个在右边。

如果优化 \\(D_{\mathrm{KL}}(P\Vert Q)\\)，\\(P\\) 在两个峰上都有质量，\\(Q\\) 漏掉任何一个峰都会被 \\(P\\) 加权惩罚。所以 \\(Q\\) 倾向于覆盖两个峰，即使中间区域也给了一些概率。

如果优化 \\(D_{\mathrm{KL}}(Q\Vert P)\\)，求和时由 \\(Q\\) 加权。只要 \\(Q\\) 自己别把质量放到 \\(P\\) 很低的地方，它可以集中在其中一个峰上。它不一定被强迫覆盖 \\(P\\) 的所有峰。

{{< figure src="/images/posts/kl-divergence-not-a-distance/kl-continuous-modes.svg" caption="<span class=\"figure-number\">Figure 2: </span>连续分布里更容易看出两个方向的差异：forward KL 更怕漏掉 P 的峰，reverse KL 更怕 Q 把质量放到 P 很低的谷底。" width="100%" >}}

这个说法不是定理的全部，但作为工程直觉很有用：

| 方向 | 更怕什么错误 | 常见行为 |
|---|---|---|
| \\(D_{\mathrm{KL}}(P\Vert Q)\\) | \\(Q\\) 漏掉 \\(P\\) 常见的东西 | 覆盖目标分布 |
| \\(D_{\mathrm{KL}}(Q\Vert P)\\) | \\(Q\\) 把质量放到 \\(P\\) 很低的地方 | 集中到高概率模式 |

## 回到 RLHF：KL penalty 在约束什么 {#rlhf-kl}

在 RLHF 或 PPO 风格的 language model training 里，常见写法是：

$$\underset{\theta}{\max}\ \mathbb{E}^{y\sim\pi^{\theta}(\cdot\mid x)}[r(x,y)]-\beta D_{\mathrm{KL}}(\pi^{\theta}(\cdot\mid x)\Vert\pi^{\mathrm{ref}}(\cdot\mid x))$$

这里：

- \\(\pi_\theta\\)：正在训练的新 policy，也就是当前语言模型；
- \\(\pi_{\mathrm{ref}}\\)：冻结的 reference policy，通常来自 SFT 模型；
- \\(r(x,y)\\)：reward model 或 verifier 给回答的分数；
- \\(\beta\\)：KL penalty 的强度。

这个方向可以读成：

**当前模型实际会采样出来的回答，不要离 reference 模型太远。**

为什么是这个方向？因为 RL rollout 本来就是从当前 policy \\(\pi_\theta\\) 采样的。训练时看到的是当前模型真的会输出的 token 轨迹，于是可以对这些轨迹计算：

$$\log \pi_\theta(y\mid x)-\log \pi_{\mathrm{ref}}(y\mid x)$$

如果当前模型开始输出 reference 模型认为极不自然的句子，这一项会变大，KL penalty 就会把它拉回来。

如果反过来写成 \\(D_{\mathrm{KL}}(\pi_{\mathrm{ref}}\Vert\pi_\theta)\\)，语义会变成：

**reference 模型会输出的东西，当前模型也要覆盖。**

这不是没有意义，但它约束的是另一个问题。它会更像“不要丢掉 reference 的覆盖面”，而不是“当前模型别跑到 reference 不认可的奇怪区域”。在 PPO/RLHF 的采样流程里，我们主要观察当前 policy 的轨迹，所以 \\(D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\\) 更自然，也更容易用采样估计。

## 为什么 KL penalty 不能太小，也不能太大 {#kl-strength}

KL penalty 的作用像一根绳子：

- 绳子太松：模型可能钻 reward model 的漏洞，变得啰嗦、投机、格式怪异，甚至损坏通用能力；
- 绳子太紧：模型几乎不能偏离 SFT reference，RL 学不到偏好或可验证奖励带来的改进。

所以 RLHF 里经常要调 \\(\beta\\)，或者使用 adaptive KL controller，让实际 KL 维持在某个目标范围附近。

这也解释了为什么 SFT 和 RLHF 是配合关系：SFT 先给出一个合理 reference，RLHF 再在它附近优化偏好。如果 reference 本身很差，KL penalty 只会把模型拴在一个不好的区域；如果没有 KL penalty，RL 又容易把模型推到 reward model 的盲区。

## 一句话总结 {#summary}

KL divergence 的核心不是“两个分布差多少”，而是：

**我把左边分布当成真实/目标，用右边分布去近似它，会在左边认为重要的位置犯多少错误。**

所以：

- \\(D_{\mathrm{KL}}(P\Vert Q)\\) 惩罚 \\(Q\\) 漏掉 \\(P\\) 关心的地方；
- \\(D_{\mathrm{KL}}(Q\Vert P)\\) 惩罚 \\(Q\\) 自己跑到 \\(P\\) 不支持的地方；
- cross entropy 训练常等价于最小化 \\(D_{\mathrm{KL}}(P_{\text{data}}\Vert Q_\theta)\\)；
- RLHF 的 KL penalty 常用 \\(D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\\)，是在约束当前 policy 的实际输出别偏离 reference 太远。

一旦你把“谁在左边，谁负责加权”记住，KL 的方向就不再是符号细节，而是训练目标本身。

## 延伸阅读 {#further-reading}

- Cover and Thomas, *Elements of Information Theory*, Chapter 2.
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.
