
前向传播给出预测，反向传播计算梯度，梯度下降更新参数。但中间还有一个非常关键的问题：

> 预测结果到底怎样才算“错”？错多少？

这个问题由 **loss function（损失函数）** 回答。它把模型输出和真实标签变成一个标量：

$$\text{loss} = L(\hat{y}, y)$$

训练时我们不是直接优化“准确率”“好不好看”“像不像人类回答”这些抽象目标，而是优化某个可微、可计算、能产生梯度的代理目标。loss function 的选择，本质上是在告诉模型：什么错误更严重，什么方向更值得更新。

这篇文章可以接在前面的 [前向传播与反向传播]({{< relref "/posts/forward-and-backward-propagation/" >}})、[批量梯度下降与随机梯度下降]({{< relref "/posts/batch-vs-stochastic-gradient-descent/" >}})、[Activation Function]({{< relref "/posts/activation-functions-neural-networks/" >}}) 之后看。activation function 决定网络怎样产生表示，loss function 决定这些表示要被训练成什么样。

## 先看选择表 {#cheat-sheet}

| 任务 | 常见输出 | 常用 loss | 直觉 |
| --- | --- | --- | --- |
| 回归：预测房价、温度、长度 | 一个连续数值 | MSE / MAE / Huber | 预测值离真实值越远，惩罚越大 |
| 二分类：垃圾邮件/非垃圾邮件 | 一个 logit 或概率 | Binary Cross Entropy | 正类样本希望概率接近 1，负类样本希望概率接近 0 |
| 多分类：猫/狗/车三选一 | 每个类别一个 logit | Cross Entropy | 正确类别的概率越低，惩罚越大 |
| 多标签分类：一张图可同时有猫和车 | 每个标签一个 logit | 多个 BCE 的和/平均 | 每个标签都是独立的 yes/no 判断 |
| 语言模型 next-token prediction | 词表上每个 token 一个 logit | Cross Entropy / NLL | 下一个真实 token 的概率越低，惩罚越大 |
| 分布匹配、蒸馏、RL 策略约束 | 两个概率分布 | KL Divergence | 一个分布偏离另一个分布多少 |
| margin 分类、排序 | 分数差 | Hinge / Ranking loss | 不只要分对，还要拉开安全边距 |
| 表征学习、检索、embedding | 向量距离或相似度 | Contrastive / Triplet loss | 相似样本靠近，不相似样本远离 |

一个简化规则：

- 目标是**连续数值**：先考虑 MSE、MAE、Huber。
- 目标是**互斥类别**：先考虑 cross entropy。
- 目标是**多个独立标签**：先考虑 binary cross entropy。
- 目标是**分布对齐**：先考虑 KL divergence。
- 目标是**相对关系**，例如谁更像、谁排前面：考虑 margin、contrastive、triplet 这类 loss。

## Loss 不是指标，而是训练信号 {#loss-vs-metric}

假设一个二分类模型输出“这封邮件是垃圾邮件”的概率。真实标签是 1，模型输出 0.51 和 0.99 时，accuracy 都算对。但训练时这两个预测不应该被同等对待：

- 0.51：勉强对，模型还很不确定。
- 0.99：非常确定，方向很好。

accuracy 只有对/错，几乎不给“还差多少”的信息。cross entropy 会给出更细的惩罚：

$$L = -\log(\hat{p})$$

如果真实类别概率 \\(\hat{p}=0.99\\)，loss 约为 0.01；如果 \\(\hat{p}=0.51\\)，loss 约为 0.67；如果 \\(\hat{p}=0.01\\)，loss 约为 4.61。越自信地犯错，惩罚越大。

这就是很多训练过程用 loss 优化、用 metric 汇报的原因：

- **loss**：连续、可微、能产生梯度，适合训练。
- **metric**：更贴近人类关心的结果，适合评估和展示。

## 回归：MSE、MAE 和 Huber {#regression-loss}

回归任务的标签是连续值。例如预测房价：真实值是 100，模型预测 90、110、200 都是数值误差。

### MSE：大错误会被平方放大 {#mse}

Mean Squared Error（MSE）是：

$$L = \frac{1}{n}\sum_i(\hat{y}^{(i)} - y^{(i)})^2$$

看三个预测：

| 真实值 \\(y\\) | 预测 \\(\hat{y}\\) | 误差 | squared error |
| --- | --- | --- | --- |
| 100 | 90 | -10 | 100 |
| 100 | 110 | 10 | 100 |
| 100 | 200 | 100 | 10000 |

MSE 的特点是：大错误会被平方放大。这个性质有时很好，因为离谱预测应该被强烈修正；但如果数据里有异常值，MSE 也会让模型过度关注这些 outlier。

MSE 常用于：

- 普通数值回归。
- 噪声接近高斯分布的预测问题。
- autoencoder 这类重构任务中的像素/特征重构 baseline。

### MAE：对异常值更稳，但梯度不够平滑 {#mae}

Mean Absolute Error（MAE）是：

$$L = \frac{1}{n}\sum_i|\hat{y}^{(i)} - y^{(i)}|$$

同样的三个预测，absolute error 分别是 10、10、100。异常值仍然更大，但不会被平方放大到 10000。

MAE 常用于：

- 希望对 outlier 更稳健的回归。
- 误差本身的绝对大小更容易解释的业务场景。

它的缺点是 0 点附近不可导，实际框架可以用 subgradient 处理，但优化手感通常不如 MSE 平滑。

### Huber：小误差像 MSE，大误差像 MAE {#huber}

Huber loss 把两者接起来：

$$L_\delta(e) = \begin{cases} \frac{1}{2}e^2, & |e|\le\delta \\\\ \delta(|e|-\frac{1}{2}\delta), & |e|>\delta \end{cases}$$

其中 \\(e=\hat{y}-y\\)。小误差时使用平方惩罚，优化平滑；大误差时变成近似线性，减少 outlier 的影响。

Huber 常用于：

- 数据有噪声，但又不想完全放弃 MSE 平滑性的回归。
- reinforcement learning 里的 value function 估计，例如 DQN 常用 smooth L1 / Huber 风格的损失。

## 分类：为什么 Cross Entropy 这么常见 {#classification-loss}

分类任务不是问“数值差多少”，而是问“正确类别的概率是多少”。

假设有三个类别：cat、dog、car。模型输出 logits：

$$z = [2.0,\ 1.0,\ -1.0]$$

softmax 把 logits 转成概率：

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

大致得到：

$$p = [0.71,\ 0.26,\ 0.03]$$

如果真实类别是 dog，那么正确类别概率是 0.26。cross entropy 对 one-hot 标签的形式是：

$$L = -\sum_i y_i\log(p_i)$$

因为 one-hot 里只有 dog 的 \\(y_i=1\\)，所以：

$$L = -\log(0.26) \approx 1.35$$

如果模型把 dog 的概率提高到 0.90，loss 变成：

$$-\log(0.90) \approx 0.105$$

这就是 cross entropy 的核心直觉：**只看真实类别被分到多少概率；概率越低，惩罚越大**。

### Cross entropy 和 NLL 的关系 {#ce-nll}

在深度学习框架里，你经常会看到：

- `CrossEntropyLoss`
- `NLLLoss`
- `log_softmax + nll_loss`

它们的边界是：

- softmax：把 logits 变成概率。
- log softmax：把 logits 变成 log probability。
- negative log likelihood（NLL）：取真实类别的 log probability，再加负号。

所以多分类 cross entropy 可以看成：

$$\text{CE}(\text{logits}, y) = -\log\left(\operatorname{softmax}(\text{logits})_y\right)$$

实际使用 PyTorch 时，`CrossEntropyLoss` 通常直接接收 **raw logits**，内部做 `log_softmax` 和 NLL。不要先手动 softmax 再传进去，否则数值稳定性和梯度都可能变差。

### 二分类：Binary Cross Entropy {#binary-cross-entropy}

二分类可以只输出一个概率 \\(\hat{p}\\)：正类概率。标签 \\(y\\in\{0,1\}\\)。Binary Cross Entropy（BCE）是：

$$L = -\left[y\log(\hat{p}) + (1-y)\log(1-\hat{p})\right]$$

看两个样本：

| 标签 | 预测正类概率 | loss |
| --- | --- | --- |
| 1 | 0.9 | \\(-\log(0.9)\approx 0.105\\) |
| 1 | 0.1 | \\(-\log(0.1)\approx 2.303\\) |

真实是正类时，预测 0.9 的惩罚很小，预测 0.1 的惩罚很大。真实是负类时，公式会转为惩罚 \\(1-\hat{p}\\)。

BCE 常用于：

- 二分类：yes/no、spam/not spam、fraud/not fraud。
- 多标签分类：每个标签都是一个独立二分类。例如一张图可以同时有 cat、car、person。

多标签分类不要用普通 softmax cross entropy，因为 softmax 假设类别互斥：概率总和为 1。多标签场景下，多个标签可以同时为真，所以通常是每个标签一个独立 logit。实现时优先用 `BCEWithLogitsLoss` 这类 with-logits 版本，让 loss 内部完成 sigmoid 和 BCE；如果模型已经输出概率，再使用普通 BCE。

## KL Divergence：比较两个分布 {#kl-divergence}

Cross entropy 通常拿 one-hot 标签和模型预测分布比较。但有些时候，目标本身也是一个分布。

例如 teacher model 给出的 soft label 是：

$$q = [0.70,\ 0.20,\ 0.10]$$

student model 输出：

$$p = [0.60,\ 0.30,\ 0.10]$$

我们不只关心正确类别是哪一个，还关心 student 的整个分布是否像 teacher。KL divergence 常写作：

$$D_{\mathrm{KL}}(q\|p)=\sum_i q_i\log\frac{q_i}{p_i}$$

它衡量的是：如果真实/参考分布是 \\(q\\)，但你用 \\(p\\) 来表示，会多付出多少信息代价。

KL divergence 常用于：

- knowledge distillation：让 student 模仿 teacher 的 soft distribution。
- variational inference / VAE：约束近似后验接近先验或目标分布。
- RLHF / policy optimization：限制新策略不要偏离 reference policy 太远。

需要注意，KL divergence 不对称：

$$D_{\mathrm{KL}}(q\|p) \ne D_{\mathrm{KL}}(p\|q)$$

方向不同，训练行为也会不同。

## Margin 和表征学习：不只看标签，还看相对关系 {#ranking-representation-loss}

有些任务里，我们不关心一个类别概率，而关心相对顺序或 embedding 几何。

### Hinge loss：分对还不够，要有 margin {#hinge}

线性 SVM 常见的 hinge loss 是：

$$L = \max(0,\ 1 - y f(x))$$

其中 \\(y\in\{-1,1\}\\)，\\(f(x)\\) 是模型打分。如果 \\(y f(x) \ge 1\\)，说明不仅分对了，而且离决策边界有足够 margin，loss 为 0。如果只是勉强分对，仍然会被惩罚。

Hinge / margin loss 常用于：

- SVM 风格分类。
- 排序任务：正样本分数应该比负样本高至少一个 margin。
- metric learning 中需要明确拉开距离的场景。

### Contrastive / Triplet loss：塑造 embedding 空间 {#contrastive-triplet}

检索、人脸识别、语义相似度这类任务通常不只是输出一个类别，而是学习一个 embedding 空间。

Triplet loss 使用三元组：

- anchor：当前样本。
- positive：与 anchor 相似的样本。
- negative：与 anchor 不相似的样本。

目标是：

$$d(a,p) + m < d(a,n)$$

也就是 anchor 到 positive 的距离应该比 anchor 到 negative 的距离小，并且至少小一个 margin \\(m\\)。常见形式是：

$$L = \max(0,\ d(a,p)-d(a,n)+m)$$

这类 loss 常用于：

- image/text retrieval。
- face verification。
- sentence embedding 和向量数据库检索。
- contrastive learning，例如让同一个样本的不同增强视图靠近，让不同样本远离。

## 实践中怎么选 {#practical-choice}

可以从输出层和标签形态倒推 loss：

| 你的标签长什么样 | 输出层通常长什么样 | loss |
| --- | --- | --- |
| 一个实数 | linear output | MSE / MAE / Huber |
| 一个 0/1 标签 | one raw logit | BCE with logits |
| 一个互斥类别 id | class logits | Cross entropy |
| 多个 0/1 标签 | one raw logit per label | BCE with logits |
| 一个概率分布 | logits or log probabilities | KL / cross entropy |
| 相似/不相似 pair | embedding vectors | contrastive loss |
| anchor-positive-negative | embedding vectors | triplet loss |

几个常见坑：

- **分类不要随手用 MSE**。MSE 关心数值距离，但分类真正关心的是正确类别概率。cross entropy 通常给出更合适的梯度。
- **多分类和多标签不要混淆**。互斥类别用 softmax cross entropy；多个标签可同时成立时，用 sigmoid + BCE。
- **优先使用 with-logits 版本**。例如 `BCEWithLogitsLoss` 比手动 `sigmoid + BCELoss` 更数值稳定。
- **class imbalance 需要额外处理**。极度不平衡的数据可能需要 class weights、focal loss、重采样或 threshold 调整。
- **loss 下降不等于业务指标一定变好**。loss 是代理目标，要同时看验证集 metric。

## 总结 {#summary}

Loss function 是训练目标的数学接口。它把“什么算错”翻译成“梯度应该往哪里走”。

- MSE、MAE、Huber 适合连续值回归，差别在于如何处理大误差和异常值。
- Binary cross entropy 适合二分类和多标签分类。
- Cross entropy 适合互斥多分类，也是语言模型 next-token prediction 的核心损失。
- KL divergence 适合比较两个分布，常见于蒸馏、VAE 和策略约束。
- Hinge、contrastive、triplet 这类 loss 适合排序、margin 和 embedding 空间学习。

如果只记一个原则：**先看标签和输出代表什么，再选 loss**。loss 不是一个随便替换的 API 参数，而是模型学习问题定义的一部分。
