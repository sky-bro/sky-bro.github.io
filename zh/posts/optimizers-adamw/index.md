
在前面的文章里，我们已经把训练过程拆成了几件事：

- [loss function]({{< relref "/posts/loss-functions-cross-entropy/" >}}) 定义什么叫错；
- [前向传播与反向传播]({{< relref "/posts/forward-and-backward-propagation/" >}}) 计算每个参数的梯度；
- [梯度下降]({{< relref "/posts/batch-vs-stochastic-gradient-descent/" >}}) 根据梯度更新参数。

但真正写训练代码时，我们通常不会直接写：

```python
param = param - lr * param.grad
```

而是写：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
optimizer.step()
```

这里的 optimizer 到底做了什么？AdamW 为什么这么常见？是不是今天训练神经网络基本就用 AdamW，别的 optimizer 都过时了？

一个短答案是：

> AdamW 是现代深度学习里非常强的默认选择，尤其在 Transformer、LLM、diffusion、vision-language model 这类大模型训练中很常见。但它不是唯一选择。SGD with momentum、Adafactor、LAMB/LARS、Lion、Sophia 等仍然在不同约束下有自己的位置。

这篇文章先用一个小例子理解 optimizer，再顺着 SGD → Momentum → RMSProp → Adam → AdamW 的发展线索看它们分别解决什么问题。

{{< figure src="/images/posts/optimizers-adamw/optimizer-evolution-map.svg" caption="<span class=\"figure-number\">图 1： </span>Optimizer 发展地图：主线从 SGD 走到 AdamW，旁支方法通常服务于内存、大 batch 或特定研究配方。" width="100%" >}}

## 先看选择表 {#cheat-sheet}

| Optimizer | 核心想法 | 常见位置 | 主要代价 |
| --- | --- | --- | --- |
| SGD | 沿负梯度方向走一步 | 教学、小模型、部分 CNN/vision 训练 | 对学习率敏感，收敛可能慢 |
| SGD + Momentum | 累积历史方向，减少震荡 | CNN、需要强泛化 baseline 的监督学习 | 仍需要较仔细调学习率 |
| RMSProp / Adagrad | 给不同参数不同步长 | RNN、稀疏特征、早期深度学习 | 只利用二阶矩或累积平方梯度，主流大模型中较少作为默认 |
| Adam | Momentum + 自适应步长 | 通用深度学习 baseline | L2 weight decay 与自适应缩放耦合，泛化/正则化不够干净 |
| AdamW | Adam + decoupled weight decay | Transformer、LLM、diffusion、现代大模型默认强基线 | 需要额外一份一阶矩和二阶矩状态，显存约为参数量的 2 倍以上 |
| Adafactor | 分解二阶矩，节省 optimizer state | 超大模型、显存/内存受限训练 | 行为和超参数不如 AdamW 直观 |
| LAMB / LARS | layer-wise trust ratio | 超大 batch 训练 | 更复杂，主要服务大 batch 稳定性 |
| Lion | 只保留 momentum，用 sign update | 部分视觉/生成模型实验 | 不是所有任务都稳定优于 AdamW |
| Sophia / 二阶近似 | 使用曲率信息调整更新 | LLM 预训练研究 | 实现复杂，生态默认程度不如 AdamW |

实用结论：

- 不知道该用什么时，**AdamW 是很好的起点**。
- 训练传统 CNN 或追求某些监督学习泛化 baseline 时，**SGD + momentum** 仍然值得比较。
- 模型太大、optimizer state 太贵时，考虑 **Adafactor、8-bit optimizer、ZeRO/FSDP optimizer sharding**。
- 超大 batch 或特定预训练设置下，LAMB/LARS/Sophia/Lion 这类方法可能有价值，但通常不是第一步。

## 从一次参数更新到 SGD {#from-update-to-sgd}

### Optimizer 更新的是什么 {#what-optimizer-updates}

假设只有一个参数 \\(w\\)，loss 是：

$$L(w) = (w - 3)^2$$

当 \\(w=0\\) 时：

$$\frac{dL}{dw} = 2(w-3) = -6$$

梯度是 -6，表示如果 \\(w\\) 增大，loss 会下降。最普通的 SGD 更新是：

$$w_{t+1} = w_t - \eta g_t$$

其中 \\(g_t\\) 是当前梯度，\\(\eta\\) 是 learning rate。若 \\(\eta=0.1\\)，则：

$$w_1 = 0 - 0.1\times(-6) = 0.6$$

optimizer 的最小职责就是：拿到每个参数的梯度，决定这一步把参数改多少。

但真实神经网络不是一个参数，而是几百万到几万亿个参数。不同参数的梯度尺度可能差很多；mini-batch 带来的梯度有噪声；loss surface 有狭长峡谷、平坦区、陡峭区。后续 optimizer 的核心改进，基本都围绕三个问题：方向是否稳定、步长是否适配、正则化是否干净。

{{< figure src="/images/posts/optimizers-adamw/sgd-momentum-adam-intuition.svg" caption="<span class=\"figure-number\">图 2： </span>SGD、Momentum 和 Adam 的核心直觉：SGD 跟随当前梯度，Momentum 平滑历史方向，Adam 同时用一阶矩决定方向、用二阶矩归一化步长。" width="100%" >}}

### SGD：最朴素，也最容易看清楚 {#sgd}

SGD 的更新就是：

$$\theta_{t+1} = \theta_t - \eta g_t$$

其中 \\(\theta\\) 是所有参数，\\(g_t=\nabla_\theta L_t(\theta_t)\\) 是当前 mini-batch 上的梯度。

SGD 的优点是简单、状态少、行为直接。它几乎不保存额外状态，只需要参数和梯度。缺点也明显：如图 2 左侧，如果 loss surface 像一个狭长山谷，梯度可能在陡峭方向来回震荡，在真正该前进的方向走得很慢。这就引出 momentum。

## 从 Momentum 到 Adam {#from-momentum-to-adam}

### Momentum：不要只看当前这一步 {#momentum}

Momentum 给 optimizer 加一个速度变量 \\(v\\)：

$$v_t = \beta v_{t-1} + (1-\beta)g_t$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

直觉上，\\(v_t\\) 是梯度的指数移动平均。若某个方向的梯度长期一致，它会被积累；若某个方向来回变号，它会互相抵消。

如图 2 中间所示，Momentum 会让来回变号的方向互相抵消，让长期一致的方向累积起来。因此更新方向更偏向“持续有效”的方向，而不是被单个 mini-batch 的噪声带偏。

这就是 SGD + momentum 仍然有生命力的原因：它便宜、稳定、额外状态只有一份 velocity。在一些 CNN/vision 任务中，它仍然是强 baseline，也常被认为有不错的泛化表现。

### 自适应学习率：每个参数不该同样走 {#adaptive-learning-rate}

Momentum 解决的是“方向噪声”。另一个问题是：不同参数的梯度尺度可能完全不同。如果所有参数共享同一个有效 learning rate，小梯度参数可能几乎不动，大梯度参数可能步子太大。

Adagrad、RMSProp 这类方法会记录梯度平方的历史，并用它缩放更新。RMSProp 的典型形式是：

$$s_t = \rho s_{t-1} + (1-\rho)g_t^2$$

$$\theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{s_t}+\epsilon}$$

如果某个参数长期梯度很大，\\(s_t\\) 会变大，分母变大，实际步长变小；如果某个参数长期梯度很小，分母也小，实际步长相对放大。

这类方法的核心思想是：

> 不同参数拥有不同历史梯度尺度，所以不应该共享完全相同的有效 learning rate。

Adam 正是把 momentum 和这种自适应缩放合到了一起。

### Adam：一阶矩加二阶矩 {#adam}

Adam 可以理解成两份移动平均：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

其中：

- \\(m_t\\)：一阶矩，类似 momentum，记录平均梯度方向。
- \\(v_t\\)：二阶矩，记录梯度平方尺度。

刚开始时 \\(m_0=0, v_0=0\\)，移动平均会偏小，所以 Adam 使用 bias correction：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

最后更新：

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

这行公式可以拆成一句话：

> 沿着历史平均梯度方向走，但用历史梯度平方来归一化每个参数的步长。

图 2 右侧展示了 Adam 的关键：某一维的平均梯度很大，但如果它的历史平方梯度也很大，就会被 \\(\sqrt{\hat{v}_t}\\) 缩小；小梯度但稳定的方向不会被完全淹没。这就是 Adam 好用的关键：它通常比 SGD 对 learning rate 没那么敏感，早期收敛快，能适应不同参数的梯度尺度。

但 Adam 有一个重要问题：weight decay 的处理不够干净。

## AdamW 的地位：默认强基线，不是唯一答案 {#adamw-position}

### AdamW：Adam 的关键修正是 decoupled weight decay {#adamw}

Weight decay 的目的，是把权重往 0 拉一点，抑制过大的参数：

$$\theta \leftarrow \theta - \eta \lambda \theta$$

在普通 SGD 里，把 L2 regularization 加到 loss 里，和在更新时做 weight decay，在形式上是等价的。因为 SGD 直接用梯度更新：

$$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t)$$

但 Adam 不是直接用梯度。Adam 会把梯度放进 \\(m_t\\) 和 \\(v_t\\)，再做自适应缩放。如果把 \\(\lambda\theta\\) 混进 Adam 的梯度里，它也会被二阶矩缩放：

$$\theta_{t+1} = \theta_t - \eta \frac{\text{Adam moments of }(g_t+\lambda\theta_t)}{\sqrt{\text{second moment}}+\epsilon}$$

这意味着“把权重往 0 拉”的力度不再只是由 \\(\lambda\\) 决定，还会被每个参数的历史梯度尺度影响。正则化和自适应学习率纠缠在一起。

AdamW 的改动是：**不要把 weight decay 当成梯度的一部分，而是在 Adam 更新之外单独衰减权重**。

{{< figure src="/images/posts/optimizers-adamw/adam-vs-adamw-weight-decay.svg" caption="<span class=\"figure-number\">图 3： </span>Adam 和 AdamW 的区别：Adam 会把 weight decay 混入自适应 moment 路径；AdamW 把 weight decay 作为单独路径应用，正则化强度更干净。" width="100%" >}}

可以把它写成：

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_t$$

或者理解成两步：

1. 用 Adam 的自适应规则根据梯度更新参数；
2. 额外做一次 decoupled weight decay，把参数按比例往 0 拉。

这就是 AdamW 里的 W：Adam with decoupled Weight decay。

#### 为什么这件事很重要 {#why-adamw-matters}

AdamW 不是“Adam 换了个名字”。它修正的是 Adam 在正则化上的一个边界问题：

- Adam 负责根据梯度历史决定每个参数这一步该怎么走；
- weight decay 负责控制权重规模；
- 两件事解耦之后，\\(\lambda\\) 才更像一个独立的 regularization strength。

这也是为什么现代 Transformer 训练里常见的默认组合是：

```python
optimizer = AdamW(
    params,
    lr=...,
    betas=(0.9, 0.95),  # or (0.9, 0.999), depending on setup
    weight_decay=...
)
```

实际训练中还会配合 learning-rate warmup、cosine decay、gradient clipping、parameter groups。例如 bias、LayerNorm/RMSNorm 的 scale 参数常常不做 weight decay。

### AdamW 是不是唯一默认答案 {#is-adamw-the-only-answer}

AdamW 的地位可以这样理解：

> 它是现代深度学习里最常用、最稳妥的默认强基线之一，但不是理论上或工程上唯一正确的 optimizer。

什么时候 AdamW 很自然？

- Transformer / LLM / diffusion / multimodal model。
- 你希望先得到一个强 baseline，而不是先研究 optimizer。
- 训练预算有限，希望减少调参不确定性。
- 任务没有强烈证据说明别的 optimizer 更合适。

什么时候不一定首选 AdamW？

- **传统 CNN 监督训练**：SGD + momentum 仍然常被比较，尤其在一些视觉 benchmark 上泛化表现很好。
- **内存非常紧**：AdamW 需要保存 \\(m\\) 和 \\(v\\) 两份 optimizer state。混合精度训练中还可能有 fp32 master weights，显存压力更大。Adafactor、8-bit Adam、optimizer sharding 可能更合适。
- **超大 batch 训练**：LAMB、LARS 这类 layer-wise optimizer 可能帮助稳定大 batch scaling。
- **研究性预训练**：Lion、Sophia、Shampoo 等方法可能在特定设置下更快或更省，但需要额外验证。
- **稀疏特征或在线学习**：Adagrad/FTRL 这类方法在推荐/广告等场景仍有位置。

所以更实际的说法不是“现在只有 AdamW”，而是：

> AdamW 是多数现代神经网络训练的默认起点；其他 optimizer 是在特定模型、数据、batch size、内存预算或泛化目标下的有意识选择。

## 训练配方和发展脉络 {#recipe-and-history}

### Optimizer 之外还有训练配方 {#training-recipe}

很多时候，训练效果不是 optimizer 单独决定的，而是 optimizer + schedule + regularization + batch size 的组合。

以 AdamW 为例，真正的 recipe 往往包括：

- learning rate：最关键的超参数之一；
- betas：控制一阶/二阶矩记忆多长；
- weight decay：正则化强度；
- warmup：训练初期逐渐升高 learning rate，避免一开始更新过猛；
- decay schedule：cosine decay、linear decay、constant 等；
- gradient clipping：限制异常大的梯度；
- parameter groups：对 norm、bias、embedding 等参数使用不同 weight decay 或 learning rate。

一个常见误区是：换 optimizer 就能解决所有训练不稳定。实际更常见的是 learning rate、warmup、batch size、初始化、归一化层、loss scale 或数据问题导致训练不稳。

Optimizer 决定“拿到梯度后怎么走”；但它不能替代好的 loss、合理的数据、稳定的模型结构和正确的训练 schedule。

### 发展脉络总结 {#historical-map}

可以把 optimizer 的发展理解成不断修补 SGD 的几个弱点：方向噪声、不同参数的尺度差异、正则化和自适应更新的耦合，以及大模型训练中的 optimizer state 成本。图 1 就是这条脉络的压缩版。

如果只记一条主线：

> SGD 告诉我们沿梯度下降；Momentum 让方向更稳；RMSProp/Adagrad 让每个参数有自己的步长；Adam 把这两类想法合并；AdamW 把 weight decay 从 Adam 的自适应梯度里解耦出来，因此成了现代深度学习的默认强基线。

### 延伸阅读 {#further-reading}

- Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.
- Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017.
- Noam Shazeer and Mitchell Stern, [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235), 2018.
- Xiangning Chen et al., [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675), 2023.
- Hong Liu et al., [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342), 2023.
