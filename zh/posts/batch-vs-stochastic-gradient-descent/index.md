
## 引言 {#introduction}

从上一篇前向传播与反向传播的文章中，我们已知梯度下降的更新公式：

\\[
\theta \leftarrow \theta - \eta \nabla C
\\]

其中 \\(\nabla C\\) 是**整个训练数据集**上的损失梯度。但计算 \\(\nabla C\\) 意味着要对每一个训练样本运行前向传播和反向传播，然后取平均。当数据集有数百万样本时，这是极其昂贵的。

我们如何计算 \\(\nabla C\\) 以及何时更新 \\(\theta\\)，就引出了梯度下降的不同变体：**批量**（batch）、**随机**（stochastic）和**小批量**（mini-batch）。


## Batch Gradient Descent（批量梯度下降） {#batch-gradient-descent}

Batch Gradient Descent（BGD）在每次更新前使用**所有**训练样本来计算梯度。

给定包含 \\(m\\) 个样本的数据集，梯度为所有单个梯度的平均值：

\\[
\nabla C(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla C\_i(\theta)
\\]

更新步骤为：

\\[
\theta \leftarrow \theta - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla C\_i(\theta)
\\]

**特点：**

- 每次更新都使用完整数据集，梯度方向准确
- 损失平滑稳定地向最小值下降
- 对于凸损失函数，保证收敛到全局最小值

**缺点：**

- 每次迭代很慢——必须处理所有 \\(m\\) 个样本
- 需要将整个数据集加载到内存
- 对于非常大的数据集，单次迭代可能不可行


## Stochastic Gradient Descent（随机梯度下降） {#stochastic-gradient-descent}

Stochastic Gradient Descent（SGD）在**每个**训练样本之后就更新参数：

\\[
\theta \leftarrow \theta - \eta \cdot \nabla C\_i(\theta)
\\]

训练数据通常被打乱（shuffle），逐个样本处理，然后在下一轮（epoch）重新打乱。

**特点：**

- 更新非常快——每步只需一个样本
- 单个样本梯度的噪声有助于跳出浅的局部最小值
- 内存高效——同时只需一个样本在内存中

**缺点：**

- 梯度方向波动剧烈，导致损失在最优值附近**震荡**
- 永远不会真正收敛——在最小值周围不断跳动
- 必须使用学习率调度（learning rate scheduling）来逐渐减小震荡


## Mini-Batch Gradient Descent（小批量梯度下降） {#mini-batch-gradient-descent}

Mini-Batch Gradient Descent 是**折中方案**：不用单个样本也不用全部样本，而是使用一个大小为 \\(k\\) 的小批量（常用 32、64、128 或 256）：

\\[
\nabla C(\theta) = \frac{1}{k} \sum_{j \in \text{batch}} \nabla C\_j(\theta)
\\]

\\[
\theta \leftarrow \theta - \eta \cdot \frac{1}{k} \sum_{j \in \text{batch}} \nabla C\_j(\theta)
\\]

{{< figure src="/images/posts/batch-vs-stochastic-gradient-descent/mini-batch-concept.svg" caption="<span class=\"figure-number\">图 1： </span>mini-batch 概念：将数据集分成大小为 4 的批次，每个批次触发一次参数更新" >}}

**为什么 mini-batch 是默认选择：**

1. **向量化（vectorization）**：现代硬件（GPU/TPU）针对批量矩阵运算进行了优化。一次处理 128 个样本比 128 次单独更新快得多。
2. **降低噪声**：对一个批次取平均，平滑了梯度噪声，收敛比纯 SGD 更稳定。
3. **内存友好**：128 个样本的批次可以轻松放入 GPU 内存，而数百万样本不行。
4. **一个 epoch = 多次更新**：当 \\(m = 10000\\)、\\(k = 100\\) 时，一个 epoch 产生 100 次更新，优化器有更多机会调整方向。


## 对比 {#comparison}

{{< figure src="/images/posts/batch-vs-stochastic-gradient-descent/convergence-comparison.svg" caption="<span class=\"figure-number\">图 2： </span>BGD（平滑）、SGD（噪声）和 mini-batch（适度噪声但高效）的收敛行为" >}}

| | Batch GD | Stochastic GD | Mini-Batch GD |
|---|---|---|---|
| **每次更新的样本数** | \\(m\\)（全部） | 1 | \\(k\\)（如 32） |
| **每 epoch 的更新次数** | 1 | \\(m\\) | \\(m / k\\) |
| **收敛方式** | 平滑稳定 | 噪声震荡 | 适度噪声 |
| **单次更新速度** | 慢 | 快 | 中等（但可向量化） |
| **内存占用** | 高 | 低 | 中等 |
| **能跳出局部最小值** | 否 | 是 | 部分 |
| **典型使用场景** | 小数据集 | 在线学习 | 深度学习（默认） |


## 总结 {#summary}

- **Batch Gradient Descent**：准确但慢——适合可以承受完整遍历的小数据集。
- **Stochastic Gradient Descent**：快但噪声大——适用于在线学习或需要快速粗略进展的场景。
- **Mini-Batch Gradient Descent**：实际默认方案——在精度、速度和硬件效率之间取得平衡。大多数深度学习框架（PyTorch、TensorFlow 等）底层使用的都是 mini-batch SGD。

在实践中，batch size \\(k\\) 的选择是一个超参数，取决于模型大小、GPU 内存以及对梯度噪声和吞吐量之间的权衡。
