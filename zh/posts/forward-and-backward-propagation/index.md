
## 深度神经网络 {#deep-neural-network}

{{< figure src="/images/posts/forward-backward-propagation/neural_network_example.svg" caption="<span class=\"figure-number\">图 1： </span>神经网络示例" width="70%" >}}

每个层可以用数学形式描述如下，从输入层逐层计算到最终输出层的过程称为**前向传播（forward propagation）**：

\begin{align}
\label{eq:3}
z^{(l)} &= W^{(l)}a^{(l-1)} + b^{(l)} && (\text{if } l > 1) \\\\
a^{(l)} &= \begin{cases}
  \sigma(z^{(l)}) & \text{if } l > 1 \\\\
  x & \text{if } l = 1
\end{cases}
\end{align}

其中包含线性部分 \\(z^{(l)}\\) 和激活部分 \\(a^{(l)}\\)。若使用 sigmoid 作为激活函数：\\(\sigma(z^{(l)}) = \frac{1}{1 + e^{-z^{(l)}}} \\)

以第 2 层为例，计算过程如下：

\begin{align}
\label{eq:4}
  z^{(2)} &= W^{(2)}x + b^{(2)} \\\\
          &= \begin{bmatrix}
               w\_{1,1}^{(2)} & w\_{1,2}^{(2)} & w\_{1,3}^{(2)} \\\\
               w\_{2,1}^{(2)} & w\_{2,2}^{(2)} & w\_{2,3}^{(2)} \\\\
               w\_{3,1}^{(2)} & w\_{3,2}^{(2)} & w\_{3,3}^{(2)} \\\\
               w\_{4,1}^{(2)} & w\_{4,2}^{(2)} & w\_{4,3}^{(2)} \\\\
             \end{bmatrix}
             \begin{bmatrix}
               x\_1 \\\\
               x\_2 \\\\
               x\_3
             \end{bmatrix}
             +
             \begin{bmatrix}
               b\_1^{(2)} \\\\
               b\_2^{(2)} \\\\
               b\_3^{(2)} \\\\
               b\_4^{(2)}
             \end{bmatrix} \\\\
  \begin{bmatrix}
    z\_{1}^{(2)} \\\\
    z\_{2}^{(2)} \\\\
    z\_{3}^{(2)} \\\\
    z\_{4}^{(2)}
  \end{bmatrix}
  &= \begin{bmatrix}
       w\_{1,1}^{(2)} x\_1 + w\_{1,2}^{(2)} x\_2 + w\_{1,3}^{(2)} x\_3 + b\_1^{(2)} \\\\
       w\_{2,1}^{(2)} x\_1 + w\_{2,2}^{(2)} x\_2 + w\_{2,3}^{(2)} x\_3 + b\_2^{(2)} \\\\
       w\_{3,1}^{(2)} x\_1 + w\_{3,2}^{(2)} x\_2 + w\_{3,3}^{(2)} x\_3 + b\_3^{(2)} \\\\
       w\_{4,1}^{(2)} x\_1 + w\_{4,2}^{(2)} x\_2 + w\_{4,3}^{(2)} x\_3 + b\_4^{(2)} \\\\
     \end{bmatrix} \\\\
a^{(2)} &= \sigma(z^{(2)}) \\\\
       &= \begin{bmatrix}
       \sigma(z\_{1}^{(2)}) \\\\
       \sigma(z\_{2}^{(2)}) \\\\
       \sigma(z\_{3}^{(2)}) \\\\
       \sigma(z\_{4}^{(2)})
       \end{bmatrix} \\\\
      &= \begin{bmatrix}
\frac{1}{1 + e^{-z\_{1}^{(2)}}} \\\\
\frac{1}{1 + e^{-z\_{2}^{(2)}}} \\\\
\frac{1}{1 + e^{-z\_{3}^{(2)}}} \\\\
\frac{1}{1 + e^{-z\_{4}^{(2)}}}
       \end{bmatrix}
\end{align}

### loss（cost）函数 {#loss--cost--function}

选择损失函数，例如 MSE（均方误差）：

\begin{align}
\label{eq:5}
C &= \frac{1}{2} \\| a^{(L)} - y \\|\_{2}^2
\end{align}

### gradient descent（梯度下降） {#gradient-descent}

{{< alert theme="info" dir="ltr" >}}

使用梯度下降找到最优参数 \\(\theta\\)（权重 \\(w\\) 和偏置 \\(b\\) 都是 \\(\theta\\) 的一部分），使损失最小化。

\\[
\theta^{\*} = arg \min\_{\theta} C(\theta)
\\]

{{< /alert >}}

为了使 \\(C\\) 减小，在每次迭代中应用以下公式（每次迭代累积 \\(C\\) 的变化量 \\(\Delta C\\)）：

\\[
\Delta C = \nabla C \Delta \theta
\\]

沿 \\(\nabla C\\)（\\(C\\) 的梯度，即**增长最快的方向**）的反方向更新 \\(\theta\\)，学习率为 \\(\eta\\)：

\begin{align}
\label{eq:10}
\theta &= \theta + \Delta \theta \\\\
   &= \theta -\eta \nabla C \\\\
\end{align}

计算 \\(\nabla C\\) 是我们需要的一切，而**反向传播**正是为此而生。

## Forward Propagation（前向传播） {#forward-propagation}

前向传播从第 1 层到第 \\(L\\) 层，逐层计算 \\(z^{(l)}\\) 和 \\(a^{(l)}\\)：

1. 从输入开始：\\(a^{(1)} = x\\)
2. 对于每一层 \\(l = 2, 3, \ldots, L\\)：
   1. 计算线性组合：\\(z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}\\)
   2. 应用激活函数：\\(a^{(l)} = \sigma(z^{(l)})\\)
3. 最终输出为 \\(a^{(L)}\\)，然后计算损失 \\(C = \frac{1}{2} \| a^{(L)} - y \|_2^2\\)

换句话说，前向传播就是评估神经网络所表示的复合函数的过程：将输入数据逐层送入网络，直到产生预测结果。

{{< figure src="/images/posts/forward-backward-propagation/forward-propagation.svg" caption="<span class=\"figure-number\">图 2： </span>前向传播：数据从输入流向输出，每一层先进行线性变换再经过激活函数" >}}

## Backward Propagation（反向传播） {#backward-propagation}

{{< alert theme="info" dir="ltr" >}}

反向传播是使用链式法则（chain rule）计算损失对每个参数的梯度 \\(\nabla C\\) 的过程。

{{< /alert >}}

前向传播产生预测 \\(a^{(L)}\\) 和损失 \\(C\\) 之后，反向传播计算每个参数对该损失的贡献程度。我们需要所有参数的梯度：\\(\frac{\partial C}{\partial W^{(l)}}\\) 和 \\(\frac{\partial C}{\partial b^{(l)}}\\)。

定义第 \\(l\\) 层的误差信号（也称 delta）为：

\begin{align}
\label{eq:8}
\delta^{(l)} &= \frac{\partial C}{\partial z^{(l)}} \\\\
&= \frac{\partial C}{\partial a^{(l)}} \odot \frac{\partial a^{(l)}}{\partial z^{(l)}} \\\\
&= \begin{cases}
  (a^{(L)} - y) \odot \sigma'(z^{(L)}) & \text{if}\ l = L \\\\
  (\frac{\partial z^{(l+1)}}{\partial a^{(l)}} \frac{\partial C}{\partial z^{(l+1)}}) \odot \sigma'(z^{(l)}) & \text{if}\ l<L \\\\
              \end{cases} \\\\
&= \begin{cases}
  (a^{(L)} - y) \odot \sigma'(z^{(L)}) & \text{if } l = L \\\\
  ((W^{(l+1)})^{T}\delta^{(l+1)}) \odot \sigma'(z^{(l)}) & \text{if } l<L \\\\
              \end{cases} \\\\
\frac{\partial C}{\partial W^{(l)}} &= \frac{\partial C}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial W^{(l)}} \\\\
&= \delta^{(l)}(a^{(l-1)})^{T} \\\\
\frac{\partial C}{\partial b^{(l)}} &= \frac{\partial C}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial b^{(l)}} \\\\
&= \delta^{(l)} \\\\
\end{align}

{{< figure src="/images/posts/forward-backward-propagation/backward-propagation.svg" caption="<span class=\"figure-number\">图 3： </span>反向传播：误差信号从输出层向输入层回传，沿途计算每一层的梯度" >}}

以神经网络（\\(L = 3\\)）为例：

\begin{align}
\label{eq:11}
\delta^{(l)} &= \begin{cases}
  (a^{(L)} - y) \odot \sigma'(z^{(L)}) & \text{if } l = L \\\\
  ((W^{(l+1)})^{T}\delta^{(l+1)}) \odot \sigma'(z^{(l)}) & \text{if } l < L
\end{cases} \\\\
&= \begin{cases}
    \begin{bmatrix}
      a\_{1}^{(3)} - y\_{1} \\\\
      a\_{2}^{(3)} - y\_{2}
    \end{bmatrix} \odot \begin{bmatrix}
      \sigma'(z\_{1}^{(3)}) \\\\
      \sigma'(z\_{2}^{(3)})
    \end{bmatrix} & \text{if } l = L = 3 \\\\
    (\begin{bmatrix}
      w\_{1,1}^{(3)} & w\_{2,1}^{(3)} \\\\
      w\_{1,2}^{(3)} & w\_{2,2}^{(3)} \\\\
      w\_{1,3}^{(3)} & w\_{2,3}^{(3)} \\\\
      w\_{1,4}^{(3)} & w\_{2,4}^{(3)}
\end{bmatrix}
\begin{bmatrix}
  \delta\_{1}^{(3)} \\\\
  \delta\_{2}^{(3)}
\end{bmatrix}
) \odot \begin{bmatrix}
  \sigma'(z\_{1}^{(2)}) \\\\
  \sigma'(z\_{2}^{(2)}) \\\\
  \sigma'(z\_{3}^{(2)}) \\\\
  \sigma'(z\_{4}^{(2)})
\end{bmatrix} & \text{if } l = 2 \\\\
\end{cases} \\\\
  \frac{\partial C}{\partial W^{(l)}} &= \delta^{(l)} (a^{(l-1)})^{T} \\\\
        &= \begin{cases}
          \begin{bmatrix}
            \delta\_{1}^{(3)} \\\\
            \delta\_{2}^{(3)}
          \end{bmatrix} \begin{bmatrix}
            a\_{1}^{(2)} & a\_{2}^{(2)} & a\_{3}^{(2)}
          \end{bmatrix} & \text{if } l = 3 \\\\
          \begin{bmatrix}
            \delta\_{1}^{(2)} \\\\
            \delta\_{2}^{(2)} \\\\
            \delta\_{3}^{(2)} \\\\
            \delta\_{4}^{(2)}
          \end{bmatrix} \begin{bmatrix}
            x\_{1} & x\_{2} &x\_{3}
          \end{bmatrix} & \text{if } l = 2
        \end{cases} \\\\
  \frac{\partial C}{\partial b^{(l)}} &= \delta^{(l)} \\\\
        &= \begin{cases}
          \begin{bmatrix}
            \delta\_{1}^{(3)} \\\\
            \delta\_{2}^{(3)}
          \end{bmatrix} & \text{if } l = 3 \\\\
          \begin{bmatrix}
            \delta\_{1}^{(2)} \\\\
            \delta\_{2}^{(2)} \\\\
            \delta\_{3}^{(2)} \\\\
            \delta\_{4}^{(2)}
          \end{bmatrix} & \text{if } l = 2
        \end{cases}
\end{align}

## 完整训练流程 {#full-training-loop}

反向传播计算出所有梯度后，完整的梯度下降算法如下：

1. **初始化（initialize）**：用小的随机值初始化所有权重 \\(W^{(l)}\\) 和偏置 \\(b^{(l)}\\)
2. **重复（repeat）** 直到收敛（或固定轮数）：
   1. **前向传播**：逐层计算 \\(a^{(l)}\\)（\\(l = 1 \ldots L\\)），然后计算损失 \\(C\\)
   2. **反向传播**：使用链式法则从输出层到输入层计算 \\(\delta^{(l)}\\)（\\(l = L, L-1, \ldots, 2\\)）
   3. **计算梯度**：\\(\frac{\partial C}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^{T}\\)，\\(\frac{\partial C}{\partial b^{(l)}} = \delta^{(l)}\\)
   4. **更新参数**：对于每一层，\\(W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial C}{\partial W^{(l)}}\\) 和 \\(b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial C}{\partial b^{(l)}}\\)
