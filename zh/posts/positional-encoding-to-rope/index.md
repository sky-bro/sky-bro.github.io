
## 引言 {#introduction}

Transformer 的自注意力有一个看似反直觉的性质：如果不给 token 额外的位置提示，它本身并不知道一句话里的词序。

比如下面两句话：

- 我 喜欢 你
- 你 喜欢 我

它们的 token 集合几乎一样，但语义完全不同。RNN 会按时间步读取输入，CNN 会用局部窗口保留邻近关系，而标准 self-attention 对一组输入向量做的是全局两两匹配。只看 attention 公式：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

如果我们把输入 token 的顺序整体打乱，再以同样方式打乱输出，attention 的计算结构并不会自然抵抗这种置换。这叫做**置换等变性**：attention 擅长建立内容之间的关系，但位置顺序需要额外注入。

这篇文章沿着一条逐步收紧的线索讲位置编码：

```mermaid
flowchart LR
    A[没有位置] --> B[绝对位置编码]
    B --> C[可外推的正弦位置编码]
    C --> D[相对位置应该影响注意力分数]
    D --> E[用复数旋转表达位置]
    E --> F[RoPE: 旋转后的 Q/K 点积只依赖相对距离]
```

核心问题不是“RoPE 的公式长什么样”，而是：**为什么位置可以变成旋转？为什么旋转以后，attention 会自然看到相对距离？**

## 位置编码的基本问题 {#position-basics}

### 为什么 self-attention 需要位置 {#why-position}

先把每个 token 看成一个向量 \\(x_i\\)。如果没有位置编码，Transformer 对 \\(x_i\\) 做线性投影：

$$q_{i} = W_Q x_{i},\quad k_{i} = W_K x_{i},\quad v_{i} = W_V x_{i}$$

然后用 \\(q_i^T k_j\\) 衡量 token \\(i\\) 对 token \\(j\\) 的关注程度。

注意这里的 \\(q_i\\)、\\(k_j\\)、\\(v_j\\) 都只来自 token 内容本身。位置 \\(i\\)、\\(j\\) 没有进入公式。因此模型能知道“我”和“你”的内容不同，却不能仅凭这个公式知道谁在前、谁在后。

最直接的补丁是给每个位置一个向量 \\(p_i\\)，把输入改成：

$$h_{i} = x_{i} + p_{i}$$

这样 \\(h_i\\) 同时包含 token 内容和位置。后续的 \\(Q,K,V\\) 都从 \\(h_i\\) 投影出来，位置也就被带进了 attention。

这就是绝对位置编码的基本思想：**每个位置有自己的坐标牌，token 先戴上坐标牌，再进入模型。**

### 第一站：可学习的绝对位置编码 {#learned-absolute}

最朴素的做法是维护一张位置表：

$$P \in \mathbb{R}^{L_{\max} \times d}$$

第 \\(i\\) 个位置直接查表得到 \\(p_i = P[i]\\)，然后加到 token embedding 上：

$$h_{i} = x_{i} + P[i]$$

这和词表 embedding 很像。优点是简单，模型可以自己学每个位置应该长什么样。缺点也很明显：

- 训练时只见过 \\(0 \ldots L_{\max}-1\\)，更长的位置没有对应表项；
- 位置向量之间没有显式结构，模型不一定学到“相邻位置应该更相似”；
- attention 看到的是混合后的 \\(x_i + p_i\\)，相对距离 \\(i-j\\) 并不是一等公民。

对于固定长度分类任务，这通常够用。但对长上下文语言模型，问题会更尖锐：我们不只是想知道“这是第 137 个 token”，还想知道“当前 token 和那个 token 相距 5、50、500 个位置”。

这就引出一个自然要求：位置编码最好带有某种可计算结构，而不是纯查表。

### 第二站：正弦绝对位置编码 {#sinusoidal-absolute}

原始 Transformer 使用的是正弦/余弦位置编码。对位置 \\(pos\\) 和维度索引 \\(i\\)，定义：

$$PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)$$

也就是说，每两个维度组成一组 \\((\sin,\cos)\\)，不同组使用不同频率。低频维度变化慢，适合表达长距离；高频维度变化快，适合区分近距离。

把它想成很多个时钟：

- 一个时钟每走一步转很小角度，能覆盖长周期；
- 一个时钟每走一步转较大角度，对局部位置很敏感；
- 多个时钟组合起来，就能给每个位置一个多尺度的指纹。

正弦位置编码比可学习查表多了一个关键结构：对于固定频率 \\(\omega\\)，位置 \\(pos\\) 对应的二维向量是：

$$\begin{bmatrix}\cos(pos\omega) \\\\ \sin(pos\omega)\end{bmatrix}$$

这已经不是任意向量，而是单位圆上的一个点。位置增加 1，就相当于在圆上多转 \\(\omega\\) 弧度。

{{< alert theme="info" >}}

这里已经出现了 RoPE 的影子：正弦位置编码本质上是在不同频率的二维平面里，用角度表示位置。

{{< /alert >}}

不过原始做法仍然是**加法**：

$$h_{i} = x_{i} + PE(i)$$

位置先被加进 token 表示，之后再投影成 \\(Q,K,V\\)。这能让模型知道绝对位置，但相对位置关系仍要由后续层自己学出来。我们能不能让 attention 的点积本身直接带上 \\(i-j\\)？

## 从绝对位置走向相对位置 {#absolute-to-relative}

attention 中真正决定权重的是分数：

$$s_{ij}=q_{i}^T k_{j}$$

如果位置编码只是在输入层相加，那么位置如何影响 \\(s_{ij}\\) 是间接的。一个更贴近语言建模的目标是：

$$s_{ij} = f(x_{i},x_{j},i-j)$$

也就是说，token 内容当然重要，但两个 token 的**相对距离**也应该直接参与打分。

为什么相对距离更自然？因为许多语言模式对平移不敏感：

- “形容词修饰后面的名词”关注的是附近关系，不是绝对第几个 token；
- 代码中的括号匹配关心距离和层级，不关心它出现在文件第几行；
- 自回归生成时，当前 token 总是在序列末尾，但它需要回看不同距离的历史 token。

绝对位置像是在问：“你住在哪个门牌号？”

相对位置更像是在问：“你离我多远？”

RoPE 的目标可以概括成一句话：**不把位置加到 token 上，而是用位置旋转 query 和 key，使它们的点积自然包含相对位置。**

## RoPE 的旋转结构 {#rope-rotation}

### 旋转之前：二维平面里的点积 {#dot-product-before-rotation}

先只看二维向量。设 query 和 key 分别是：

$$q=\begin{bmatrix}q_{1} \\\\ q_{2}\end{bmatrix},\quad k=\begin{bmatrix}k_{1} \\\\ k_{2}\end{bmatrix}$$

普通点积是：

$$q^T k = q_{1}k_{1} + q_{2}k_{2}$$

现在给位置 \\(m\\) 一个旋转角 \\(m\theta\\)，给位置 \\(n\\) 一个旋转角 \\(n\theta\\)。二维旋转矩阵为：

$$R_\alpha=\begin{bmatrix}\cos\alpha & -\sin\alpha \\\\ \sin\alpha & \cos\alpha\end{bmatrix}$$

RoPE 在这一组二维维度上做的事可以写成：

$$\tilde{q}\_{m} = R\_{m\theta}q,\quad \tilde{k}\_{n} = R\_{n\theta}k$$

attention 分数使用旋转后的点积：

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = (R\_{m\theta}q)^T(R\_{n\theta}k)$$

利用旋转矩阵的性质 \\(R_a^T R_b = R_{b-a}\\)，得到：

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = q^T R\_{(n-m)\theta}k$$

关键来了：右边只出现了 \\(n-m\\)，而不是单独的 \\(m\\) 和 \\(n\\)。

这就是 RoPE 最重要的结构性结果：**两个绝对位置分别旋转以后，它们的点积只依赖相对距离。**

### 欧拉公式：为什么旋转可以写成复数乘法 {#euler-formula}

上面用旋转矩阵已经能解释 RoPE。但它和欧拉公式的关系会让这个结构更清楚。

欧拉公式是：

$$e^{i\alpha}=\cos\alpha+i\sin\alpha$$

它说明复平面上的单位复数 \\(e^{i\alpha}\\) 就是一个角度为 \\(\alpha\\) 的旋转。把二维向量 \\((x_1,x_2)\\) 看成复数：

$$z=x_{1}+i x_{2}$$

让它乘上 \\(e^{i\alpha}\\)：

$$z' = z e^{i\alpha}$$

展开：

$$\begin{aligned}z' &= (x_{1}+i x_{2})(\cos\alpha+i\sin\alpha) \\\\ &= (x_{1}\cos\alpha - x_{2}\sin\alpha) + i(x_{1}\sin\alpha + x_{2}\cos\alpha)\end{aligned}$$

把实部和虚部分开：

$$\begin{bmatrix}x'\_{1} \\\\ x'\_{2}\end{bmatrix}=\begin{bmatrix}\cos\alpha & -\sin\alpha \\\\ \sin\alpha & \cos\alpha\end{bmatrix}\begin{bmatrix}x\_{1} \\\\ x\_{2}\end{bmatrix}$$

这正是二维旋转矩阵。

所以 RoPE 的复数视角很简洁：

$$\tilde{q}\_{m} = q \cdot e^{im\theta},\quad \tilde{k}\_{n} = k \cdot e^{in\theta}$$

当我们比较它们时，一个位置相位会抵消掉另一个位置相位，剩下相位差：

$$e^{in\theta} / e^{im\theta} = e^{i(n-m)\theta}$$

这就是“相对位置来自相位差”的直觉来源。

严格说，attention 里实际计算的仍然是实数点积，不是任意复数乘法。复数视角只是把同一件事写得更紧凑：一个复数对应两个实数 hidden dimensions，相位差会通过这两个维度的旋转进入最终的实数点积。

{{< figure src="/images/posts/positional-encoding-to-rope/rope-phase-difference.svg" caption="<span class=\"figure-number\">Figure 1: </span>RoPE 的核心不是记住绝对角度，而是让 query/key 的绝对旋转在点积中变成相对相位差。" width="92%" >}}

{{< notice info "一句话理解 RoPE" >}}

RoPE 把每对 hidden dimensions 看成复平面上的一个数；位置不是加法偏置，而是相位旋转。两个 token 做注意力点积时，绝对相位相互抵消，只留下相对相位差。

{{< /notice >}}

### RoPE 在高维向量中怎么做 {#high-dimensional-rope}

真实模型的 head dimension 不止 2，而是 \\(d\\)。RoPE 的做法是：拿出一段偶数维的旋转维度 \\(d_{\text{rot}}\\)，把它拆成 \\(d_{\text{rot}}/2\\) 个二维平面。大多数模型会让每个 head 的旋转维度本身就是偶数；如果总维度是奇数，工程上通常会只旋转其中最大的偶数部分，剩下 1 维不旋转，或者直接选择偶数 head dimension。

$$[(x_{0},x_{1}),(x_{2},x_{3}),\ldots,(x_{d_{\text{rot}}-2},x_{d_{\text{rot}}-1})]$$

每个二维平面使用一个频率 \\(\theta_i\\)。常见定义是：

$$\theta_{i} = 10000^{-2i/d_{\text{rot}}}$$

这里的 `10000` 不是数学常数，而是一个频率基底。它控制不同二维平面的波长跨度：基底越大，最低频变化越慢，更偏向长距离；基底越小，频率整体更密，更偏向局部区分。原始 Transformer 选择 `10000` 是一个经验上好用的多尺度范围，后来的长上下文模型会调整这个基底或缩放位置索引，本质上都是在改“这些时钟转得多快”。

对位置 \\(m\\)，第 \\(i\\) 对维度旋转角为 \\(m\theta_i\\)。写成矩阵乘法最直观：

$$\begin{bmatrix}\tilde{x}\_{2i} \\\\ \tilde{x}\_{2i+1}\end{bmatrix}=\begin{bmatrix}\cos(m\theta\_{i}) & -\sin(m\theta\_{i}) \\\\ \sin(m\theta\_{i}) & \cos(m\theta\_{i})\end{bmatrix}\begin{bmatrix}x\_{2i} \\\\ x\_{2i+1}\end{bmatrix}$$

展开后就是常见的 elementwise 公式：

$$\begin{aligned}\tilde{x}\_{2i} &= x\_{2i}\cos(m\theta\_{i})-x\_{2i+1}\sin(m\theta\_{i}) \\\\ \tilde{x}\_{2i+1} &= x\_{2i}\sin(m\theta\_{i})+x\_{2i+1}\cos(m\theta\_{i})\end{aligned}$$

注意，“两两组合”不只有相邻配对一种写法。RoPE 原论文公式通常按 \\((x_0,x_1),(x_2,x_3)\\) 这种相邻维度来讲；LLaMA 系列实现里常见的 `rotate_half` 则把向量切成前后两半，让前半的每个维度和后半对应维度配对。两者只是内存布局不同：只要 \\(\cos/\sin\\) 的排列和配对方式一致，数学上仍然是在若干二维平面里做旋转。

{{< figure src="/images/posts/positional-encoding-to-rope/rope-pairing-layouts.svg" caption="<span class=\"figure-number\">Figure 2: </span>RoPE 的本质是二维平面旋转；相邻配对和 split-half 配对是两种常见布局，后者更贴近 LLaMA 的 rotate_half 实现。" width="94%" >}}

实际计算时也不会真的构造一个巨大的块对角旋转矩阵。实现通常预先为每个 position 和每个频率缓存 \\(\cos(m\theta_i)\\)、\\(\sin(m\theta_i)\\)，然后广播到 \\(Q,K\\) 上做逐元素运算：

$$\operatorname{RoPE}(x,m)=x\odot\cos_m+\operatorname{rotate}(x)\odot\sin_m$$

其中 \\(\operatorname{rotate}(x)\\) 对相邻配对来说就是把每对 \\((a,b)\\) 变成 \\((-b,a)\\)；对 split-half 布局来说就是把 \\([x_1,x_2]\\) 变成 \\([-x_2,x_1]\\)。所以“旋转矩阵”是理解工具，不是高性能实现里的真实算子。

对 query 和 key 都做这个变换：

$$\tilde{q}\_{m} = \operatorname{RoPE}(q\_{m},m),\quad \tilde{k}\_{n} = \operatorname{RoPE}(k\_{n},n)$$

再计算 attention：

$$s\_{mn} = \tilde{q}\_{m}^T \tilde{k}\_{n}$$

Value 通常不旋转。原因是位置主要用于决定“该关注谁”，也就是影响 \\(QK^T\\) 的分数；一旦权重确定，\\(V\\) 承载的是被聚合的内容。

这也带来一个实现边界：decode 时如果使用 KV cache，旧的 key 必须保留写入 cache 时对应的位置，新 query 也必须使用当前序列里的绝对 position offset。如果每个 decode chunk 都从 0 重新编号，模型就会把相距很远的 token 当成相位上很近的 token 来比较。RoPE 的计算发生在 attention 内部，但 position counter 必须对整条序列保持一致。

### 一个小例子：同一个内容，距离不同，分数不同 {#small-example}

为了看清楚相对位置如何进入点积，考虑二维情况下：

$$q=\begin{bmatrix}1 \\\\ 0\end{bmatrix},\quad k=\begin{bmatrix}1 \\\\ 0\end{bmatrix}$$

如果两者在同一位置，\\(m=n\\)，相对角度为 0：

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = q^T R\_{0} k = 1$$

如果 key 比 query 晚一个单位，\\(n-m=1\\)，相对角度为 \\(\theta\\)：

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = q^T R\_{\theta} k = \cos\theta$$

如果相差 \\(r\\) 个单位：

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = \cos(r\theta)$$

这只是一个极简例子。真实模型中 \\(q\\) 和 \\(k\\) 不会这么简单，多组频率也会共同参与打分。但它揭示了核心机制：**距离改变了相位差，相位差改变了点积，点积改变了 attention 权重。**

## RoPE 的位置直觉与长上下文 {#rope-intuition}

### RoPE 和正弦位置编码的关系 {#relationship-to-sinusoidal}

现在可以回头比较正弦位置编码和 RoPE。

两者都使用多频率的 \\(\sin\\) 和 \\(\cos\\)，也都可以用欧拉公式理解。但它们注入位置的方式不同：

| 方法 | 位置如何进入模型 | attention 分数中的位置关系 |
| --- | --- | --- |
| 可学习绝对位置编码 | \\(x_i + p_i\\) | 间接学习 |
| 正弦绝对位置编码 | \\(x_i + PE(i)\\) | 有周期结构，但仍是间接学习 |
| 相对位置偏置 | 给 \\(s_{ij}\\) 加上距离相关 bias | 直接依赖 \\(i-j\\) |
| RoPE | 按位置旋转 \\(q_i,k_j\\) | 点积天然依赖 \\(i-j\\) |

正弦位置编码像是在 token 上贴一个“位置标签”。RoPE 则更像是在 attention 的坐标系里转动 query/key：位置不再是额外标签，而是参与相似度计算的几何变换。

这也是为什么 RoPE 常被说成同时具备绝对和相对位置的信息：

- query/key 的旋转角来自各自绝对位置；
- 两者点积时，真正影响匹配的是相对角度差。

### 为什么这对长上下文有帮助 {#long-context}

RoPE 不是魔法。它不会单独解决所有长上下文问题，模型仍然受训练长度、注意力复杂度、数据分布和外推策略影响。但它提供了几个很重要的归纳偏置：

- **平移结构**：点积依赖相对距离，适合语言中大量局部模式；
- **多尺度频率**：不同维度覆盖不同距离尺度；
- **无需位置表**：可以计算任意位置的旋转角，不受 learned position table 长度限制；
- **实现局部**：只需要在 attention 里旋转 \\(Q,K\\)，不改变整体 Transformer 结构。

长上下文扩展方法经常会继续围绕 RoPE 做文章，例如调整频率基底、缩放位置索引、分段处理相位等。它们背后的共同问题是：训练时见过的相位范围有限，推理时如果上下文变长，模型是否还能解释那些更远距离对应的相位差？

换句话说，RoPE 给了我们一个漂亮的几何坐标系，但坐标系如何外推，仍然是工程和训练共同决定的。

## 总结 {#summary}

从绝对位置编码到 RoPE，可以按下面这条线理解：

1. self-attention 本身不携带顺序信息，所以需要位置编码；
2. 可学习绝对位置编码最简单，但缺少结构，也不擅长长度外推；
3. 正弦位置编码把位置表示成多频率的 \\(\sin/\cos\\)，本质上是在单位圆上编码角度；
4. 相对位置更贴近 attention 的需求，因为注意力分数应该知道两个 token 相距多远；
5. RoPE 不把位置加到 token 上，而是按位置旋转 query 和 key；
6. 根据 \\(R_a^T R_b = R_{b-a}\\)，旋转后的点积只依赖相对距离；
7. 欧拉公式 \\(e^{i\alpha}=\cos\alpha+i\sin\alpha\\) 说明二维旋转等价于复数乘法，因此 RoPE 也可以理解为给 query/key 加上位置相位。
8. 真实 decode 使用 KV cache 时，cached keys 和新 queries 必须使用一致的绝对 position offset，否则相对相位差会错。

如果只保留一个 mental model：**RoPE 把位置编码从“加一个向量”变成“转一个角度”。绝对位置决定各自转到哪里，相对位置决定它们在 attention 点积中相差多少相位。**

## 参考资料 {#references}

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
- EleutherAI, [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/), 2021.
