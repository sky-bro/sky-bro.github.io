
## introduction {#introduction}

from the previous post on forward and backward propagation, we know the gradient descent update rule:

\\[
\theta \leftarrow \theta - \eta \nabla C
\\]

where \\(\nabla C\\) is the gradient of the loss over the **entire training dataset**. but computing \\(\nabla C\\) means running forward and backward propagation for every single training example, then averaging the results. when the dataset has millions of samples, this is extremely expensive.

how we compute \\(\nabla C\\) and when we update \\(\theta\\) leads to different variants of gradient descent: **batch**, **stochastic**, and **mini-batch**.


## batch gradient descent {#batch-gradient-descent}

batch gradient descent (BGD) computes the gradient using **all** training examples before each update.

given a dataset with \\(m\\) samples, the gradient is the average over all individual gradients:

\\[
\nabla C(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla C\_i(\theta)
\\]

the update step is:

\\[
\theta \leftarrow \theta - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla C\_i(\theta)
\\]

**characteristics:**

- every update uses the full dataset, so the gradient direction is accurate
- the loss decreases smoothly and steadily toward the minimum
- guaranteed to converge to the global minimum for convex loss functions

**drawbacks:**

- each iteration is slow — must process all \\(m\\) samples
- requires loading the entire dataset into memory
- for very large datasets, a single iteration may be infeasible


## stochastic gradient descent {#stochastic-gradient-descent}

stochastic gradient descent (SGD) updates the parameters after **each individual** training example:

\\[
\theta \leftarrow \theta - \eta \cdot \nabla C\_i(\theta)
\\]

the training data is typically shuffled and processed one sample at a time, then reshuffled for the next epoch.

**characteristics:**

- updates are very fast — only one sample needed per step
- the noise from individual gradients can help escape shallow local minima
- memory efficient — only one sample needs to be in memory at a time

**drawbacks:**

- the gradient direction fluctuates wildly, causing the loss to **oscillate** near the optimum
- never truly converges — keeps bouncing around the minimum
- learning rate scheduling is essential to reduce oscillations over time


## mini-batch gradient descent {#mini-batch-gradient-descent}

mini-batch gradient descent is the **middle ground**: instead of one sample or all samples, use a small batch of \\(k\\) samples (commonly 32, 64, 128, or 256):

\\[
\nabla C(\theta) = \frac{1}{k} \sum_{j \in \text{batch}} \nabla C\_j(\theta)
\\]

\\[
\theta \leftarrow \theta - \eta \cdot \frac{1}{k} \sum_{j \in \text{batch}} \nabla C\_j(\theta)
\\]

{{< figure src="/images/posts/batch-vs-stochastic-gradient-descent/mini-batch-concept.svg" caption="<span class=\"figure-number\">Figure 1: </span>mini-batch concept: dataset split into batches of size 4, each batch triggers one parameter update" >}}

**why mini-batch is the default choice:**

1. **vectorization**: modern hardware (GPU/TPU) is optimized for batched matrix operations. processing 128 samples at once is much faster than 128 individual updates.
2. **reduced noise**: averaging over a batch smooths out the gradient noise, leading to more stable convergence than pure SGD.
3. **memory-friendly**: a batch of 128 samples fits easily in GPU memory, unlike millions of samples.
4. **one epoch = multiple updates**: with \\(m = 10000\\) and \\(k = 100\\), one epoch produces 100 updates, giving the optimizer many chances to adjust direction.


## comparison {#comparison}

{{< figure src="/images/posts/batch-vs-stochastic-gradient-descent/convergence-comparison.svg" caption="<span class=\"figure-number\">Figure 2: </span>convergence behavior of BGD (smooth), SGD (noisy), and mini-batch (moderately noisy but efficient)" >}}

| | Batch GD | Stochastic GD | Mini-Batch GD |
|---|---|---|---|
| **samples per update** | \\(m\\) (all) | 1 | \\(k\\) (e.g. 32) |
| **updates per epoch** | 1 | \\(m\\) | \\(m / k\\) |
| **convergence** | smooth, steady | noisy, oscillates | moderate noise |
| **speed per update** | slow | fast | medium (but vectorized) |
| **memory** | high | low | medium |
| **can escape local minima** | no | yes | partially |
| **typical use case** | small datasets | online learning | deep learning (default) |


## summary {#summary}

- **batch gradient descent**: accurate but slow — good for small datasets where you can afford full passes.
- **stochastic gradient descent**: fast and noisy — useful for online learning or when you need quick, rough progress.
- **mini-batch gradient descent**: the practical default — balances accuracy, speed, and hardware efficiency. most deep learning frameworks (PyTorch, TensorFlow, etc.) use mini-batch SGD under the hood.

in practice, the choice of batch size \\(k\\) is a hyperparameter that depends on model size, GPU memory, and the desired trade-off between gradient noise and throughput.
