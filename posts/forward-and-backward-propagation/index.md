
## Deep Neural Network {#deep-neural-network}

{{< figure src="/images/posts/forward-backward-propagation/neural_network_example.svg" caption="<span class=\"figure-number\">Figure 1: </span>neural network example" width="70%" >}}

each layer can be described mathematically as below, the process to compute through the layers to the final layer is what we call **forward propagation**:

\begin{align}
\label{eq:3}
z^{(l)} &= W^{(l)}a^{(l-1)} + b^{(l)} && (\text{if } l > 1) \\\\
a^{(l)} &= \begin{cases}
  \sigma(z^{(l)}) & \text{if } l > 1 \\\\
  x & \text{if } l = 1
\end{cases}
\end{align}

which is composed of a linear part \\(z^{(l)}\\) and an activation part \\(a^{(l)}\\), if use sigmoid as the activation function: \\(\sigma(z^{(l)}) = \frac{1}{1 + e^{-z^{(l)}}} \\)

for example, computation of layer 2 can be described as:

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

### loss (cost) function {#loss--cost--function}

choose a loss (cost) function, for example MSE (mean square error)

\begin{align}
\label{eq:5}
C &= \frac{1}{2} \\| a^{(L)} - y \\|\_{2}^2
\end{align}

### gradient descent {#gradient-descent}

{{< alert theme="info" dir="ltr" >}}

we use gradient descent to find the optimum parameters \\(\theta\\) (weights \\(w\\) and bias \\(b\\) are both part of \\(\theta\\)), so that the loss is minimized .

\\[
\theta^{\*} = arg \min\_{\theta} C(\theta)
\\]

{{< /alert >}}

To make \\(C\\) decrease, we apply the formula below in each iteration (each iteration we accumulate some delta of \\(C\\): \\(\Delta C\\)):

\\[
\Delta C = \nabla C \Delta \theta
\\]

update \\(\theta\\) in the opposite direction of \\(\nabla C\\) (the gradient of \\(C\\), a constant, the **fastest direction** to increase \\(C\\)), and with a learning rate \\(\eta\\)

\begin{align}
\label{eq:10}
\theta &= \theta + \Delta \theta \\\\
   &= \theta -\eta \nabla C \\\\
\end{align}

Computing \\(\nabla C\\) is all we need, and we use **backward propagation** to do it.

## Forward Propagation {#forward-propagation}

forward propagation computes \\(z^{(l)}\\) and \\(a^{(l)}\\) for each layer sequentially from layer 1 to layer \\(L\\):

1. start with the input: \\(a^{(1)} = x\\)
2. for each layer \\(l = 2, 3, \ldots, L\\):
   1. compute the linear combination: \\(z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}\\)
   2. apply the activation: \\(a^{(l)} = \sigma(z^{(l)})\\)
3. the final output is \\(a^{(L)}\\), then compute the loss \\(C = \frac{1}{2} \| a^{(L)} - y \|_2^2\\)

in other words, forward propagation is the process of evaluating the composite function that represents the neural network, feeding input data through the network layer by layer until a prediction is produced.

{{< figure src="/images/posts/forward-backward-propagation/forward-propagation.svg" caption="<span class=\"figure-number\">Figure 2: </span>forward propagation: data flows from input to output, each layer applies a linear transform followed by an activation function" >}}

## Backward Propagation {#backward-propagation}

{{< alert theme="info" dir="ltr" >}}

backward propagation is the process of using the chain rule to compute the gradients \\(\nabla C\\) of the loss with respect to each parameter.

{{< /alert >}}

after forward propagation produces a prediction \\(a^{(L)}\\) and the loss \\(C\\), backward propagation computes how much each parameter contributed to that loss. we need the gradient of all parameters: \\(\frac{\partial C}{\partial W^{(l)}}\\) and \\(\frac{\partial C}{\partial b^{(l)}}\\).

define the error signal (also called delta) at layer \\(l\\) as:

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

{{< figure src="/images/posts/forward-backward-propagation/backward-propagation.svg" caption="<span class=\"figure-number\">Figure 3: </span>backward propagation: error signals flow from output layer back to input, computing gradients for each layer along the way" >}}

take the neural network (where \\(L = 3\\)) as an example:

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

## the full training loop {#full-training-loop}

with all gradients computed by backward propagation, the complete gradient descent algorithm is:

1. **initialize** all weights \\(W^{(l)}\\) and biases \\(b^{(l)}\\) with small random values
2. **repeat** until convergence (or for a fixed number of epochs):
   1. **forward pass**: compute \\(a^{(l)}\\) for each layer \\(l = 1 \ldots L\\), then compute the loss \\(C\\)
   2. **backward pass**: compute \\(\delta^{(l)}\\) for each layer \\(l = L, L-1, \ldots, 2\\) using the chain rule
   3. **compute gradients**: \\(\frac{\partial C}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^{T}\\), \\(\frac{\partial C}{\partial b^{(l)}} = \delta^{(l)}\\)
   4. **update parameters**: for each layer, \\(W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial C}{\partial W^{(l)}}\\) and \\(b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial C}{\partial b^{(l)}}\\)
