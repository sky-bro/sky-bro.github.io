
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


## Backward Propagation {#backward-propagation}

{{< alert theme="info" dir="ltr" >}}

process of using chain rule to compute the gradients \\(\nabla C\\).

{{< /alert >}}

we need the gradient of all our parameters: \\(\frac{\partial C}{\partial W^{(l)}}, \frac{\partial C}{\partial b^{(l)}}\\).

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
