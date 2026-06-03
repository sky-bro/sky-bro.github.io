
## Introduction {#introduction}

Self-attention has a surprising weakness: by itself, it does not know word order.

Consider these two sentences:

- I like you
- You like me

They contain almost the same tokens, but their meanings are different. RNNs read inputs step by step, and CNNs preserve local neighborhoods through convolution windows. Standard self-attention, however, mostly compares all token vectors with all other token vectors. Its core formula is:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

If we permute the input tokens and permute the output in the same way, the attention computation does not naturally resist that permutation. Attention is good at modeling content relationships, but order must be injected separately.

This post follows a gradual path:

```mermaid
flowchart LR
    A[No position] --> B[Absolute positional embeddings]
    B --> C[Sinusoidal positional encoding]
    C --> D[Relative position should affect scores]
    D --> E[Position as complex rotation]
    E --> F[RoPE: Q/K dot products depend on relative distance]
```

The main question is not just "what is RoPE's formula?" It is: **why can position be represented as rotation, and why does that make attention see relative distance?**

## The Basic Problem {#position-basics}

### Why self-attention needs position {#why-position}

Let each token be represented by a vector \\(x_i\\). Without positional information, a Transformer projects it into query, key, and value vectors:

$$q_{i} = W_Q x_{i},\quad k_{i} = W_K x_{i},\quad v_{i} = W_V x_{i}$$

The score between token \\(i\\) and token \\(j\\) is based on \\(q_i^T k_j\\).

But \\(q_i\\), \\(k_j\\), and \\(v_j\\) only come from token content. The positions \\(i\\) and \\(j\\) are not part of the formula. The model can know that "I" and "you" are different tokens, but it cannot know who came first from this computation alone.

The simplest fix is to give each position a vector \\(p_i\\) and add it to the token embedding:

$$h_{i} = x_{i} + p_{i}$$

Now \\(h_i\\) contains both content and position. The later \\(Q,K,V\\) projections inherit position information from \\(h_i\\).

This is the basic idea of absolute positional encoding: **each token wears a position tag before entering the model.**

### First stop: learned absolute positional embeddings {#learned-absolute}

The most direct approach is a learned table:

$$P \in \mathbb{R}^{L_{\max} \times d}$$

Position \\(i\\) looks up \\(p_i = P[i]\\), and the model uses:

$$h_{i} = x_{i} + P[i]$$

This is simple and flexible. The model learns whatever each position should mean. But it has clear limits:

- positions beyond the trained maximum length have no table entry;
- the position vectors do not have an explicit geometric structure;
- relative distance \\(i-j\\) is not a first-class object in the attention score.

For fixed-length classification, this can be enough. For long-context language models, the model often needs something more structured: not just "this is position 137", but "this token is 5, 50, or 500 tokens away from the current token."

### Second stop: sinusoidal positional encoding {#sinusoidal-absolute}

The original Transformer used sinusoidal positional encodings. For position \\(pos\\) and dimension index \\(i\\):

$$PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Every two dimensions form a \\((\sin,\cos)\\) pair. Different pairs use different frequencies. Low-frequency dimensions change slowly and cover long distances; high-frequency dimensions change quickly and distinguish nearby positions.

Think of it as many clocks:

- one clock turns by a tiny angle each step and covers long periods;
- another clock turns faster and is sensitive to local differences;
- together, these clocks give each position a multi-scale fingerprint.

For a fixed frequency \\(\omega\\), position \\(pos\\) corresponds to a point on the unit circle:

$$\begin{bmatrix}\cos(pos\omega) \\\\ \sin(pos\omega)\end{bmatrix}$$

Increasing position by 1 simply rotates that point by \\(\omega\\) radians.

{{< alert theme="info" >}}

This already contains the seed of RoPE: sinusoidal encodings represent position as angles in many two-dimensional planes.

{{< /alert >}}

The original method still uses addition:

$$h_{i} = x_{i} + PE(i)$$

Position is mixed into token representations before they become \\(Q,K,V\\). Can we make the attention score itself directly depend on \\(i-j\\)?

## From Absolute to Relative Position {#absolute-to-relative}

The attention score is:

$$s_{ij}=q_{i}^T k_{j}$$

If position is only added at the input layer, its effect on \\(s_{ij}\\) is indirect. For language modeling, a more natural target is:

$$s_{ij} = f(x_{i},x_{j},i-j)$$

Content matters, but the relative distance between two tokens should also affect the score.

Relative distance is natural because many language patterns are translation-invariant:

- an adjective often modifies a nearby noun, regardless of the absolute token index;
- matching brackets in code care about distance and nesting, not line number;
- in autoregressive generation, the current token is always at the end but must look back across different distances.

Absolute position asks: "what is your address?"

Relative position asks: "how far are you from me?"

RoPE's goal is: **do not add position to the token; rotate query and key by position, so their dot product naturally contains relative position.**

## The Rotation Structure of RoPE {#rope-rotation}

### Dot products after two rotations {#dot-product-before-rotation}

Start with a two-dimensional query and key:

$$q=\begin{bmatrix}q_{1} \\\\ q_{2}\end{bmatrix},\quad k=\begin{bmatrix}k_{1} \\\\ k_{2}\end{bmatrix}$$

Their ordinary dot product is:

$$q^T k = q_{1}k_{1} + q_{2}k_{2}$$

Now give position \\(m\\) the angle \\(m\theta\\), and position \\(n\\) the angle \\(n\theta\\). The two-dimensional rotation matrix is:

$$R_\alpha=\begin{bmatrix}\cos\alpha & -\sin\alpha \\\\ \sin\alpha & \cos\alpha\end{bmatrix}$$

RoPE applies:

$$\tilde{q}\_{m} = R\_{m\theta}q,\quad \tilde{k}\_{n} = R\_{n\theta}k$$

The attention score uses the rotated dot product:

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = (R\_{m\theta}q)^T(R\_{n\theta}k)$$

Using \\(R_a^T R_b = R_{b-a}\\), we get:

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = q^T R\_{(n-m)\theta}k$$

The result contains \\(n-m\\), not \\(m\\) and \\(n\\) separately.

This is RoPE's key structural result: **two absolute rotations become a relative rotation inside the dot product.**

### Euler's formula: rotation as complex multiplication {#euler-formula}

The matrix derivation is enough, but Euler's formula makes the geometry clearer:

$$e^{i\alpha}=\cos\alpha+i\sin\alpha$$

A unit complex number \\(e^{i\alpha}\\) is a rotation by angle \\(\alpha\\). Treat a two-dimensional vector \\((x_1,x_2)\\) as a complex number:

$$z=x_{1}+i x_{2}$$

Multiplying by \\(e^{i\alpha}\\) gives:

$$z' = z e^{i\alpha}$$

Expanding it:

$$\begin{aligned}z' &= (x_{1}+i x_{2})(\cos\alpha+i\sin\alpha) \\\\ &= (x_{1}\cos\alpha - x_{2}\sin\alpha) + i(x_{1}\sin\alpha + x_{2}\cos\alpha)\end{aligned}$$

Separating real and imaginary parts:

$$\begin{bmatrix}x'\_{1} \\\\ x'\_{2}\end{bmatrix}=\begin{bmatrix}\cos\alpha & -\sin\alpha \\\\ \sin\alpha & \cos\alpha\end{bmatrix}\begin{bmatrix}x\_{1} \\\\ x\_{2}\end{bmatrix}$$

This is exactly the rotation matrix.

In complex notation, RoPE is compact:

$$\tilde{q}\_{m} = q \cdot e^{im\theta},\quad \tilde{k}\_{n} = k \cdot e^{in\theta}$$

When the two are compared, one phase cancels the other and leaves a phase difference:

$$e^{in\theta} / e^{im\theta} = e^{i(n-m)\theta}$$

That is the intuition behind "relative position comes from phase difference."

Strictly speaking, attention still uses a real-valued dot product, not arbitrary complex multiplication. The complex view is a compact way to describe the same pairwise real rotations: each complex component corresponds to two real hidden dimensions, and the phase difference shows up in the real dot product after those rotations.

{{< figure src="/images/posts/positional-encoding-to-rope/rope-phase-difference.svg" caption="<span class=\"figure-number\">Figure 1: </span>RoPE does not need to memorize absolute angles; the dot product turns two absolute rotations into a relative phase difference." width="92%" >}}

{{< notice info "One-sentence model" >}}

RoPE treats every pair of hidden dimensions as a point in the complex plane. Position is not an additive bias; it is a phase rotation. When two tokens interact through a dot product, the absolute phases cancel into a relative phase gap.

{{< /notice >}}

### RoPE in high-dimensional vectors {#high-dimensional-rope}

A real attention head has dimension \\(d\\), not 2. RoPE takes an even rotary dimension \\(d_{\text{rot}}\\) and splits it into \\(d_{\text{rot}}/2\\) two-dimensional planes. Most models choose an even head dimension or an even rotary sub-dimension. If a dimension were odd, implementation usually rotates the largest even part and leaves one dimension unrotated, or simply avoids the odd case.

$$[(x_{0},x_{1}),(x_{2},x_{3}),\ldots,(x_{d_{\text{rot}}-2},x_{d_{\text{rot}}-1})]$$

Each plane uses a frequency \\(\theta_i\\). A common definition is:

$$\theta_{i} = 10000^{-2i/d_{\text{rot}}}$$

The value `10000` is not a mathematical constant. It is a frequency base. A larger base spreads the frequencies over a wider range and makes the slowest clocks change more slowly; a smaller base packs frequencies more tightly and emphasizes local distinction. Later long-context methods often modify the base or scale positions. They are adjusting how fast these clocks rotate.

For position \\(m\\), pair \\(i\\) rotates by angle \\(m\theta_i\\). The matrix form is the clearest:

$$\begin{bmatrix}\tilde{x}\_{2i} \\\\ \tilde{x}\_{2i+1}\end{bmatrix}=\begin{bmatrix}\cos(m\theta\_{i}) & -\sin(m\theta\_{i}) \\\\ \sin(m\theta\_{i}) & \cos(m\theta\_{i})\end{bmatrix}\begin{bmatrix}x\_{2i} \\\\ x\_{2i+1}\end{bmatrix}$$

Expanded elementwise:

$$\begin{aligned}\tilde{x}\_{2i} &= x\_{2i}\cos(m\theta\_{i})-x\_{2i+1}\sin(m\theta\_{i}) \\\\ \tilde{x}\_{2i+1} &= x\_{2i}\sin(m\theta\_{i})+x\_{2i+1}\cos(m\theta\_{i})\end{aligned}$$

There are multiple ways to pair dimensions. The RoPE paper notation often uses adjacent pairs like \\((x_0,x_1),(x_2,x_3)\\). LLaMA-style `rotate_half` implementations commonly split the vector in half and pair corresponding dimensions from the two halves. The math is the same if the \\(\cos/\sin\\) layout matches the pairing.

{{< figure src="/images/posts/positional-encoding-to-rope/rope-pairing-layouts.svg" caption="<span class=\"figure-number\">Figure 2: </span>RoPE is a set of two-dimensional rotations. Adjacent pairs and split-half pairs are different tensor layouts for the same idea." width="94%" >}}

In real kernels, we do not build a giant block-diagonal rotation matrix. Implementations cache \\(\cos(m\theta_i)\\) and \\(\sin(m\theta_i)\\), broadcast them to \\(Q,K\\), and compute:

$$\operatorname{RoPE}(x,m)=x\odot\cos_m+\operatorname{rotate}(x)\odot\sin_m$$

For adjacent pairs, \\(\operatorname{rotate}(a,b)=(-b,a)\\). For split-half layout, \\(\operatorname{rotate}([x_1,x_2])=[-x_2,x_1]\\). The rotation matrix is the conceptual model, not the high-performance operator.

RoPE is applied to query and key:

$$\tilde{q}\_{m} = \operatorname{RoPE}(q\_{m},m),\quad \tilde{k}\_{n} = \operatorname{RoPE}(k\_{n},n)$$

Then attention uses:

$$s\_{mn} = \tilde{q}\_{m}^T \tilde{k}\_{n}$$

Values are usually not rotated. Position mostly decides **who to attend to**, which is controlled by \\(QK^T\\). Once the weights are known, \\(V\\) carries the content to aggregate.

This also explains an implementation boundary in decoding. During cached generation, old keys in the KV cache must keep the positions they had when they were written, and new queries must use their current absolute position offset. If every decode chunk restarted positions from zero, the model would compare query and key phases as if distant tokens were adjacent. RoPE is local to the attention computation, but the position counter is global to the sequence.

### A tiny example {#small-example}

In two dimensions, let:

$$q=\begin{bmatrix}1 \\\\ 0\end{bmatrix},\quad k=\begin{bmatrix}1 \\\\ 0\end{bmatrix}$$

If they are at the same position, \\(m=n\\), the relative angle is 0:

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = q^T R\_{0} k = 1$$

If the key is one position after the query, \\(n-m=1\\), the relative angle is \\(\theta\\):

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = q^T R\_{\theta} k = \cos\theta$$

If they are \\(r\\) positions apart:

$$\tilde{q}\_{m}^T\tilde{k}\_{n} = \cos(r\theta)$$

Real queries and keys are richer than this example, and many frequencies act together. But the mechanism is visible: **distance changes phase difference; phase difference changes the dot product; the dot product changes attention weights.**

## RoPE Intuition and Long Contexts {#rope-intuition}

### Relationship to sinusoidal positional encoding {#relationship-to-sinusoidal}

Both sinusoidal positional encoding and RoPE use multi-frequency \\(\sin\\) and \\(\cos\\), and both can be understood through Euler's formula. The difference is how position enters the model:

| Method | How position enters | Position relation in attention scores |
| --- | --- | --- |
| Learned absolute embedding | \\(x_i + p_i\\) | learned indirectly |
| Sinusoidal absolute encoding | \\(x_i + PE(i)\\) | structured but still indirect |
| Relative position bias | add distance-dependent bias to \\(s_{ij}\\) | directly depends on \\(i-j\\) |
| RoPE | rotate \\(q_i,k_j\\) by position | dot product naturally depends on \\(i-j\\) |

Sinusoidal encoding is like attaching a position label to each token. RoPE instead rotates the coordinate system used by query and key: position becomes part of similarity computation.

This is why RoPE is often described as containing both absolute and relative information:

- the rotation angle comes from each token's absolute position;
- the dot product is affected by the relative angle between the two positions.

### Why this helps with long contexts {#long-context}

RoPE is not magic. It does not solve every long-context problem by itself. Models still depend on training length, attention complexity, data distribution, and extrapolation strategy. But RoPE gives useful inductive biases:

- **translation structure**: scores depend on relative distance;
- **multi-scale frequencies**: different dimensions cover different distance scales;
- **no learned position table**: rotations can be computed for arbitrary positions;
- **local implementation**: only \\(Q,K\\) inside attention need to be rotated.

Long-context methods often modify RoPE by changing the frequency base, scaling position indices, or treating phases differently across regions. They are all asking the same question: if the model was trained on one phase range, how should it interpret longer-range phases at inference time?

RoPE gives us a clean geometric coordinate system. How well that coordinate system extrapolates is still a joint result of modeling, data, and engineering.

## Summary {#summary}

The path from absolute positional encoding to RoPE is:

1. self-attention does not contain order by itself, so it needs positional information;
2. learned absolute embeddings are simple but lack structure and extrapolation;
3. sinusoidal encodings represent position as multi-frequency angles on unit circles;
4. relative position is more natural for attention because scores should know how far two tokens are;
5. RoPE rotates query and key instead of adding a position vector;
6. because \\(R_a^T R_b = R_{b-a}\\), the rotated dot product depends on relative distance;
7. Euler's formula \\(e^{i\alpha}=\cos\alpha+i\sin\alpha\\) explains why two-dimensional rotation is complex multiplication.
8. in real decoding, cached keys and new queries must use consistent absolute position offsets, otherwise the relative phase gap becomes wrong.

If you keep only one mental model: **RoPE turns "add a position vector" into "rotate by an angle." Absolute positions decide where query and key rotate; relative position decides the phase gap that attention sees.**

## References {#references}

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
- EleutherAI, [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/), 2021.
