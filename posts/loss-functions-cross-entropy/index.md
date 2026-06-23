
Forward propagation produces a prediction. Backpropagation computes gradients. Gradient descent updates parameters. But one question sits between those steps:

> what exactly counts as being wrong, and how wrong is it?

That is the job of a **loss function**. It turns a model output and a target into one scalar:

$$\text{loss} = L(\hat{y}, y)$$

During training, we usually do not optimize abstract goals such as "looks good", "is accurate", or "answers like a human" directly. We optimize a differentiable, computable proxy objective that can produce gradients. Choosing a loss function means telling the model which mistakes are expensive and which update direction is useful.

This post fits naturally after [forward and backward propagation]({{< relref "/posts/forward-and-backward-propagation/" >}}), [batch vs stochastic gradient descent]({{< relref "/posts/batch-vs-stochastic-gradient-descent/" >}}), and [activation functions]({{< relref "/posts/activation-functions-neural-networks/" >}}). Activation functions shape what a network can represent; loss functions shape what those representations are trained to become.

## Start With The Selection Table {#cheat-sheet}

| Task | Typical output | Common loss | Intuition |
| --- | --- | --- | --- |
| Regression: price, temperature, length | one continuous value | MSE / MAE / Huber | larger prediction errors get larger penalties |
| Binary classification: spam or not spam | one logit or probability | Binary Cross Entropy | positive examples want probability near 1; negatives want it near 0 |
| Multiclass classification: cat/dog/car | one logit per class | Cross Entropy | penalize low probability on the correct class |
| Multilabel classification: an image can contain cat and car | one logit per label | sum/mean of BCE terms | each label is an independent yes/no decision |
| Language-model next-token prediction | one logit per vocabulary token | Cross Entropy / NLL | penalize low probability on the real next token |
| Distribution matching, distillation, policy constraints | two probability distributions | KL Divergence | measure how far one distribution is from another |
| Margin classification and ranking | score differences | Hinge / Ranking loss | being correct is not enough; keep a safety margin |
| Representation learning, retrieval, embeddings | vector distances or similarities | Contrastive / Triplet loss | pull similar examples together and push dissimilar ones apart |

A simplified rule:

- If the target is a **continuous value**, start with MSE, MAE, or Huber.
- If the target is **one mutually exclusive class**, start with cross entropy.
- If the target has **multiple independent labels**, start with binary cross entropy.
- If the target is **a distribution**, consider KL divergence.
- If the target is a **relative relation**, such as which item is closer or ranked higher, consider margin, contrastive, or triplet losses.

## Loss Is Not The Same As A Metric {#loss-vs-metric}

Suppose a binary classifier outputs the probability that an email is spam. The true label is 1. If the model predicts 0.51 or 0.99, accuracy marks both as correct. But training should not treat them equally:

- 0.51: barely correct, still uncertain.
- 0.99: highly confident and in the right direction.

Accuracy is only right or wrong; it gives almost no information about how far the model still has to move. Cross entropy gives a finer penalty:

$$L = -\log(\hat{p})$$

If the true-class probability \\(\hat{p}=0.99\\), the loss is about 0.01. If \\(\hat{p}=0.51\\), the loss is about 0.67. If \\(\hat{p}=0.01\\), the loss is about 4.61. Confident mistakes are punished heavily.

This is why training often optimizes a loss but reports metrics:

- **loss**: continuous, differentiable, gradient-producing, good for training.
- **metric**: closer to what humans or product requirements care about, good for evaluation.

## Regression: MSE, MAE, And Huber {#regression-loss}

In regression, the label is a continuous value. For example, if the true house price is 100, predictions of 90, 110, and 200 are numerical errors.

### MSE: Large Errors Get Squared {#mse}

Mean Squared Error is:

$$L = \frac{1}{n}\sum_i(\hat{y}^{(i)} - y^{(i)})^2$$

For three predictions:

| True value \\(y\\) | Prediction \\(\hat{y}\\) | Error | Squared error |
| --- | --- | --- | --- |
| 100 | 90 | -10 | 100 |
| 100 | 110 | 10 | 100 |
| 100 | 200 | 100 | 10000 |

MSE strongly amplifies large errors. That is useful when extreme mistakes really should be corrected aggressively. But if the dataset contains outliers, MSE can make the model pay too much attention to them.

MSE is common for:

- ordinary numerical regression.
- prediction problems where noise is close to Gaussian.
- reconstruction baselines in autoencoders, such as pixel or feature reconstruction.

### MAE: More Robust To Outliers, Less Smooth {#mae}

Mean Absolute Error is:

$$L = \frac{1}{n}\sum_i|\hat{y}^{(i)} - y^{(i)}|$$

For the same three predictions, the absolute errors are 10, 10, and 100. The outlier is still worse, but it is not amplified into 10000.

MAE is common when:

- regression should be more robust to outliers.
- the absolute size of the error is easy to interpret in the domain.

Its drawback is that it is not differentiable at 0. Frameworks handle this with subgradients, but optimization often feels less smooth than MSE.

### Huber: MSE For Small Errors, MAE For Large Errors {#huber}

Huber loss connects the two:

$$L_\delta(e) = \begin{cases} \frac{1}{2}e^2, & |e|\le\delta \\\\ \delta(|e|-\frac{1}{2}\delta), & |e|>\delta \end{cases}$$

where \\(e=\hat{y}-y\\). Small errors use a squared penalty, which is smooth. Large errors become almost linear, reducing the influence of outliers.

Huber is common for:

- noisy regression where you still want MSE-like smoothness near the optimum.
- value-function estimation in reinforcement learning; DQN-style methods often use smooth L1 / Huber-like losses.

## Classification: Why Cross Entropy Is Everywhere {#classification-loss}

Classification is not mainly about numerical distance. It asks: how much probability did the model assign to the correct class?

Suppose there are three classes: cat, dog, car. The model outputs logits:

$$z = [2.0,\ 1.0,\ -1.0]$$

Softmax turns logits into probabilities:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Approximately:

$$p = [0.71,\ 0.26,\ 0.03]$$

If the true class is dog, the correct-class probability is 0.26. Cross entropy with a one-hot target is:

$$L = -\sum_i y_i\log(p_i)$$

Only the dog entry has \\(y_i=1\\), so:

$$L = -\log(0.26) \approx 1.35$$

If the model raises dog's probability to 0.90, the loss becomes:

$$-\log(0.90) \approx 0.105$$

That is the core intuition of cross entropy: **look at the probability assigned to the true class; the lower it is, the larger the penalty**.

### Cross Entropy And NLL {#ce-nll}

In deep learning frameworks, you will often see:

- `CrossEntropyLoss`
- `NLLLoss`
- `log_softmax + nll_loss`

The boundary is:

- softmax: turns logits into probabilities.
- log softmax: turns logits into log probabilities.
- negative log likelihood: selects the log probability of the true class and negates it.

So multiclass cross entropy can be written as:

$$\text{CE}(\text{logits}, y) = -\log\left(\operatorname{softmax}(\text{logits})_y\right)$$

In PyTorch, `CrossEntropyLoss` usually expects **raw logits** and internally applies `log_softmax` plus NLL. Do not manually apply softmax first unless the API explicitly asks for probabilities; it can hurt numerical stability and gradients.

### Binary Classification: Binary Cross Entropy {#binary-cross-entropy}

Binary classification can output one probability \\(\hat{p}\\): the probability of the positive class. The label is \\(y\in\{0,1\}\\). Binary Cross Entropy is:

$$L = -\left[y\log(\hat{p}) + (1-y)\log(1-\hat{p})\right]$$

For two positive examples:

| Label | Predicted positive probability | Loss |
| --- | --- | --- |
| 1 | 0.9 | \\(-\log(0.9)\approx 0.105\\) |
| 1 | 0.1 | \\(-\log(0.1)\approx 2.303\\) |

If the true label is positive, predicting 0.9 gets a small penalty and predicting 0.1 gets a large penalty. If the true label is negative, the formula switches to penalizing \\(1-\hat{p}\\).

BCE is common for:

- binary classification: yes/no, spam/not spam, fraud/not fraud.
- multilabel classification: each label is an independent binary decision. For example, an image can contain cat, car, and person at the same time.

Do not use ordinary softmax cross entropy for multilabel classification. Softmax assumes classes are mutually exclusive: probabilities sum to 1. In a multilabel setting, several labels can be true at once, so the usual setup is one independent logit per label. In implementation, prefer a with-logits loss such as `BCEWithLogitsLoss`, which applies sigmoid and BCE internally; if the model already outputs probabilities, use ordinary BCE.

## KL Divergence: Comparing Two Distributions {#kl-divergence}

Cross entropy often compares a one-hot label against a model distribution. Sometimes the target itself is a distribution.

For example, a teacher model may produce this soft label:

$$q = [0.70,\ 0.20,\ 0.10]$$

The student model outputs:

$$p = [0.60,\ 0.30,\ 0.10]$$

Now we care not only about which class is correct, but whether the student's whole distribution resembles the teacher's. KL divergence is commonly written as:

$$D_{\mathrm{KL}}(q\|p)=\sum_i q_i\log\frac{q_i}{p_i}$$

It measures the extra information cost of representing the reference distribution \\(q\\) using \\(p\\).

KL divergence is common in:

- knowledge distillation: make a student imitate a teacher's soft distribution.
- variational inference / VAE: constrain an approximate posterior toward a prior or target distribution.
- RLHF / policy optimization: keep a new policy from drifting too far from a reference policy.

KL divergence is asymmetric:

$$D_{\mathrm{KL}}(q\|p) \ne D_{\mathrm{KL}}(p\|q)$$

Changing the direction can change training behavior.

## Margins And Representation Learning {#ranking-representation-loss}

Some tasks do not care about a class probability. They care about relative order or embedding geometry.

### Hinge Loss: Correct Is Not Enough {#hinge}

The hinge loss used in linear SVMs is:

$$L = \max(0,\ 1 - y f(x))$$

where \\(y\in\{-1,1\}\\), and \\(f(x)\\) is the model score. If \\(y f(x) \ge 1\\), the example is not only correctly classified but also far enough from the decision boundary, so the loss is 0. If it is barely correct, it is still penalized.

Hinge / margin losses are common in:

- SVM-style classification.
- ranking tasks, where a positive item should score at least one margin above a negative item.
- metric-learning setups that need explicit distance separation.

### Contrastive And Triplet Loss: Shaping An Embedding Space {#contrastive-triplet}

Retrieval, face recognition, and semantic similarity tasks often learn an embedding space instead of outputting a single class.

Triplet loss uses three examples:

- anchor: the current example.
- positive: an example similar to the anchor.
- negative: an example dissimilar to the anchor.

The target relation is:

$$d(a,p) + m < d(a,n)$$

The anchor should be closer to the positive than to the negative by at least margin \\(m\\). A common loss is:

$$L = \max(0,\ d(a,p)-d(a,n)+m)$$

These losses are common in:

- image/text retrieval.
- face verification.
- sentence embeddings and vector-database retrieval.
- contrastive learning, where different augmented views of the same example are pulled together and different examples are pushed apart.

## How To Choose In Practice {#practical-choice}

Work backward from the label shape and output layer:

| What your label looks like | Typical output layer | Loss |
| --- | --- | --- |
| one real number | linear output | MSE / MAE / Huber |
| one 0/1 label | one raw logit | BCE with logits |
| one mutually exclusive class id | class logits | Cross entropy |
| multiple 0/1 labels | one raw logit per label | BCE with logits |
| one probability distribution | logits or log probabilities | KL / cross entropy |
| similar/dissimilar pair | embedding vectors | contrastive loss |
| anchor-positive-negative | embedding vectors | triplet loss |

Common pitfalls:

- **Do not casually use MSE for classification**. MSE cares about numerical distance; classification cares about probability on the correct class. Cross entropy usually gives a better training signal.
- **Do not confuse multiclass and multilabel classification**. Mutually exclusive classes use softmax cross entropy; labels that can be true together use sigmoid plus BCE.
- **Prefer with-logits versions**. For example, `BCEWithLogitsLoss` is more numerically stable than manual `sigmoid + BCELoss`.
- **Class imbalance needs extra handling**. Highly imbalanced data may need class weights, focal loss, resampling, or threshold tuning.
- **Lower loss does not guarantee better business metrics**. Loss is a proxy objective, so always check validation metrics too.

## Summary {#summary}

A loss function is the mathematical interface of the training objective. It translates "what counts as wrong" into "which way should the gradients move".

- MSE, MAE, and Huber are for continuous regression; they differ in how they treat large errors and outliers.
- Binary cross entropy is for binary and multilabel classification.
- Cross entropy is for mutually exclusive multiclass classification and is the core next-token loss in language models.
- KL divergence compares two distributions and appears in distillation, VAEs, and policy constraints.
- Hinge, contrastive, and triplet losses are useful for ranking, margins, and embedding-space learning.

If you remember one principle: **look at what the label and output mean before choosing the loss**. A loss function is not just a swappable API argument; it is part of the problem definition.
