My simple note on the RSA algorithm.

<!--more-->

## Introduction

We know that **RSA** is an **asymmetric encryption algorithm**, meaning that the communication partners Alice and Bob hold different keys, instead of same keys as in symmetric encryption.

In rsa, Alice first computes the product $n$ of two different large prime numbers $p$ and $q$, and uses $p$ and $q$ to derive two keys $e$ and $d$, one for herself and one for others. Then she makes $(e, n)$ public and destroies the $p$ and $q$.

## Encryption and Decryption

> $e$ and $d$ can both be used for encryption or decryption: $e$ can be used to decpryt what's encrypted with $d$, $d$ can be used to decpryt what's encrypted with $e$

* To Encrypt

    If Alice encrypts a message `m`, she need to compute $m^d \equiv c\\ (\text{mod}\\ n)$

* To Decrypt

    Bob knows $(e, n)$ (everyone knows, because this is public), he computes $c^e \equiv m^{de} \equiv m\\ (\text{mod}\\ n)$ and gets the original message `m`.

So the correctness of RSA lies in $m^{de}\equiv m\\ (\text{mod}\\ n)$. We'll understand why in the Proof of Correctness Section.

## How to Generate e and d

* $\phi(n)=(p-1)\times (q-1)$
* choose an integer $e$ which is a smaller than coprime to $\phi(n)$, A popular choice is $e = 2^{16} + 1 = 10001\text{h} = 65537$
* the modular inverse of $e$ modulo $\phi(n)$ is $d$, it can be computed with [Extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)
* in PKCS #1 v2.0, $\phi(n)$ has been replaced with $\lambda(n) = \text{lcm}(p-1,q-1)$

### Euler's totient function

$\phi(n)$: the number of positive integers up to $n$ that are relatively prime to $n$.

For example, $\phi(9)=6$, because there are 6 numbers relatively prime to 9: $\\{1,2,4,5,7,8\\}$

More generally, for any number $N$, it can be represented as the product of some prime numbers: $N = p_1^{k_1}\times p_2^{k_2}\times p_3^{k_3}\times ...$, and $\phi(N)=N\times(1-\frac{1}{p_1})\times(1-\frac{1}{p_2})\times(1-\frac{1}{p_3})...$

### Compute Modular Inverse

There are three ways to compute modular inverse, please refer to this [post](TODO)

## Proof of Correctness

### Euler's totient theorem

* $m^{\phi(n)}=1\\ (\text{mod}\\ n)$ when $m$ and $n$ are coprime
* special case: $m^{n-1}=1\\ (\text{mod}\\ n)$ when $n$ is a prime number and $m$ and $n$ are coprime ($m$ is not a multiple of $n$) -- **Fermat’s Little Theorem**

### Chinese Remainder Theorem

TODO

### Correctness of RSA

* we know that $ed = 1\\ (\text{mod}\\ \lambda(pq))$, $\lambda(pq) = \text{lcm}(p-1, q-1)$.
* $ed = 1\\ (\text{mod}\\ \lambda(pq))$, that is $ed=h(p-1)+1 = k(q-1)+1$
* In order to prove $m^{ed}\equiv m\\ (\text{mod}\\ pq)$, it's equivalent to prove $m^{ed}\equiv m\\ (\text{mod}\\ p)$ and  $m^{ed}\equiv m\\ (\text{mod}\\ q)$ separately (using **Chinese Remainder Theorem**)
  * prove $m^{ed}\equiv m\\ (\text{mod}\\ p)$
    $$
    \begin{align*}
    m^{ed} &= m^{h(p-1)+1}\\ (\text{mod}\\ p)\\\\
    &= 0\\ \text{(when m is multiple of p) or}\\ (1)^{h}\times m\\ \text{(when m and p are coprime)}\\\\
    &= m\\ (\text{mod}\\ p)
    \end{align*}
    $$
  * prove $m^{ed}\equiv m\\ (\text{mod}\\ q)$
    $$
    \begin{align*}
    m^{ed} &= m^{k(q-1)+1}\\ (\text{mod}\\ q)\\\\
    &= 0\\ \text{(when m is multiple of q) or}\\ (1)^{k}\times m\\ \text{(when m and q are coprime)}\\\\
    &= m\\ (\text{mod}\\ q)
    \end{align*}
    $$

## refs

* [wiki: RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem))
* [wiki: Euler's totient function](https://en.wikipedia.org/wiki/Euler%27s_totient_function)
* [wiki: Euler's totient theorem](https://en.wikipedia.org/wiki/Euler%27s_totient_theorem)
* [wiki: Chinese remainder theorem](https://en.wikipedia.org/wiki/Chinese_remainder_theorem)
* [slide: Correctness Proof of RSA](https://www.cse.cuhk.edu.hk/~taoyf/course/bmeg3120/notes/rsa-proof.pdf), I think the proof is wrong, (when applying Fermat’s Little Theorem, didn't consider the case when m is multiple of p).
* [wiki: Extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)
* [同余定理+逆元】知识点讲解](https://blog.csdn.net/LOOKQAQ/article/details/81282342)
