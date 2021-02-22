BIT can be used to compute the prefix **sum** of an array in $log(n)$ time and takes only $O(n)$ space.

> [source code](https://github.com/sky-bro/AC/tree/master/Algorithms/BIT)

<!--more-->

## Introduction

Binary Indexed Tree (BIT, or Fenwick Tree), is used for computing the prefix *sum* (some associative binary operation) of an array, and it can modify an element by *adding* some value to that element.

Unlike [Segment Tree](../segment-tree-iterative/), Fenwick Tree cannot let you query a range, it's only used for querying the prefix sum. (But depends on the *sum* operation, you can sometines get a range sum from two prefix sums).

Fenwick Tree takes only $O(n)$ space, whereas segment tree takes $O(2*n)$ space, or $O(4*n)$ if you use lazy propagation.

Fenwick Tree has two functions. They are `sum` and `add`, as shown below.

## `add` increase an element

```cpp
void add(int i, T v) {  // adds v to A[i]
    while (i <= n) A[i] += v, i += i & -i;
}
```

> `(i & -i)` will get the lowest bit of i, for example `(6 & -6) = 0b10 = 2`

Elements are 1-indexed here, so `A[0]` is not used.

## `sum` get a prefix sum

```cpp
T sum(int i) {  // prefix sum: A[1] + A[2] + ... + A[i]
    T v{};
    while (i) v += A[i], i -= i & -i;
    return v;
}
```

TODO

## template

> [source code](https://github.com/sky-bro/AC/tree/master/Algorithms/BIT)

```cpp
template <typename T>
class fenwick {
public:
  int n;
  vector<T> A;
  fenwick(int n): n(n), A(n+1) {} // A[0] not used
  T sum(int i) {  // prefix sum: A[1] + A[2] + ... + A[i]
    T v{};
    while (i) v += A[i], i -= i & -i;
    return v;
  }
  void add(int i, T v) {  // adds v to A[i]
    while (i <= n) A[i] += v, i += i & -i;
  }
};
```

## Refs

* [wiki: Fenwick Tree](https://en.wikipedia.org/wiki/Fenwick_tree)
