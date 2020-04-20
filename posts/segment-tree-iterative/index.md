
A segment tree is a binary tree where each node represents an interval. Each node stores some property of its corresponding interval: like the maximum/minimum num, the sum of that interval.

<!--more-->

## Applicable Problems

For any array `a`, where every element belongs to some monoid $(S, \oplus)$ we can build a segment tree to answer the following queries (problems):
– `Get(l, r)` — returns $a_l \oplus a_{l+1} \oplus \dotsb a_r$
– `Change(p, x)` — set $a_p = x$

{{< notice info >}}

* **Monoids** are semigroups with identity.
* $\oplus$ is an associative binary operation.
* **Identity element** for some pair $(S, \oplus)$ is such an element $e \in S$ that for every $a \in S$ condition $a ⊕ e = e \oplus a = a$ holds;

{{< /notice >}}

## Example Problem

{{< alert theme="success" >}}

[leetcode 307: Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)
Given an integer array `nums`, find the sum of the elements between indices `i` and `j` (`i ≤ j`), inclusive.
The `update(i, val)` function modifies nums by updating the element at index `i` to `val`.

{{< /alert >}}

For this problem, the **identity element** is **0**, and the binary operation is **+** between integers.
~~And for simplicity we use the **identity element** to extend the length of the original array to some integer power of 2.~~

{{< figure src="/images/posts/segment tree/padded.svg" caption="Padded Segment Tree" alt="Padded Segment Tree" >}}

if size of the array is `n`, then we only need an array of `2*n` length to store the segment tree. (only in iterative version, property of a [Complete Binary Tree](https://en.wikipedia.org/wiki/Binary_tree#Arrays))

{{< figure src="/images/posts/segment tree/iterative02.svg" caption="Segment Tree Built Iteratively" alt="Segment Tree Built Iteratively" >}}

### Define the binary operation

Here we will just use `+` for our operation, you can if you need define a merge function for your special operation $\oplus$.

```C++
inline int merge(int a, int b) {
  return a + b;
}
```

### Build the Tree

We want to construct an array like above (the original array is `{1, 2, 3, 4, 5, 6}`), the essential idea of a segment tree is that a node at index $i$ (index starts from 1, you can also try starting from 0, but starting from 1 is simpler) can have two children at indexes $(2 \ast i)$ and $(2 \ast i + 1)$.

```C++
NumArray(vector<int>& nums) {
  n = nums.size();
  segment_tree.resize(2 * n);
  for (int i = 0; i < n; ++i) {
    segment_tree[i + n] = nums[i];
  }
  for (int i = n - 1; i > 0; --i) {
    segment_tree[i] = segment_tree[i<<1] + segment_tree[i<<1|1];
  }
}
```

### Query a range sum

```C++
int sumRange(int i, int j) { // sum range [i, j]
    int sum = 0;
    for (i += n, j += n+1; i < j; i >>= 1, j >>= 1) {
        if (i&1) sum += segment_tree[i++];
        if (j&1) sum += segment_tree[--j];
    }
    return sum;
}
```

### Update an element/elements

```C++
void update(int i, int val) {
    i += n;
    if (segment_tree[i] == val) return;
    for (segment_tree[i] = val; i > 1; i >>= 1) {
        segment_tree[i>>1] = segment_tree[i] + segment_tree[i^1];
    }
}
```

### Complete Solution to the Problem

{{< expand "Leetcode 307 Solution (C++)" >}}

```C++
class NumArray {
private:
    int n;
    vector<int> segment_tree;
public:
    NumArray(vector<int>& nums) {
        n = nums.size();
        segment_tree.resize(2*n);
        for (int i = 0; i < n; ++i) {
            segment_tree[i+n] = nums[i];
        }
        for (int i = n-1; i > 0; --i) {
            segment_tree[i] = segment_tree[i<<1] + segment_tree[i<<1|1];
        }
    }
    void update(int i, int val) {
        i += n;
        if (segment_tree[i] == val) return;
        for (segment_tree[i] = val; i > 1; i >>= 1) {
            segment_tree[i>>1] = segment_tree[i] + segment_tree[i^1];
        }
    }
    int sumRange(int i, int j) { // sum range [i, j]
        int sum = 0;
        for (i += n, j += n+1; i < j; i >>= 1, j >>= 1) {
            if (i&1) sum += segment_tree[i++];
            if (j&1) sum += segment_tree[--j];
        }
        return sum;
    }
};
```

{{< /expand >}}

## Refs

* [youtube: Efficient Segment Tree Tutorial](https://www.youtube.com/watch?v=Oq2E2yGadnU)
* [codeforces: Efficient and easy segment trees](https://codeforces.com/blog/entry/18051): best
* [Segment tree Theory and applications](http://maratona.ic.unicamp.br/MaratonaVerao2016/material/segment_tree_lecture.pdf)
* [wiki: binary tree - in an array](https://en.wikipedia.org/wiki/Binary_tree#Arrays)
