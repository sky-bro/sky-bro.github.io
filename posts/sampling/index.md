How to select k out of N objects.

<!--more-->

## Reservoir sampling

> [source code](https://github.com/sky-bro/AC/tree/master/Algorithms/Sampling/Reservoir%20Sampling)

Reservoir sampling is a family of randomized algorithms for choosing a simple random sample, without replacement, of k items from a population of unknown size n in a **single pass** over the items.

* The size of the population n is not known and is typically very large. (scan from left to right, without looking back)
* At any point, the current state is a simple random sample without replacement of size k over the part of the population seen so far.

### Simple algorithm (*Algorithm R*)

```c++
/**
 * A simple and popular but slow algorithm, commonly known as Algorithm R
 * time complexity O(n)
 */
void reservoir_sample(vector<int> &S, vector<int> &R, int k) {
    R.resize(k);
    copy(S.begin(), S.begin()+k, R.begin());
    for (int i = k, n = S.size(); i < n; ++i) {
        int j = rand_int(0, i);
        if (j < k) R[j] = S[i]; // choose ith with probability (k/(i+1))
                                // keep ith with
                                // probability not being swapped by remmaing elements [i+1...n-1]
                                // (k/(i+1)) * (1-1/(i+2))*...*(1-1/n) = k/n
    }
}
```

### An optimal algorithm (*Algorithm L*)

```c++
/**
 * Algorithm L improves upon this algorithm by
 * computing how many items are discarded before
 * the next item enters the reservoir
 * time complexity O(k(1+log(n/k)))
 */
void reservoir_sample(vector<int> &S, vector<int> &R, int k) {
    R.resize(k);
    copy(S.begin(), S.begin()+k, R.begin());
    double W = exp(log(rand_real(0, 1))/k);
    for (int i = k, n = S.size(); i < n;) {
        i += floor(log(rand_real(0, 1))/log(1-W)) + 1;
        if (i < n) {
            R[rand_int(0, k-1)] = S[i];
            W *= exp(log(rand_real(0, 1))/k);
        }
    }
}
```

### Problems for practice

## Refs

* [wiki: Reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)
* [C++ randomly sample k numbers from range 0:n-1 (n > k) without replacement](https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement)
