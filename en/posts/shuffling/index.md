How to generate a random sequence (permutation) of a finite sequence.

<!--more-->

## Fisher–Yates shuffle

The original form was described by Ronald Fisher and Frank Yates:

1. Write down the numbers from 1 through N.
2. Pick a random number k between one and the number of un-struck numbers remaining (inclusive).
3. Counting from the low end, strike out the kth number not yet struck out, and write it down at the end of a separate list.
4. Repeat from step 2 until all the numbers have been struck out.
5. The sequence of numbers written down in step 3 is now a random permutation of the original numbers

The modern version of the Fisher–Yates shuffle, designed for computer use, was introduced by Richard Durstenfeld in 1964 and popularized by Donald E. Knuth in The Art of Computer Programming as "Algorithm P (Shuffling)".

```python
# -- To shuffle an array a of n elements (indices 0..n-1):
from random import randrange
def shuffle(items): # 0-based
    i = len(items)
    while i > 1:
        j = randrange(i) # 0, 1, 2, ..., i-1
        i -= 1
        items[i], items[j] = items[j], items[i] # swap(items[i], items[j])
```

(clang's libc++ uses this idea, see `/usr/include/c++/v1/algorithm` on your computer or [github: llvm-project](https://github.com/llvm/llvm-project/blob/main/libcxx/include/algorithm#L3255))

You can easily write a equivalent left to right version of this.

```python
# -- To shuffle an array a of n elements (indices 0..n-1):
from random import randrange
def shuffle(items): # 0-based
    n = len(items)
    i = 0
    while i < n - 1:
        j = randrange(i, n) # i, i+1, ..., n-1
        items[i], items[j] = items[j], items[i] # swap(items[i], items[j])
        i += 1
```

## The "inside-out" shuffle

(gcc's libstdc++ uses this, see `/usr/include/c++/<gcc version>/bits/stl_algo.h` on your computer or [github: gcc-mirror](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/stl_algo.h#L3729))

To simultaneously initialize and shuffle an array, use this:

```python
# -- To shuffle an array a of n elements (indices 0..n-1):
from random import randrange
def shuffle(items): # 0-based
    n = len(items)
    i = 2
    while i <= n:
        j = randrange(i) # 0, 1, ..., i-1
        items[i-1], items[j] = items[j], items[i-1] # swap(items[i-1], items[j])
        i += 1
```

At any loop, `items[0:i]` is a shuffle of `items[0:i]`, you don't even need to know the value of n in advance.

The correctness of the inside-out shuffle can be proved by induction.

## Refs

* [wiki: Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
