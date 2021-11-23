max_heap: `A[0]` is the maximum value
min_heap: `A[0]` is the minimum value
source code: [AC/Algorithms/Heap](https://github.com/sky-bro/AC/tree/master/Algorithms/Heap)

<!--more-->

## intro

* `go_up(int i)`: from node i go up to at most node 0, swap with parent if node i is bigger (time: $O(log(n))$)
* `go_down(int i)`: from node i go down to its leaf child at most, swap with the larger child every time goes down (time: $O(log(n))$)
* `heapify(n)`: adjust `A[0..n-1]` as a heap with `go_down` (time: $O(n)$, refer: [How is make_heap in C++ implemented to have complexity of 3N?](https://stackoverflow.com/a/5057675/14335187))
* `push(int x)`: `A[sz] = x`, then `go_up(sz++)` (time: $O(log(n))$, same as `go_up`)
* `pop()`: `swap(A[0], A[--sz])`, then `go_down(0)` (time: $O(log(n))$, same as `go_down`)

## code

[complete source code on github](https://github.com/sky-bro/AC/tree/master/Algorithms/Heap)

```c++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5;
int sz = 0;
// here is a max_heap, A[0] is the maximum value
int A[N];

void go_down(int i) {
    for (int j = 2 * i + 1; j < sz; j = 2 * (i = j) + 1) { // j: left child of i
        if (j < sz - 1 && A[j] < A[j + 1]) ++j; // go to right child
        if (A[i] >= A[j]) break;
        swap(A[i], A[j]);
    }
}

void go_up(int i) {
    for (int p = (i - 1) / 2; i; p = ((i = p) - 1) / 2) {
        if (A[i] <= A[p]) break;
        swap(A[i], A[p]);
    }
}

void heapify(int n) {
    if (n == 1) return;
    int i = (n - 2) / 2;
    while (true) {
        go_down(i);
        if (i-- == 0) return;
    }
}

void pop() {
    if (sz <= 1) {
        sz = 0;
        return;
    }
    swap(A[0], A[--sz]);
    go_down(0);
}

void push(int x) {
    A[sz] = x;
    go_up(sz++);
}
```

## Refs

* [How is make_heap in C++ implemented to have complexity of 3N?](https://stackoverflow.com/a/5057675/14335187)
* STL source code: make_heap, push_heap, pop_heap, ...
