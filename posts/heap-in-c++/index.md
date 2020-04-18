
let's learn how to use heap in C++

<!-- more -->

## Related functions

|     func      | description                          |
| :----------- | :----------------------------------- |
|   make_heap   | Make heap from range                 |
|   push_heap   | Push element into heap range         |
|   pop_heap    | Pop element from heap range          |
|   sort_heap   | Sort elements of heap                |
|    is_heap    | Test if range is heap                |
| is_heap_until | Find first element not in heap order |

### make_heap

> make heap from range

We need to have a heap before we operate on a heap, `make_heap` let's us rearrange elements in a range.

Usually we use a vector to hold these range of elements:

```c++
#include <iostream>     // std::cout
#include <algorithm>    // std::make_heap, std::pop_heap, std::push_heap, std::sort_heap
#include <vector>       // std::vector

std::vector<int> heap{1,2,3,4,5,6,7};
std::make_heap(heap.begin(), heap.end()); // 7 5 6 4 2 1 3
```

For now, you only need to know that after `make_heap`, the biggest element is `*begin(heap)` or `heap[0]`. Other elements are not necessarily sorted, but arranged in a certain way, to know the detail, see [heap in C++ 2](./#).

### push_heap

> Push element into heap range

Given a heap in the range `[first,last-1)`, this function extends the range considered a heap to `[first,last)` by placing the value in `(last-1)` into its corresponding location within it.

Meaning to add an element to a heap:

```C++
heap.push_back(8); // 7 5 6 4 2 1 3 8
std::push_heap(heap.begin(), heap.end()); // 8 7 6 5 2 1 3 4
```

`push_heap` will adjust the elements in the new range `(old range + 1)`, and place the added element to where it fits.

### pop_heap

> Pop element from heap range

Given a heap in the range `[first,last)`, this function rearranges the elements in the heap range `[first,last)` in such a way that the part considered a heap is shortened by one: The element with the highest value is moved to `(last-1)`.

Meaning to pop out the largest element from the heap:

```C++
std::pop_heap(heap.begin(), heap.end()); // 7 5 6 4 2 1 3 8
```

new heap range: `[first, last-1)`

### is_heap

> Test if range is heap

simple as description, code:

```C++
// heap: 7 5 6 4 2 1 3 8
std::is_heap(heap.begin(), heap.end()); // false
std::is_heap(heap.begin(), heap.end()-1); // true
```

### is_heap_until

> Find first element not in heap order

We've already popped out 8 from the heap, so:

```C++
// 7 5 6 4 2 1 3 8
auto it = std::is_heap_until(heap.begin(), heap.end());
std::cout << *it << std::endl; // 8
```

### sort_heap

> Sort elements of heap

Sorts the elements in the heap range `[first,last)` into ascending order.

This is actually quite simple, you may have as well implement this function yourself, just keep doing `pop_heap` until the heap is shortened to 1.

```C++
std::sort_heap(heap.begin(), heap.end() - 1, cmp); // 1 2 3 4 5 6 7 8
```

## Code Examples

```c++
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
void printArr(const std::vector<T> &arr) {
  for (const T &t : arr) std::cout << t << " ";
  std::cout << std::endl;
}

// std::random_device rd;
// std::mt19937_64 urng(rd());

int main(int argc, char const *argv[]) {
  std::cout << "original array" << std::endl;
  std::vector<int> heap(7);
  std::iota(heap.begin(), heap.end(), 1);
  printArr(heap); // 1 2 3 4 5 6 7

  // std::cout << "shuffle" << std::endl;
  // std::shuffle(heap.begin(), heap.end(), urng);
  // printArr(heap);

  // auto cmp = std::greater<int>();
  // auto cmp = std::less<int>();

  std::cout << "make heap" << std::endl;
  std::make_heap(heap.begin(), heap.end());
  printArr(heap); // 7 5 6 4 2 1 3

  std::cout << "push back 8" << std::endl;
  heap.push_back(8);
  printArr(heap); // 7 5 6 4 2 1 3 8

  std::cout << "push heap" << std::endl;
  std::push_heap(heap.begin(), heap.end());
  printArr(heap); // 8 7 6 5 2 1 3 4

  std::cout << "pop heap" << std::endl;
  std::pop_heap(heap.begin(), heap.end());
  printArr(heap); // 7 5 6 4 2 1 3 8

  std::cout << "is heap" << std::endl;
  std::cout << std::is_heap(heap.begin(), heap.end()) << std::endl; // 0

  std::cout << "is heap until" << std::endl;
  auto it = std::is_heap_until(heap.begin(), heap.end());
  std::cout << *it << std::endl; // 8

  std::cout << "sort heap (begin, end-1)" << std::endl;
  std::sort_heap(heap.begin(), heap.end() - 1);
  printArr(heap); // 1 2 3 4 5 6 7 8
  return 0;
}
```

## Refs

* [cplusplus.com: see heap algorithms](https://www.cplusplus.com/reference/algorithm/)
