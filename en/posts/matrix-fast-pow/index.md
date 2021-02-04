Use matrix and fast pow together can make some problems much easier.

<!--more-->

## Fast Pow

old friend fast pow:

```c++
template <typename T>
T pow(T x, int n) {
  T ret = 1;
  while (n) {
    if (n & 1) ret *= x;
    x *= x;
    n >>= 1;
  }
  return ret;
}
```

{{< notice info >}}
`Matrix<ll, 10, 10, MOD> ret = 1` will use constructor `Matrix(T x, bool isMainDiagonal = true)`.
{{< /notice >}}

## A Matrix Template

```c++
template <typename T, std::size_t R, std::size_t C = R,
          std::size_t M = INT32_MAX>
class Matrix {
 public:
  T m[R][C];

  Matrix() { memset(m, 0, sizeof(m)); }
  /**
   * construct a matrix whose diagonal (fill at most min(R, C) number as x) is
   * filled with number x, and the rest filled with 0's
   * @param x number to be filled at the diagonal
   * @param isMainDiagonal fill main diagonal if true, else fill the
   * antidiagonal
   */
  Matrix(T x, bool isMainDiagonal = true) : Matrix() {
    if (isMainDiagonal)
      for (std::size_t i = 0; i < R && i < C; ++i) m[i][i] = x;
    else
      for (std::size_t i = 0, j = C - 1; i < R && j >= 0; --j, ++i) m[i][j] = x;
  }

  template <std::size_t C2>
  Matrix<T, R, C2, M> operator*(const Matrix<T, C, C2, M> &other) const {
    Matrix<T, R, C2, M> res;
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t k = 0; k < C; ++k)
        for (std::size_t j = 0; j < C2; ++j)
          res.m[i][j] = (res.m[i][j] + m[i][k] * other.m[k][j] % M) % M;
    return res;
  }

  Matrix<T, R, C, M> &operator*=(const Matrix<T, C, C, M> &other) {
    return *this = *this * other;
  }

  void fill(T x) {
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j) m[i][j] = x;
  }

  T sum() const {
    T res = 0;
    for (std::size_t i = 0; i < R; ++i)
      for (std::size_t j = 0; j < C; ++j) res = (res + m[i][j]) % M;
    return res;
  }
};
```

## Examples

### leetcode: 509 Fibonacci Number

> link: [509 Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
> time: $O(log(n))$, space: $O(1)$

#### description

$$
\begin{align*}
F(0) &= 0 \\\\
F(1) &= 1 \\\\
F(n) &= F(n - 1) + F(n - 2),\\;for\\; n > 1.
\end{align*}
$$
now given `n`, compute `F(n)`;

#### idea

keep track of previous two numbers `F(i-2), F(i-1)`, and update them at each increase of `i`, till we get to `n`.

the transition can be represented by a matrix, mulitply all transition matrixes together (use fast pow here) then transit once from initial state `F(0) = 0, F(1) = 1`.

#### solution

> code on github: [sky-bro/AC/leetcode.com/0509 Fibonacci Number/](https://github.com/sky-bro/AC/tree/master/leetcode.com/0509%20Fibonacci%20Number)

```c++
class Solution {
 public:
  int fib(int N) {
    Matrix<ll, 1, 2, MOD> res;
    res.m[0][1] = 1;
    if (N < 2) return N;
    Matrix<ll, 2, 2, MOD> x;  // transition matrix
    x.fill(1);
    x.m[0][0] = 0;
    return (res * pow(x, N - 1)).m[0][1];
  }
};
```

### leetcode: 935 Knight Dialer

> link: [935 Knight Dialer](https://leetcode.com/problems/knight-dialer/)
> time: $O(log(n))$, space: $O(1)$

#### description

```c
/*
1 2 3
4 5 6
7 8 9
* 0 #
*/
```

how many different sequence of numbers can a chess knight dial (`% 109 + 7`). the knight can only stand on numeric cells, and may start from any digit.

#### idea

This is a simple dp problem, at every move, knight standing at 1 can go to 6 and 8 (or knight standing at 6 and 8 can go to 1), so `dp[1] = dp[6] + dp[8]`, but we need to update all `dp[0..9]` same time (we don't want to overwrite dp values in the last step), so the time and space complexity would be $O(n)$ and $O(1)$.

Another way to update dp values is by multiplying a transition matrix, and in every transition, this matrix is gonna be the same, so we just multiply all transitions together (where can use fast pow) and transit from the starting point (when `n=1`: `[1,1,1,1,1,1,1,1,1,1]`) only once.
The time complexity would be reduced to $O(log(n))$.

#### solution

> code on github: [sky-bro/AC/leetcode.com/0935 Knight Dialer/](https://github.com/sky-bro/AC/tree/master/leetcode.com/0935%20Knight%20Dialer)

```c++
class Solution {
 public:
  int knightDialer(int n) {
    if (n == 1) return 10;
    Matrix<ll, 1, 10, MOD> res;
    res.fill(1);
    /*
    1 2 3
    4 5 6
    7 8 9
    * 0 #
    */
    Matrix<ll, 10, 10, MOD> x;
    vector<vector<int>> trans = { {4, 6}, {6, 8}, {7, 9}, {4, 8}, {0, 3, 9}, {}, {0, 1, 7}, {2, 6}, {1, 3}, {2, 4} };
    for (int i = 0; i < 10; ++i)
      for (int j : trans[i]) x.m[i][j] = 1;
    return (res * pow(x, n - 1)).sum();
  }
};
```

### more for exercise

* todo...

## refs

* [Matrix - By DanAlex](https://codeforces.com/blog/entry/21189)
* [Exponentiation by squaring](https://simple.wikipedia.org/wiki/Exponentiation_by_squaring)
