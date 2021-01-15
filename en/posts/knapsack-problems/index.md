knapsack problem and its variations

<!--more-->

## 0-1 knapsack problem

### description

Given a set of $n$ items, each with a weight $w_{i}$ and a value $v_{i}$, along with a maximum weight capacity $W$, try pick some of these items (total weight not surpassing $W$), so that the total value is maximized, find the maximum total value.

### solution to acwing: 01背包问题

> [problem link](https://www.acwing.com/problem/content/2/)
> time: $O(NV)$, space: $O(V)$
> for every item, total weight of items in our knapsack decrease (so that we take each item at most once) from maximum (knapsack volume) to current item's weight.
> `dp[v]` is the maximum value we get with total weight of $v$ (or less) if we do not take in this item;
> `dp[v-p.first] + p.second` is the maximum value we get with total weight of $v$ (or less) if we take this item.

```c++
#include <bits/stdc++.h>

using namespace std;

static int x = []() { std::ios::sync_with_stdio(false); cin.tie(0); return 0; }();
typedef long long ll;

int N, V;

int solve(vector<pair<int, int>> &A) {
  vector<int> dp(V + 1);
  for (auto &p : A) {
    for (int v = V; v >= p.first; --v) {
      dp[v] = max(dp[v], dp[v - p.first] + p.second);
    }
  }
  return *max_element(dp.begin(), dp.end());
}

int main(int argc, char const *argv[]) {
  cin >> N >> V;  // volume
  vector<pair<int, int>> A(N);
  for (int i = 0; i < N; ++i) {
    cin >> A[i].first >> A[i].second;  // (weight, value)... not (value, weight)
  }
  cout << solve(A) << endl;
  return 0;
}
```

### count ways / check valid way exists

> [leetcode: 416 Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
> time: $O(NV)$, space: $O(V)$
> now we need to count the number of ways to form exact $V$ total value
> similar as above, but initialize `dp[0]=1`, and rest as 0.
> if we do not need to count the number of ways, but just check if there exists one valid way, use `or` operation indead of `+`.

```c++
// cannot count ways, gets overflow, use '|' operation is enough
class Solution {
 public:
  bool canPartition(vector<int>& nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum & 1) return false;
    vector<int> dp(sum / 2 + 1);
    dp[0] = 1;
    for (int num : nums) {
      for (int i = sum / 2; i >= num; --i) {
        dp[i] |= dp[i - num];
        // dp[i] += dp[i - num]; // overflow, if number of ways is too big
      }
    }
    return dp[sum / 2];
  }
};
```

```c++
// for 0-1 knapsack problem, using bitset is more simple -- update all bits same time
// use bitset
// https://leetcode.com/problems/partition-equal-subset-sum/discuss/90590/Simple-C%2B%2B-4-line-solution-using-a-bitset
class Solution {
 public:
  bool canPartition(vector<int>& nums) {
    bitset<10001> bits(1);
    int sum = accumulate(nums.begin(), nums.end(), 0);
    for (int num : nums) bits |= bits << num;
    return !(sum & 1) && bits[sum >> 1];
  }
};
```

## unbounded knapsack problem

### description

similar to the 0-1 knapsack problem, but now there are infinite number of each item, so we can take multiple of each.

### solution to acwing: 完全背包问题

> [problem link](https://www.acwing.com/problem/content/3/)
> time: $O(NV)$, space: $O(V)$
> only difference from the 0-1 knapsack problem solution is that now for each item, total weight of items in our knapsack increases (so we can re-put this item in our knapsack) from current item's weight to maximum (knapsack's volume)

```c++
#include <bits/stdc++.h>

using namespace std;

static int x = []() { std::ios::sync_with_stdio(false); cin.tie(0); return 0; }();
typedef long long ll;

int N, V;

int solve(vector<pair<int, int>> &A) {
  vector<int> dp(V + 1);
  for (auto &p : A) {
    for (int v = p.first; v <= V; ++v) { // only difference from 0-1 knapsack problem
      dp[v] = max(dp[v], dp[v - p.first] + p.second);
    }
  }
  return dp[V];
}

int main(int argc, char const *argv[]) {
  cin >> N >> V;  // volume
  vector<pair<int, int>> A(N);
  for (int i = 0; i < N; ++i) {
    cin >> A[i].first >> A[i].second;  // (weight, value)... not (value, weight)
  }
  cout << solve(A) << endl;
  return 0;
}
```

### count ways / check valid way exists

> [leetcode: 518 Coin Change 2](https://leetcode.com/problems/coin-change-2/)
> similar to the previous one, but we update from weight $w_i$ to $V$

```c++
class Solution {
 public:
  int change(int t, vector<int>& cs) {
    // int dp[t + 1] = {1};
    vector<int> dp(t + 1);
    dp[0] = 1;
    for (auto c : cs)
      for (auto j = c; j <= t; ++j) dp[j] += dp[j - c];
    return dp[t];
  }
};
```

## bounded knapsack problem

### description

n types of goods, there are $s_i$ number of type $i$ goods and each with weight $w_i$ and value $v_i$.

### solution to acwing: 多重背包问题 I

> [problem link](https://www.acwing.com/problem/content/4/)
> time: $O(NVlogM)$, space: $O(V)$
> reduce bounded knapsack problem to 0-1 knapsack problem
> for type $i$ goods (M -- total number), split them into groups ($ceil(log_2M)$ groups) of number $1, 2, 4, 8, ... 2^k, M-2^k$. and treat these groups as 0-1 knapsack problem items.

```c++
#include <bits/stdc++.h>

using namespace std;

static int x = []() { std::ios::sync_with_stdio(false); cin.tie(0); return 0; }();
typedef long long ll;

int N, V;

int solve(vector<int> &weights, vector<int> &values, vector<int> &nums) {
  vector<int> dp(V + 1);
  for (int i = 0; i < N; ++i) {
    int num = min(nums[i], V / weights[i]);
    for (int k = 1; num; k *= 2) {
      if (k > num) k = num;
      num -= k;
      for (int j = V; j >= weights[i] * k; --j) {
        dp[j] = max(dp[j], dp[j - weights[i] * k] + values[i] * k);
      }
    }
  }
  return dp[V];
}

int main(int argc, char const *argv[]) {
  cin >> N >> V;  // volume
  vector<int> weights(N), values(N), nums(N);
  for (int i = 0; i < N; ++i) {
    cin >> weights[i] >> values[i] >>
        nums[i];  // (weight, value)... not (value, weight)
  }
  cout << solve(weights, values, nums) << endl;
  return 0;
}
```

## more knapsack problems for exercise

todo...

## refs

* [wiki: knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem)
* [web.ntnu.edu.tw/~algo/Knapsack Problem](http://web.ntnu.edu.tw/~algo/KnapsackProblem.html) -- very nice organized (written in traditional chinese)
