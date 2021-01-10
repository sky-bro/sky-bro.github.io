Description from wiki: a disjoint-set data structure, also called a union–find data structure or merge–find set, is a data structure that stores a collection of disjoint (non-overlapping) sets

> you can easily solve [leetcode: 547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/) w/ UF.
> get UF template at [github/sky-bro/AC/Algorithms/Union Find](https://github.com/sky-bro/AC/tree/master/Algorithms/Union%20Find)

<!--more-->

## Template

UF datastructure is very simple, it has only two key functions: `U` and `F`.

* use `U` (union) to connect two nodes, so they belong to the same set;
* use `F` (find) to find the root of a node (or the group id of the node)

We use `ids` to store the parent of every node, `ids[i]` is the i-th node's parent. if `ids[i] == i` means `i` is the root if its group, we can say the group id is `i`.

So, different root/group id means nodes are in different groups, whereas same root/group id means two nodes are in the same group.

### simle

```c++
class UF {
 private:
  vector<int> ids;

 public:
  UF(int n) {
    ids.resize(n);
    iota(ids.begin(), ids.end(), 0);
  }

  int F(int x) { return ids[x] == x ? x : (ids[x] = F(ids[x])); }

  void U(int p, int q) { ids[F(p)] = ids[F(q)]; }
};
```

### full

part from `U` and `F`, based on your needs, you can add these functions:

* `int group_count()`, count the number of sets/groups -- initialize `group_cnt` as n (each node as a group), every time `U` unions two different groups, `--group_count`.
* `int count(int x)`, count the size of the group of node x -- when `U` unions two different groups, add one group's size to another (the new root).
* `bool connected(int p, int q)`, check if two nodes are in the same group.

```c++
class UF {
 private:
  vector<int> ids, cnts;
  int cnt;

 public:
  UF(int n) {
    ids.resize(n);
    iota(ids.begin(), ids.end(), 0);
    cnt = n;
    cnts.resize(n, 1);
  }

  int F(int x) { return ids[x] == x ? x : (ids[x] = F(ids[x])); }

  void U(int p, int q) {
    int pid = F(p), qid = F(q);
    if (pid != qid) {
      ids[qid] = pid;
      --cnt;
      cnts[pid] += cnts[qid];
    }
  }

  int group_count() { return cnt; }
  int count(int x) { return cnts[F(x)]; }
  bool connected(int p, int q) { return F(p) == F(q); }
};
```

## Solution to lc problem 547

> you can also solve this problem with dfs: [github/sky-bro/AC/leetcode.com/0547 Friend Circles/main.cpp](https://github.com/sky-bro/AC/tree/master/leetcode.com/0547%20Friend%20Circles), lc has changed the name of this problem to "Number of Provinces"

```c++
// union find
class Solution {
 private:
  int n, cnt;
  vector<int> ids;
  void U(int p, int q) {
    int pid = F(p), qid = F(q);
    if (pid != qid) {
      --cnt;
      ids[pid] = ids[qid];
    }
  }
  int F(int x) { return x == ids[x] ? x : (ids[x] = F(ids[x])); }

 public:
  int findCircleNum(vector<vector<int>>& isConnected) {
    n = cnt = isConnected.size();
    ids.resize(n);
    iota(ids.begin(), ids.end(), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (isConnected[i][j]) {
          U(i, j);
        }
      }
    }
    return cnt;
  }
};
```
