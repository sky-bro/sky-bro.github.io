
A de Bruijn sequence of order n on a size-k alphabet A is a cyclic sequence in which every possible length-n string on A occurs exactly once as a substring.

For a *de Bruijn sequence* of order n on a size-k alphabet $A$, we denote it by $B(k, n)$

<!--more-->

## Basic Properties

* $B(k, n)$ has length $k^n$ (also the number of distinct strings of length n on A)
* De Bruijn sequences are optimally short with respect to the property of containing every string of length n exactly once
* The number of distinct de Bruijn sequences $B(k, n)$ is
  $$\frac{(k!)^{k^{n-1}}}{k^n}$$

## An Example Sequence

*let's use $B(2, 4)$ as an example*

Sequence `0000111101100101` (cyclic sequcence) belongs to $B(2,4)$.
It contains every string of length n exactly once:

```text
   {0  0  0  0} 1  1  1  1  0  1  1  0  0  1  0  1
    0 {0  0  0  1} 1  1  1  0  1  1  0  0  1  0  1
    0  0 {0  0  1  1} 1  1  0  1  1  0  0  1  0  1
    0  0  0 {0  1  1  1} 1  0  1  1  0  0  1  0  1
    0  0  0  0 {1  1  1  1} 0  1  1  0  0  1  0  1
    0  0  0  0  1 {1  1  1  0} 1  1  0  0  1  0  1
    0  0  0  0  1  1 {1  1  0  1} 1  0  0  1  0  1
    0  0  0  0  1  1  1 {1  0  1  1} 0  0  1  0  1
    0  0  0  0  1  1  1  1 {0  1  1  0} 0  1  0  1
    0  0  0  0  1  1  1  1  0 {1  1  0  0} 1  0  1
    0  0  0  0  1  1  1  1  0  1 {1  0  0  1} 0  1
    0  0  0  0  1  1  1  1  0  1  1 {0  0  1  0} 1
    0  0  0  0  1  1  1  1  0  1  1  0 {0  1  0  1}
    0} 0  0  0  1  1  1  1  0  1  1  0  0 {1  0  1 ...
... 0  0} 0  0  1  1  1  1  0  1  1  0  0  1 {0  1 ...
... 0  0  0} 0  1  1  1  1  0  1  1  0  0  1  0 {1 ...
```

## How to Construct the Sequence

Can be constructed by taking an Eulerian cycle of an (n − 1)-dimensional de Bruijn graph: ([Hierholzer’s Algorithm](https://www.geeksforgeeks.org/hierholzers-algorithm-directed-graph/))
{{< figure src="/images/posts/De Bruijn sequence/example01.jpg" caption="A de Bruijn graph" alt="A de Bruijn graph" >}}

```c++
// copied from geeksforgeeks
#include <bits/stdc++.h>
using namespace std;

unordered_set<string> seen;
vector<int> edges;

// Modified DFS in which no edge
// is traversed twice
void dfs(string node, int& k, string& A)
{
  for (int i = 0; i < k; ++i) {
    string str = node + A[i];
    if (seen.find(str) == seen.end()) {
      seen.insert(str);
      dfs(str.substr(1), k, A);
      edges.push_back(i);
    }
  }
}

// Function to find a de Bruijn sequence
// of order n on k characters
string deBruijn(int n, int k, string A)
{

  // Clearing global variables
  seen.clear();
  edges.clear();

  string startingNode = string(n - 1, A[0]);
  dfs(startingNode, k, A);

  string S;

  // Number of edges
  int l = pow(k, n);
  for (int i = 0; i < l; ++i)
    S += A[edges[i]];
  S += startingNode;

  return S;
}

// Driver code
int main()
{
  int n = 3, k = 2;
  string A = "01";
  cout << deBruijn(n, k, A);
  return 0;
} 
```

## Related Problems

* [leetcode 753: Cracking the Safe](https://leetcode.com/problems/cracking-the-safe/)
  example solution from [leetcode discuss](https://leetcode.com/problems/cracking-the-safe/discuss/110260/De-Bruijn-sequence-C%2B%2B)

  ```c++
  class Solution {
    int n, k, v;
    vector<vector<bool> > visited;
    string sequence;

  public:
    string crackSafe(int n, int k) {
      if (k == 1) return string(n, '0');
      this->n = n;
      this->k = k;
      v = 1;
      for (int i = 0; i < n - 1; ++i) v *= k;
      visited.resize(v, vector<bool>(k, false));
      dfs(0);
      return sequence + sequence.substr(0, n - 1);
    }

    void dfs(int u) {
      for (int i = 0; i < k; ++i) {
        if (!visited[u][i]) {
          visited[u][i] = true;
          dfs((u * k + i) % v);
          sequence.push_back('0' + i);
        }
      }
    }
  };
  ```

## Refs

* [wiki: De Bruijn sequence](https://en.wikipedia.org/wiki/De_Bruijn_sequence)
* [geeksforgeeks: De Bruijn sequence | Set 1](https://www.geeksforgeeks.org/de-bruijn-sequence-set-1/)
