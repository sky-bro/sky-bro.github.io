
## Dijkstra's algorithm {#dijkstra-s-algorithm}

for a given node, find the shortest path between that node and every other.

{{< alert theme="info" dir="ltr" >}}

idea: use **best first search** to explore (visit adjacent nodes of some node) the unexplored node with shortest/best distance.

{{< /alert >}}


### shortest path {#shortest-path}

code example: from [AC/Algorithms/Dijkstra](https://github.com/sky-bro/AC/tree/master/Algorithms/Dijkstra)

```cpp
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

using namespace std;

/**
 * compute shortest distance from src to all other nodes
 */
void dijkstra(const vector<unordered_map<int, int>>& G, int src,
              vector<int>& dist) {
  int n = G.size();
  dist.clear();
  dist.resize(n, -1);
  vector<bool> vis(n);
  priority_queue<pair<int, int>> pq;  // {-weight, label}
  pq.emplace(0, src);
  dist[src] = 0;
  while (!pq.empty()) {
    auto p = pq.top();
    pq.pop();
    vis[p.second] = true;
    for (auto neighbor : G[p.second]) {
      int v = neighbor.first;
      int d = neighbor.second - p.first;
      if (vis[neighbor.first]) continue;
      if (dist[v] == -1 || dist[v] > d) {
        pq.emplace(-d, v);
        dist[v] = d;
      }
    }
  }
}

template <typename T>
void printArr(const vector<T>& arr) {
  for (const T& t : arr) cout << t << " ";
  cout << endl;
}

int main(int argc, char const* argv[]) {
  vector<vector<int>> edges = {
      {0, 1, 100}, {1, 2, 100}, {0, 2, 500}};  // {src, dst, weight}, ...
  int N = 3;
  vector<unordered_map<int, int>> G(N);
  for (const auto& edge : edges) {
    G[edge[0]][edge[1]] = edge[2];
  }
  vector<int> dist;
  int src = 0;
  dijkstra(G, src, dist);
  printArr(dist);
  return 0;
}
```


### path reconstruction {#path-reconstruction}

every time a shorter dist for a node is found (relaxation), record the source node to this node of current edge.


### negative cycle detection {#negative-cycle-detection}

do not allow negative weight edges, cannot detect negative cycle


## Bellman-Ford/SPFA algorithm {#bellman-ford-spfa-algorithm}

like Dijkstra's algorithm, finds the shortest path between a source node to all other nodes but allows negative weights and negative cycles.

refer to:

-   <https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm>
-   <https://github.com/sky-bro/AC/tree/master/Algorithms/bellman-ford>
-   <https://www.luogu.com.cn/blog/tale365/bellman-ford-suan-fa>

{{< alert theme="info" dir="ltr" >}}

Dijkstra's algorithm uses a priority queue to greedily select the closest vertex that has not yet been processed, and performs this relaxation process on all of its outgoing edges;

by contrast, the Bellman–Ford algorithm simply relaxes all the edges, and does this V-1 times

{{< /alert >}}


### shortest path {#shortest-path}

```cpp
bool solve() {
  int n, m, START = 1;
  cin >> n >> m;
  // read graph
  vector<unordered_map<int, int>> graph(n + 1);
  for (int i = 0; i < m; ++i) {
    int s, t, w;
    cin >> s >> t >> w;
    graph[s][t] = graph[s].count(t) ? min(w, graph[s][t]) : w;
    if (w >= 0) {
      graph[t][s] = graph[t].count(s) ? min(w, graph[t][s]) : w;
    }
  }

  // the order (0 indexed) of a shortest path, if >= n, then negative cycle detected
  vector<int> cnt(n + 1);
  vector<int> dist(n + 1, INT32_MAX);
  dist[START] = 0;

  priority_queue<int> q;
  q.push(START);
  vector<bool> in_queue(n);
  while (!q.empty()) {
    int source = q.top();
    q.pop();
    in_queue[source] = false;
    // iterate through all neighbors
    for (auto neighborWeight: graph[source]) {
      // relaxation
      if (neighborWeight.second + dist[source] < dist[neighborWeight.first]) {
        dist[neighborWeight.first] = neighborWeight.second + dist[source];
        cnt[neighborWeight.first] = cnt[source] + 1;
        if (cnt[neighborWeight.first] >= n) {
          // loop detected
          return true;
        }
        if (!in_queue[neighborWeight.first]) {
          in_queue[neighborWeight.first] = true;
          q.push(neighborWeight.first);
        }
      }
    }
  }
  return false;
}
```


### path reconstruction {#path-reconstruction}

as in Dijkstra's algorithm, in each relaxation of an edge, record the source node of the edge.


### negative cycle detection {#negative-cycle-detection}

-   Bellman-Ford: after relaxing edges \\(V-1\\) iterations, in the $V$-th relaxation, if more shorter distances can be found, then there's a cycle.
-   SPFA: in each relaxation, record the shortest path length, if path length &gt;= number of nodes, meaning some nodes have appeared twice on this path, i.e. a negative cycle.


### code practices {#code-practices}

-   [P2136 拉近距离](https://www.luogu.com.cn/problem/P2136)
-   [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/): variant of SPFA, path at most of length K
-   [P3385 【模板】负环](https://www.luogu.com.cn/problem/P3385)


## Floyd's algorithm {#floyd-s-algorithm}

find shortest paths in  a directed weighted graph (positive or negative edge weights, no negative cycles)


### shortest path {#shortest-path}

```cpp
// https://www.luogu.com.cn/problem/B3647
void solve() {
  // number of nodes and edges
  int n, m;
  cin >> n >> m;

  // all positive weights, use -1 to indicate no edge
  vector<vector<int>> G(n, vector<int>(n, NO_EDGE));
  vector<vector<int>> prev(n, vector<int>(n));

  for (int i = 0; i < n; ++i) {
    G[i][i] = 0;
  }
  for (int i = 0; i < m; ++i) {
    int from, to, weight;
    cin >> from >> to >> weight;
    --from;
    --to;
    G[from][to] = min(weight, G[from][to]);
    prev[from][to] = from;
    // non directed
    G[to][from] = min(weight, G[to][from]);
    prev[to][from] = to;
  }

  // floyd
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (G[i][k] != NO_EDGE && G[k][j] != NO_EDGE &&
            (G[i][j] == NO_EDGE || G[i][j] > G[i][k] + G[k][j])) {
          G[i][j] = G[i][k] + G[k][j];
          prev[i][j] = prev[k][j];
        }
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << G[i][j] << " ";
    }
    cout << endl;
  }
}
```


### path reconstruction {#path-reconstruction}

every time a shorter path between i and j is found by adding node k, update the previous node of j on the shortest path between i and j to that of k and j.


### detect negative cycle {#detect-negative-cycle}

when a negative cycle exists, the shortest path between a node and itself will be negative. So, in each iteration, we detect if the shortest distance matrix's diagonal contains any negative value.


## SUMMARY {#summary}

| Algorithm    | Use cases                                                        | Time Complexity                                                 | Space Complexity           |
|--------------|------------------------------------------------------------------|-----------------------------------------------------------------|----------------------------|
| Dijkstra     | single source shortest path, positive weight                     | \\(O(\vert{}E\vert{} + \vert{}V\vert{}\*\log{\vert{}V\vert})\\) | \\(O(\vert{}V\vert)\\)     |
| Bellman-Ford | single source shortest path, negative weight, negative cycle     | \\(O(\vert{}V\vert{}\*\vert{}E\vert)\\)                         | \\(O(\vert{}V\vert)\\)     |
| floyd        | shortest path between all nodes, negative weight, negative cycle | \\(O(\vert{}V\vert{}^{3})\\)                                    | \\(O(\vert{}V\vert^{2})\\) |
