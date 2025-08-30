## 08 — Graphs: BFS, DFS, Toposort, DSU

Representations
- Adjacency list (preferred for sparse graphs)
- Edge list (useful for DSU/MST)
- Matrix (dense graphs)

Build adjacency list
```java
import java.util.*;

public class GraphBuild {
  public static List<List<Integer>> buildDirected(int n, int[][] edges) {
    List<List<Integer>> g = new ArrayList<>();
    for (int i = 0; i < n; i++) g.add(new ArrayList<>());
    for (int[] e : edges) g.get(e[0]).add(e[1]);
    return g;
  }
  public static List<List<Integer>> buildUndirected(int n, int[][] edges) {
    List<List<Integer>> g = new ArrayList<>();
    for (int i = 0; i < n; i++) g.add(new ArrayList<>());
    for (int[] e : edges) { g.get(e[0]).add(e[1]); g.get(e[1]).add(e[0]); }
    return g;
  }
}
```

BFS (shortest path in unweighted graphs)
```java
import java.util.*;

public class BFS {
  public static int[] distances(List<List<Integer>> g, int start) {
    int n = g.size();
    int[] dist = new int[n]; Arrays.fill(dist, -1);
    Deque<Integer> dq = new ArrayDeque<>(); dq.add(start); dist[start] = 0;
    while (!dq.isEmpty()) {
      int u = dq.poll();
      for (int v : g.get(u)) if (dist[v] == -1) { dist[v] = dist[u] + 1; dq.add(v); }
    }
    return dist;
  }
}
```

DFS (components, cycle detection)
```java
import java.util.*;

public class DFS {
  public static void dfs(List<List<Integer>> g, int u, boolean[] seen) {
    seen[u] = true;
    for (int v : g.get(u)) if (!seen[v]) dfs(g, v, seen);
  }
  public static boolean hasCycleDirected(List<List<Integer>> g) {
    int n = g.size();
    int[] state = new int[n]; // 0=unseen,1=visiting,2=done
    for (int i = 0; i < n; i++) if (state[i] == 0 && cycleDfs(g, i, state)) return true;
    return false;
  }
  private static boolean cycleDfs(List<List<Integer>> g, int u, int[] st) {
    st[u] = 1;
    for (int v : g.get(u)) {
      if (st[v] == 1) return true;
      if (st[v] == 0 && cycleDfs(g, v, st)) return true;
    }
    st[u] = 2; return false;
  }
}
```

Topological sort
- DFS postorder reverse, or Kahn’s algorithm (BFS by indegree)
```java
import java.util.*;

public class Toposort {
  public static List<Integer> kahn(List<List<Integer>> g) {
    int n = g.size();
    int[] indeg = new int[n];
    for (int u = 0; u < n; u++) for (int v : g.get(u)) indeg[v]++;
    Deque<Integer> dq = new ArrayDeque<>();
    for (int i = 0; i < n; i++) if (indeg[i] == 0) dq.add(i);
    List<Integer> order = new ArrayList<>();
    while (!dq.isEmpty()) {
      int u = dq.poll(); order.add(u);
      for (int v : g.get(u)) if (--indeg[v] == 0) dq.add(v);
    }
    return order.size() == n ? order : Collections.emptyList();
  }
}
```

Disjoint Set Union (Union-Find) for connectivity
```java
public class DSU {
  private final int[] parent, size;
  public DSU(int n){ parent = new int[n]; size = new int[n]; for(int i=0;i<n;i++){parent[i]=i; size[i]=1;} }
  public int find(int x){ return parent[x]==x?x:(parent[x]=find(parent[x])); }
  public boolean union(int a,int b){ int ra=find(a), rb=find(b); if(ra==rb) return false; if(size[ra]<size[rb]){int t=ra;ra=rb;rb=t;} parent[rb]=ra; size[ra]+=size[rb]; return true; }
}
```

Exercises
- Number of connected components in an undirected graph
- Detect cycle in directed/undirected graph
- Course schedule (toposort) and reconstruct an order
- Count provinces using DSU


