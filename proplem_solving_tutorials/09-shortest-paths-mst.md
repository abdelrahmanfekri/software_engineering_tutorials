## 09 — Shortest Paths and MST

Shortest paths
- Dijkstra: non-negative weights, O((N+M) log N)
- 0-1 BFS: edges weights in {0,1}, O(N+M)
- Bellman–Ford: handles negative edges, detects negative cycles, O(NM)
- Floyd–Warshall: all-pairs, O(N^3)

Dijkstra with adjacency list
```java
import java.util.*;

public class Dijkstra {
  public static long[] shortestPaths(int n, int[][] edges) {
    List<List<int[]>> g = new ArrayList<>();
    for (int i = 0; i < n; i++) g.add(new ArrayList<>());
    for (int[] e : edges) g.get(e[0]).add(new int[]{e[1], e[2]});
    long INF = (long)1e18;
    long[] dist = new long[n]; Arrays.fill(dist, INF); dist[0] = 0;
    PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
    pq.offer(new long[]{0, 0});
    boolean[] vis = new boolean[n];
    while (!pq.isEmpty()) {
      long[] cur = pq.poll(); long d = cur[0]; int u = (int)cur[1];
      if (vis[u]) continue; vis[u] = true;
      for (int[] e : g.get(u)) {
        int v = e[0], w = e[1];
        if (dist[v] > d + w) { dist[v] = d + w; pq.offer(new long[]{dist[v], v}); }
      }
    }
    return dist;
  }
}
```

0-1 BFS
```java
import java.util.*;

public class ZeroOneBFS {
  public static int[] shortest01(int n, List<List<int[]>> g, int start) {
    int[] dist = new int[n]; Arrays.fill(dist, Integer.MAX_VALUE); dist[start] = 0;
    Deque<Integer> dq = new ArrayDeque<>(); dq.add(start);
    while(!dq.isEmpty()){
      int u = dq.pollFirst();
      for(int[] e: g.get(u)){
        int v=e[0], w=e[1];
        int nd = dist[u] + w;
        if(nd < dist[v]){ dist[v]=nd; if(w==0) dq.addFirst(v); else dq.addLast(v); }
      }
    }
    return dist;
  }
}
```

Minimum Spanning Tree (MST)
- Kruskal: sort edges by weight + DSU
- Prim: build tree by expanding with minimal edge using PQ

Kruskal
```java
import java.util.*;

public class KruskalMST {
  static class DSU { int[] p, s; DSU(int n){ p=new int[n]; s=new int[n]; for(int i=0;i<n;i++){p[i]=i;s[i]=1;} }
    int f(int x){ return p[x]==x?x:(p[x]=f(p[x])); }
    boolean u(int a,int b){ a=f(a); b=f(b); if(a==b) return false; if(s[a]<s[b]){int t=a;a=b;b=t;} p[b]=a; s[a]+=s[b]; return true; }}
  public static long mstWeight(int n, int[][] edges) {
    Arrays.sort(edges, Comparator.comparingInt(e -> e[2]));
    DSU dsu = new DSU(n);
    long total = 0; int used = 0;
    for (int[] e : edges) if (dsu.u(e[0], e[1])) { total += e[2]; if (++used == n - 1) break; }
    return used == n - 1 ? total : -1; // -1 if disconnected
  }
}
```

Exercises
- Shortest path on a grid with obstacles (Dijkstra vs BFS depending on weights)
- Network delay time (Dijkstra)
- Minimum cost to connect all points (Prim with Manhattan distance)


