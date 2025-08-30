## 16 — Templates & Snippets (Java 21)

Fast I/O (simple)
```java
import java.io.*;
import java.util.*;

public class FastIO {
  static class FastScanner {
    private final InputStream in; private final byte[] buffer = new byte[1 << 16];
    private int ptr = 0, len = 0;
    FastScanner(InputStream is) { in = is; }
    private int read() throws IOException {
      if (ptr >= len) { len = in.read(buffer); ptr = 0; if (len <= 0) return -1; }
      return buffer[ptr++];
    }
    String next() throws IOException {
      StringBuilder sb = new StringBuilder();
      int c; while ((c = read()) <= ' ') { if (c == -1) return null; }
      do { sb.append((char)c); c = read(); } while (c > ' ');
      return sb.toString();
    }
    int nextInt() throws IOException { return Integer.parseInt(next()); }
    long nextLong() throws IOException { return Long.parseLong(next()); }
  }
}
```

Pair and Triplet
```java
public record Pair<A, B>(A first, B second) {}
public record Triplet<A, B, C>(A first, B second, C third) {}
```

Disjoint Set Union (Union-Find)
```java
public class DisjointSetUnion {
  private final int[] parent, size;
  public DisjointSetUnion(int n) {
    parent = new int[n]; size = new int[n];
    for (int i = 0; i < n; i++) { parent[i] = i; size[i] = 1; }
  }
  public int find(int x) { return parent[x] == x ? x : (parent[x] = find(parent[x])); }
  public boolean union(int a, int b) {
    int ra = find(a), rb = find(b);
    if (ra == rb) return false;
    if (size[ra] < size[rb]) { int t = ra; ra = rb; rb = t; }
    parent[rb] = ra; size[ra] += size[rb];
    return true;
  }
}
```

Segment Tree (sum)
```java
import java.util.*;

public class SegmentTree {
  private final int n; private final long[] tree;
  public SegmentTree(int[] arr) {
    int size = 1; while (size < arr.length) size <<= 1; n = size; tree = new long[n << 1];
    for (int i = 0; i < arr.length; i++) tree[n + i] = arr[i];
    for (int i = n - 1; i > 0; i--) tree[i] = tree[i << 1] + tree[i << 1 | 1];
  }
  public void update(int index, long value) {
    int pos = n + index; tree[pos] = value; pos >>= 1;
    while (pos > 0) { tree[pos] = tree[pos << 1] + tree[pos << 1 | 1]; pos >>= 1; }
  }
  public long query(int l, int r) { // inclusive l, exclusive r
    long res = 0; int left = l + n, right = r + n;
    while (left < right) {
      if ((left & 1) == 1) res += tree[left++];
      if ((right & 1) == 1) res += tree[--right];
      left >>= 1; right >>= 1;
    }
    return res;
  }
}
```

Fenwick (Binary Indexed Tree)
```java
public class FenwickTree {
  private final long[] bit; private final int n;
  public FenwickTree(int n) { this.n = n; bit = new long[n + 1]; }
  public void add(int index, long delta) { for (int i = index + 1; i <= n; i += i & -i) bit[i] += delta; }
  public long sumPrefix(int index) { long s = 0; for (int i = index + 1; i > 0; i -= i & -i) s += bit[i]; return s; }
  public long sumRange(int l, int r) { return sumPrefix(r) - (l > 0 ? sumPrefix(l - 1) : 0); }
}
```

BFS/DFS templates
```java
import java.util.*;

public class GraphTemplates {
  public static List<List<Integer>> buildUndirected(int n, int[][] edges) {
    List<List<Integer>> g = new ArrayList<>();
    for (int i = 0; i < n; i++) g.add(new ArrayList<>());
    for (int[] e : edges) { g.get(e[0]).add(e[1]); g.get(e[1]).add(e[0]); }
    return g;
  }
  public static int[] bfsDistances(List<List<Integer>> g, int start) {
    int n = g.size(); int[] dist = new int[n]; Arrays.fill(dist, -1);
    Deque<Integer> dq = new ArrayDeque<>(); dq.add(start); dist[start] = 0;
    while (!dq.isEmpty()) {
      int u = dq.poll();
      for (int v : g.get(u)) if (dist[v] == -1) { dist[v] = dist[u] + 1; dq.add(v); }
    }
    return dist;
  }
}
```

Comparators & heaps
```java
import java.util.*;

public class Heaps {
  public static PriorityQueue<int[]> maxHeapBySecond() {
    return new PriorityQueue<>((a, b) -> Integer.compare(b[1], a[1]));
  }
}
```


