## 12 — DP Advanced & Optimizations

Tree DP (example: tree diameter)
```java
import java.util.*;

public class TreeDiameter {
  static class E { int to, w; E(int t,int w){this.to=t;this.w=w;} }
  static int best;
  public static int diameter(int n, int[][] edges) {
    List<List<E>> g = new ArrayList<>(); for(int i=0;i<n;i++) g.add(new ArrayList<>());
    for (int[] e : edges) { g.get(e[0]).add(new E(e[1], e[2])); g.get(e[1]).add(new E(e[0], e[2])); }
    best = 0; dfs(0, -1, g); return best;
  }
  private static int dfs(int u, int p, List<List<E>> g) {
    int top1 = 0, top2 = 0;
    for (E e : g.get(u)) if (e.to != p) {
      int h = dfs(e.to, u, g) + e.w;
      if (h > top1) { top2 = top1; top1 = h; } else if (h > top2) top2 = h;
    }
    best = Math.max(best, top1 + top2);
    return top1;
  }
}
```

Bitmask DP (TSP)
```java
import java.util.*;

public class TSP {
  public static int tsp(int[][] dist) {
    int n = dist.length; int N = 1 << n; int INF = 1_000_000_000;
    int[][] dp = new int[N][n]; for (int[] row : dp) Arrays.fill(row, INF);
    dp[1][0] = 0; // start at 0
    for (int mask = 1; mask < N; mask++) {
      for (int u = 0; u < n; u++) if ((mask & (1 << u)) != 0) {
        int cur = dp[mask][u]; if (cur >= INF) continue;
        for (int v = 0; v < n; v++) if ((mask & (1 << v)) == 0) {
          int nm = mask | (1 << v);
          dp[nm][v] = Math.min(dp[nm][v], cur + dist[u][v]);
        }
      }
    }
    int ans = INF; int full = N - 1;
    for (int u = 0; u < n; u++) ans = Math.min(ans, dp[full][u] + dist[u][0]);
    return ans;
  }
}
```

Digit DP (count numbers ≤ N with digits constraints)
```java
import java.util.*;

public class DigitDP {
  static int[][][] memo;
  static char[] s;
  public static int countNoAdjacentEqual(int n) {
    s = Integer.toString(n).toCharArray();
    memo = new int[s.length + 1][11][2];
    for (int[][] a : memo) for (int[] b : a) Arrays.fill(b, -1);
    return dfs(0, 10, 1);
  }
  private static int dfs(int i, int prev, int tight) {
    if (i == s.length) return 1;
    int t = memo[i][prev][tight]; if (t != -1) return t;
    int limit = tight == 1 ? s[i] - '0' : 9; int ans = 0;
    for (int d = 0; d <= limit; d++) {
      if (d == prev) continue;
      ans += dfs(i + 1, d, (tight == 1 && d == limit) ? 1 : 0);
    }
    return memo[i][prev][tight] = ans;
  }
}
```

Optimizations overview
- Space optimization: reuse 1D or 2 rows
- Monotonic queue optimization: DP of form dp[i] = min_j (A[j]) + B[i] - C[j] with j within window
- Divide & Conquer DP: quadrangle inequality / monotonicity to reduce transition cost
- Knuth optimization: for optimal partitions when quadrangle inequality holds

Exercises
- Tree DP: number of nodes in each subtree
- Bitmask DP: minimum Hamiltonian path cost
- Digit DP: count numbers with sum of digits ≤ S


