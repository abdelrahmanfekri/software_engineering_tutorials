## 13 — Backtracking & Bitmasking

Backtracking template
```java
import java.util.*;

public class BacktrackTemplate {
  public static List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    dfs(nums, 0, new ArrayList<>(), res); return res;
  }
  private static void dfs(int[] nums, int i, List<Integer> cur, List<List<Integer>> res) {
    if (i == nums.length) { res.add(new ArrayList<>(cur)); return; }
    dfs(nums, i + 1, cur, res);
    cur.add(nums[i]); dfs(nums, i + 1, cur, res); cur.remove(cur.size() - 1);
  }
}
```

Permutations
```java
import java.util.*;

public class Permutations {
  public static List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    boolean[] used = new boolean[nums.length];
    dfs(nums, used, new ArrayList<>(), res); return res;
  }
  private static void dfs(int[] nums, boolean[] used, List<Integer> cur, List<List<Integer>> res) {
    if (cur.size() == nums.length) { res.add(new ArrayList<>(cur)); return; }
    for (int i = 0; i < nums.length; i++) if (!used[i]) {
      used[i] = true; cur.add(nums[i]);
      dfs(nums, used, cur, res);
      cur.remove(cur.size() - 1); used[i] = false;
    }
  }
}
```

N-Queens
```java
import java.util.*;

public class NQueens {
  public static List<List<String>> solveNQueens(int n) {
    List<List<String>> res = new ArrayList<>();
    boolean[] col = new boolean[n], d1 = new boolean[2*n], d2 = new boolean[2*n];
    int[] pos = new int[n]; Arrays.fill(pos, -1);
    dfs(0, n, col, d1, d2, pos, res); return res;
  }
  private static void dfs(int r, int n, boolean[] col, boolean[] d1, boolean[] d2, int[] pos, List<List<String>> res) {
    if (r == n) { res.add(render(pos, n)); return; }
    for (int c = 0; c < n; c++) if (!col[c] && !d1[r - c + n] && !d2[r + c]) {
      col[c] = d1[r - c + n] = d2[r + c] = true; pos[r] = c;
      dfs(r + 1, n, col, d1, d2, pos, res);
      col[c] = d1[r - c + n] = d2[r + c] = false; pos[r] = -1;
    }
  }
  private static List<String> render(int[] pos, int n) {
    List<String> board = new ArrayList<>();
    for (int r = 0; r < n; r++) {
      char[] row = new char[n]; Arrays.fill(row, '.');
      row[pos[r]] = 'Q'; board.add(new String(row));
    }
    return board;
  }
}
```

Bitmask enumeration
```java
public class BitmaskEnum {
  public static void iterateSubsets(int mask) {
    for (int sub = mask; sub > 0; sub = (sub - 1) & mask) {
      // process sub
    }
  }
}
```

Exercises
- Combination sum (with/without reuse)
- Word search (backtracking on grid)
- Partition to k equal sum subsets (bitmask + memo)


