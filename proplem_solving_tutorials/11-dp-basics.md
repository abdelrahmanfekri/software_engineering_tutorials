## 11 — Dynamic Programming (Basics)

DP checklist
- Define state: what parameters fully describe a subproblem?
- Transition: relation between states
- Base cases: minimal problems with known answers
- Order: top-down (memoization) or bottom-up (tabulation)
- Complexity: states × transitions

0/1 Knapsack
```java
public class Knapsack01 {
  public static int maxValue(int[] weight, int[] value, int capacity) {
    int n = weight.length; int[] dp = new int[capacity + 1];
    for (int i = 0; i < n; i++) {
      for (int c = capacity; c >= weight[i]; c--) {
        dp[c] = Math.max(dp[c], dp[c - weight[i]] + value[i]);
      }
    }
    return dp[capacity];
  }
}
```

Coin Change (min coins)
```java
import java.util.*;

public class CoinChangeMin {
  public static int minCoins(int[] coins, int amount) {
    int INF = 1_000_000_000; 
    int[] dp = new int[amount + 1]; 
    Arrays.fill(dp, INF); dp[0] = 0;
    for (int coin : coins) 
    for (int a = coin; a <= amount; a++) 
    dp[a] = Math.min(dp[a], dp[a - coin] + 1);
    return dp[amount] >= INF ? -1 : dp[amount];
  }
}
```

Longest Increasing Subsequence (O(n log n))
```java
import java.util.*;

public class LIS {
  public static int lengthOfLIS(int[] nums) {
    int[] tails = new int[nums.length]; int size = 0;
    for (int x : nums) {
      int i = Arrays.binarySearch(tails, 0, size, x);
      if (i < 0) i = -(i + 1);
      tails[i] = x; if (i == size) size++;
    }
    return size;
  }
}
```

Grid paths (obstacles)
```java
public class UniquePaths2 {
  public static int uniquePathsWithObstacles(int[][] grid) {
    int m = grid.length, n = grid[0].length; int[] dp = new int[n];
    dp[0] = grid[0][0] == 0 ? 1 : 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == 1) dp[j] = 0; else if (j > 0) dp[j] += dp[j - 1];
      }
    }
    return dp[n - 1];
  }
}
```

Exercises
- House robber (linear and circular)
- Decode ways
- Edit distance (two-strings DP)


