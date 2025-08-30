## 04 — Binary Search, Binary Search on Answer, Prefix & Difference Sums

Binary Search Essentials
- Maintain a sorted predicate: F F F T T T; find first T (lower bound)
- Avoid infinite loops: compute mid as `low + (high - low) / 2`

Lower/Upper bound implementation
```java
public class Bounds {
  public static int lowerBound(int[] sorted, int target) {
    int low = 0, high = sorted.length; // [low, high)
    while (low < high) {
      int mid = low + (high - low) / 2;
      if (sorted[mid] < target) low = mid + 1; else high = mid;
    }
    return low;
  }
  public static int upperBound(int[] sorted, int target) {
    int low = 0, high = sorted.length; // [low, high)
    while (low < high) {
      int mid = low + (high - low) / 2;
      if (sorted[mid] <= target) low = mid + 1; else high = mid;
    }
    return low;
  }
}
```

Binary Search on Answer
- When answer is monotonic over a numeric domain (e.g., minimal capacity to ship in D days)
- Define feasible(mid): can we meet constraints with value mid?

Prefix Sums
- Precompute sums: `prefix[i] = sum(nums[0..i-1])`
- Range sum [l, r]: `prefix[r+1] - prefix[l]`

Difference Array for range updates
```java
public class DifferenceArray {
  public static int[] applyRangeIncrements(int n, int[][] updates) {
    int[] diff = new int[n + 1];
    for (int[] u : updates) {
      int l = u[0], r = u[1], delta = u[2];
      diff[l] += delta;
      if (r + 1 < diff.length) diff[r + 1] -= delta;
    }
    int[] result = new int[n];
    int run = 0;
    for (int i = 0; i < n; i++) {
      run += diff[i];
      result[i] = run;
    }
    return result;
  }
}
```

Exercises
- Find smallest divisor such that sum(ceil(a[i]/x)) ≤ threshold
- Ship packages within D days (binary search capacity)
- Range add queries using difference array

