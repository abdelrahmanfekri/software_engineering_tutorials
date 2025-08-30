## 10 — Greedy, Intervals, Scheduling

Greedy recipe
- Sort by a key that preserves optimal substructure
- Maintain an invariant; use exchange argument for correctness

Activity selection (maximize non-overlapping)
```java
import java.util.*;

public class ActivitySelection {
  public static int maxNonOverlapping(int[][] intervals) {
    Arrays.sort(intervals, Comparator.comparingInt(a -> a[1]));
    int count = 0, end = Integer.MIN_VALUE;
    for (int[] it : intervals) {
      if (it[0] >= end) { count++; end = it[1]; }
    }
    return count;
  }
}
```

Merge intervals
```java
import java.util.*;

public class MergeIntervals {
  public static int[][] merge(int[][] intervals) {
    if (intervals.length == 0) return intervals;
    Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
    List<int[]> res = new ArrayList<>();
    int[] cur = intervals[0].clone();
    for (int i = 1; i < intervals.length; i++) {
      if (intervals[i][0] <= cur[1]) cur[1] = Math.max(cur[1], intervals[i][1]);
      else { res.add(cur); cur = intervals[i].clone(); }
    }
    res.add(cur);
    return res.toArray(new int[0][]);
  }
}
```

Minimum arrows to burst balloons (intervals)
```java
import java.util.*;

public class MinArrows {
  public static int findMinArrowShots(int[][] points) {
    if (points.length == 0) return 0;
    Arrays.sort(points, Comparator.comparingInt(a -> a[1]));
    int arrows = 1; long end = points[0][1];
    for (int i = 1; i < points.length; i++) {
      if (points[i][0] > end) { arrows++; end = points[i][1]; }
    }
    return arrows;
  }
}
```

Job sequencing with deadlines and profit
```java
import java.util.*;

public class JobSequencing {
  static class Job { int d, p; Job(int d,int p){this.d=d; this.p=p;} }
  public static int maxProfit(List<Job> jobs) {
    jobs.sort((a,b) -> b.p - a.p);
    int maxD = 0; for (Job j : jobs) maxD = Math.max(maxD, j.d);
    int[] parent = new int[maxD + 1]; for (int i = 0; i <= maxD; i++) parent[i] = i;
    java.util.function.IntUnaryOperator find = new java.util.function.IntUnaryOperator(){
      @Override public int applyAsInt(int x){ return parent[x]==x?x:(parent[x]=applyAsInt(parent[x])); }
    };
    int profit = 0;
    for (Job j : jobs) {
      int t = find.applyAsInt(j.d);
      if (t > 0) { profit += j.p; parent[t] = t - 1; }
    }
    return profit;
  }
}
```

Exercises
- Non-overlapping intervals (min removals)
- Meeting rooms (minimum number of rooms)
- Partition labels (greedy partitioning of string)


