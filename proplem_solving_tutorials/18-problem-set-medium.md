## 18 — Problem Set (Medium) with Full Solutions

1) Longest Substring Without Repeating Characters
- Approach: sliding window with frequency map of last index
```java
import java.util.*;

public class PS18_LongestUniqueSubstring {
  public static int lengthOfLongestSubstring(String s) {
    int[] last = new int[256]; Arrays.fill(last, -1);
    int start = 0, ans = 0;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (last[c] >= start) start = last[c] + 1;
      last[c] = i;
      ans = Math.max(ans, i - start + 1);
    }
    return ans;
  }
}
```

2) Top K Frequent Elements
- Approach: frequency map + min-heap of size k
```java
import java.util.*;

public class PS18_TopKFrequent {
  public static int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> f = new HashMap<>();
    for (int x : nums) f.put(x, f.getOrDefault(x, 0) + 1);
    PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    for (Map.Entry<Integer,Integer> e : f.entrySet()) {
      pq.offer(new int[]{e.getKey(), e.getValue()}); if (pq.size() > k) pq.poll();
    }
    int[] res = new int[pq.size()]; int i = 0; while(!pq.isEmpty()) res[i++] = pq.poll()[0];
    return res;
  }
}
```

3) K Closest Points to Origin
- Approach: max-heap of size k on distance
```java
import java.util.*;

public class PS18_KClosestPoints {
  public static int[][] kClosest(int[][] points, int k) {
    PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> Integer.compare(dist(b), dist(a)));
    for (int[] p : points) { pq.offer(p); if (pq.size() > k) pq.poll(); }
    int[][] res = new int[pq.size()][]; int i = 0; while(!pq.isEmpty()) res[i++] = pq.poll();
    return res;
  }
  private static int dist(int[] p){ return p[0]*p[0] + p[1]*p[1]; }
}
```

4) Number of Islands
- Approach: BFS/DFS over grid; mark visited; count components
```java
public class PS18_NumberOfIslands {
  public static int numIslands(char[][] grid) {
    int m = grid.length, n = grid[0].length, ans = 0;
    boolean[][] seen = new boolean[m][n];
    int[] dr = {1,-1,0,0}, dc = {0,0,1,-1};
    java.util.ArrayDeque<int[]> dq = new java.util.ArrayDeque<>();
    for (int r = 0; r < m; r++) for (int c = 0; c < n; c++) if (grid[r][c]=='1' && !seen[r][c]) {
      ans++; seen[r][c] = true; dq.add(new int[]{r,c});
      while(!dq.isEmpty()){
        int[] cur = dq.poll();
        for (int d = 0; d < 4; d++){
          int nr = cur[0] + dr[d], nc = cur[1] + dc[d];
          if (nr>=0&&nr<m&&nc>=0&&nc<n && grid[nr][nc]=='1' && !seen[nr][nc]){ seen[nr][nc]=true; dq.add(new int[]{nr,nc}); }
        }
      }
    }
    return ans;
  }
}
```

5) Course Schedule (can finish?)
- Approach: topological sort; if topo order exists → true
```java
import java.util.*;

public class PS18_CourseSchedule {
  public static boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> g = new ArrayList<>(); for(int i=0;i<numCourses;i++) g.add(new ArrayList<>());
    int[] indeg = new int[numCourses];
    for (int[] p : prerequisites) { g.get(p[1]).add(p[0]); indeg[p[0]]++; }
    Deque<Integer> dq = new ArrayDeque<>(); for(int i=0;i<numCourses;i++) if(indeg[i]==0) dq.add(i);
    int taken = 0; while(!dq.isEmpty()){ int u=dq.poll(); taken++; for(int v: g.get(u)) if(--indeg[v]==0) dq.add(v); }
    return taken == numCourses;
  }
}
```


