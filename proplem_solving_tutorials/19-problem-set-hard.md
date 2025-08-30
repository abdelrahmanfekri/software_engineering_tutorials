## 19 — Problem Set (Hard) with Full Solutions

1) Trapping Rain Water
- Approach: two pointers with leftMax/rightMax
```java
public class PS19_TrappingRainWater {
  public static int trap(int[] h){ int n=h.length, l=0, r=n-1, lm=0, rm=0, ans=0; while(l<r){ if(h[l]<h[r]){ if(h[l]>=lm) lm=h[l]; else ans+=lm-h[l]; l++; } else { if(h[r]>=rm) rm=h[r]; else ans+=rm-h[r]; r--; } } return ans; }
}
```

2) Longest Increasing Path in a Matrix
- Approach: DFS + memo (top-down DP)
```java
public class PS19_LongestIncreasingPath {
  static int[] dr={1,-1,0,0}, dc={0,0,1,-1};
  public static int longestIncreasingPath(int[][] m){ int R=m.length, C=m[0].length, ans=0; int[][] memo=new int[R][C]; for(int r=0;r<R;r++) for(int c=0;c<C;c++) ans=Math.max(ans, dfs(m,r,c,memo)); return ans; }
  static int dfs(int[][] m,int r,int c,int[][] memo){ if(memo[r][c]!=0) return memo[r][c]; int best=1; for(int k=0;k<4;k++){ int nr=r+dr[k], nc=c+dc[k]; if(0<=nr&&nr<m.length&&0<=nc&&nc<m[0].length&&m[nr][nc]>m[r][c]) best=Math.max(best,1+dfs(m,nr,nc,memo)); } return memo[r][c]=best; }
}
```

3) Minimum Window Substring
- Approach: variable-size sliding window with frequency counts
```java
import java.util.*;

public class PS19_MinWindowSubstring {
  public static String minWindow(String s, String t) {
    int[] need = new int[128]; int required = 0;
    for(char c : t.toCharArray()) if (need[c]++ == 0) required++;
    int[] have = new int[128]; int formed = 0;
    int L = 0, bestLen = Integer.MAX_VALUE, bestL = 0;
    for (int R = 0; R < s.length(); R++) {
      char c = s.charAt(R);
      if (++have[c] == need[c]) formed++;
      while (formed == required) {
        if (R - L + 1 < bestLen) { bestLen = R - L + 1; bestL = L; }
        char lc = s.charAt(L++);
        if (have[lc]-- == need[lc]) formed--;
      }
    }
    return bestLen == Integer.MAX_VALUE ? "" : s.substring(bestL, bestL + bestLen);
  }
}
```

4) Word Ladder (Shortest transformation sequence length)
- Approach: BFS over generic patterns
```java
import java.util.*;

public class PS19_WordLadder {
  public static int ladderLength(String beginWord, String endWord, java.util.List<String> wordList) {
    Set<String> dict = new HashSet<>(wordList); if(!dict.contains(endWord)) return 0;
    Map<String, List<String>> bucket = new HashMap<>();
    int L = beginWord.length();
    for(String w: dict){ for(int i=0;i<L;i++){ String key = w.substring(0,i)+'*'+w.substring(i+1); bucket.computeIfAbsent(key,k->new ArrayList<>()).add(w);} }
    Deque<String> dq=new ArrayDeque<>(); dq.add(beginWord);
    Map<String,Integer> dist=new HashMap<>(); dist.put(beginWord,1);
    while(!dq.isEmpty()){
      String w = dq.poll(); int d = dist.get(w);
      for(int i=0;i<L;i++){
        String key=w.substring(0,i)+'*'+w.substring(i+1);
        for(String nxt: bucket.getOrDefault(key, List.of())) if(!dist.containsKey(nxt)){ dist.put(nxt,d+1); dq.add(nxt); if(nxt.equals(endWord)) return d+1; }
      }
    }
    return 0;
  }
}
```

5) Median of Two Sorted Arrays
- Approach: binary search partition O(log(min(n,m)))
```java
public class PS19_MedianTwoSorted {
  public static double findMedianSortedArrays(int[] A, int[] B) {
    if (A.length > B.length) return findMedianSortedArrays(B, A);
    int m=A.length, n=B.length, half=(m+n+1)/2, lo=0, hi=m;
    while(lo<=hi){
      int i=(lo+hi)/2, j=half-i;
      int Aleft = (i==0)? Integer.MIN_VALUE : A[i-1];
      int Aright= (i==m)? Integer.MAX_VALUE : A[i];
      int Bleft = (j==0)? Integer.MIN_VALUE : B[j-1];
      int Bright= (j==n)? Integer.MAX_VALUE : B[j];
      if (Aleft<=Bright && Bleft<=Aright){
        int leftMax = Math.max(Aleft, Bleft);
        int rightMin = Math.min(Aright, Bright);
        return ((m+n)%2==1) ? leftMax : (leftMax + rightMin) / 2.0;
      } else if (Aleft > Bright) hi = i - 1; else lo = i + 1;
    }
    return 0;
  }
}
```


