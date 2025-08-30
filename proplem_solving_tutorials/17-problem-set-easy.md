## 17 — Problem Set (Easy) with Full Solutions

Format
- Each problem includes: statement, approach, complexity, and Java solution

1) Two Sum (indices)
- Problem: Given `nums` and `target`, return indices i < j such that nums[i] + nums[j] = target.
- Approach: One pass with `HashMap<value, index>`
- Complexity: O(N) time, O(N) space
```java
import java.util.*;

public class PS17_TwoSum {
  public static int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> indexByValue = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
      int need = target - nums[i];
      Integer j = indexByValue.get(need);
      if (j != null) return new int[]{j, i};
      indexByValue.put(nums[i], i);
    }
    return new int[]{-1, -1};
  }
}
```

2) Valid Parentheses
- Approach: stack; push openers, pop on closers
- Complexity: O(N)
```java
import java.util.*;

public class PS17_ValidParentheses {
  public static boolean isValid(String s) {
    Deque<Character> st = new ArrayDeque<>();
    for (char c : s.toCharArray()) {
      if (c=='('||c=='['||c=='{') st.push(c);
      else {
        if (st.isEmpty()) return false;
        char o = st.pop();
        if (o=='('&&c!=')' || o=='['&&c!=']' || o=='{'&&c!='}') return false;
      }
    }
    return st.isEmpty();
  }
}
```

3) Merge Two Sorted Lists
- Approach: iterative merge with dummy head
- Complexity: O(N+M)
```java
public class PS17_MergeTwoLists {
  static class ListNode { int val; ListNode next; ListNode(int v){val=v;} }
  public static ListNode merge(ListNode a, ListNode b) {
    ListNode dummy = new ListNode(0), cur = dummy;
    while (a != null && b != null) {
      if (a.val <= b.val) { cur.next = a; a = a.next; }
      else { cur.next = b; b = b.next; }
      cur = cur.next;
    }
    cur.next = (a != null) ? a : b; return dummy.next;
  }
}
```

4) Binary Search (lower bound)
- Approach: standard
```java
public class PS17_LowerBound {
  public static int lowerBound(int[] a, int x) {
    int l=0, r=a.length; while(l<r){ int m=l+(r-l)/2; if(a[m]<x) l=m+1; else r=m; } return l;
  }
}
```

5) Max Profit (Best Time to Buy/Sell Stock I)
- Approach: track min so far and max diff
```java
public class PS17_MaxProfit {
  public static int maxProfit(int[] prices){ int min= Integer.MAX_VALUE, ans=0; for(int p: prices){ if(p<min) min=p; ans=Math.max(ans, p-min);} return ans; }
}
```


