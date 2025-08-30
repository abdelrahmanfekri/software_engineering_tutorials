## 02 — Core Data Structures in Java

Key structures and when to use them
- Arrays: fixed-size contiguous; O(1) index; best for tight loops
- ArrayList: dynamic array; amortized O(1) append; O(1) access
- LinkedList: O(1) add/remove at ends; O(N) random access; use rarely
- Deque: `ArrayDeque` preferred; O(1) amortized push/pop both ends
- Stack: use `ArrayDeque` as stack; avoid `Stack` class
- Queue: `ArrayDeque` for BFS; `PriorityQueue` for heaps
- HashMap/HashSet: average O(1) operations; unordered
- TreeMap/TreeSet: ordered by comparator; O(log N) operations
- PriorityQueue: min-heap by default; pass comparator for max-heap

PriorityQueue example (top-k)
```java
import java.util.*;

public class TopKFrequent {
  public static List<Integer> topK(int[] nums, int k) {
    Map<Integer, Integer> frequencyByNumber = new HashMap<>();
    for (int value : nums) frequencyByNumber.put(value, frequencyByNumber.getOrDefault(value, 0) + 1);

    PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    for (Map.Entry<Integer, Integer> entry : frequencyByNumber.entrySet()) {
      minHeap.offer(new int[]{entry.getKey(), entry.getValue()});
      if (minHeap.size() > k) minHeap.poll();
    }

    List<Integer> result = new ArrayList<>();
    while (!minHeap.isEmpty()) result.add(minHeap.poll()[0]);
    Collections.reverse(result);
    return result;
  }
}
```

TreeSet/TreeMap example (ordered queries)
```java
import java.util.*;

public class OrderedQueries {
  public static Integer predecessor(TreeSet<Integer> set, int x) {
    return set.floor(x - 1);
  }
  public static Integer successor(TreeSet<Integer> set, int x) {
    return set.ceiling(x + 1);
  }
}
```

Deque for sliding window maximum
```java
import java.util.*;

public class SlidingWindowMaximum {
  public static int[] maxSlidingWindow(int[] nums, int windowSize) {
    if (windowSize <= 0) return new int[0];
    int n = nums.length;
    int[] result = new int[Math.max(0, n - windowSize + 1)];
    Deque<Integer> indexDeque = new ArrayDeque<>();

    for (int rightIndex = 0; rightIndex < n; rightIndex++) {
      while (!indexDeque.isEmpty() && indexDeque.peekFirst() <= rightIndex - windowSize) {
        indexDeque.pollFirst();
      }
      while (!indexDeque.isEmpty() && nums[indexDeque.peekLast()] <= nums[rightIndex]) {
        indexDeque.pollLast();
      }
      indexDeque.offerLast(rightIndex);
      if (rightIndex >= windowSize - 1) {
        result[rightIndex - windowSize + 1] = nums[indexDeque.peekFirst()];
      }
    }
    return result;
  }
}
```

Exercises
- Implement a max-heap using `PriorityQueue` and a comparator
```java
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
```
- Use `TreeMap` to keep a sliding window multiset with counts
- Build a deque-based solution for the first negative number in every subarray of size k


