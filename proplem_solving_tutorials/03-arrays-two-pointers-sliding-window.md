## 03 — Arrays, Two Pointers, Sliding Window

Two Pointers
- Works best on sorted arrays or when advancing pointers monotonically
- Common tasks: pair sum, remove duplicates, partitioning, merging

Example: Remove duplicates in-place from sorted array
```java
public class RemoveDuplicatesSorted {
  public static int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;
    int writeIndex = 1;
    for (int readIndex = 1; readIndex < nums.length; readIndex++) {
      if (nums[readIndex] != nums[readIndex - 1]) {
        nums[writeIndex++] = nums[readIndex];
      }
    }
    return writeIndex;
  }
}
```

Sliding Window
- Fixed-size: add new, remove old; maintain aggregates (sum/max/min)
- Variable-size: expand until valid, shrink to maintain invariant (min window, longest window)

Example: Longest subarray with sum ≤ K (non-negative numbers)
```java
public class LongestSubarrayWithSumAtMostK {
  public static int longestLength(int[] nums, int maxSum) {
    int n = nums.length, left = 0, currentSum = 0, best = 0;
    for (int right = 0; right < n; right++) {
      currentSum += nums[right];
      while (currentSum > maxSum && left <= right) {
        currentSum -= nums[left++];
      }
      best = Math.max(best, right - left + 1);
    }
    return best;
  }
}
```

Pattern recognition
- If order doesn’t matter but sorted helps → sort + two pointers
- If we need max/min over subarrays → deque monotonic or heap
- If constraints update over a window → sliding window with frequency map

Exercises
- Longest substring with at most K distinct characters (string variant)
- Minimum window to reach sum ≥ S (positive numbers)
- Count subarrays with product < K (transform with logs or sliding window trick)


