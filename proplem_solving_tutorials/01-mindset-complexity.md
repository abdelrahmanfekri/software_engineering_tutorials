## 01 — Mindset, Framework, Complexity

Goals
- Build a repeatable approach that works under time pressure
- Master Big-O and constraints to choose the right strategy

Problem-solving framework
1) Understand: restate the problem, define inputs/outputs, examples
2) Constraints: N limits, time/memory limits, value ranges
3) Brute force: define the naive approach and its complexity
4) Optimize: apply patterns (hashing, sorting, two pointers, DP, greedy, graph)
5) Prove correctness: invariants, exchange arguments, DP definitions
6) Implement: write clean code with checks
7) Verify: test edge cases, complexity, and corner constraints

Complexity cheat sheet
- O(1) < O(log N) < O(√N) < O(N) < O(N log N) < O(N^2) < O(N^3) < O(2^N) < O(N!)
- Typical limits:
  - N ≤ 1e5 → O(N log N) or better
  - N ≤ 1e6 → O(N) with small constant factors
  - N ≤ 2e5 edges → Dijkstra O((N+M) log N)
  - Grid up to 1e3 x 1e3 → BFS/DFS is OK; avoid O(N^2 M^2)

Space complexity
- Know memory per primitive: int (4B), long (8B), double (8B), boolean (1B~)
- Arrays: N elements × type size; adjacency lists: ~2M edges in memory

Mathematical tools to recognize
- Logarithms and binary search on answers
- Pigeonhole principle, invariants/monovariants
- Prefix sums/differences for range updates/queries
- Modular arithmetic: avoid overflow and use long; when needed, use modulo

Edge-case checklist
- Empty or minimal inputs, all equal elements, strictly increasing/decreasing sequences
- Duplicates vs unique, negatives vs positives, zero cases
- Off-by-ones in ranges and indices

Java tips for speed and clarity
- Prefer `ArrayDeque` over `Stack`/`LinkedList` for queues/deques
- Use `StringBuilder` for joins; avoid string concatenation in loops
- For heavy loops, avoid boxing/unboxing; use primitives
- Prefer `HashMap`/`HashSet` for average O(1) ops; `TreeMap`/`TreeSet` for ordered O(log N)

Example: Brute force to optimized

Problem: Given an array of integers and target T, does any pair sum to T?
- Brute force: double loop O(N^2)
- Optimized: use a set in one pass O(N)

```java
import java.util.*;

public class PairSumExists {
  public static boolean existsPairWithSum(int[] values, int targetSum) {
    Set<Integer> seenValues = new HashSet<>();
    for (int currentValue : values) {
      int needed = targetSum - currentValue;
      if (seenValues.contains(needed)) return true;
      seenValues.add(currentValue);
    }
    return false;
  }
}
```

Exercises
- Show time/space complexity for both approaches
- Extend to count unique pairs; handle duplicates carefully
- Return indices instead of boolean; ensure stable/first indices


