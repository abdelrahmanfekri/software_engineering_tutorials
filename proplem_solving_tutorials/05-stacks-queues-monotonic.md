## 05 — Stacks, Queues, Monotonic Structures

Use cases
- Balanced parentheses, expression evaluation → stack
- Next greater/smaller element → monotonic stack
- Sliding window max/min → deque (monotonic queue)

Valid parentheses
```java
import java.util.*;

public class ValidParentheses {
  public static boolean isValid(String s) {
    Deque<Character> stack = new ArrayDeque<>();
    for (char c : s.toCharArray()) {
      if (c == '(' || c == '[' || c == '{') stack.push(c);
      else {
        if (stack.isEmpty()) return false;
        char open = stack.pop();
        if ((open == '(' && c != ')') || (open == '[' && c != ']') || (open == '{' && c != '}')) return false;
      }
    }
    return stack.isEmpty();
  }
}
```

Next greater element
```java
import java.util.*;

public class NextGreaterElement {
  public static int[] nextGreater(int[] nums) {
    int n = nums.length;
    int[] answer = new int[n];
    Arrays.fill(answer, -1);
    Deque<Integer> stack = new ArrayDeque<>(); // indices, decreasing by value
    for (int i = 0; i < n; i++) {
      while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
        answer[stack.pop()] = nums[i];
      }
      stack.push(i);
    }
    return answer;
  }
}
```

Exercises
- Daily Temperatures (monotonic stack)
- Evaluate Reverse Polish Notation (stack)
- Sliding window minimum (deque)


