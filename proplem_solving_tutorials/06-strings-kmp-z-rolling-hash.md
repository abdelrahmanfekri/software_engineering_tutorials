## 06 — Strings: KMP, Z-function, Rolling Hash

Core patterns
- Frequency maps (anagrams), prefix functions (KMP), Z-function, rolling hash (Rabin–Karp)
- Sliding window on strings for substrings

KMP prefix-function (pi array)
```java
public class KMP {
  public static int[] prefixFunction(String s) {
    int n = s.length();
    int[] pi = new int[n];
    for (int i = 1; i < n; i++) {
      int j = pi[i - 1];
      while (j > 0 && s.charAt(i) != s.charAt(j)) j = pi[j - 1];
      if (s.charAt(i) == s.charAt(j)) j++;
      pi[i] = j;
    }
    return pi;
  }
  public static int findFirst(String text, String pattern) {
    if (pattern.isEmpty()) return 0;
    String joined = pattern + "#" + text;
    int[] pi = prefixFunction(joined);
    int m = pattern.length();
    for (int i = 0; i < joined.length(); i++) {
      if (pi[i] == m) return i - 2 * m; // start index in text
    }
    return -1;
  }
}
```

Z-function
```java
public class ZFunction {
  public static int[] zArray(String s) {
    int n = s.length();
    int[] z = new int[n];
    for (int i = 1, l = 0, r = 0; i < n; i++) {
      if (i <= r) z[i] = Math.min(r - i + 1, z[i - l]);
      while (i + z[i] < n && s.charAt(z[i]) == s.charAt(i + z[i])) z[i]++;
      if (i + z[i] - 1 > r) { l = i; r = i + z[i] - 1; }
    }
    return z;
  }
}
```

Rolling hash (simple base/mod)
```java
import java.util.*;

public class RollingHash {
  static final long MOD = 1_000_000_007L;
  static final long BASE = 911382323L;

  public static long[] buildPrefix(String s) {
    int n = s.length();
    long[] prefix = new long[n + 1];
    long power = 1;
    long[] pow = new long[n + 1];
    pow[0] = 1;
    for (int i = 0; i < n; i++) {
      prefix[i + 1] = (prefix[i] * BASE + s.charAt(i)) % MOD;
      pow[i + 1] = (pow[i] * BASE) % MOD;
    }
    return prefix; // store pow externally as needed
  }
}
```

Exercises
- Find all occurrences of a pattern in a text (KMP or Z)
- Check if a string is a rotation of another (double string trick)
- Longest palindromic substring (expand around center / Manacher reference)


