# Generics Deep Dive

Key topics:
- Generic classes and methods
- Wildcards `? extends` / `? super` (PECS principle)
- Type erasure, reifiable vs non-reifiable types
- Bounded type parameters and multiple bounds
- Capturing wildcards and helper methods
- Practical patterns and pitfalls

## Generic Classes and Methods
```java
public class Pair<L, R> {
    private final L left;
    private final R right;
    public Pair(L left, R right) { this.left = left; this.right = right; }
    public L left() { return left; }
    public R right() { return right; }
}

class GenericUtils {
    public static <T extends Number> double sum(Iterable<T> numbers) {
        double total = 0.0;
        for (T n : numbers) { total += n.doubleValue(); }
        return total;
    }
}
```

## Wildcards (PECS)
Producer Extends, Consumer Super.
```java
import java.util.*;

class WildcardExamples {
    // copy from producer to consumer safely
    public static <T> void copy(List<? extends T> src, List<? super T> dst) {
        for (T t : src) dst.add(t);
    }

    // max using extends-bounded wildcard
    public static <T extends Comparable<? super T>> T max(List<? extends T> list) {
        return list.stream().max(Comparable::compareTo).orElseThrow();
    }
}
```

## Type Erasure and Reifiability
- Generics are erased at runtime, so `List<String>` and `List<Integer>` share the same runtime type `List`.
- Arrays are reifiable (`String[]` knows it contains `String`), generics are usually not.
- Avoid creating generic arrays: use `List<T>` or `@SuppressWarnings("unchecked")` carefully.

```java
import java.lang.reflect.Array;

class ArrayFactory {
    @SuppressWarnings("unchecked")
    public static <T> T[] newArray(Class<T> componentType, int length) {
        return (T[]) Array.newInstance(componentType, length);
    }
}
```

## Multiple Bounds and Captures
```java
interface Identified { String id(); }

class Bounds {
    // T must be Number and Comparable
    public static <T extends Number & Comparable<T>> T min(T a, T b) {
        return a.compareTo(b) <= 0 ? a : b;
    }

    // Helper to capture wildcard and mutate destination
    public static void addAllNumbers(List<? super Integer> dst, List<Integer> src) {
        dst.addAll(src);
    }
}
```

## Effective Practices
- Prefer wildcards in APIs (`List<? extends Foo>`) and concrete generics in implementations (`List<Foo>`).
- Keep type parameters minimal and meaningful.
- Use `Class<T>` tokens when you need runtime type info.


