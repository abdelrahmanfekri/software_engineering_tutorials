# Foreign Function & Memory (FFM) and Vector APIs

Both are preview/incubator in Java 21.

Compile/run examples with:
```bash
# FFM (preview)
javac --enable-preview --release 21 FfmDemo.java
java --enable-preview FfmDemo

# Vector API (incubator)
javac --add-modules jdk.incubator.vector VectorDemo.java
java --add-modules jdk.incubator.vector VectorDemo
```

## Foreign Function & Memory API
```java
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class FfmDemo {
    public static void main(String[] args) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(ValueLayout.JAVA_INT, 1);
            seg.set(ValueLayout.JAVA_INT, 0, 42);
            int x = seg.get(ValueLayout.JAVA_INT, 0);
            System.out.println("Value=" + x);
        }
    }
}
```

## Vector API (SIMD)
```java
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class VectorDemo {
    static final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
    public static void main(String[] args) {
        float[] a = new float[SPEC.length()];
        float[] b = new float[SPEC.length()];
        for (int i = 0; i < SPEC.length(); i++) { a[i] = i; b[i] = 2*i; }
        var va = FloatVector.fromArray(SPEC, a, 0);
        var vb = FloatVector.fromArray(SPEC, b, 0);
        var vc = va.add(vb);
        float[] c = new float[SPEC.length()];
        vc.intoArray(c, 0);
        System.out.println(java.util.Arrays.toString(c));
    }
}
```


