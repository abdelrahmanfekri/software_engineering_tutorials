# JVM Internals, GC & Performance

Topics:
- Class loading, JIT (C2), escape analysis
- Garbage Collectors: G1 (default), ZGC, Shenandoah
- Java Memory Model: `volatile`, happens-before
- Profiling & Diagnostics: JFR, jcmd, jmap, jstack
- VarHandles for low-level concurrency

## Choose a GC
Run-time flags:
```bash
java -XX:+UseG1GC -Xms512m -Xmx512m App
java -XX:+UseZGC -Xms2g -Xmx2g App
java -XX:+UseShenandoahGC App
```

## Java Flight Recorder (JFR) Programmatic Recording
```java
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import java.nio.file.Path;

public class JfrDemo {
    public static void main(String[] args) throws Exception {
        Path file = Path.of("profile.jfr");
        try (Recording r = new Recording()) {
            r.start();
            // workload
            for (int i = 0; i < 1_000_000; i++) Math.sqrt(i);
            r.stop();
            r.dump(file);
        }
        for (RecordedEvent e : RecordingFile.readAllEvents(file)) {
            if (e.getEventType().getName().contains("CPU")).
                System.out.println(e.getStartTime() + ": " + e.getEventType().getLabel());
        }
    }
}
```

## jcmd/jmap/jstack Cheatsheet
```bash
jcmd <pid> VM.flags
jcmd <pid> GC.heap_info
jcmd <pid> JFR.start name=profile settings=profile delay=10s duration=60s filename=app.jfr
jmap -heap <pid>
jstack <pid> | less
```

## VarHandles
```java
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class VarHandleDemo {
    static class Counter { volatile int value; }
    private static final VarHandle VALUE_HANDLE;
    static {
        try { VALUE_HANDLE = MethodHandles.lookup().findVarHandle(Counter.class, "value", int.class); }
        catch (ReflectiveOperationException e) { throw new ExceptionInInitializerError(e); }
    }
    public static void main(String[] args) {
        Counter c = new Counter();
        VALUE_HANDLE.setOpaque(c, 1);
        boolean success = VALUE_HANDLE.compareAndSet(c, 1, 2);
        System.out.println("CAS success: " + success + ", value=" + c.value);
    }
}
```


