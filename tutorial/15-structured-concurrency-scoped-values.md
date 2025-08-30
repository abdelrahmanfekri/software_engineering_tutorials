# Structured Concurrency & Scoped Values (Preview)

These are preview features in Java 21. Compile/run with:
```bash
javac --enable-preview --release 21 --add-modules jdk.incubator.concurrent StructuredDemo.java
java --enable-preview --add-modules jdk.incubator.concurrent StructuredDemo
```

## Structured Concurrency (JEP 453)
```java
import jdk.incubator.concurrent.StructuredTaskScope;
import java.util.concurrent.ExecutionException;

public class StructuredDemo {
    public static void main(String[] args) throws Exception {
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            var user = scope.fork(() -> fetchUser());
            var orders = scope.fork(() -> fetchOrders());
            scope.join();          // wait all
            scope.throwIfFailed(); // propagate first failure
            System.out.println(user.get() + " " + orders.get());
        }
    }
    static String fetchUser() throws InterruptedException { Thread.sleep(200); return "User"; }
    static String fetchOrders() throws InterruptedException { Thread.sleep(300); return "Orders"; }
}
```

## Scoped Values (JEP 446)
```java
public class ScopedValuesDemo {
    // Preview API in java.lang
    static final java.lang.ScopedValue<String> REQUEST_ID = java.lang.ScopedValue.newInstance();

    public static void main(String[] args) {
        java.lang.ScopedValue.where(REQUEST_ID, "req-1234").run(() -> {
            handleRequest();
        });
    }

    static void handleRequest() {
        String id = java.lang.ScopedValue.getWhere(REQUEST_ID);
        System.out.println("Handling with id=" + id);
    }
}
```


