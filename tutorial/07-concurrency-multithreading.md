# Concurrency & Multithreading

## Modern Concurrency with Virtual Threads
```java
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class ModernConcurrency {
    // Virtual Threads (Java 21) - Project Loom
    public void virtualThreadsDemo() {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = IntStream.range(0, 10000)
                .mapToObj(i -> executor.submit(() -> {
                    try {
                        Thread.sleep(1000); // Simulated I/O
                        return "Task " + i + " completed";
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return "Task " + i + " interrupted";
                    }
                }))
                .toList();

            futures.forEach(future -> {
                try {
                    System.out.println(future.get());
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    // CompletableFuture for async programming
    public void completableFutureDemo() {
        CompletableFuture<String> future1 = CompletableFuture
            .supplyAsync(() -> fetchDataFromAPI("user"))
            .thenCompose(user -> CompletableFuture.supplyAsync(() -> fetchDataFromAPI("orders/" + user)))
            .thenApply(orders -> "Processed: " + orders)
            .exceptionally(throwable -> "Error: " + throwable.getMessage());

        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future3 = CompletableFuture.supplyAsync(() -> "World");

        CompletableFuture<String> combined = future2.thenCombine(future3, (a, b) -> a + " " + b);

        CompletableFuture.allOf(future1, combined)
            .thenRun(() -> System.out.println("All operations completed"));
    }

    private String fetchDataFromAPI(String endpoint) {
        try { Thread.sleep(500); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        return "Data from " + endpoint;
    }
}
```

## Synchronization and Locks
```java
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;

public class SynchronizationDemo {
    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
    private final Lock readLock = rwLock.readLock();
    private final Lock writeLock = rwLock.writeLock();

    private final AtomicInteger counter = new AtomicInteger(0);
    private final AtomicReference<String> message = new AtomicReference<>("Initial");

    private volatile String data = "initial data";

    public String readData() {
        readLock.lock();
        try { return data; } finally { readLock.unlock(); }
    }

    public void writeData(String newData) {
        writeLock.lock();
        try { data = newData; } finally { writeLock.unlock(); }
    }

    public void atomicOperations() {
        boolean updated = counter.compareAndSet(0, 1);
        int newValue = counter.updateAndGet(val -> val * 2 + 1);
        message.updateAndGet(String::toUpperCase);
    }

    private final StampedLock stampedLock = new StampedLock();

    public String readWithStampedLock() {
        long stamp = stampedLock.tryOptimisticRead();
        String currentData = data;
        if (!stampedLock.validate(stamp)) {
            stamp = stampedLock.readLock();
            try { currentData = data; } finally { stampedLock.unlockRead(stamp); }
        }
        return currentData;
    }
}
```


