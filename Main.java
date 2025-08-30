import java.util.concurrent.*;
import java.util.stream.IntStream;

public class Main{
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

    public static void main(String[] args) {
        Main main = new Main();
        main.virtualThreadsDemo();
        main.completableFutureDemo();
    }
}