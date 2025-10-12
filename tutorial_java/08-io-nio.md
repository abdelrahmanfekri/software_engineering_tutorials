# I/O and NIO

## Modern File Operations
```java
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;

public class ModernIO {
    public void fileOperations() throws IOException {
        Path path = Paths.get("example.txt");

        // Write to file
        Files.write(path, "Hello, World!".getBytes(),
            StandardOpenOption.CREATE, StandardOpenOption.WRITE);

        // Read from file
        String content = Files.readString(path);
        List<String> lines = Files.readAllLines(path);

        // Stream file lines
        try (Stream<String> linesStream = Files.lines(path)) {
            linesStream.filter(line -> !line.trim().isEmpty())
                .map(String::toUpperCase)
                .forEach(System.out::println);
        }

        // Walk file tree
        try (Stream<Path> paths = Files.walk(Paths.get("."))) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".java"))
                 .forEach(System.out::println);
        }

        // File attributes
        BasicFileAttributes attrs = Files.readAttributes(path, BasicFileAttributes.class);
        System.out.println("Created: " + attrs.creationTime());
        System.out.println("Size: " + attrs.size());

        // Watch service for file system events
        watchDirectory(Paths.get("."));
    }

    private void watchDirectory(Path dir) throws IOException {
        WatchService watcher = FileSystems.getDefault().newWatchService();
        dir.register(watcher,
            StandardWatchEventKinds.ENTRY_CREATE,
            StandardWatchEventKinds.ENTRY_DELETE,
            StandardWatchEventKinds.ENTRY_MODIFY);

        // In a real application, run in a separate thread
        try {
            WatchKey key;
            while ((key = watcher.take()) != null) {
                for (WatchEvent<?> event : key.pollEvents()) {
                    System.out.println("Event: " + event.kind() + " for " + event.context());
                }
                key.reset();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // NIO.2 for high-performance I/O
    public void nioOperations() throws IOException {
        try (var channel = FileChannel.open(Paths.get("largefile.dat"), StandardOpenOption.READ)) {
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            while (buffer.hasRemaining()) {
                byte b = buffer.get();
                // Process byte
            }
        }
    }
}
```


