# Exception Handling

## Modern Exception Handling
```java
import java.io.*;
import java.nio.file.*;
import java.sql.SQLException;
import java.util.logging.Logger;

public class ExceptionHandling {
    private static final Logger logger = Logger.getLogger(ExceptionHandling.class.getName());

    // Try-with-resources (Java 7+, enhanced in Java 9+)
    public void readFileModern(String filename) {
        try (var reader = Files.newBufferedReader(Paths.get(filename));
             var writer = Files.newBufferedWriter(Paths.get("output.txt"))) {

            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line.toUpperCase());
                writer.newLine();
            }
        } catch (IOException e) {
            logger.severe("File operation failed: " + e.getMessage());
            throw new ProcessingException("Failed to process file: " + filename, e);
        }
    }

    // Multi-catch (Java 7+)
    public void handleMultipleExceptions() {
        try {
            // Some risky operations
            riskyOperation();
        } catch (IOException | SQLException | InterruptedException e) {
            logger.warning("Operation failed: " + e.getClass().getSimpleName());
        } catch (Exception e) {
            logger.severe("Unexpected error: " + e.getMessage());
            throw new RuntimeException("Unexpected error occurred", e);
        }
    }

    // Custom exceptions with modern features
    public static class ProcessingException extends RuntimeException {
        private final String context;

        public ProcessingException(String message, Throwable cause) {
            super(message, cause);
            this.context = "Processing";
        }

        public ProcessingException(String message, String context, Throwable cause) {
            super(message, cause);
            this.context = context;
        }

        public String getContext() { return context; }

        // Suppressed exceptions (Java 7+)
        public void addContext(Exception contextException) { addSuppressed(contextException); }
    }

    private void riskyOperation() throws IOException, SQLException, InterruptedException {
        // Placeholder for risky operations
    }
}
```


