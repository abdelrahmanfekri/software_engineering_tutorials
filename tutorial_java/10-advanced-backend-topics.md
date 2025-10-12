# Advanced Topics for Backend Development

## HTTP Client (Java 11+)
```java
import java.net.http.*;
import java.net.URI;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

public class HttpClientDemo {
    private final HttpClient client = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_2)
        .connectTimeout(Duration.ofSeconds(10))
        .build();

    // Synchronous HTTP request
    public String fetchDataSync(String url) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .timeout(Duration.ofSeconds(30))
            .GET()
            .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() == 200) {
            return response.body();
        } else {
            throw new RuntimeException("HTTP error: " + response.statusCode());
        }
    }

    // Asynchronous HTTP request
    public CompletableFuture<String> fetchDataAsync(String url) {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .header("Content-Type", "application/json")
            .GET()
            .build();

        return client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body);
    }

    // POST with JSON
    public CompletableFuture<String> postJsonAsync(String url, String jsonData) {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonData))
            .build();

        return client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
            .thenApply(response -> {
                if (response.statusCode() >= 200 && response.statusCode() < 300) {
                    return response.body();
                } else {
                    throw new RuntimeException("HTTP error: " + response.statusCode());
                }
            });
    }
}
```

## JSON Processing (Jackson)
```java
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import java.time.LocalDateTime;
import java.util.List;

public class JsonProcessing {
    private final ObjectMapper objectMapper = new ObjectMapper()
        .registerModule(new JavaTimeModule()); // For Java Time API

    public String toJson(Object object) {
        try { return objectMapper.writeValueAsString(object); }
        catch (Exception e) { throw new RuntimeException("Failed to serialize to JSON", e); }
    }

    public <T> T fromJson(String json, Class<T> clazz) {
        try { return objectMapper.readValue(json, clazz); }
        catch (Exception e) { throw new RuntimeException("Failed to deserialize JSON", e); }
    }

    public <T> List<T> fromJsonList(String json, Class<T> elementType) {
        try {
            var listType = objectMapper.getTypeFactory().constructCollectionType(List.class, elementType);
            return objectMapper.readValue(json, listType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to deserialize JSON list", e);
        }
    }

    public record UserDto(Long id, String name, String email, LocalDateTime createdAt) {}
    public record CreateUserRequest(String name, String email) {}
    public record ApiResponse<T>(boolean success, T data, String message) {}
}
```

## Database Operations with JDBC
```java
import javax.sql.DataSource;
import java.sql.*;
import java.time.LocalDateTime;
import java.util.*;

public class DatabaseOperations {
    private final DataSource dataSource;
    public DatabaseOperations(DataSource dataSource) { this.dataSource = dataSource; }

    public Optional<User> findUserById(Long id) {
        String sql = "SELECT id, name, email, created_at FROM users WHERE id = ?";
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql)) {
            stmt.setLong(1, id);
            try (ResultSet rs = stmt.executeQuery()) {
                if (rs.next()) {
                    return Optional.of(new User(
                        rs.getLong("id"),
                        rs.getString("name"),
                        rs.getString("email"),
                        rs.getTimestamp("created_at").toLocalDateTime()
                    ));
                }
            }
        } catch (SQLException e) {
            throw new DataAccessException("Failed to find user by id: " + id, e);
        }
        return Optional.empty();
    }

    public List<User> findAllUsers() {
        String sql = "SELECT id, name, email, created_at FROM users ORDER BY created_at DESC";
        List<User> users = new ArrayList<>();
        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                users.add(new User(
                    rs.getLong("id"),
                    rs.getString("name"),
                    rs.getString("email"),
                    rs.getTimestamp("created_at").toLocalDateTime()
                ));
            }
        } catch (SQLException e) {
            throw new DataAccessException("Failed to fetch all users", e);
        }
        return users;
    }

    public User createUser(String name, String email) {
        String sql = "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?) RETURNING id";
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql)) {
            stmt.setString(1, name);
            stmt.setString(2, email);
            stmt.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));
            try (ResultSet rs = stmt.executeQuery()) {
                if (rs.next()) {
                    Long id = rs.getLong("id");
                    return new User(id, name, email, LocalDateTime.now());
                } else {
                    throw new DataAccessException("Failed to create user - no ID returned");
                }
            }
        } catch (SQLException e) {
            throw new DataAccessException("Failed to create user", e);
        }
    }

    public void createUsersBatch(List<CreateUserRequest> requests) {
        String sql = "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)";
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql)) {
            conn.setAutoCommit(false);
            for (CreateUserRequest request : requests) {
                stmt.setString(1, request.name());
                stmt.setString(2, request.email());
                stmt.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));
                stmt.addBatch();
            }
            int[] results = stmt.executeBatch();
            conn.commit();
            System.out.println("Created " + results.length + " users");
        } catch (SQLException e) {
            throw new DataAccessException("Failed to create users in batch", e);
        }
    }

    public record User(Long id, String name, String email, LocalDateTime createdAt) {}
    public record CreateUserRequest(String name, String email) {}

    public static class DataAccessException extends RuntimeException {
        public DataAccessException(String message) { super(message); }
        public DataAccessException(String message, Throwable cause) { super(message, cause); }
    }
}
```

## Caching Strategies
```java
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;

public class CachingStrategies {
    public static class SimpleCache<K, V> {
        private final ConcurrentHashMap<K, CacheEntry<V>> cache = new ConcurrentHashMap<>();
        private final Duration ttl;
        public SimpleCache(Duration ttl) { this.ttl = ttl; }

        public Optional<V> get(K key) {
            CacheEntry<V> entry = cache.get(key);
            if (entry != null && !entry.isExpired()) {
                return Optional.of(entry.value);
            } else if (entry != null) {
                cache.remove(key);
            }
            return Optional.empty();
        }

        public void put(K key, V value) { cache.put(key, new CacheEntry<>(value, LocalDateTime.now().plus(ttl))); }
        public void invalidate(K key) { cache.remove(key); }
        public void clear() { cache.clear(); }

        private record CacheEntry<V>(V value, LocalDateTime expiry) { boolean isExpired() { return LocalDateTime.now().isAfter(expiry); } }
    }
}
```

## Validation and Data Processing
```java
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class ValidationAndProcessing {
    private static final Pattern EMAIL_PATTERN = Pattern.compile("^[A-Za-z0-9+_.-]+@([A-Za-z0-9.-]+\\.[A-Za-z]{2,})$");

    public static class Validator {
        public static ValidationResult validateUser(CreateUserRequest request) {
            var errors = new ArrayList<String>();
            if (request.name() == null || request.name().trim().isEmpty()) {
                errors.add("Name is required");
            } else if (request.name().length() < 2) {
                errors.add("Name must be at least 2 characters long");
            } else if (request.name().length() > 100) {
                errors.add("Name must not exceed 100 characters");
            }
            if (request.email() == null || request.email().trim().isEmpty()) {
                errors.add("Email is required");
            } else if (!EMAIL_PATTERN.matcher(request.email()).matches()) {
                errors.add("Invalid email format");
            }
            return new ValidationResult(errors.isEmpty(), errors);
        }

        public static <T> ValidationBuilder<T> validate(T object) { return new ValidationBuilder<>(object); }
    }

    public static class ValidationBuilder<T> {
        private final T object;
        private final List<String> errors = new ArrayList<>();
        public ValidationBuilder(T object) { this.object = object; }
        public ValidationBuilder<T> check(Predicate<T> condition, String errorMessage) {
            if (!condition.test(object)) { errors.add(errorMessage); }
            return this;
        }
        public ValidationResult build() { return new ValidationResult(errors.isEmpty(), errors); }
    }

    public record ValidationResult(boolean isValid, List<String> errors) {
        public void throwIfInvalid() {
            if (!isValid) { throw new ValidationException("Validation failed: " + String.join(", ", errors)); }
        }
    }

    public static class ValidationException extends RuntimeException {
        private final List<String> errors;
        public ValidationException(String message) { super(message); this.errors = List.of(message); }
        public ValidationException(List<String> errors) { super("Validation failed: " + String.join(", ", errors)); this.errors = errors; }
        public List<String> getErrors() { return errors; }
    }

    public static class DataProcessor {
        public static <T, R> List<R> processAndTransform(List<T> input, Predicate<T> filter, Function<T, R> transformer, Predicate<R> validator) {
            return input.stream().filter(filter).map(transformer).filter(validator).collect(Collectors.toList());
        }

        public static <T, R> BatchProcessResult<R> processBatch(List<T> items, Function<T, R> processor) {
            List<R> successful = new ArrayList<>();
            List<ProcessingError> errors = new ArrayList<>();
            for (int i = 0; i < items.size(); i++) {
                try { successful.add(processor.apply(items.get(i))); }
                catch (Exception e) { errors.add(new ProcessingError(i, items.get(i), e)); }
            }
            return new BatchProcessResult<>(successful, errors);
        }
    }

    public record BatchProcessResult<T>(List<T> successful, List<ProcessingError> errors) {}
    public record ProcessingError(int index, Object item, Exception error) {}

    public record CreateUserRequest(String name, String email) {}
}
```

## Logging and Monitoring
```java
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.CompletableFuture;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.sql.DataSource;
import java.sql.*;
import java.net.http.*;
import java.net.URI;

public class LoggingAndMonitoring {
    private static final Logger logger = Logger.getLogger(LoggingAndMonitoring.class.getName());

    public static class StructuredLogger {
        private final Logger logger;
        public StructuredLogger(Class<?> clazz) { this.logger = Logger.getLogger(clazz.getName()); }
        public void logUserAction(String userId, String action, String resource) {
            logger.info(String.format("USER_ACTION user_id=%s action=%s resource=%s timestamp=%s", userId, action, resource, Instant.now()));
        }
        public void logPerformance(String operation, Duration duration) {
            logger.info(String.format("PERFORMANCE operation=%s duration_ms=%d timestamp=%s", operation, duration.toMillis(), Instant.now()));
        }
        public void logError(String operation, Exception error, String context) {
            logger.log(Level.SEVERE, String.format("ERROR operation=%s error_type=%s message=%s context=%s", operation, error.getClass().getSimpleName(), error.getMessage(), context), error);
        }
    }

    public static class PerformanceMonitor {
        private final StructuredLogger logger;
        public PerformanceMonitor(StructuredLogger logger) { this.logger = logger; }
        public <T> T monitor(String operation, java.util.function.Supplier<T> supplier) {
            Instant start = Instant.now();
            try {
                T result = supplier.get();
                Duration duration = Duration.between(start, Instant.now());
                logger.logPerformance(operation, duration);
                return result;
            } catch (Exception e) {
                Duration duration = Duration.between(start, Instant.now());
                logger.logError(operation, e, "duration_ms=" + duration.toMillis());
                throw e;
            }
        }
        public void monitorAsync(String operation, Runnable runnable) {
            CompletableFuture.runAsync(() -> monitor(operation, () -> { runnable.run(); return null; }));
        }
    }

    public static class HealthChecker {
        public static class HealthStatus {
            private final String component; private final boolean healthy; private final String message; private final Instant timestamp;
            public HealthStatus(String component, boolean healthy, String message) { this.component = component; this.healthy = healthy; this.message = message; this.timestamp = Instant.now(); }
            public String getComponent() { return component; }
            public boolean isHealthy() { return healthy; }
            public String getMessage() { return message; }
            public Instant getTimestamp() { return timestamp; }
        }

        public CompletableFuture<HealthStatus> checkDatabase(javax.sql.DataSource dataSource) {
            return CompletableFuture.supplyAsync(() -> {
                try (Connection conn = dataSource.getConnection(); Statement stmt = conn.createStatement(); ResultSet rs = stmt.executeQuery("SELECT 1")) {
                    return new HealthStatus("database", true, "Connection successful");
                } catch (SQLException e) {
                    return new HealthStatus("database", false, "Connection failed: " + e.getMessage());
                }
            });
        }

        public CompletableFuture<HealthStatus> checkExternalService(String serviceUrl) {
            return CompletableFuture.supplyAsync(() -> {
                try {
                    HttpRequest request = HttpRequest.newBuilder().uri(URI.create(serviceUrl + "/health")).timeout(Duration.ofSeconds(5)).GET().build();
                    HttpClient client = HttpClient.newHttpClient();
                    HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                    if (response.statusCode() == 200) { return new HealthStatus("external-service", true, "Service reachable"); }
                    else { return new HealthStatus("external-service", false, "Service returned " + response.statusCode()); }
                } catch (Exception e) {
                    return new HealthStatus("external-service", false, "Service unreachable: " + e.getMessage());
                }
            });
        }
    }
}
```


