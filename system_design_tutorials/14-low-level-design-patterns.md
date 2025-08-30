## 14. Low-Level Design Patterns (with Java snippets)

Translate HLD into robust, testable code. Prefer clarity over cleverness.

### DDD Tactical Patterns
- **Entities, Value Objects, Aggregates with invariants**
  - **Entities**: Objects with identity that can change over time
  - **Value Objects**: Immutable objects defined by their attributes
  - **Aggregates**: Clusters of related entities with clear boundaries
  - **Invariants**: Business rules that must always be true
  - **Example**: User (entity) with Email (value object), Order (aggregate) with OrderItems

- **Repositories for persistence boundaries; domain events for side effects**
  - **Repositories**: Abstract data access, hide persistence details
  - **Persistence boundaries**: Clear separation between domain and data layers
  - **Domain events**: Events that occur within the domain
  - **Side effects**: Actions triggered by domain events
  - **Example**: OrderRepository for order persistence, OrderCreatedEvent for notifications

**Key insight**: DDD patterns create clear boundaries and make business logic explicit. Use them to organize complex domains.

### Resilience Utilities
```java
public final class Retry {
  public static <T> T withExponentialBackoff(Supplier<T> op, int maxAttempts, Duration base) throws Exception {
    int attempt = 0;
    while (true) {
      try { return op.get(); }
      catch (Exception ex) {
        attempt++;
        if (attempt >= maxAttempts) throw ex;
        long jitter = ThreadLocalRandom.current().nextLong(base.toMillis());
        long sleep = (long) (Math.pow(2, attempt - 1) * base.toMillis()) + jitter;
        Thread.sleep(Math.min(sleep, 15_000));
      }
    }
  }
}
```

**Study this code**: It shows a robust retry pattern with exponential backoff and jitter. Use it for external service calls.

### Idempotency Wrapper
```java
class IdempotencyStore { // e.g., backed by Redis/DB with TTL
  boolean tryBegin(String key) { /* set NX */ return true; }
  void complete(String key) { /* set status=done */ }
}

class PaymentsService {
  private final IdempotencyStore store;
  PaymentsService(IdempotencyStore store) { this.store = store; }
  public PaymentResponse charge(String idempotencyKey, PaymentRequest req) {
    if (!store.tryBegin(idempotencyKey)) return new PaymentResponse("DUPLICATE");
    try { /* perform charge; write outbox */ return new PaymentResponse("OK"); }
    finally { store.complete(idempotencyKey); }
  }
}
```

**Key insight**: Idempotency is critical for reliability. This pattern ensures operations are safe to retry.

### Caching Facade (Cache-Aside)
```java
class CacheFacade<K, V> {
  private final Cache<K, V> cache; private final Repository<K, V> repo;
  V get(K key) {
    V v = cache.getIfPresent(key);
    if (v != null) return v;
    v = repo.find(key);
    if (v != null) cache.put(key, v); // add jittered TTL in real impl
    return v;
  }
}
```

**Study this pattern**: It implements cache-aside with proper fallback to the repository. Add TTL and error handling in production.

### Patterns to Reach For
- **Strategy, Command, Adapter, Decorator, Builder**
  - **Strategy**: Encapsulate algorithms and make them interchangeable
  - **Command**: Encapsulate requests as objects
  - **Adapter**: Make incompatible interfaces work together
  - **Decorator**: Add behavior to objects dynamically
  - **Builder**: Construct complex objects step by step

- **Producer/Consumer with bounded queues; Worker pools; Actor-style isolation where fit**
  - **Producer/Consumer**: Decouple producers from consumers
  - **Bounded queues**: Prevent memory overflow
  - **Worker pools**: Reuse threads for better performance
  - **Actor-style**: Isolate state and behavior

- **Expand/contract toggles around persistence mappers**
  - **Expand/contract**: Add new fields before removing old ones
  - **Toggles**: Feature flags for gradual rollout
  - **Persistence mappers**: Handle schema evolution gracefully

**Choose patterns based on**: Your specific needs, not because they're popular. Simplicity is often better than complexity.

### Testing
- **Unit tests around aggregates and invariants; contract tests for clients/providers**
  - **Aggregate tests**: Test business logic and invariants
  - **Invariant tests**: Ensure business rules are always enforced
  - **Contract tests**: Verify API contracts between services
  - **Example**: Test that Order total equals sum of OrderItem prices

- **Property-based tests for critical transformations**
  - **Property-based tests**: Generate test data automatically
  - **Critical transformations**: Test important business logic
  - **Example**: Test that order calculation is always correct for any valid input

**Key insight**: Good testing catches bugs early and enables confident refactoring. Focus on testing business logic, not implementation details.

### Interview Checklist
- **Show idempotency, retry, and caching in code; transaction boundaries explicit**
  - Demonstrate understanding of resilience patterns
  - Show you can implement them correctly
- **Clean interfaces, separation of concerns, and testability**
  - Explain your design decisions
  - Show you understand good design principles

### Repository Interface Example
```java
public interface OrderRepository {
  Optional<Order> findById(long id);
  void save(Order order);
}

public final class JdbcOrderRepository implements OrderRepository {
  private final DataSource ds;
  public JdbcOrderRepository(DataSource ds) { this.ds = ds; }
  public Optional<Order> findById(long id) { /* SELECT ... */ return Optional.empty(); }
  public void save(Order order) { /* INSERT/UPDATE with tx */ }
}
```

**Study this pattern**: It shows clean separation between interface and implementation. Use dependency injection for flexibility.

### Error Handling Patterns
- **Result types**: Return success/failure instead of throwing exceptions
- **Circuit breakers**: Stop calling failing services
- **Fallbacks**: Provide alternative behavior when primary fails
- **Graceful degradation**: Reduce functionality instead of failing completely

**Why this matters**: Good error handling improves user experience and system reliability.

### Configuration Management
- **Environment-specific config**: Different configs for dev, staging, prod
- **Feature flags**: Control feature rollout independently of deployment
- **Dynamic config**: Update configuration without restarting
- **Validation**: Validate configuration at startup

**Key insight**: Configuration management affects deployment flexibility and operational efficiency.

### Logging and Observability
- **Structured logging**: Use JSON or structured formats
- **Correlation IDs**: Link related log entries
- **Log levels**: Appropriate levels for different types of information
- **Performance logging**: Log timing information for performance analysis

**Why this matters**: Good logging enables debugging and monitoring in production.

### Performance Considerations
- **Lazy loading**: Load data only when needed
- **Connection pooling**: Reuse database connections
- **Batch operations**: Group multiple operations together
- **Async processing**: Process non-critical operations asynchronously

**Key insight**: Performance optimizations should be based on measurements, not assumptions.

### Security Patterns
- **Input validation**: Validate all user inputs
- **Output encoding**: Encode outputs to prevent injection
- **Principle of least privilege**: Give minimum necessary access
- **Secure defaults**: Secure configuration by default

**Why this matters**: Security should be built into the design, not added later.

### Additional Resources for Deep Study
- **Books**: "Effective Java" by Joshua Bloch (Java best practices)
- **Patterns**: Gang of Four design patterns, Martin Fowler's patterns
- **Practice**: Implement these patterns in real projects
- **Real-world**: Study how companies implement these patterns in production

**Study strategy**: Understand the patterns, practice implementing them, then study real-world usage to understand practical constraints.


