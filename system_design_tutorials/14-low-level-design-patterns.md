## 14. Low-Level Design Patterns (with Java snippets)

Translate HLD into robust, testable code. Prefer clarity over cleverness.

### DDD Tactical Patterns
- Entities, Value Objects, Aggregates with invariants
- Repositories for persistence boundaries; domain events for side effects

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

### Patterns to Reach For
- Strategy, Command, Adapter, Decorator, Builder
- Producer/Consumer with bounded queues; Worker pools; Actor-style isolation where fit
- Expand/contract toggles around persistence mappers

### Testing
- Unit tests around aggregates and invariants; contract tests for clients/providers
- Property-based tests for critical transformations

### Interview Checklist
- Show idempotency, retry, and caching in code; transaction boundaries explicit
- Clean interfaces, separation of concerns, and testability


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


