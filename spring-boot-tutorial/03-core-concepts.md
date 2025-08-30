### Core Concepts — Starters, Auto-config, DI, Beans

### Starters and auto-configuration
- Starters aggregate dependencies (e.g., `spring-boot-starter-web`).
- Auto-config inspects the classpath and environment to create sensible defaults.

### Dependency injection (DI)
Prefer constructor injection for immutability and testability.
```java
@org.springframework.stereotype.Service
public class PriceService {
  private final TaxService taxService;

  public PriceService(TaxService taxService) { this.taxService = taxService; }

  public int withTax(int base) { return base + taxService.tax(base); }
}

@org.springframework.stereotype.Component
class TaxService {
  int tax(int base) { return (int)(base * 0.1); }
}
```

### Configuration and beans
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {
  @Bean
  public IdGenerator idGenerator() { return new IdGenerator(); }
}

class IdGenerator { public String next() { return java.util.UUID.randomUUID().toString(); } }
```

### Conditional beans and profiles
```java
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springfr amework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
class FeatureConfig {
  @Bean
  @ConditionalOnProperty(name = "feature.x.enabled", havingValue = "true", matchIfMissing = false)
  XService xService() { return new XService(); }
}
class XService {}
```

### Logging
Spring Boot uses Logback by default. Configure via `application.yml` or `logback-spring.xml`.
```yaml
logging:
  level:
    root: info
    com.example: debug
```

Next: `04-config-and-profiles.md`.


