### Testing with JUnit 5 and Testcontainers

### Unit tests
```java
@org.junit.jupiter.api.Test
void priceWithTax() {
  var svc = new PriceService(new TaxService());
  org.assertj.core.api.Assertions.assertThat(svc.withTax(1000)).isGreaterThan(1000);
}
```

### Slice tests
- `@DataJpaTest` for repositories (uses in-memory DB unless overridden)
- `@WebMvcTest` for controllers (mock MVC)

### Integration test with Postgres Testcontainer
```java
package com.example.bookstore;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@SpringBootTest
@Testcontainers
class BookstoreApplicationIT {
  @Container
  static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:16-alpine")
      .withDatabaseName("bookstore")
      .withUsername("postgres")
      .withPassword("postgres");

  static {
    System.setProperty("spring.datasource.url", postgres.getJdbcUrl());
    System.setProperty("spring.datasource.username", postgres.getUsername());
    System.setProperty("spring.datasource.password", postgres.getPassword());
  }

  @Test void contextLoads() {}
}
```

Alternatively use `jdbc:tc:postgresql:16:///bookstore` in `application-test.yml`.

Next: `09-observability-actuator.md`.


