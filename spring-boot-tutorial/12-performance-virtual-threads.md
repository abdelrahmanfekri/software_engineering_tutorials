### Performance — Java 21 Virtual Threads and Structured Concurrency

Virtual threads make blocking code scale better by dramatically reducing thread cost.

### Enable virtual threads in Spring Boot
In Boot 3.2+ you can opt-in for virtual threads for common executors:
```yaml
spring:
  threads:
    virtual:
      enabled: true
```

### Configure Tomcat to use virtual threads for request handling
```java
import org.apache.catalina.Executor;
import org.apache.catalina.core.StandardThreadExecutor;
import org.apache.catalina.core.StandardWrapper;
import org.apache.catalina.startup.Tomcat;
import org.springframework.boot.web.embedded.tomcat.TomcatProtocolHandlerCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Configuration
class VirtualThreadConfig {
  @Bean
  TomcatProtocolHandlerCustomizer<?> protocolHandlerVirtualThreads() {
    ExecutorService vts = Executors.newVirtualThreadPerTaskExecutor();
    return protocolHandler -> protocolHandler.setExecutor(vts);
  }
}
```

### Structured concurrency (Java 21)
```java
import java.time.Duration;
import java.util.concurrent.StructuredTaskScope;

public class PriceAggregator {
  record Prices(int base, int shipping) {}
  public Prices fetch() throws Exception {
    try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
      var base  = scope.fork(() -> slowCall(100));
      var ship  = scope.fork(() -> slowCall(20));
      scope.joinUntil(java.time.Instant.now().plus(Duration.ofSeconds(2)));
      scope.throwIfFailed();
      return new Prices(base.get(), ship.get());
    }
  }
  private int slowCall(int v) throws Exception { Thread.sleep(200); return v; }
}
```

Next: `13-native-image-graalvm.md`.


