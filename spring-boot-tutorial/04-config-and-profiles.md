### Configuration and Profiles

### application.yml
```yaml
spring:
  profiles:
    default: dev
app:
  Title: "Bookstore API"
  rateLimitPerMinute: 120
```

### Typed configuration with `@ConfigurationProperties`
```java
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "app")
public class AppProps {
  private String title;
  private int rateLimitPerMinute;
  public String getTitle() { return title; }
  public void setTitle(String title) { this.title = title; }
  public int getRateLimitPerMinute() { return rateLimitPerMinute; }
  public void setRateLimitPerMinute(int v) { this.rateLimitPerMinute = v; }
}
```

### Profile-specific config
`application-dev.yml`
```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/bookstore
    username: postgres
    password: postgres
```

`application-test.yml`
```yaml
spring:
  datasource:
    url: jdbc:tc:postgresql:16:///bookstore
  jpa:
    hibernate:
      ddl-auto: validate
```

Activate profile:
```bash
SPRING_PROFILES_ACTIVE=prod ./gradlew bootRun
```

Next: `05-rest-and-validation.md`.


