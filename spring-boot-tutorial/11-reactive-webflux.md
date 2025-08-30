### Reactive Programming with Spring WebFlux

Use WebFlux when you need high concurrency with non-blocking IO or streaming.

### Functional routing style
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;
import static org.springframework.web.reactive.function.server.RequestPredicates.GET;

@Configuration
class Routes {
  @Bean RouterFunction<ServerResponse> httpRoutes(HelloHandler handler) {
    return route(GET("/reactive/hello"), handler::hello);
  }
}

@org.springframework.stereotype.Component
class HelloHandler {
  public reactor.core.publisher.Mono<ServerResponse> hello(org.springframework.web.reactive.function.server.ServerRequest req) {
    return ServerResponse.ok().bodyValue("Hello reactive!");
  }
}
```

### WebClient
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
class WebClientConfig {
  @Bean WebClient webClient() { return WebClient.builder().baseUrl("https://api.example.com").build(); }
}
```

For blocking DB access with JPA, prefer MVC + virtual threads; for fully reactive stack, use R2DBC.

Next: `12-performance-virtual-threads.md`.


