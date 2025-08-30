### REST APIs, DTOs, Validation, Error Handling, RestClient

### Simple controller and DTOs
```java
package com.example.bookstore.api;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

record CreateBookRequest(@NotBlank String title, @NotBlank String author, @Positive int priceCents) {}
record BookResponse(Long id, String title, String author, int priceCents) {}

@RestController
@RequestMapping("/api/books")
public class BookController {
  private final BookService service;
  public BookController(BookService service) { this.service = service; }

  @PostMapping
  public ResponseEntity<BookResponse> create(@Valid @RequestBody CreateBookRequest req) {
    return new ResponseEntity<>(service.create(req), HttpStatus.CREATED);
  }

  @GetMapping
  public List<BookResponse> list() { return service.list(); }
}
```

### Validation
- Use `jakarta.validation` annotations on DTOs.
- Add `@Valid` to controller method parameters.

### Global error handling
```java
package com.example.bookstore.api;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

import java.util.Map;

@ControllerAdvice
public class ApiExceptionHandler {
  @ExceptionHandler(MethodArgumentNotValidException.class)
  public ResponseEntity<?> handleValidation(MethodArgumentNotValidException ex) {
    var errors = ex.getBindingResult().getFieldErrors().stream()
        .collect(java.util.stream.Collectors.toMap(
            fe -> fe.getField(),
            fe -> fe.getDefaultMessage(),
            (a, b) -> a
        ));
    return ResponseEntity.badRequest().body(Map.of("error", "validation_failed", "fields", errors));
  }

  @ExceptionHandler(IllegalArgumentException.class)
  public ResponseEntity<?> handleBadRequest(IllegalArgumentException ex) {
    return ResponseEntity.badRequest().body(Map.of("error", ex.getMessage()));
  }
}
```

### Calling other services — RestClient (modern alternative to RestTemplate)
```java
import org.springframework.boot.web.client.RestClientBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
class RestClientConfig {
  @Bean RestClient restClient(RestClientBuilder builder) {
    return builder.baseUrl("https://api.example.com").build();
  }
}
```

Next: `06-data-jpa-and-flyway.md`.


