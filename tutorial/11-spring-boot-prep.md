# Spring Boot 3.5+ Preparation

## Modern Spring Boot Features
```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

@SpringBootApplication
public class ModernSpringBootApplication {
    public static void main(String[] args) {
        SpringApplication.run(ModernSpringBootApplication.class, args);
    }
}

@RestController
@RequestMapping("/api/v1/users")
@Validated
class UserController {
    private final UserService userService;
    public UserController(UserService userService) { this.userService = userService; }

    @GetMapping
    public ResponseEntity<List<UserDto>> getAllUsers() { return ResponseEntity.ok(userService.findAllUsers()); }

    @GetMapping("/{id}")
    public ResponseEntity<UserDto> getUserById(@PathVariable @Min(1) Long id) {
        return userService.findUserById(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<UserDto> createUser(@Valid @RequestBody CreateUserRequest request) {
        UserDto user = userService.createUser(request);
        return ResponseEntity.status(201).body(user);
    }

    @PutMapping("/{id}")
    public ResponseEntity<UserDto> updateUser(@PathVariable @Min(1) Long id, @Valid @RequestBody UpdateUserRequest request) {
        return userService.updateUser(id, request).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable @Min(1) Long id) {
        boolean deleted = userService.deleteUser(id);
        return deleted ? ResponseEntity.noContent().build() : ResponseEntity.notFound().build();
    }
}

@Service
@Transactional(readOnly = true)
class UserService {
    private final UserRepository userRepository;
    private final NotificationService notificationService;
    public UserService(UserRepository userRepository, NotificationService notificationService) { this.userRepository = userRepository; this.notificationService = notificationService; }

    public Optional<UserDto> findUserById(Long id) { return userRepository.findById(id).map(this::convertToDto); }
    public List<UserDto> findAllUsers() { return userRepository.findAll().stream().map(this::convertToDto).toList(); }

    @Transactional
    public UserDto createUser(CreateUserRequest request) {
        User user = new User();
        user.setName(request.name());
        user.setEmail(request.email());
        user.setCreatedAt(LocalDateTime.now());
        User savedUser = userRepository.save(user);
        notificationService.sendWelcomeEmailAsync(savedUser.getEmail());
        return convertToDto(savedUser);
    }

    @Transactional
    public Optional<UserDto> updateUser(Long id, UpdateUserRequest request) {
        return userRepository.findById(id).map(user -> {
            user.setName(request.name());
            user.setEmail(request.email());
            user.setUpdatedAt(LocalDateTime.now());
            return convertToDto(userRepository.save(user));
        });
    }

    @Transactional
    public boolean deleteUser(Long id) {
        if (userRepository.existsById(id)) { userRepository.deleteById(id); return true; }
        return false;
    }

    private UserDto convertToDto(User user) { return new UserDto(user.getId(), user.getName(), user.getEmail(), user.getCreatedAt()); }
}

record CreateUserRequest(@NotBlank @Size(min = 2, max = 100) String name, @NotBlank @Email String email) {}
record UpdateUserRequest(@NotBlank @Size(min = 2, max = 100) String name, @NotBlank @Email String email) {}
record UserDto(Long id, String name, String email, LocalDateTime createdAt) {}

@Service
class NotificationService {
    public CompletableFuture<Void> sendWelcomeEmailAsync(String email) {
        return CompletableFuture.runAsync(() -> {
            try { Thread.sleep(1000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            System.out.println("Welcome email sent to: " + email);
        });
    }
}
```

## Error Handling and Security Basics
```java
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import jakarta.validation.ConstraintViolationException;
import java.time.Instant;
import java.util.List;

@ControllerAdvice
class GlobalExceptionHandler {
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidationExceptions(MethodArgumentNotValidException ex) {
        List<String> errors = ex.getBindingResult().getFieldErrors().stream().map(error -> error.getField() + ": " + error.getDefaultMessage()).toList();
        return ResponseEntity.badRequest().body(new ErrorResponse("Validation failed", errors, Instant.now()));
    }

    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ErrorResponse> handleConstraintViolation(ConstraintViolationException ex) {
        List<String> errors = ex.getConstraintViolations().stream().map(v -> v.getPropertyPath() + ": " + v.getMessage()).toList();
        return ResponseEntity.badRequest().body(new ErrorResponse("Constraint violation", errors, Instant.now()));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(Exception ex) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(new ErrorResponse("Internal server error", List.of(ex.getMessage()), Instant.now()));
    }

    public record ErrorResponse(String message, List<String> details, Instant timestamp) {}
}
```

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
class SecurityConfig {
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .authorizeHttpRequests(authz -> authz
                .requestMatchers("/actuator/health").permitAll()
                .requestMatchers("/api/v1/public/**").permitAll()
                .anyRequest().authenticated()
            )
            .httpBasic(Customizer.withDefaults());
        return http.build();
    }
    @Bean public PasswordEncoder passwordEncoder() { return new BCryptPasswordEncoder(); }
}
```

## Testing and Production Readiness
```yaml
# application-prod.yml
server:
  port: 8080
  servlet:
    context-path: /api
  compression:
    enabled: true
  http2:
    enabled: true

spring:
  datasource:
    url: ${DATABASE_URL}
    username: ${DATABASE_USERNAME}
    password: ${DATABASE_PASSWORD}
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000

  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        jdbc:
          batch_size: 20
        order_inserts: true
        order_updates: true

  cache:
    type: caffeine
```


