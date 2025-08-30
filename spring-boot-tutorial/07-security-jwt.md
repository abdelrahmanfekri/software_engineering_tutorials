### Security with Spring Security and JWT

### Dependencies
Already added: `spring-boot-starter-security`. Add a JWT library, e.g. Auth0:
```kotlin
dependencies {
  implementation("com.auth0:java-jwt:4.4.0")
}
```

### Security configuration (stateless, JWT auth)
```java
package com.example.bookstore.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableMethodSecurity
public class SecurityConfig {
  @Bean PasswordEncoder passwordEncoder() { return new BCryptPasswordEncoder(); }

  @Bean
  SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
    http
      .csrf(csrf -> csrf.disable())
      .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
      .authorizeHttpRequests(auth -> auth
        .requestMatchers("/actuator/**", "/api/auth/**").permitAll()
        .anyRequest().authenticated()
      )
      .httpBasic(Customizer.withDefaults());
    // Add JWT filter before UsernamePasswordAuthenticationFilter here
    return http.build();
  }
}
```

### JWT service
```java
package com.example.bookstore.security;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.exceptions.JWTVerificationException;
import com.auth0.jwt.interfaces.DecodedJWT;
import org.springframework.stereotype.Component;

import java.time.Instant;

@Component
public class JwtService {
  private final Algorithm algorithm = Algorithm.HMAC256("change-me");

  public String createToken(String username) {
    return JWT.create()
        .withSubject(username)
        .withIssuedAt(Instant.now())
        .withExpiresAt(Instant.now().plusSeconds(3600))
        .sign(algorithm);
  }

  public DecodedJWT verify(String token) throws JWTVerificationException {
    return JWT.require(algorithm).build().verify(token);
  }
}
```

### Authentication endpoints
Implement endpoints `/api/auth/register`, `/api/auth/login` that issue a JWT on successful login. Store users in a DB table with `BCryptPasswordEncoder` hashed passwords.

### Method security
Annotate protected methods with `@PreAuthorize("hasRole('ADMIN')")` as needed.

Next: `08-testing-and-testcontainers.md`.


