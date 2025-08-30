### Create a Spring Boot 3.5+ Project (Java 21)

### Option A — Spring Initializr (web UI)
- Open `https://start.spring.io` and choose:
  - Project: Gradle (Kotlin DSL)
  - Language: Java
  - Spring Boot: 3.5.x
  - Packaging: Jar
  - Java: 21
  - Dependencies: Web, Validation, JPA, PostgreSQL, Flyway, Security, Actuator, Testcontainers, Kafka, Lombok (optional)

Download and extract the project.

### Option B — Spring CLI
```bash
spring init \
  --boot-version=3.5.0 \
  --build=gradle-kotlin \
  --java-version=21 \
  --dependencies=web,validation,data-jpa,postgresql,flyway,security,actuator,testcontainers,kafka \
  bookstore
```

### Gradle Kotlin DSL — minimal `build.gradle.kts`
```kotlin
plugins {
    id("org.springframework.boot") version "3.5.0"
    id("io.spring.dependency-management") version "1.1.6"
    kotlin("jvm") version "2.0.20" apply false // if using Kotlin
    id("org.graalvm.buildtools.native") version "0.10.2"
}

group = "com.example"
version = "0.0.1-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_21

repositories { mavenCentral() }

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-validation")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")
    implementation("org.springframework.boot:spring-boot-starter-security")
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    implementation("org.springframework.kafka:spring-kafka")
    implementation("com.fasterxml.jackson.core:jackson-databind")
    implementation("org.flywaydb:flyway-core")

    runtimeOnly("org.postgresql:postgresql")

    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testImplementation("org.testcontainers:junit-jupiter")
    testImplementation("org.testcontainers:postgresql")
}

tasks.test { useJUnitPlatform() }
```

### Main application class
```java
package com.example.bookstore;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BookstoreApplication {
  public static void main(String[] args) {
    SpringApplication.run(BookstoreApplication.class, args);
  }
}
```

### Base configuration `src/main/resources/application.yml`
```yaml
spring:
  application:
    name: bookstore
  datasource:
    url: jdbc:postgresql://localhost:5432/bookstore
    username: postgres
    password: postgres
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        format_sql: true
  flyway:
    enabled: true

server:
  port: 8080

management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,env,beans
```

### Run
```bash
./gradlew bootRun
# or
./gradlew build && java -jar build/libs/bookstore-0.0.1-SNAPSHOT.jar
```

Open `http://localhost:8080/actuator/health`.

Next: `03-core-concepts.md`.


