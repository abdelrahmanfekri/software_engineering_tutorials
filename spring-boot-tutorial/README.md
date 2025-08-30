### Spring Boot 3.5+ with Java 21 — Hands-on Tutorial

This is a practical, end-to-end learning path for Spring Boot 3.5+ using Java 21. It is designed for self-paced learning with code-first examples, exercises, and a capstone project.

Use these lessons in order, or jump to topics as needed. Each chapter includes: key ideas, runnable snippets, commands, and quick labs.

### Prerequisites
- Java 21 (Temurin/GraalVM)
- Gradle or Maven
- Docker (for Testcontainers and deployment)

### How to use
- Start with `00-index.md` and follow the chapters in sequence.
- Each chapter can be applied to a single sample app you build as you go.
- Use the capstone in `16-case-study-labs.md` to integrate everything.

### Table of Contents
- 00-index.md — Overview and learning plan
- 01-setup.md — Install Java 21, SDKMAN, Gradle/Maven, Boot CLI
- 02-create-project.md — Create a new Spring Boot 3.5 project (Gradle/Maven)
- 03-core-concepts.md — Starters, Auto-config, DI, Bean lifecycle
- 04-config-and-profiles.md — `application.yml`, `@ConfigurationProperties`, profiles
- 05-rest-and-validation.md — Controllers, DTOs, Validation, Errors, RestClient
- 06-data-jpa-and-flyway.md — Entities, Repos, H2/Postgres, Flyway migrations
- 07-security-jwt.md — Spring Security, JWT, stateless APIs, method security
- 08-testing-and-testcontainers.md — Unit, slice, integration tests, Testcontainers
- 09-observability-actuator.md — Actuator, metrics, tracing with OpenTelemetry
- 10-messaging-and-events.md — Domain events, Kafka basics, outbox pattern
- 11-reactive-webflux.md — Reactive stack, WebClient, functional endpoints
- 12-performance-virtual-threads.md — Virtual threads, tuning, structured concurrency
- 13-native-image-graalvm.md — AOT, native images, footprint, tradeoffs
- 14-docker-and-deploy.md — Boot jars, layered jars, `bootBuildImage`, Kubernetes
- 15-best-practices-and-checklists.md — Packaging, logging, errors, security
- 16-case-study-labs.md — Bookstore API capstone and lab exercises

### What you will build
- A production-style REST API featuring: validation, JPA, Flyway, JWT security, tests with Testcontainers, metrics/tracing, and containerized deploy.

### License
Use freely for learning and internal training.


