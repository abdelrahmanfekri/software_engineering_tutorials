### Capstone: Bookstore API — Case Study and Labs

You will build a Bookstore API incrementally using what you learned.

### Requirements
- CRUD for books and authors
- Search by author and title
- JWT-protected write operations; public reads
- Postgres database with Flyway migrations
- Integration tests with Testcontainers
- Actuator, Prometheus metrics, request tracing
- Docker image build and run

### Steps
1. Scaffold project (see `02-create-project.md`).
2. Create entities `Book` and `Author` with a relation; add Flyway migration.
3. Implement repositories and services; map to DTOs.
4. Create controllers with validation and global error handler.
5. Add Spring Security + JWT: `/api/auth/**` endpoints; protect POST/PUT/DELETE.
6. Write tests: `@DataJpaTest`, `@WebMvcTest`, and `@SpringBootTest` with Testcontainers.
7. Add Actuator; export Prometheus metrics.
8. Add a Kafka event `books.created` on book creation (optional advanced).
9. Enable virtual threads; measure throughput with `wrk` or `hey`.
10. Build container image and run with Postgres via `docker-compose`.

### Lab challenges
- Add pagination and sorting for list endpoints.
- Implement optimistic locking on `Book`.
- Add rate limiting middleware (e.g., bucket4j).
- Introduce `@Retryable` for transient failures.
- Add API documentation with Spring REST Docs or OpenAPI.

### Acceptance checks
- All endpoints behave and validate correctly.
- Tests pass locally and in CI.
- Application metrics and traces visible in your observability stack.

Congrats — you now have a realistic Spring Boot 3.5 + Java 21 service!


