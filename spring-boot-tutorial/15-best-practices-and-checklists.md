### Best Practices and Checklists

### API design
- **Versioning**: Prefix routes with `/api/v1`.
- **Validation**: Use `jakarta.validation`; return structured errors.
- **Error handling**: Centralize with `@ControllerAdvice`.
- **Idempotency**: For POST that might retry, support idempotency keys.

### Data and migrations
- **Migrations**: All schema changes via Flyway.
- **Indexes**: Add for query fields; verify with EXPLAIN.
- **Transactions**: Keep short; avoid N+1 with fetch joins.

### Security
- **Secrets**: Use environment variables or vault, never commit.
- **JWT**: Short-living tokens; rotate signing keys.
- **CORS/CSRF**: Configure appropriately for SPA/backend.

### Observability
- **Actuator**: Expose only needed endpoints.
- **Metrics**: Business KPIs as custom counters/gauges.
- **Tracing**: Propagate context to downstream calls.

### Performance
- **Virtual threads**: Great for blocking IO.
- **Caching**: Use `@Cacheable` for expensive reads.
- **Resilience**: Timeouts, retries, circuit breakers.

### Packaging & deploy
- **Layered jars** or **Buildpacks**.
- **Health probes**: readiness/liveness endpoints.
- **Configuration**: Externalize with profiles.

### Code structure
- `api` (controllers, DTOs)
- `domain` (entities, repositories, services)
- `security`, `config`, `infra`

Next: `16-case-study-labs.md`.


