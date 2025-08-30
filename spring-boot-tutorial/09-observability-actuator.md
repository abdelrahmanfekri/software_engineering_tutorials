### Observability — Actuator, Metrics, Tracing

### Actuator
Expose health, metrics, env for diagnostics.
```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,env,beans,threaddump,httpexchanges
  endpoint:
    health:
      probes:
        enabled: true
```

### Metrics with Micrometer
- Default JVM and HTTP metrics are auto-configured.
- Add Prometheus registry:
```kotlin
dependencies { implementation("io.micrometer:micrometer-registry-prometheus") }
```

Expose scrape endpoint:
```yaml
management:
  endpoints:
    web:
      exposure:
        include: prometheus
```

### Tracing with OpenTelemetry
Add dependencies:
```kotlin
dependencies {
  implementation("io.micrometer:micrometer-tracing-bridge-otel")
  implementation("io.opentelemetry:opentelemetry-exporter-otlp")
}
```

Configure OTLP endpoint (e.g., to Grafana Tempo):
```yaml
management:
  otlp:
    tracing:
      endpoint: http://localhost:4318/v1/traces
```

Next: `10-messaging-and-events.md`.


