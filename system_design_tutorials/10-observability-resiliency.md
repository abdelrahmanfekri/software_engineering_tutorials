## 10. Observability and Resiliency

You cannot operate what you cannot observe; resilience turns failures into degradations instead of outages.

### Observability
- Metrics: RED (Rate, Errors, Duration), USE for infra; SLIs for SLOs
- Logs: structured, sampled; PII handling; correlation/tracing IDs
- Traces: OpenTelemetry; spans across services, DB, cache, queue; exemplars
- Dashboards and alerts: SLO-based, low-noise, actionable; runbooks

### Resilience Patterns
- Timeouts per hop; retries with exponential backoff + jitter; budgets and caps
- Circuit breaker, bulkhead isolation, hedged requests for tail latency
- Load shedding and admission control; priority queues; backpressure

### Chaos and DR
- Fault injection, game days; steady-state SLO validation
- Backups, restores; RPO/RTO targets; region evacuation playbooks

### Interview Checklist
- Golden signals, tracing, and actionable alerts
- Retries/timeouts/circuit breakers and backpressure strategy
- DR strategy and validation via chaos/restore drills


### RED/USE Dashboards (examples)
- API: requests/sec, error %, latency histograms (p50/p95/p99)
- DB: queries/sec, slow queries, lock waits, buffer cache hit
- Kafka: produce/fetch latency, consumer lag per partition, rebalances

### Retry and Circuit Breaker Settings (starting points)
- Timeouts: 95th percentile × 1.5 with upper cap; retries: 2 with exp backoff + jitter
- Breaker: open on consecutive failures or high error rate in short window; half-open probes

### Disaster Recovery Tiers
- RPO/RTO targets per system; backups verified via periodic restore drills; cross-region replication for critical data.


