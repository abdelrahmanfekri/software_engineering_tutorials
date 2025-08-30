## 10. Observability and Resiliency

You cannot operate what you cannot observe; resilience turns failures into degradations instead of outages.

### Observability
- **Metrics: RED (Rate, Errors, Duration), USE for infra; SLIs for SLOs**
  - **RED**: Request rate, error rate, duration (latency)
  - **USE**: Utilization, saturation, errors (for infrastructure)
  - **SLIs**: Service Level Indicators that measure SLOs
  - **Example**: API requests/sec, error percentage, p95 latency

- **Logs: structured, sampled; PII handling; correlation/tracing IDs**
  - **Structured**: Machine-readable format (JSON) instead of plain text
  - **Sampled**: Log subset to reduce storage and processing costs
  - **PII handling**: Don't log sensitive personal information
  - **Correlation IDs**: Link related log entries across services
  - **Example**: `{"level": "info", "message": "User login", "user_id": "123", "correlation_id": "abc123"}`

- **Traces: OpenTelemetry; spans across services, DB, cache, queue; exemplars**
  - **OpenTelemetry**: Standard for distributed tracing
  - **Spans**: Individual operations within a request
  - **Service boundaries**: Track requests across multiple services
  - **Exemplars**: Sample traces for performance analysis
  - **Example**: Request flow: API → Service → Database → Cache

**Key insight**: Observability is not just monitoring. It's about understanding system behavior and debugging problems quickly.

### Resilience Patterns
- **Timeouts per hop; retries with exponential backoff + jitter; budgets and caps**
  - **Timeouts per hop**: Set timeouts for each service call
  - **Exponential backoff**: Increase delay between retries exponentially
  - **Jitter**: Add randomness to prevent synchronized retries
  - **Budgets and caps**: Limit total retry time and attempts
  - **Example**: 1s timeout, 2 retries with 1s, 2s delays

- **Circuit breaker, bulkhead isolation, hedged requests for tail latency**
  - **Circuit breaker**: Stop calling failing services to prevent cascading failures
  - **Bulkhead isolation**: Separate resources to prevent failure propagation
  - **Hedged requests**: Send multiple requests and use the fastest response
  - **Example**: Circuit breaker opens after 5 consecutive failures

- **Load shedding and admission control; priority queues; backpressure**
  - **Load shedding**: Drop non-critical requests under load
  - **Admission control**: Limit incoming requests based on system capacity
  - **Priority queues**: Process high-priority requests first
  - **Backpressure**: Signal upstream to slow down when overloaded

**Why this matters**: Resilience patterns prevent failures from cascading and ensure graceful degradation. They're essential for production systems.

### Chaos and DR
- **Fault injection, game days; steady-state SLO validation**
  - **Fault injection**: Intentionally cause failures to test resilience
  - **Game days**: Planned chaos engineering exercises
  - **Steady-state SLO validation**: Verify SLOs during chaos testing
  - **Example**: Kill random database nodes and verify system continues working

- **Backups, restores; RPO/RTO targets; region evacuation playbooks**
  - **Backups**: Regular data backups with verification
  - **Restores**: Test backup restoration procedures
  - **RPO**: Recovery Point Objective (maximum data loss)
  - **RTO**: Recovery Time Objective (maximum downtime)
  - **Region evacuation**: Plans for moving to backup regions

**Key insight**: Chaos engineering validates your resilience assumptions. Disaster recovery ensures business continuity.

### Interview Checklist
- **Golden signals, tracing, and actionable alerts**
  - Explain your observability strategy
  - Show you understand what to monitor
- **Retries/timeouts/circuit breakers and backpressure strategy**
  - Demonstrate understanding of resilience patterns
  - Show you have a plan for handling failures
- **DR strategy and validation via chaos/restore drills**
  - Explain your disaster recovery approach
  - Show you understand the importance of testing

### RED/USE Dashboards (examples)
- **API: requests/sec, error %, latency histograms (p50/p95/p99)**
  - **Requests/sec**: Throughput measurement
  - **Error %**: Reliability measurement
  - **Latency histograms**: Performance measurement
  - **Percentiles**: p50 (median), p95, p99 (tail latency)

- **DB: queries/sec, slow queries, lock waits, buffer cache hit**
  - **Queries/sec**: Database throughput
  - **Slow queries**: Performance bottlenecks
  - **Lock waits**: Concurrency issues
  - **Buffer cache hit**: Memory efficiency

- **Kafka: produce/fetch latency, consumer lag per partition, rebalances**
  - **Produce/fetch latency**: Message processing performance
  - **Consumer lag**: Processing backlog
  - **Rebalances**: Consumer group stability

**Use these metrics**: They provide comprehensive visibility into system health and performance.

### Retry and Circuit Breaker Settings (starting points)
- **Timeouts: 95th percentile × 1.5 with upper cap; retries: 2 with exp backoff + jitter**
  - **Timeout calculation**: Base on actual performance with safety margin
  - **Upper cap**: Prevent extremely long timeouts
  - **Retry count**: Balance reliability vs resource usage
  - **Example**: p95 = 200ms, timeout = 300ms, max 500ms

- **Breaker: open on consecutive failures or high error rate in short window; half-open probes**
  - **Consecutive failures**: Open after N failures in a row
  - **Error rate**: Open if error rate exceeds threshold
  - **Half-open**: Test if service has recovered
  - **Example**: Open after 5 consecutive failures, close after 30s

**Start with these**: They're proven patterns that work in most systems.

### Disaster Recovery Tiers
- **RPO/RTO targets per system**: Different systems have different recovery requirements
- **Backups verified via periodic restore drills**: Test backup procedures regularly
- **Cross-region replication for critical data**: Geographic redundancy for high availability

**Key insight**: DR planning should be based on business impact, not technical complexity.

### Alerting Strategy
- **Actionable alerts**: Only alert when human action is required
- **Alert fatigue**: Too many alerts reduce response effectiveness
- **Escalation**: Clear escalation paths for different alert types
- **Runbooks**: Documented procedures for common alerts

**Why this matters**: Good alerting ensures problems are detected and resolved quickly. Poor alerting leads to missed issues and alert fatigue.

### Performance Budgets
- **Latency budgets**: Allocate latency across service calls
- **Throughput budgets**: Plan for expected traffic patterns
- **Resource budgets**: Plan for CPU, memory, and storage usage
- **Cost budgets**: Plan for infrastructure costs

**Key insight**: Performance budgets help make trade-offs explicit and prevent performance regressions.

### SLO-Based Alerting
- **Error budget alerts**: Alert when approaching error budget limits
- **Latency SLO alerts**: Alert when latency exceeds targets
- **Availability alerts**: Alert when availability drops below targets
- **Trend alerts**: Alert on deteriorating trends before SLO violations

**Why this matters**: SLO-based alerting focuses on business impact rather than technical metrics.

### Additional Resources for Deep Study
- **Books**: "Site Reliability Engineering" by Google (comprehensive coverage)
- **Papers**: "The Tail at Scale" by Dean and Barroso (performance and resilience)
- **Practice**: Set up monitoring and chaos engineering in test environments
- **Real-world**: Study how companies like Netflix and Google handle observability and resilience

**Study strategy**: Understand the principles, practice with real tools, then study real-world implementations to understand practical constraints.


