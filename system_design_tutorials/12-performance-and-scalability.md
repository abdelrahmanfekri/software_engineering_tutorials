## 12. Performance and Scalability

Optimize for user-perceived latency and sustainable throughput.

### Performance Diagnostics
- Establish baselines; A/B or shadow tests; p50/p95/p99 tracking
- CPU vs IO bound; flamegraphs; lock contention; GC logs

### Application Optimizations
- Efficient serialization (Protobuf/JSON with Jackson tuned)
- Batching and pipelining; vectorized operations where applicable
- Cache locality; reduce allocations; minimize synchronous hops

### Database and Storage
- Query plans; avoid N+1; covering indexes; partition pruning
- Write amplification awareness; bulk loads; compaction windows

### Scalability
- Vertical vs horizontal; stateful vs stateless; sticky sessions trade-offs
- Partitioning and sharding; autoscaling triggers and SLO-aware scaling

### Load and Capacity Testing
- Synthetic and replay; chaos under load; saturation points; regression budgets

### Interview Checklist
- Identify bottlenecks quantitatively; propose concrete tuning steps
- Clear sharding/partition plan; realistic autoscaling and limits
- Testing approach to validate improvements and guard regressions


### Java/JVM-Specific Tips
- Prefer G1/ZGC; tune heap regions; analyze GC logs; avoid large object churn
- Use virtual threads (Project Loom) for IO-heavy workloads; beware pinning and platform limits
- Profile with async-profiler/Flight Recorder; verify wins with benchmarks

### C10k/C10M Patterns
- Avoid per-connection threads; leverage async IO or virtual threads; batch writes; use connection pooling and keep-alive.


