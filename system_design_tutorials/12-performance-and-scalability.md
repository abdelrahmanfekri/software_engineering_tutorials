## 12. Performance and Scalability

Optimize for user-perceived latency and sustainable throughput.

### Performance Diagnostics
- **Establish baselines; A/B or shadow tests; p50/p95/p99 tracking**
  - **Baselines**: Measure performance under normal conditions
  - **A/B tests**: Compare performance between different versions
  - **Shadow tests**: Send traffic to new version without affecting users
  - **Percentile tracking**: Monitor p50 (median), p95, p99 (tail) latency
  - **Example**: Baseline p95 = 200ms, new version p95 = 150ms

**Key insight**: You can't optimize what you don't measure. Establish baselines before making changes.

### Application Optimizations
- **Efficient serialization (Protobuf/JSON with Jackson tuned); batching and pipelining**
  - **Protobuf**: Binary serialization, smaller payloads, faster parsing
  - **JSON with Jackson**: Optimize JSON parsing and generation
  - **Batching**: Group multiple operations together
  - **Pipelining**: Send multiple requests without waiting for responses
  - **Example**: Batch 100 database inserts instead of 100 individual inserts

- **Vectorized operations where applicable; cache locality; reduce allocations**
  - **Vectorized operations**: Process multiple data elements simultaneously
  - **Cache locality**: Keep related data close together in memory
  - **Reduce allocations**: Minimize object creation and garbage collection
  - **Example**: Use SIMD instructions for numerical computations

**Why this matters**: Application-level optimizations can provide significant performance improvements with minimal infrastructure changes.

### Database and Storage
- **Query plans; avoid N+1; covering indexes; partition pruning**
  - **Query plans**: Understand how database executes queries
  - **N+1 problem**: One query + N queries for related data
  - **Covering indexes**: Index contains all columns needed for query
  - **Partition pruning**: Database only scans relevant partitions
  - **Example**: Use JOIN instead of separate queries, create covering indexes

- **Write amplification awareness; bulk loads; compaction windows**
  - **Write amplification**: Actual writes vs logical writes
  - **Bulk loads**: Load data in batches for better performance
  - **Compaction windows**: Schedule maintenance during low-traffic periods
  - **Example**: Use COPY for bulk data loading in PostgreSQL

**Key insight**: Database performance is often the bottleneck. Good query design and indexing are critical.

### Scalability
- **Vertical vs horizontal; stateful vs stateless; sticky sessions trade-offs**
  - **Vertical scaling**: Increase resources on existing machines
  - **Horizontal scaling**: Add more machines
  - **Stateful vs stateless**: Stateful services are harder to scale
  - **Sticky sessions**: Route same user to same server
  - **Example**: Start with vertical scaling, move to horizontal when needed

- **Partitioning and sharding; autoscaling triggers and SLO-aware scaling**
  - **Partitioning**: Split data across multiple storage systems
  - **Sharding**: Distribute data across multiple databases
  - **Autoscaling triggers**: Scale based on metrics (CPU, memory, queue depth)
  - **SLO-aware scaling**: Scale to maintain service level objectives
  - **Example**: Scale when CPU > 70% or response time > 200ms

**Why this matters**: Scalability enables growth without performance degradation. Plan for scalability from the beginning.

### Load and Capacity Testing
- **Synthetic and replay; chaos under load; saturation points; regression budgets**
  - **Synthetic tests**: Generate artificial load patterns
  - **Replay tests**: Replay real traffic patterns
  - **Chaos under load**: Test resilience while under stress
  - **Saturation points**: Identify performance limits
  - **Regression budgets**: Allowable performance degradation
  - **Example**: Replay production traffic patterns, inject failures during load

**Key insight**: Load testing validates performance assumptions and identifies bottlenecks before production.

### Interview Checklist
- **Identify bottlenecks quantitatively; propose concrete tuning steps**
  - Show you can measure and identify performance problems
  - Demonstrate understanding of optimization techniques
- **Clear sharding/partition plan; realistic autoscaling and limits**
  - Explain your scaling strategy
  - Show you understand scaling challenges
- **Testing approach to validate improvements and guard regressions**
  - Demonstrate understanding of performance testing
  - Show you have a plan for preventing regressions

### Java/JVM-Specific Tips
- **Prefer G1/ZGC; tune heap regions; analyze GC logs; avoid large object churn**
  - **G1/ZGC**: Modern garbage collectors with better performance
  - **Heap regions**: Tune garbage collector parameters
  - **GC logs**: Analyze garbage collection behavior
  - **Large object churn**: Minimize creation of large objects
  - **Example**: Use G1GC with tuned regions, analyze GC logs for optimization

- **Use virtual threads (Project Loom) for IO-heavy workloads; beware pinning and platform limits**
  - **Virtual threads**: Lightweight threads for IO operations
  - **IO-heavy workloads**: Perfect for database and HTTP calls
  - **Pinning**: Virtual threads can't yield during blocking operations
  - **Platform limits**: Understand virtual thread constraints
  - **Example**: Use virtual threads for database connection pools

- **Profile with async-profiler/Flight Recorder; verify wins with benchmarks**
  - **Async-profiler**: Low-overhead profiling tool
  - **Flight Recorder**: Built-in Java profiling
  - **Benchmarks**: Measure performance improvements
  - **Example**: Profile application, optimize hot paths, benchmark improvements

**Key insight**: JVM tuning can provide significant performance improvements. Understand your garbage collector and profiling tools.

### C10k/C10M Patterns
- **Avoid per-connection threads; leverage async IO or virtual threads; batch writes**
  - **Per-connection threads**: Don't create one thread per connection
  - **Async IO**: Non-blocking IO operations
  - **Virtual threads**: Lightweight threads for IO
  - **Batch writes**: Group multiple operations together
  - **Example**: Use async IO with virtual threads for high-concurrency servers

- **Use connection pooling and keep-alive**
  - **Connection pooling**: Reuse connections instead of creating new ones
  - **Keep-alive**: Keep connections open for multiple requests
  - **Example**: Database connection pool with keep-alive

**Why this matters**: High-concurrency systems require different patterns than traditional request-per-thread models.

### Performance Budgets
- **Latency budgets**: Allocate latency across service calls
  - **Service A**: 50ms budget
  - **Service B**: 100ms budget
  - **Database**: 30ms budget
  - **Total**: 180ms budget

- **Throughput budgets**: Plan for expected traffic patterns
  - **Peak traffic**: 10,000 requests/second
  - **Average traffic**: 2,000 requests/second
  - **Capacity planning**: Provision for peak with headroom

**Key insight**: Performance budgets make trade-offs explicit and prevent performance regressions.

### Caching Strategies
- **Application cache**: Cache frequently accessed data in memory
- **Database cache**: Use database query cache and buffer pool
- **CDN cache**: Cache static content at edge locations
- **Cache invalidation**: Plan how to keep cache data fresh

**Why this matters**: Caching can dramatically improve performance. Plan your caching strategy carefully.

### Monitoring and Alerting
- **Performance metrics**: Track key performance indicators
- **SLO monitoring**: Monitor service level objectives
- **Performance alerts**: Alert on performance degradation
- **Trend analysis**: Identify performance trends over time

**Key insight**: Continuous monitoring is essential for maintaining performance. Set up alerts for performance issues.

### Additional Resources for Deep Study
- **Books**: "High Performance MySQL" by Baron Schwartz (database performance)
- **Papers**: "The Tail at Scale" by Dean and Barroso (tail latency)
- **Practice**: Profile applications and optimize performance bottlenecks
- **Real-world**: Study how companies optimize performance at scale

**Study strategy**: Understand the fundamentals, practice with profiling tools, then study real-world optimizations to understand practical constraints.


