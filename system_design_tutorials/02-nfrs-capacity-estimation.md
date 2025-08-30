## 2. NFRs, SLOs, and Capacity Estimation

Design choices follow from Non-Functional Requirements (NFRs). Quantify first; design second.

### NFRs and Definitions
- **Availability**: target 99.9% (or higher). Error budgets drive rollout velocity.
  - **What it means**: Percentage of time the system is operational and responding to requests
  - **Why it matters**: Affects user experience and business continuity. Higher availability = more complex infrastructure
  - **Trade-off**: Higher availability often means higher cost and complexity

- **Latency**: p50/p95/p99; identify tail tolerance. Set budgets per endpoint.
  - **What it means**: Response time percentiles. p95 means 95% of requests complete within this time
  - **Why it matters**: User experience is directly tied to latency. Tail latency (p99) often indicates system problems
  - **Example**: p50: 100ms, p95: 500ms, p99: 2s means 1% of requests take over 2 seconds

- **Throughput**: sustained vs peak QPS/RPS; background jobs vs interactive.
  - **What it means**: Requests per second the system can handle
  - **Why it matters**: Determines infrastructure sizing and scaling strategy
  - **Key insight**: Peak throughput is often 5-10x average, plan accordingly

- **Durability**: data loss tolerances, RPO/RTO, backup/restore windows.
  - **What it means**: How much data can be lost and how quickly it can be recovered
  - **RPO**: Recovery Point Objective (maximum acceptable data loss)
  - **RTO**: Recovery Time Objective (maximum acceptable downtime)

- **Consistency**: strong, bounded-staleness, eventual; per entity or per action.
  - **Strong**: All reads see the latest write
  - **Bounded-staleness**: Reads may be stale by at most X seconds
  - **Eventual**: Reads will eventually see the latest write
  - **Why it matters**: Affects user experience and system complexity

- **Cost**: infra, storage classes, egress, ops toil.
  - **What it means**: Total cost of ownership including infrastructure, operations, and maintenance
  - **Why it matters**: Business constraints often drive technical decisions

### SLI/SLO/SLA
- **SLI**: measurable signal (success rate, latency under 200ms, freshness < 60s)
  - **What it means**: Service Level Indicator - the actual metric you measure
  - **Example**: "95% of API requests complete within 200ms"

- **SLO**: objective (99.9% success, 95% < 200ms)
  - **What it means**: Service Level Objective - your target for the SLI
  - **Example**: "We target 95% of requests under 200ms"

- **SLA**: external contract with remedies
  - **What it means**: Service Level Agreement - what you promise to customers
  - **Example**: "We guarantee 99.9% uptime or provide service credits"

**Key insight**: SLIs measure, SLOs target, SLAs promise. Start with SLIs, set SLOs, then negotiate SLAs.

### Back-of-the-Envelope Cheatsheet
- **Seconds/day** = 86,400; hours/day = 24.
- **Peak factor**: 2–10× avg, depending on diurnal/seasonal burstiness.
  - **Diurnal**: Daily patterns (e.g., more traffic during business hours)
  - **Seasonal**: Weekly/monthly patterns (e.g., more traffic on weekends)
- **DAU to QPS**: DAU × actions/user/day ÷ 86,400 × peak factor.
- **Storage/day**: events/day × payload × replication factor (e.g., 3)
- **Bandwidth**: QPS × payload × 8 ÷ compression.
- **Cache hit**: target 80–95%; origin QPS = incoming × (1 − hit).

**Why this matters**: These calculations show you can translate business requirements into technical specifications. Interviewers want to see quantitative thinking.

### Example: 10M DAU Photo App
- **Uploads**: 1/photo/day → 10M/day → ~116 QPS avg → ×5 peak → ~580 QPS
- **Feed fetch**: 8/day → 80M/day → ~926 QPS avg → ×6 peak → ~5.5k QPS
- **Photo size**: 1.5 MB avg; storage/day = 15 TB × 3 replicas = 45 TB
- **CDN offload**: 95% hit → origin egress down to 5%

**Key insights**:
- Read traffic is often much higher than write traffic
- Storage costs include replication factor
- CDN dramatically reduces origin load

### Headroom and Safety
- **Capacity buffers**: 30–50% headroom for peaks and failures
  - **Why**: Systems need buffer for unexpected traffic spikes and infrastructure failures
  - **Example**: If you need 1000 QPS, provision for 1300-1500 QPS
- **Autoscaling**: cool-downs, predictive vs reactive; limits to avoid thrash
  - **Cool-downs**: Wait time before scaling down to avoid rapid scale up/down cycles
  - **Predictive**: Scale based on patterns (e.g., time of day)
  - **Reactive**: Scale based on current metrics

### Partitioning and Hotspots
- **Partition key choice**: uniform hashing vs semantic (userId, topicId)
  - **Uniform hashing**: Distributes load evenly but may not match access patterns
  - **Semantic**: Matches access patterns but may create hotspots
- **Hot keys**: celebs, trending topics; mitigate with random suffixing or client sharding
  - **Random suffixing**: Add random number to partition key to distribute load
  - **Client sharding**: Route different clients to different partitions
- **Skew detection**: top-K keys, p99 partition lag, queue depth per partition
  - **Why**: Hot partitions can cause performance cliffs and availability issues

### Data Growth and Retention
- **Retention policies by data class**: Different data has different retention requirements
  - **User data**: Keep as long as user is active
  - **Logs**: Keep for debugging (days to weeks)
  - **Analytics**: Keep for business intelligence (months to years)
- **TTL for logs/analytics**: Automatically delete old data
- **Archival tiers**: Move old data to cheaper storage (S3/Glacier, Nearline/Coldline)
- **Index pruning**: Remove old indexes to improve performance

### Interview Checklist
- **State NFRs with numbers**: Don't just say "high availability," say "99.9% availability"
- **Compute QPS/storage/bandwidth**: Show your work with back-of-the-envelope calculations
- **Call out peak factors and cache goals**: Demonstrate understanding of real-world patterns
- **Identify hot partitions**: Show you understand scaling challenges

### Availability Math (error budgets per month)
- **99.9%** → ~43.2 minutes down/month
- **99.99%** → ~4.32 minutes down/month
- **99.999%** → ~26 seconds down/month
- **Budget use drives rollout speed and mitigation urgency**.

**Key insight**: Error budgets determine how aggressive you can be with deployments. More aggressive = higher risk of violating SLOs.

### Worked Example: Chat App
- **5M DAU, 40 messages/user/day** ⇒ 200M/day ⇒ ~2,315 QPS avg; ×6 peak ⇒ ~13.9k QPS
- **Payload ~350 bytes msg** ⇒ ingress ~39 Mbps at peak before compression
- **WebSocket fanout**: 1:1 DMs vs small groups; plan per-convo partitioning; presence updates bursty
- **Cache target**: 90% for recent conversation lists; origin load ~10% of requests

**Key insights**:
- Real-time systems have different scaling challenges than batch systems
- Presence updates can be bursty and require careful handling
- WebSocket connections need to be distributed across multiple servers

### Storage Growth Template
- **Logs/metrics/traces often dominate**: Plan separate retention and storage classes
  - **Why**: Observability data often grows faster than business data
  - **Strategy**: Use different storage classes and retention policies
- **Hot (SSD) for indices and current data**: Fast access for recent data
- **Warm/cold (object storage) beyond horizon**: Cheaper storage for old data

### Headroom and Sizing
- **Instance-level**: target 60–70% CPU, 65–75% heap at peak
  - **Why**: Leave room for garbage collection, temporary spikes, and monitoring overhead
- **Cluster-level**: 30–50% spare capacity for zone loss + deploy waves
  - **Zone loss**: If one availability zone fails, others must handle the load
  - **Deploy waves**: Rolling deployments need extra capacity

### Additional Resources for Deep Study
- **Books**: "Site Reliability Engineering" by Google (SLOs and error budgets)
- **Papers**: "The Tail at Scale" by Dean and Barroso (latency and percentiles)
- **Practice**: Use tools like Grafana to understand real-world metrics
- **Real-world**: Study how companies like Netflix and Uber handle scale

**Study strategy**: Understand the math, practice with real examples, then study how companies implement these concepts in production.


