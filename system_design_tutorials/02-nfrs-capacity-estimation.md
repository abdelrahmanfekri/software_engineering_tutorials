## 2. NFRs, SLOs, and Capacity Estimation

Design choices follow from Non-Functional Requirements (NFRs). Quantify first; design second.

### NFRs and Definitions
- Availability: target 99.9% (or higher). Error budgets drive rollout velocity.
- Latency: p50/p95/p99; identify tail tolerance. Set budgets per endpoint.
- Throughput: sustained vs peak QPS/RPS; background jobs vs interactive.
- Durability: data loss tolerances, RPO/RTO, backup/restore windows.
- Consistency: strong, bounded-staleness, eventual; per entity or per action.
- Cost: infra, storage classes, egress, ops toil.

### SLI/SLO/SLA
- SLI: measurable signal (success rate, latency under 200ms, freshness < 60s)
- SLO: objective (99.9% success, 95% < 200ms)
- SLA: external contract with remedies

### Back-of-the-Envelope Cheatsheet
- Seconds/day = 86,400; hours/day = 24.
- Peak factor: 2–10× avg, depending on diurnal/seasonal burstiness.
- DAU to QPS: DAU × actions/user/day ÷ 86,400 × peak factor.
- Storage/day: events/day × payload × replication factor (e.g., 3)
- Bandwidth: QPS × payload × 8 ÷ compression.
- Cache hit: target 80–95%; origin QPS = incoming × (1 − hit).

### Example: 10M DAU Photo App
- Uploads: 1/photo/day → 10M/day → ~116 QPS avg → ×5 peak → ~580 QPS
- Feed fetch: 8/day → 80M/day → ~926 QPS avg → ×6 peak → ~5.5k QPS
- Photo size: 1.5 MB avg; storage/day = 15 TB × 3 replicas = 45 TB
- CDN offload: 95% hit → origin egress down to 5%

### Headroom and Safety
- Capacity buffers: 30–50% headroom for peaks and failures
- Autoscaling: cool-downs, predictive vs reactive; limits to avoid thrash

### Partitioning and Hotspots
- Partition key choice: uniform hashing vs semantic (userId, topicId)
- Hot keys: celebs, trending topics; mitigate with random suffixing or client sharding
- Skew detection: top-K keys, p99 partition lag, queue depth per partition

### Data Growth and Retention
- Retention policies by data class; TTL for logs/analytics
- Archival tiers (S3/Glacier, Nearline/Coldline); index pruning

### Interview Checklist
- State NFRs with numbers; compute QPS/storage/bandwidth
- Call out peak factors and cache goals; identify hot partitions
- Explain capacity headroom and autoscaling policy


### Availability Math (error budgets per month)
- 99.9% → ~43.2 minutes down/month
- 99.99% → ~4.32 minutes down/month
- 99.999% → ~26 seconds down/month
- Budget use drives rollout speed and mitigation urgency.

### Worked Example: Chat App
- 5M DAU, 40 messages/user/day ⇒ 200M/day ⇒ ~2,315 QPS avg; ×6 peak ⇒ ~13.9k QPS
- Payload ~350 bytes msg ⇒ ingress ~39 Mbps at peak before compression
- WebSocket fanout: 1:1 DMs vs small groups; plan per-convo partitioning; presence updates bursty
- Cache target: 90% for recent conversation lists; origin load ~10% of requests

### Storage Growth Template
- Logs/metrics/traces often dominate: plan separate retention and storage classes
- Hot (SSD) for indices and current data; warm/cold (object storage) beyond horizon

### Headroom and Sizing
- Instance-level: target 60–70% CPU, 65–75% heap at peak
- Cluster-level: 30–50% spare capacity for zone loss + deploy waves


