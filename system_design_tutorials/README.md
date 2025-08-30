# System Design Interview Track (3+ YOE)

A practical, interview-focused track for engineers with 3–5 years of experience. It balances high-level architecture with low-level design details, emphasizes modern microservices, and calls out real-world challenges, trade-offs, and technologies.

## How to use
- Start from 1 and progress sequentially. Skim, then deep dive with the checklists and exercises.
- For each topic, prepare a short “interview script” you can verbalize in 1–2 minutes.
- Apply concepts to 2–3 case studies (e.g., feed, chat, payments) to build muscle memory.

## Table of Contents
1. [Interview Approach & Mindset](01-interview-approach-mindset.md)
2. [NFRs, SLOs, and Capacity Estimation](02-nfrs-capacity-estimation.md)
3. [Networking, Load Balancing, and Rate Limiting](03-networking-load-balancing.md)
4. [Storage, Indexing, and Search](04-storage-databases.md)
5. [Caching and CDNs](05-caching-cdn.md)
6. [Queues, Streams, and Event-Driven Systems](06-queues-streams.md)
7. [Microservices Architecture](07-architecture-microservices.md)
8. [Distributed Systems Patterns](08-distributed-patterns.md)
9. [API Design and Contracts](09-api-design-contracts.md)
10. [Observability and Resiliency](10-observability-resiliency.md)
11. [Security and Compliance](11-security-and-compliance.md)
12. [Performance and Scalability](12-performance-and-scalability.md)
13. [Infrastructure, Kubernetes, and CI/CD](13-infrastructure-kubernetes-ci-cd.md)
14. [Low-Level Design Patterns](14-low-level-design-patterns.md)
15. [Case Studies and Exercises](15-case-studies-and-exercises.md)
16. [Fintech: Payments and Ledger Design](16-fintech-payments-ledger.md)

## Interview day checklist (short)
- Clarify product scope, users, and success metrics.
- Quantify scale: QPS, data size/day, read/write ratio, latency targets (p50/p95/p99), availability SLO.
- Propose HLD: API, data model, core components, request flow, storage, caches, messaging.
- Address NFRs: performance, reliability, scalability, security, cost.
- Stress points: hotspots, single points of failure, backpressure, skew, consistency.
- Deep dive: 1–2 subsystems (e.g., cache strategy + write path + idempotency).
- Trade-offs: name alternatives and why you chose yours.
- Operability: observability, rollout, capacity plan, failure handling.


