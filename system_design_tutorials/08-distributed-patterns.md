## 8. Distributed Systems Patterns

Understand fundamental trade-offs and building blocks.

### CAP and PACELC
- CAP: during partition, choose availability vs consistency; most choose AP with bounded staleness
- PACELC: Else (no partition) choose latency vs consistency; informs read/write quorums

### Quorums and Replication
- N replicas; read quorum R, write quorum W; strong reads require R + W > N
- Leader/follower vs leaderless (Dynamo-style) with read-repair

### Time and Ordering
- Clock skew is real; use NTP; avoid relying on system time for ordering
- Logical clocks (Lamport), vector clocks for causal relationships
- Monotonic ID generators (Snowflake/ULID) with precautions

### Consensus and Coordination
- Raft/etcd/ZooKeeper for configuration, locks, elections
- Avoid overusing distributed locks; prefer idempotency and commutativity

### CRDTs and Conflict Resolution
- Convergent data types for AP systems (counters, sets); last-write-wins caveats

### Retry, Backoff, and Jitter
- Exponential backoff with jitter to prevent thundering herds

### Interview Checklist
- State partition/latency trade-offs explicitly
- Explain replication, quorums, and consistency model
- Show failure handling for leader loss and network partitions


### Quorum Math Examples
- N=3, R=2, W=2 ⇒ R+W=4>3 (strong reads); tolerate 1 node failure
- N=5, R=2, W=3 ⇒ R+W=5 (strong); higher write cost, lower read cost

### Snowflake/ULID IDs
- 64-bit Snowflake: time | datacenter | worker | sequence; beware clock regress
- ULID: lexicographically sortable, good for storage locality; still handle collisions

### Vector Clocks (intuition)
- Maintain per-node counters; detect concurrent updates and resolve via app policy or CRDTs


