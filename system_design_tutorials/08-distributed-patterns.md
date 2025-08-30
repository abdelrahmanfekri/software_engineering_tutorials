## 8. Distributed Systems Patterns

Understand fundamental trade-offs and building blocks.

### CAP and PACELC
- **CAP: during partition, choose availability vs consistency; most choose AP with bounded staleness**
  - **CAP Theorem**: In a distributed system, you can have at most two of: Consistency, Availability, Partition tolerance
  - **Partition**: Network failure that separates nodes
  - **Consistency**: All nodes see the same data
  - **Availability**: System responds to requests
  - **Most choose AP**: Availability + Partition tolerance, with eventual consistency

- **PACELC: Else (no partition) choose latency vs consistency; informs read/write quorums**
  - **PACELC**: Partition-Else Latency-Consistency
  - **No partition**: Choose between latency and consistency
  - **Latency**: Fast responses but potentially stale data
  - **Consistency**: Fresh data but higher latency
  - **Quorums**: Number of nodes required for read/write operations

**Key insight**: CAP is about partition tolerance, PACELC is about normal operation. Both inform your system design choices.

### Quorums and Replication
- **N replicas; read quorum R, write quorum W; strong reads require R + W > N**
  - **N replicas**: Total number of copies of data
  - **R (read quorum)**: Number of replicas needed for a read
  - **W (write quorum)**: Number of replicas needed for a write
  - **Strong reads**: R + W > N ensures at least one replica has the latest data
  - **Example**: N=3, R=2, W=2 means 2 replicas for reads and writes

- **Leader/follower vs leaderless (Dynamo-style) with read-repair**
  - **Leader/follower**: One node handles writes, others replicate
  - **Leaderless**: Any node can handle reads/writes
  - **Read-repair**: Fix inconsistencies during read operations
  - **Example**: DynamoDB uses leaderless with read-repair

**Why this matters**: Quorum design determines your consistency model and failure tolerance. Poor quorum design leads to data inconsistency.

### Time and Ordering
- **Clock skew is real; use NTP; avoid relying on system time for ordering**
  - **Clock skew**: Different nodes have different times
  - **NTP**: Network Time Protocol for clock synchronization
  - **System time ordering**: Don't use system time for causal ordering
  - **Example**: Two events might have wrong order due to clock skew

- **Logical clocks (Lamport), vector clocks for causal relationships**
  - **Lamport clocks**: Simple logical timestamps for ordering
  - **Vector clocks**: Track causal relationships between events
  - **Causal ordering**: Events that depend on each other are ordered correctly
  - **Example**: Message A → Message B, B should have higher timestamp than A

**Key insight**: Physical time is unreliable in distributed systems. Use logical clocks for ordering and causality.

### Consensus and Coordination
- **Raft/etcd/ZooKeeper for configuration, locks, elections**
  - **Raft**: Consensus algorithm for replicated state machines
  - **etcd**: Distributed key-value store using Raft
  - **ZooKeeper**: Coordination service for distributed systems
  - **Configuration**: Store system configuration
  - **Locks**: Distributed locking for coordination
  - **Elections**: Choose leader among multiple candidates

- **Avoid overusing distributed locks; prefer idempotency and commutativity**
  - **Distributed locks**: Expensive and can cause deadlocks
  - **Idempotency**: Same operation multiple times has same effect
  - **Commutativity**: Operations can be reordered without changing result
  - **Example**: Use idempotent operations instead of locks for deduplication

**Why this matters**: Consensus is expensive and complex. Use it only when necessary, prefer simpler alternatives when possible.

### CRDTs and Conflict Resolution
- **Convergent data types for AP systems (counters, sets); last-write-wins caveats**
  - **CRDT**: Conflict-free Replicated Data Type
  - **Convergent**: All replicas eventually reach the same state
  - **Counters**: Monotonic counters that can only increase
  - **Sets**: Sets that support add/remove operations
  - **Last-write-wins**: Simple but can lose updates

**Key insight**: CRDTs provide eventual consistency without complex conflict resolution. Use them when your data model supports it.

### Retry, Backoff, and Jitter
- **Exponential backoff with jitter to prevent thundering herds**
  - **Exponential backoff**: Increase delay between retries exponentially
  - **Jitter**: Add randomness to prevent synchronized retries
  - **Thundering herd**: Multiple clients retrying simultaneously
  - **Example**: 1s, 2s, 4s, 8s delays with ±20% random variation

**Why this matters**: Poor retry strategies can overwhelm systems and cause cascading failures. Good retry strategies are essential for resilience.

### Interview Checklist
- **State partition/latency trade-offs explicitly**
  - Explain your CAP/PACELC choices
  - Show you understand the trade-offs
- **Explain replication, quorums, and consistency model**
  - Demonstrate understanding of replication strategies
  - Show you can design quorums for your requirements
- **Show failure handling for leader loss and network partitions**
  - Explain how your system handles failures
  - Show you have a plan for different failure scenarios

### Quorum Math Examples
- **N=3, R=2, W=2** ⇒ R+W=4>3 (strong reads); tolerate 1 node failure
  - **Strong reads**: Always read the latest data
  - **Tolerate 1 failure**: System works with 2 out of 3 nodes
- **N=5, R=2, W=3** ⇒ R+W=5 (strong); higher write cost, lower read cost
  - **Higher write cost**: Need 3 nodes for writes
  - **Lower read cost**: Only need 2 nodes for reads

**Study these examples**: They show how quorum design affects performance and failure tolerance.

### Snowflake/ULID IDs
- **64-bit Snowflake: time | datacenter | worker | sequence; beware clock regress**
  - **Time**: Timestamp in milliseconds
  - **Datacenter**: Datacenter identifier
  - **Worker**: Worker node identifier
  - **Sequence**: Sequence number within the same millisecond
  - **Clock regress**: System clock going backwards

- **ULID: lexicographically sortable, good for storage locality; still handle collisions**
  - **Lexicographically sortable**: Can be sorted as strings
  - **Storage locality**: Good for database storage and indexing
  - **Collisions**: Handle duplicate IDs gracefully

**Key insight**: Good ID generation is critical for distributed systems. Choose based on your ordering and uniqueness requirements.

### Vector Clocks (intuition)
- **Maintain per-node counters; detect concurrent updates and resolve via app policy or CRDTs**
  - **Per-node counters**: Each node maintains a vector of counters
  - **Concurrent updates**: Updates that happen simultaneously
  - **App policy**: Application-specific conflict resolution
  - **CRDTs**: Use conflict-free data types when possible

**Why this matters**: Vector clocks help detect and resolve conflicts in distributed systems. They're essential for maintaining data consistency.

### Failure Detection and Recovery
- **Heartbeats**: Regular messages to detect node failures
- **Gossip protocols**: Spread information about node health
- **Failure detectors**: Algorithms to detect node failures
- **Recovery strategies**: How to recover from failures

**Key insight**: Failure detection is critical for distributed systems. Poor failure detection leads to system instability.

### Network Partition Handling
- **Split-brain**: Multiple nodes think they're the leader
- **Network isolation**: Nodes can't communicate with each other
- **Reconciliation**: How to merge data after partition heals
- **Conflict resolution**: How to handle conflicting updates

**Why this matters**: Network partitions are inevitable in distributed systems. Good partition handling prevents data loss and inconsistency.

### Additional Resources for Deep Study
- **Books**: "Designing Data-Intensive Applications" by Martin Kleppmann (comprehensive coverage)
- **Papers**: "Dynamo: Amazon's Highly Available Key-value Store" (distributed systems design)
- **Practice**: Build simple distributed systems and experiment with failures
- **Real-world**: Study how companies like Google, Amazon, and Netflix handle distributed systems

**Study strategy**: Understand the theory, practice with simple examples, then study real-world implementations to understand practical constraints.


