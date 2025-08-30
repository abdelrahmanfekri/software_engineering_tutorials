## 16. Fintech: Payments and Ledger Design

Design a resilient, auditable payments system with a correct ledger.

### Money and Precision
- Never use floating point; store minor units (cents) or fixed-point decimals
- Rounding rules explicit; currencies with non-2 decimal places; FX precision

### Double-Entry Ledger
- Accounts and postings; invariants: sum(postings)=0; immutable entries + corrections
- States: pending, posted, reversed; cut-off windows; closing balances

### Payment Flow
- Ingestion (idempotent): accept → authorize/hold → capture/settle → post to ledger
- Outbox to publish events; retries with exactly-once effect via idempotency keys

### Idempotency and Reconciliation
- Idempotency key per client request; dedup store with TTL
- Reconciliation jobs compare provider files vs internal ledger; dispute workflows

### Providers and Webhooks
- Signed callbacks; replay protection; eventual consistency to user via projections

### Compliance and Security
- PCI DSS scope reduction via tokenization; vault PANs; rotate keys regularly
- Audit logs: tamper-evident; user access reviews; SAR/suspicious activity workflows

### Example Ledger Schema (simplified)
```sql
-- accounts(id, owner_id, currency, status, ...)
-- entries(id, account_id, amount_minor, currency, direction, txn_id, created_at)
-- txn(id, external_id, type, state, created_at)
-- invariant: for each txn, sum(direction * amount_minor) = 0
```

### Failure Handling
- Exactly-once illusion via idempotency + outbox; avoid two-phase commit
- Partial failure: compensating entries; reversal flows

### Interview Checklist
- Idempotent ingestion; outbox; ledger invariants; reconciliation
- Provider webhook security; user-facing consistency model
- Compliance considerations (PCI, audit, data retention)


### End-to-End Diagram (Payments)
```mermaid
sequenceDiagram
  participant Client
  participant API as Payments API (Idempotency)
  participant DB as Ledger DB + Outbox
  participant K as Kafka
  participant Prov as Provider
  participant Recon as Reconciliation
  Client->>API: POST /charge (Idempotency-Key)
  API->>DB: Begin tx; insert ledger postings; insert outbox
  DB-->>API: Commit
  API-->>Client: 202 Accepted (txnId)
  DB->>K: Outbox publisher emits ChargeInitiated
  K->>Prov: Provider Adapter requests auth/capture
  Prov-->>K: Webhook Event
  K->>DB: Post settlement entries
  Recon->>DB: Compare provider files vs ledger; create adjustments if needed
```

### Reconciliation Notes
- Daily provider reports matched to internal ledger; differences queued for ops review; immutable audit trail.


