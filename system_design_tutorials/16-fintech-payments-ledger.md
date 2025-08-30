## 16. Fintech: Payments and Ledger Design

Design a resilient, auditable payments system with a correct ledger.

### Money and Precision
- **Never use floating point; store minor units (cents) or fixed-point decimals**
  - **Floating point**: Imprecise for financial calculations
  - **Minor units**: Store amounts in smallest currency unit (cents, pence)
  - **Fixed-point decimals**: Use BigDecimal or similar for precision
  - **Example**: Store $10.99 as 1099 cents, not as 10.99 float

- **Rounding rules explicit; currencies with non-2 decimal places; FX precision**
  - **Rounding rules**: Define how to handle fractional amounts
  - **Non-2 decimal places**: Some currencies have 3+ decimal places
  - **FX precision**: Foreign exchange requires high precision
  - **Example**: JPY has 0 decimal places, BHD has 3 decimal places

**Key insight**: Financial precision is non-negotiable. Use appropriate data types and be explicit about rounding.

### Double-Entry Ledger
- **Accounts and postings; invariants: sum(postings)=0; immutable entries + corrections**
  - **Accounts**: Represent entities that hold value (user accounts, system accounts)
  - **Postings**: Individual debit/credit entries
  - **Invariants**: Sum of all postings must equal zero (double-entry principle)
  - **Immutable entries**: Never modify existing entries
  - **Corrections**: Create new entries to fix mistakes, don't modify old ones
  - **Example**: Debit user account 100, credit merchant account 100

- **States: pending, posted, reversed; cut-off windows; closing balances**
  - **Pending**: Transaction initiated but not yet posted
  - **Posted**: Transaction completed and posted to ledger
  - **Reversed**: Transaction cancelled or refunded
  - **Cut-off windows**: Time windows for end-of-day processing
  - **Closing balances**: Account balances at end of period

**Why this matters**: Double-entry bookkeeping ensures data integrity and enables auditing. Poor ledger design leads to financial errors and compliance issues.

### Payment Flow
- **Ingestion (idempotent): accept → authorize/hold → capture/settle → post to ledger**
  - **Ingestion**: Accept payment request with idempotency key
  - **Authorize/hold**: Verify funds and place hold
  - **Capture/settle**: Complete the transaction
  - **Post to ledger**: Create ledger entries
  - **Example**: Accept payment → authorize $100 → capture $100 → debit user, credit merchant

- **Outbox to publish events; retries with exactly-once effect via idempotency keys**
  - **Outbox pattern**: Reliable event publishing using database transactions
  - **Events**: Publish payment events for downstream systems
  - **Retries**: Handle failures gracefully
  - **Exactly-once**: Ensure events are processed exactly once
  - **Example**: PaymentCompleted event published via outbox

**Key insight**: Payment flows must be reliable and auditable. Use patterns that ensure consistency and prevent duplicate processing.

### Idempotency and Reconciliation
- **Idempotency key per client request; dedup store with TTL**
  - **Idempotency key**: Unique identifier for each payment request
  - **Dedup store**: Store processed requests to prevent duplicates
  - **TTL**: Time-to-live for deduplication records
  - **Example**: Use client-generated UUID as idempotency key

- **Reconciliation jobs compare provider files vs internal ledger; dispute workflows**
  - **Reconciliation**: Compare external provider data with internal records
  - **Provider files**: Files from payment processors (Stripe, PayPal)
  - **Dispute workflows**: Handle chargebacks and disputes
  - **Example**: Daily reconciliation job, automated dispute routing

**Why this matters**: Idempotency prevents duplicate charges, reconciliation catches discrepancies, and dispute workflows protect against fraud.

### Providers and Webhooks
- **Signed callbacks; replay protection; eventual consistency to user via projections**
  - **Signed callbacks**: Verify webhook authenticity with signatures
  - **Replay protection**: Prevent duplicate webhook processing
  - **Eventual consistency**: User sees updates after processing
  - **Projections**: Read-optimized views of payment data
  - **Example**: Stripe webhook with HMAC signature verification

**Key insight**: External providers introduce complexity. Handle webhooks securely and plan for eventual consistency.

### Compliance and Security
- **PCI DSS scope reduction via tokenization; vault PANs; rotate keys regularly**
  - **PCI DSS**: Payment Card Industry Data Security Standard
  - **Scope reduction**: Minimize systems that handle card data
  - **Tokenization**: Replace sensitive data with tokens
  - **Vault PANs**: Store card numbers in secure vault
  - **Key rotation**: Regularly change encryption keys
  - **Example**: Store only last 4 digits, use tokens for processing

- **Audit logs: tamper-evident; user access reviews; SAR/suspicious activity workflows**
  - **Audit logs**: Record all payment-related actions
  - **Tamper-evident**: Logs cannot be modified without detection
  - **Access reviews**: Regular review of user permissions
  - **SAR**: Suspicious Activity Reports for regulatory compliance
  - **Example**: Immutable audit trail, quarterly access reviews

**Why this matters**: Financial systems have strict compliance requirements. Poor compliance leads to fines and loss of business.

### Example Ledger Schema (simplified)
```sql
-- accounts(id, owner_id, currency, status, ...)
-- entries(id, account_id, amount_minor, currency, direction, txn_id, created_at)
-- txn(id, external_id, type, state, created_at)
-- invariant: for each txn, sum(direction * amount_minor) = 0
```

**Study this schema**: It shows the essential structure for a double-entry ledger. Add indexes and constraints based on your access patterns.

### Failure Handling
- **Exactly-once illusion via idempotency + outbox; avoid two-phase commit**
  - **Exactly-once illusion**: Achieve exactly-once processing without complex protocols
  - **Idempotency**: Prevent duplicate processing
  - **Outbox**: Reliable event publishing
  - **Avoid 2PC**: Two-phase commit is complex and slow
  - **Example**: Use idempotency keys and outbox pattern

- **Partial failure: compensating entries; reversal flows**
  - **Partial failure**: Some steps succeed, others fail
  - **Compensating entries**: Reverse successful steps
  - **Reversal flows**: Handle refunds and cancellations
  - **Example**: Payment succeeds but notification fails → reverse payment

**Key insight**: Payment systems must handle failures gracefully. Plan for partial failures and have reversal mechanisms.

### Interview Checklist
- **Idempotent ingestion; outbox; ledger invariants; reconciliation**
  - Show you understand payment reliability patterns
  - Demonstrate knowledge of double-entry bookkeeping
- **Provider webhook security; user-facing consistency model**
  - Explain your security approach
  - Show you understand consistency trade-offs
- **Compliance considerations (PCI, audit, data retention)**
  - Demonstrate understanding of regulatory requirements
  - Show you have a plan for compliance

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

**Use this template**: It shows the complete payment flow from client to reconciliation. Customize based on your specific requirements.

### Reconciliation Notes
- **Daily provider reports matched to internal ledger**: Compare external and internal records
- **Differences queued for ops review**: Investigate discrepancies
- **Immutable audit trail**: Complete history of all changes

**Key insight**: Reconciliation is essential for financial accuracy. Plan for it from the beginning.

### Additional Design Considerations
- **Multi-currency support**: Handle different currencies and exchange rates
- **Settlement windows**: Process payments in batches for efficiency
- **Fee calculation**: Handle complex fee structures and splits
- **Refund processing**: Handle partial and full refunds
- **Chargeback management**: Handle disputes and chargebacks

**Why this matters**: Real payment systems are complex. Plan for these requirements early.

### Security Best Practices
- **Encryption at rest and in transit**: Protect sensitive financial data
- **Access controls**: Strict access controls for payment systems
- **Monitoring**: Monitor for suspicious activity
- **Incident response**: Plan for security incidents
- **Regular audits**: Regular security and compliance audits

**Key insight**: Financial security is critical. Implement multiple layers of security controls.

### Additional Resources for Deep Study
- **Books**: "Building Microservices" by Sam Newman (microservices for payments)
- **Standards**: PCI DSS, ISO 27001, SOC 2
- **Practice**: Build simple payment systems and test failure scenarios
- **Real-world**: Study how companies like Stripe, Square, and PayPal handle payments

**Study strategy**: Understand the fundamentals, practice with simple examples, then study real-world implementations to understand practical constraints.


