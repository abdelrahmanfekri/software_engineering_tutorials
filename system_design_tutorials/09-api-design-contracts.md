## 9. API Design and Contracts

Design clear, evolvable contracts with good ergonomics and operability.

### REST and gRPC
- REST: resources, verbs, nouns; pagination (cursor over offset), filtering, sorting
- gRPC: strongly-typed, binary, streaming; ideal for internal service RPC

### Versioning and Compatibility
- Backward-compatible changes: additive first; deprecations with grace periods
- SemVer in OpenAPI; Protobuf reserved fields/tags

### Idempotency and Safety
- Idempotency keys; PUT vs POST semantics; safe retries; dedup windows

### Errors and Telemetry
- Consistent error schema (problem+json); correlation IDs; request IDs in logs

### Security
- OAuth2/OIDC; scopes, least privilege; mTLS internal; JWT best practices

### Webhooks
- Signed payloads; retries with backoff; idempotent receivers; replay protection

### Interview Checklist
- Resource modeling, versioning strategy, idempotent writes, and error schema
- Security posture and operational concerns (quotas, observability)


### Example REST Contract (OpenAPI excerpt)
```yaml
paths:
  /v1/posts:
    post:
      operationId: createPost
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreatePostRequest'
      responses:
        '201': { description: Created }
        '400': { description: Bad Request }
        '409': { description: Conflict }
      headers:
        Idempotency-Key: { schema: { type: string } }
```

### Webhook Hardening
- HMAC signature with shared secret; include timestamp; reject replays beyond window; idempotent processing keyed by event ID.

### Error Schema (problem+json)
```json
{ "type": "https://errors.example.com/validation", "title": "Invalid field", "detail": "title is required", "instance": "/v1/posts" }
```


