## 9. API Design and Contracts

Design clear, evolvable contracts with good ergonomics and operability.

### REST and gRPC
- **REST: resources, verbs, nouns; pagination (cursor over offset), filtering, sorting**
  - **Resources**: Nouns that represent business entities (users, orders, products)
  - **Verbs**: HTTP methods (GET, POST, PUT, DELETE, PATCH)
  - **Pagination**: Use cursors instead of offsets for better performance
  - **Filtering**: Query parameters for filtering results
  - **Sorting**: Query parameters for ordering results
  - **Example**: `GET /users?cursor=abc123&limit=20&sort=name&filter=active`

- **gRPC: strongly-typed, binary, streaming; ideal for internal service RPC**
  - **Strongly-typed**: Protocol Buffers define exact data structures
  - **Binary**: More efficient than JSON for internal communication
  - **Streaming**: Support for server-side, client-side, and bidirectional streaming
  - **Internal RPC**: Perfect for service-to-service communication
  - **Example**: User service calling payment service for authorization

**Key insight**: Use REST for public APIs (easy to debug, integrate), gRPC for internal services (performance, type safety).

### Versioning and Compatibility
- **Backward-compatible changes: additive first; deprecations with grace periods**
  - **Additive first**: Add new fields before removing old ones
  - **Deprecations**: Mark old fields as deprecated with clear timelines
  - **Grace periods**: Give clients time to migrate before breaking changes
  - **Example**: Add `new_field` while keeping `old_field` for 6 months

- **SemVer in OpenAPI; Protobuf reserved fields/tags**
  - **SemVer**: Semantic versioning (major.minor.patch)
  - **OpenAPI**: Use version numbers in API documentation
  - **Protobuf**: Reserve field numbers to prevent conflicts
  - **Example**: API v1.2.3, where 1=major breaking change, 2=backward-compatible addition, 3=bug fix

**Why this matters**: API evolution is inevitable. Good versioning practices prevent breaking changes and enable independent client/server evolution.

### Idempotency and Safety
- **Idempotency keys; PUT vs POST semantics; safe retries; dedup windows**
  - **Idempotency keys**: Unique identifiers for operations to prevent duplicates
  - **PUT vs POST**: PUT is idempotent, POST is not
  - **Safe retries**: Clients can retry failed requests safely
  - **Dedup windows**: Time windows for deduplication
  - **Example**: Payment API with idempotency key to prevent double charges

**Key insight**: Idempotency is critical for reliability. Design APIs to be safe for retries and network failures.

### Errors and Telemetry
- **Consistent error schema (problem+json); correlation IDs; request IDs in logs**
  - **Problem+json**: Standard error response format (RFC 7807)
  - **Correlation IDs**: Track requests across service boundaries
  - **Request IDs**: Unique identifiers for each request
  - **Example**: `{"type": "validation_error", "title": "Invalid input", "detail": "Email is required", "correlation_id": "abc123"}`

**Why this matters**: Good error handling improves debugging and user experience. Consistent error formats make integration easier.

### Security
- **OAuth2/OIDC; scopes, least privilege; mTLS internal; JWT best practices**
  - **OAuth2**: Authorization framework for delegated access
  - **OIDC**: OpenID Connect for authentication
  - **Scopes**: Limit what resources clients can access
  - **Least privilege**: Give minimum necessary permissions
  - **mTLS**: Mutual TLS for service-to-service authentication
  - **JWT**: JSON Web Tokens with proper expiration and validation

**Key insight**: Security should be designed in from the beginning, not added later. Use industry standards and follow security best practices.

### Webhooks
- **Signed payloads; retries with backoff; idempotent receivers; replay protection**
  - **Signed payloads**: HMAC signatures to verify webhook authenticity
  - **Retries with backoff**: Exponential backoff for failed webhook deliveries
  - **Idempotent receivers**: Handle duplicate webhook deliveries gracefully
  - **Replay protection**: Prevent replay attacks using timestamps and nonces
  - **Example**: Payment webhook with signature verification and idempotency

**Why this matters**: Webhooks enable real-time integration but introduce security and reliability challenges. Good webhook design is essential for production use.

### Interview Checklist
- **Resource modeling, versioning strategy, idempotent writes, and error schema**
  - Show your API design and explain your choices
  - Demonstrate understanding of API evolution challenges
- **Security posture and operational concerns (quotas, observability)**
  - Explain your security approach
  - Show you understand operational requirements

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

**Study this example**: It shows good API design practices including idempotency, proper HTTP status codes, and clear request/response schemas.

### Webhook Hardening
- **HMAC signature with shared secret**: Verify webhook authenticity
- **Include timestamp**: Prevent replay attacks
- **Reject replays beyond window**: Set reasonable time windows
- **Idempotent processing**: Use event ID for deduplication

**Key insight**: Webhooks are security-critical. Proper hardening prevents attacks and ensures reliable delivery.

### Error Schema (problem+json)
```json
{ 
  "type": "https://errors.example.com/validation", 
  "title": "Invalid field", 
  "detail": "title is required", 
  "instance": "/v1/posts" 
}
```

**Use this format**: It's a standard that makes error handling consistent and debuggable.

### API Design Principles
- **Consistency**: Use consistent patterns across all endpoints
- **Simplicity**: Simple APIs are easier to use and maintain
- **Discoverability**: APIs should be self-documenting
- **Evolvability**: Design for future changes
- **Performance**: Consider performance implications of design choices

**Follow these principles**: They lead to better APIs that are easier to use and maintain.

### Rate Limiting and Quotas
- **Per-client limits**: Different limits for different client types
- **Burst handling**: Allow reasonable bursts while maintaining overall limits
- **Rate limit headers**: Tell clients about their current usage
- **Graceful degradation**: Reduce functionality instead of rejecting all requests

**Why this matters**: Rate limiting protects your system and ensures fair usage. Good rate limiting improves system reliability.

### API Documentation
- **OpenAPI/Swagger**: Machine-readable API documentation
- **Examples**: Provide working examples for all endpoints
- **Interactive docs**: Tools like Swagger UI for testing
- **Versioning**: Clear documentation for each API version

**Key insight**: Good documentation reduces integration time and support burden. Invest in comprehensive API documentation.

### Testing and Validation
- **Contract testing**: Verify API contracts between services
- **Schema validation**: Validate request/response schemas
- **Integration testing**: Test complete API workflows
- **Performance testing**: Verify API performance under load

**Why this matters**: Testing ensures API quality and prevents breaking changes. Good testing practices are essential for reliable APIs.

### Additional Resources for Deep Study
- **Books**: "RESTful Web Services" by Leonard Richardson (REST fundamentals)
- **Standards**: RFC 7807 (Problem Details for HTTP APIs)
- **Practice**: Design APIs for real use cases and get feedback
- **Real-world**: Study how companies like Stripe, GitHub, and Twilio design their APIs

**Study strategy**: Understand the principles, practice with real examples, then study real-world APIs to understand practical constraints.


