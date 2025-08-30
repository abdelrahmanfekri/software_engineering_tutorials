## 11. Security and Compliance

Bake security into design: authN/Z, data protection, and compliant operations.

### Identity and Access
- OAuth2/OIDC; PKCE for public clients; short-lived tokens; refresh rotation
- mTLS for service-to-service; SPIFFE/SPIRE in meshes
- Least privilege IAM; scoped API tokens; just-in-time access

### Data Protection
- In transit: TLS 1.2/1.3; HSTS; perfect forward secrecy
- At rest: envelope encryption; KMS/HSM; key rotation; field-level for PII
- Secrets: Vault/Secrets Manager; no secrets in env vars or logs

### Application Security
- Input validation, output encoding; CSRF/Clickjacking protections
- Dependency scanning, SCA/SAST/DAST; SBOMs and signed artifacts

### Compliance
- GDPR/CCPA: data subject rights, retention, minimization
- PCI DSS: network segmentation, tokenization, key management
- Auditing: tamper-evident logs; access reviews; SOX controls for financial data

### Interview Checklist
- AuthN/Z flow, token and key management, least privilege
- Encryption posture and secrets handling; compliance-aware data lifecycle
- Auditability and incident response


### Threat Modeling (STRIDE quick pass)
- Spoofing: strong auth, mTLS, signed webhooks
- Tampering: integrity checks, HMACs, WORM logs
- Repudiation: non-repudiation via signed logs and clock sync
- Information disclosure: encryption, tokenization, access reviews
- DoS: rate limits, WAF, resource quotas
- Elevation of privilege: RBAC/ABAC, break-glass with approvals

### Key Management
- Rotate regularly; envelope encryption; split duties; auditable key use; KMS/HSM-backed


