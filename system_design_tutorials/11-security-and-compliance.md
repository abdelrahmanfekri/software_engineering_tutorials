## 11. Security and Compliance

Bake security into design: authN/Z, data protection, and compliant operations.

### Identity and Access
- **OAuth2/OIDC; PKCE for public clients; short-lived tokens; refresh rotation**
  - **OAuth2**: Authorization framework for delegated access
  - **OIDC**: OpenID Connect for authentication on top of OAuth2
  - **PKCE**: Proof Key for Code Exchange for public clients
  - **Short-lived tokens**: Reduce risk of token compromise
  - **Refresh rotation**: Rotate refresh tokens to detect theft
  - **Example**: Access token expires in 1 hour, refresh token in 30 days

- **mTLS for service-to-service; SPIFFE/SPIRE in meshes**
  - **mTLS**: Mutual TLS for bidirectional authentication
  - **Service-to-service**: Secure communication between internal services
  - **SPIFFE**: Secure Production Identity Framework for Everyone
  - **SPIRE**: SPIFFE Runtime Environment for identity management
  - **Example**: Service A authenticates to Service B using mTLS certificates

- **Least privilege IAM; scoped API tokens; just-in-time access**
  - **Least privilege**: Give minimum necessary permissions
  - **Scoped API tokens**: Limit what resources tokens can access
  - **Just-in-time access**: Grant temporary access when needed
  - **Example**: Read-only token for monitoring, admin token only when making changes

**Key insight**: Identity and access control are the foundation of security. Poor IAM leads to unauthorized access and data breaches.

### Data Protection
- **In transit: TLS 1.2/1.3; HSTS; perfect forward secrecy**
  - **TLS 1.2/1.3**: Transport Layer Security for encrypted communication
  - **HSTS**: HTTP Strict Transport Security to enforce HTTPS
  - **Perfect forward secrecy**: Compromised keys don't affect past communications
  - **Example**: All API communication over HTTPS with HSTS headers

- **At rest: envelope encryption; KMS/HSM; key rotation; field-level for PII**
  - **Envelope encryption**: Encrypt data with data keys, encrypt data keys with master keys
  - **KMS**: Key Management Service for centralized key management
  - **HSM**: Hardware Security Module for secure key storage
  - **Key rotation**: Regularly change encryption keys
  - **Field-level encryption**: Encrypt sensitive fields individually
  - **Example**: Encrypt SSN and credit card fields, rotate keys every 90 days

- **Secrets: Vault/Secrets Manager; no secrets in env vars or logs**
  - **Vault**: HashiCorp's secrets management solution
  - **Secrets Manager**: Cloud provider secrets management
  - **No env vars**: Don't store secrets in environment variables
  - **No logs**: Never log sensitive information
  - **Example**: Use AWS Secrets Manager to store database credentials

**Why this matters**: Data protection prevents unauthorized access to sensitive information. Poor data protection leads to compliance violations and data breaches.

### Application Security
- **Input validation, output encoding; CSRF/Clickjacking protections**
  - **Input validation**: Validate all user inputs to prevent injection attacks
  - **Output encoding**: Encode outputs to prevent XSS attacks
  - **CSRF**: Cross-Site Request Forgery protection
  - **Clickjacking**: Prevent embedding in malicious sites
  - **Example**: Validate email format, encode HTML output, use CSRF tokens

- **Dependency scanning, SCA/SAST/DAST; SBOMs and signed artifacts**
  - **Dependency scanning**: Check for known vulnerabilities in dependencies
  - **SCA**: Software Composition Analysis
  - **SAST**: Static Application Security Testing
  - **DAST**: Dynamic Application Security Testing
  - **SBOM**: Software Bill of Materials
  - **Signed artifacts**: Verify artifact integrity and authenticity

**Key insight**: Application security prevents common attack vectors. Regular security testing catches vulnerabilities before production.

### Compliance
- **GDPR/CCPA: data subject rights, retention, minimization**
  - **GDPR**: General Data Protection Regulation (EU)
  - **CCPA**: California Consumer Privacy Act
  - **Data subject rights**: Right to access, delete, and port data
  - **Retention**: Don't keep data longer than necessary
  - **Minimization**: Collect only necessary data
  - **Example**: Allow users to download their data, delete accounts, set retention policies

- **PCI DSS: network segmentation, tokenization, key management**
  - **PCI DSS**: Payment Card Industry Data Security Standard
  - **Network segmentation**: Isolate payment systems from other systems
  - **Tokenization**: Replace sensitive data with tokens
  - **Key management**: Secure key generation, storage, and rotation
  - **Example**: Store payment tokens instead of actual card numbers

- **Auditing: tamper-evident logs; access reviews; SOX controls for financial data**
  - **Tamper-evident logs**: Logs that can't be modified without detection
  - **Access reviews**: Regular review of user access
  - **SOX**: Sarbanes-Oxley Act for financial reporting
  - **Example**: Immutable audit logs, quarterly access reviews

**Why this matters**: Compliance is not optional for many businesses. Poor compliance leads to legal issues, fines, and loss of customer trust.

### Interview Checklist
- **AuthN/Z flow, token and key management, least privilege**
  - Explain your authentication and authorization approach
  - Show you understand security best practices
- **Encryption posture and secrets handling; compliance-aware data lifecycle**
  - Demonstrate understanding of data protection
  - Show you understand compliance requirements
- **Auditability and incident response**
  - Explain how you track and respond to security incidents

### Threat Modeling (STRIDE quick pass)
- **Spoofing: strong auth, mTLS, signed webhooks**
  - **Spoofing**: Impersonating legitimate users or services
  - **Strong auth**: Multi-factor authentication, strong passwords
  - **mTLS**: Mutual authentication between services
  - **Signed webhooks**: Verify webhook authenticity

- **Tampering: integrity checks, HMACs, WORM logs**
  - **Tampering**: Modifying data in transit or at rest
  - **Integrity checks**: Verify data hasn't been modified
  - **HMACs**: Hash-based Message Authentication Codes
  - **WORM logs**: Write Once Read Many logs

- **Repudiation: non-repudiation via signed logs and clock sync**
  - **Repudiation**: Denying actions that were performed
  - **Non-repudiation**: Prove actions were performed
  - **Signed logs**: Cryptographically signed audit logs
  - **Clock sync**: Synchronized timestamps across systems

- **Information disclosure: encryption, tokenization, access reviews**
  - **Information disclosure**: Unauthorized access to sensitive data
  - **Encryption**: Encrypt sensitive data
  - **Tokenization**: Replace sensitive data with tokens
  - **Access reviews**: Regular review of access permissions

- **DoS: rate limits, WAF, resource quotas**
  - **DoS**: Denial of Service attacks
  - **Rate limits**: Limit request frequency
  - **WAF**: Web Application Firewall
  - **Resource quotas**: Limit resource usage

- **Elevation of privilege: RBAC/ABAC, break-glass with approvals**
  - **Elevation of privilege**: Gaining unauthorized access
  - **RBAC**: Role-Based Access Control
  - **ABAC**: Attribute-Based Access Control
  - **Break-glass**: Emergency access with approval

**Use STRIDE**: It's a systematic way to identify and address security threats.

### Key Management
- **Rotate regularly**: Change keys on a schedule
- **Envelope encryption**: Use data keys for encryption, master keys for data keys
- **Split duties**: Multiple people required for key operations
- **Auditable key use**: Track all key usage
- **KMS/HSM-backed**: Use secure key storage

**Key insight**: Key management is critical for data security. Poor key management can compromise all encrypted data.

### Security by Design
- **Shift left**: Address security early in development
- **Defense in depth**: Multiple layers of security controls
- **Fail secure**: System fails to secure state
- **Principle of least privilege**: Minimum necessary access
- **Zero trust**: Verify every request

**Why this matters**: Security should be designed in from the beginning, not added later. Security by design is more effective and less expensive.

### Incident Response
- **Detection**: Monitor for security incidents
- **Analysis**: Investigate and understand incidents
- **Containment**: Stop incident from spreading
- **Eradication**: Remove root cause
- **Recovery**: Restore normal operations
- **Lessons learned**: Improve security based on incidents

**Key insight**: Security incidents are inevitable. Good incident response minimizes damage and improves security.

### Additional Resources for Deep Study
- **Books**: "Security Engineering" by Ross Anderson (comprehensive coverage)
- **Standards**: OWASP Top 10, NIST Cybersecurity Framework
- **Practice**: Set up security tools and practice penetration testing
- **Real-world**: Study how companies handle security incidents and compliance

**Study strategy**: Understand the principles, practice with security tools, then study real-world security incidents to understand practical challenges.


