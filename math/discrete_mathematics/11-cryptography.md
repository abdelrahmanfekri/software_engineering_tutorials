# Cryptography

## Overview
Cryptography is the science of secure communication in the presence of adversaries. It combines mathematics, computer science, and engineering to protect information and ensure secure communication. This chapter covers symmetric and asymmetric cryptography, hash functions, digital signatures, elliptic curve cryptography, and quantum cryptography.

## Learning Objectives
- Understand symmetric key cryptography
- Learn about public key cryptography
- Master hash functions and their properties
- Understand digital signatures
- Learn about elliptic curve cryptography
- Study quantum cryptography
- Apply cryptographic techniques to secure systems

## 1. Symmetric Key Cryptography

### Definition
**Symmetric key cryptography** uses the same key for both encryption and decryption.

### Block Ciphers
**Definition**: A block cipher encrypts fixed-size blocks of plaintext to produce blocks of ciphertext.

#### Advanced Encryption Standard (AES)
- **Block size**: 128 bits
- **Key sizes**: 128, 192, or 256 bits
- **Rounds**: 10, 12, or 14 depending on key size
- **Operations**: SubBytes, ShiftRows, MixColumns, AddRoundKey

#### Data Encryption Standard (DES)
- **Block size**: 64 bits
- **Key size**: 56 bits (64 bits with parity)
- **Rounds**: 16
- **Status**: Deprecated due to small key size

### Stream Ciphers
**Definition**: A stream cipher encrypts plaintext by combining it with a pseudorandom stream.

#### RC4
- **Key size**: Variable (typically 40-256 bits)
- **Status**: Deprecated due to vulnerabilities

#### ChaCha20
- **Key size**: 256 bits
- **Nonce size**: 96 bits
- **Status**: Modern, secure stream cipher

### Modes of Operation

#### Electronic Codebook (ECB)
- **Description**: Each block encrypted independently
- **Problem**: Identical plaintext blocks produce identical ciphertext blocks
- **Use**: Not recommended for most applications

#### Cipher Block Chaining (CBC)
- **Description**: Each block XORed with previous ciphertext block
- **Initialization Vector (IV)**: Required for first block
- **Use**: Common for general-purpose encryption

#### Counter (CTR)
- **Description**: Encrypts counter values and XORs with plaintext
- **Advantage**: Parallelizable
- **Use**: High-performance applications

## 2. Public Key Cryptography

### Definition
**Public key cryptography** uses different keys for encryption and decryption.

### RSA Algorithm
**Key Generation**:
1. Choose two large primes p and q
2. Compute n = pq and φ(n) = (p-1)(q-1)
3. Choose e such that gcd(e, φ(n)) = 1
4. Find d such that ed ≡ 1 (mod φ(n))
5. Public key: (n, e), Private key: (n, d)

**Encryption**: c = m^e mod n
**Decryption**: m = c^d mod n

**Security**: Based on difficulty of factoring large integers

### Diffie-Hellman Key Exchange
**Protocol**:
1. Alice and Bob agree on public parameters (p, g)
2. Alice chooses private key a, computes A = g^a mod p
3. Bob chooses private key b, computes B = g^b mod p
4. Alice and Bob exchange A and B
5. Alice computes K = B^a mod p
6. Bob computes K = A^b mod p
7. Both have the same shared secret K

**Security**: Based on difficulty of discrete logarithm problem

### ElGamal Encryption
**Key Generation**:
1. Choose large prime p and generator g
2. Choose private key x
3. Compute public key y = g^x mod p

**Encryption**: (c₁, c₂) = (g^k mod p, m·y^k mod p)
**Decryption**: m = c₂·c₁^(-x) mod p

## 3. Hash Functions

### Definition
A **hash function** maps data of arbitrary size to a fixed-size output.

### Properties
1. **Deterministic**: Same input always produces same output
2. **Fixed output size**: Output size is constant
3. **Efficient**: Fast computation
4. **Preimage resistance**: Hard to find input for given output
5. **Second preimage resistance**: Hard to find different input with same output
6. **Collision resistance**: Hard to find two inputs with same output

### Common Hash Functions

#### SHA-1
- **Output size**: 160 bits
- **Status**: Deprecated due to collision attacks

#### SHA-256
- **Output size**: 256 bits
- **Status**: Widely used, secure

#### SHA-3 (Keccak)
- **Output size**: Variable (224, 256, 384, 512 bits)
- **Status**: Modern standard, secure

### Applications
- **Digital signatures**: Signing hash of message
- **Password storage**: Storing hashed passwords
- **Data integrity**: Verifying file integrity
- **Blockchain**: Cryptographic hashing

## 4. Digital Signatures

### Definition
A **digital signature** provides authentication, integrity, and non-repudiation for digital messages.

### RSA Digital Signature
**Signing**: s = m^d mod n
**Verification**: m = s^e mod n

### Digital Signature Algorithm (DSA)
**Key Generation**:
1. Choose prime p and q such that q | (p-1)
2. Choose generator g of order q
3. Choose private key x
4. Compute public key y = g^x mod p

**Signing**: (r, s) where r = (g^k mod p) mod q and s = k^(-1)(H(m) + xr) mod q
**Verification**: Check if r = (g^(s^(-1)H(m))y^(s^(-1)r) mod p) mod q

### Elliptic Curve Digital Signature Algorithm (ECDSA)
**Advantages**:
- Smaller key sizes
- Faster computation
- Same security level as RSA

## 5. Elliptic Curve Cryptography

### Elliptic Curves
**Definition**: An elliptic curve over field F is the set of points (x, y) satisfying y² = x³ + ax + b.

### Elliptic Curve Discrete Logarithm Problem (ECDLP)
**Problem**: Given points P and Q = kP, find k.

**Security**: Based on difficulty of ECDLP

### Elliptic Curve Diffie-Hellman (ECDH)
**Protocol**:
1. Alice and Bob agree on elliptic curve and base point G
2. Alice chooses private key a, computes A = aG
3. Bob chooses private key b, computes B = bG
4. Alice and Bob exchange A and B
5. Alice computes K = aB
6. Bob computes K = bA
7. Both have the same shared secret K

### Advantages
- **Smaller key sizes**: 256-bit ECC ≈ 3072-bit RSA
- **Faster computation**: Especially for signing
- **Lower power consumption**: Important for mobile devices

## 6. Quantum Cryptography

### Quantum Key Distribution (QKD)
**BB84 Protocol**:
1. Alice sends photons in random polarizations
2. Bob measures photons in random bases
3. Alice and Bob compare bases publicly
4. They keep only measurements with matching bases
5. They perform error correction and privacy amplification

**Security**: Based on quantum mechanics, not computational assumptions

### Post-Quantum Cryptography
**Lattice-based cryptography**: Based on difficulty of lattice problems
**Code-based cryptography**: Based on error-correcting codes
**Multivariate cryptography**: Based on solving systems of multivariate equations

## 7. Practice Problems

### Symmetric Cryptography
1. Implement AES encryption and decryption.

2. Compare ECB and CBC modes of operation.

3. Design a stream cipher using a linear feedback shift register.

### Public Key Cryptography
4. Generate RSA keys with p = 61, q = 53, e = 17.

5. Encrypt the message m = 65 using the public key from problem 4.

6. Implement Diffie-Hellman key exchange.

### Hash Functions
7. Implement SHA-256 hash function.

8. Find a collision for a simple hash function.

9. Design a hash function using the Merkle-Damgård construction.

### Digital Signatures
10. Implement RSA digital signature scheme.

11. Verify a DSA signature.

12. Compare RSA and ECDSA signature schemes.

### Elliptic Curve Cryptography
13. Find points on the elliptic curve y² = x³ + 2x + 3 over GF(7).

14. Implement elliptic curve point addition.

15. Generate ECDSA keys and sign a message.

### Quantum Cryptography
16. Simulate the BB84 protocol.

17. Analyze the security of quantum key distribution.

18. Compare classical and quantum cryptography.

## 8. Applications

### Computer Security
- **Secure communication**: HTTPS, TLS, SSH
- **Data protection**: Encryption at rest and in transit
- **Authentication**: Digital certificates, two-factor authentication
- **Access control**: Cryptographic access control

### Blockchain and Cryptocurrency
- **Bitcoin**: Elliptic curve digital signatures
- **Ethereum**: Cryptographic hashing and signatures
- **Smart contracts**: Cryptographic verification
- **Consensus mechanisms**: Proof of work, proof of stake

### Internet of Things (IoT)
- **Device authentication**: Cryptographic device identity
- **Secure communication**: Encrypted device communication
- **Data integrity**: Cryptographic data verification
- **Key management**: Secure key distribution

### Mobile Security
- **App security**: Code signing, certificate pinning
- **Mobile payments**: Cryptographic payment protocols
- **Biometric security**: Cryptographic biometric templates
- **Secure storage**: Encrypted local storage

## Key Takeaways

1. **Symmetric cryptography is fast**: But requires secure key distribution
2. **Public key cryptography is flexible**: But computationally expensive
3. **Hash functions are essential**: For integrity and authentication
4. **Digital signatures provide non-repudiation**: Crucial for legal applications
5. **Elliptic curves are efficient**: They provide same security with smaller keys
6. **Quantum cryptography is future-proof**: But requires quantum infrastructure
7. **Cryptography is constantly evolving**: New attacks and defenses emerge

## Next Steps
- Master symmetric and asymmetric cryptography
- Learn about hash functions and digital signatures
- Study elliptic curve cryptography
- Explore quantum cryptography
- Apply cryptographic techniques to secure systems
- Stay updated with cryptographic developments
- Connect cryptography to real-world security problems
