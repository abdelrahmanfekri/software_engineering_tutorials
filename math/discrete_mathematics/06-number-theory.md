# Number Theory

## Overview
Number theory is the study of integers and their properties. It's one of the oldest branches of mathematics and has become crucial in modern cryptography, computer science, and algorithm design. This chapter covers the essential concepts in number theory with applications to computer science.

## Learning Objectives
- Understand divisibility and modular arithmetic
- Master the Euclidean algorithm for finding GCD
- Learn about prime numbers and factorization
- Understand the Chinese remainder theorem
- Learn Fermat's little theorem and its applications
- Understand RSA cryptography
- Solve Diophantine equations

## 1. Divisibility

### Definition
For integers a and b, we say **a divides b** (written a | b) if there exists an integer c such that b = ac.

### Properties of Divisibility
- If a | b and b | c, then a | c (transitivity)
- If a | b and a | c, then a | (bx + cy) for any integers x, y
- If a | b and b | a, then a = ±b
- If a | b and b ≠ 0, then |a| ≤ |b|

### Division Algorithm
For any integers a and b with b > 0, there exist unique integers q and r such that:
a = bq + r, where 0 ≤ r < b

**Example**: 17 = 5 × 3 + 2 (q = 3, r = 2)

## 2. Greatest Common Divisor (GCD)

### Definition
The **greatest common divisor** of integers a and b (not both zero) is the largest integer that divides both a and b.

**Notation**: gcd(a, b)

### Properties
- gcd(a, b) = gcd(b, a)
- gcd(a, b) = gcd(-a, b) = gcd(a, -b) = gcd(-a, -b)
- gcd(a, 0) = |a|
- gcd(a, b) = gcd(b, a mod b)

### Euclidean Algorithm
```
EUCLID(a, b):
    while b ≠ 0:
        r = a mod b
        a = b
        b = r
    return a
```

**Example**: Find gcd(48, 18)
- 48 = 18 × 2 + 12
- 18 = 12 × 1 + 6
- 12 = 6 × 2 + 0
- gcd(48, 18) = 6

### Extended Euclidean Algorithm
Finds integers x and y such that ax + by = gcd(a, b).

```
EXTENDED-EUCLID(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        (d, x, y) = EXTENDED-EUCLID(b, a mod b)
        return (d, y, x - (a // b) * y)
```

## 3. Modular Arithmetic

### Definition
For integers a, b, and positive integer n, we say **a is congruent to b modulo n** (written a ≡ b (mod n)) if n | (a - b).

### Properties
- a ≡ a (mod n) (reflexive)
- If a ≡ b (mod n), then b ≡ a (mod n) (symmetric)
- If a ≡ b (mod n) and b ≡ c (mod n), then a ≡ c (mod n) (transitive)

### Arithmetic Operations
- (a + b) mod n = ((a mod n) + (b mod n)) mod n
- (a - b) mod n = ((a mod n) - (b mod n)) mod n
- (a × b) mod n = ((a mod n) × (b mod n)) mod n

### Modular Exponentiation
To compute a^b mod n efficiently:

```
MODULAR-EXPONENTIATION(a, b, n):
    result = 1
    a = a mod n
    while b > 0:
        if b is odd:
            result = (result * a) mod n
        a = (a * a) mod n
        b = b // 2
    return result
```

## 4. Prime Numbers

### Definition
A **prime number** is a positive integer greater than 1 that has no positive divisors other than 1 and itself.

### Fundamental Theorem of Arithmetic
Every positive integer greater than 1 can be written uniquely as a product of primes.

**Example**: 60 = 2² × 3 × 5

### Primality Testing

#### Trial Division
Test divisibility by all primes up to √n.

#### Fermat's Test
If n is prime and gcd(a, n) = 1, then a^(n-1) ≡ 1 (mod n).

**Note**: This is a necessary but not sufficient condition.

#### Miller-Rabin Test
More sophisticated probabilistic primality test.

### Sieve of Eratosthenes
Finds all primes up to n:

```
SIEVE(n):
    is_prime = array of size n+1, all True
    is_prime[0] = is_prime[1] = False
    for i = 2 to √n:
        if is_prime[i]:
            for j = i*i to n, step i:
                is_prime[j] = False
    return all i where is_prime[i] is True
```

## 5. Chinese Remainder Theorem

### Statement
If n₁, n₂, ..., n_k are pairwise relatively prime integers, then the system of congruences:
- x ≡ a₁ (mod n₁)
- x ≡ a₂ (mod n₂)
- ...
- x ≡ a_k (mod n_k)

has a unique solution modulo N = n₁n₂...n_k.

### Algorithm
1. Compute N = n₁n₂...n_k
2. For each i, compute N_i = N/n_i
3. Find x_i such that N_i × x_i ≡ 1 (mod n_i)
4. The solution is x ≡ Σ(a_i × N_i × x_i) (mod N)

**Example**: Solve x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
- N = 3 × 5 × 7 = 105
- N₁ = 35, N₂ = 21, N₃ = 15
- 35 × 2 ≡ 1 (mod 3), 21 × 1 ≡ 1 (mod 5), 15 × 1 ≡ 1 (mod 7)
- x ≡ 2×35×2 + 3×21×1 + 2×15×1 ≡ 140 + 63 + 30 ≡ 23 (mod 105)

## 6. Fermat's Little Theorem

### Statement
If p is prime and gcd(a, p) = 1, then a^(p-1) ≡ 1 (mod p).

### Corollary
If p is prime, then a^p ≡ a (mod p) for any integer a.

### Applications
- Primality testing
- Modular exponentiation
- Cryptography

## 7. RSA Cryptography

### Key Generation
1. Choose two large primes p and q
2. Compute n = pq and φ(n) = (p-1)(q-1)
3. Choose e such that gcd(e, φ(n)) = 1
4. Find d such that ed ≡ 1 (mod φ(n))
5. Public key: (n, e), Private key: (n, d)

### Encryption
For message m: c = m^e mod n

### Decryption
For ciphertext c: m = c^d mod n

### Example
- p = 3, q = 11, n = 33, φ(33) = 20
- Choose e = 3, find d = 7 (3 × 7 ≡ 1 (mod 20))
- Public key: (33, 3), Private key: (33, 7)
- Encrypt m = 5: c = 5³ mod 33 = 26
- Decrypt c = 26: m = 26⁷ mod 33 = 5

## 8. Diophantine Equations

### Definition
A **Diophantine equation** is an equation where only integer solutions are sought.

### Linear Diophantine Equations
ax + by = c

**Solution**: The equation has solutions if and only if gcd(a, b) | c.

**Finding solutions**: Use the extended Euclidean algorithm.

**Example**: Solve 3x + 5y = 1
- gcd(3, 5) = 1, and 1 | 1, so solutions exist
- Extended Euclidean: 3 × 2 + 5 × (-1) = 1
- One solution: x₀ = 2, y₀ = -1
- General solution: x = 2 + 5t, y = -1 - 3t for any integer t

## 9. Practice Problems

### Divisibility and GCD
1. Find gcd(123, 456) using the Euclidean algorithm.

2. Find integers x and y such that 123x + 456y = gcd(123, 456).

3. Prove that if a | b and a | c, then a | (bx + cy) for any integers x, y.

### Modular Arithmetic
4. Compute 7^100 mod 13.

5. Find the last two digits of 7^100.

6. Solve 3x ≡ 7 (mod 11).

### Prime Numbers
7. Use the Sieve of Eratosthenes to find all primes up to 50.

8. Check if 97 is prime using trial division.

9. Find the prime factorization of 1001.

### Chinese Remainder Theorem
10. Solve the system:
    - x ≡ 1 (mod 3)
    - x ≡ 2 (mod 5)
    - x ≡ 3 (mod 7)

11. Find the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5, 6.

### RSA Cryptography
12. Generate RSA keys with p = 5, q = 7, e = 5.

13. Encrypt the message m = 3 using the public key from problem 12.

### Diophantine Equations
14. Solve 6x + 9y = 15.

15. Find all positive integer solutions to 2x + 3y = 20.

## 10. Applications

### Cryptography
- **RSA**: Public key cryptography
- **Diffie-Hellman**: Key exchange
- **Elliptic curves**: Modern cryptography
- **Hash functions**: Cryptographic hashing

### Computer Science
- **Random number generation**: Pseudorandom generators
- **Hash tables**: Hash function design
- **Error detection**: Checksums and error codes
- **Algorithm analysis**: Complexity analysis

### Mathematics
- **Algebraic structures**: Groups, rings, fields
- **Analytic number theory**: Prime distribution
- **Algebraic geometry**: Diophantine geometry
- **Combinatorics**: Counting problems

## Key Takeaways

1. **Divisibility is fundamental**: It's the basis for many number theory concepts
2. **GCD is powerful**: The Euclidean algorithm is efficient and widely used
3. **Modular arithmetic is essential**: It's the foundation of cryptography
4. **Primes are special**: They have unique properties and applications
5. **Chinese remainder theorem is useful**: It solves systems of congruences
6. **Fermat's little theorem is important**: It's used in primality testing and cryptography
7. **RSA is practical**: It demonstrates the power of number theory in cryptography
8. **Diophantine equations are challenging**: They require creative problem-solving

## Next Steps
- Master the Euclidean algorithm and its extensions
- Practice modular arithmetic and exponentiation
- Learn about prime number generation and testing
- Study the Chinese remainder theorem
- Understand RSA cryptography
- Solve various Diophantine equations
- Apply number theory to cryptography and algorithms
