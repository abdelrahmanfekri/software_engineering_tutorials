# Logic and Proof Techniques

## Overview
Logic and proof techniques form the foundation of discrete mathematics and computer science. This chapter covers propositional logic, predicate logic, and various methods of mathematical proof.

## Learning Objectives
- Understand propositional and predicate logic
- Master logical connectives and truth tables
- Learn different proof techniques
- Apply mathematical induction effectively
- Recognize logical equivalence and implications

## 1. Propositional Logic

### Basic Concepts
**Proposition**: A declarative statement that is either true or false.

Examples:
- "The sky is blue" (proposition)
- "What time is it?" (not a proposition - it's a question)
- "x + 1 = 5" (not a proposition - depends on value of x)

### Logical Connectives

#### Negation (¬)
- Symbol: ¬p or ~p
- Truth table:
  | p | ¬p |
  |---|---|
  | T | F  |
  | F | T  |

#### Conjunction (∧)
- Symbol: p ∧ q
- Read as "p and q"
- Truth table:
  | p | q | p ∧ q |
  |---|---|---|
  | T | T |   T   |
  | T | F |   F   |
  | F | T |   F   |
  | F | F |   F   |

#### Disjunction (∨)
- Symbol: p ∨ q
- Read as "p or q" (inclusive or)
- Truth table:
  | p | q | p ∨ q |
  |---|---|---|
  | T | T |   T   |
  | T | F |   T   |
  | F | T |   T   |
  | F | F |   F   |

#### Implication (→)
- Symbol: p → q
- Read as "if p then q" or "p implies q"
- Truth table:
  | p | q | p → q |
  |---|---|---|
  | T | T |   T   |
  | T | F |   F   |
  | F | T |   T   |
  | F | F |   T   |

#### Biconditional (↔)
- Symbol: p ↔ q
- Read as "p if and only if q"
- Truth table:
  | p | q | p ↔ q |
  |---|---|---|
  | T | T |   T   |
  | T | F |   F   |
  | F | T |   F   |
  | F | F |   T   |

### Logical Equivalence
Two propositions are logically equivalent if they have the same truth table.

**Important Equivalences:**
- De Morgan's Laws:
  - ¬(p ∧ q) ≡ ¬p ∨ ¬q
  - ¬(p ∨ q) ≡ ¬p ∧ ¬q
- Double Negation: ¬(¬p) ≡ p
- Commutative Laws:
  - p ∧ q ≡ q ∧ p
  - p ∨ q ≡ q ∨ p
- Associative Laws:
  - (p ∧ q) ∧ r ≡ p ∧ (q ∧ r)
  - (p ∨ q) ∨ r ≡ p ∨ (q ∨ r)
- Distributive Laws:
  - p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)
  - p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r)

## 2. Predicate Logic

### Quantifiers

#### Universal Quantifier (∀)
- Symbol: ∀x P(x)
- Read as "for all x, P(x)"
- True if P(x) is true for every value of x in the domain

#### Existential Quantifier (∃)
- Symbol: ∃x P(x)
- Read as "there exists an x such that P(x)"
- True if P(x) is true for at least one value of x in the domain

### Negating Quantifiers
- ¬(∀x P(x)) ≡ ∃x ¬P(x)
- ¬(∃x P(x)) ≡ ∀x ¬P(x)

## 3. Methods of Proof

### Direct Proof
To prove p → q:
1. Assume p is true
2. Use logical reasoning to show q must be true

**Example**: Prove "If n is even, then n² is even"
- Assume n is even, so n = 2k for some integer k
- Then n² = (2k)² = 4k² = 2(2k²)
- Since 2k² is an integer, n² is even

### Proof by Contrapositive
To prove p → q, prove ¬q → ¬p instead.

**Example**: Prove "If n² is odd, then n is odd"
- Contrapositive: "If n is even, then n² is even" (proven above)
- Therefore, the original statement is true

### Proof by Contradiction
To prove p:
1. Assume ¬p is true
2. Show this leads to a contradiction
3. Conclude p must be true

**Example**: Prove "√2 is irrational"
- Assume √2 is rational, so √2 = a/b where a,b are integers with no common factors
- Then 2 = a²/b², so 2b² = a²
- This means a² is even, so a is even
- Let a = 2k, then 2b² = (2k)² = 4k², so b² = 2k²
- This means b² is even, so b is even
- But if both a and b are even, they have a common factor of 2
- This contradicts our assumption that a and b have no common factors
- Therefore, √2 is irrational

### Mathematical Induction

#### Principle of Mathematical Induction
To prove ∀n ∈ ℕ, P(n):
1. **Base Case**: Prove P(1) is true
2. **Inductive Step**: Prove ∀k ∈ ℕ, P(k) → P(k+1)
3. **Conclusion**: By induction, P(n) is true for all n ∈ ℕ

**Example**: Prove 1 + 2 + 3 + ... + n = n(n+1)/2

**Base Case**: n = 1
- Left side: 1
- Right side: 1(1+1)/2 = 1
- ✓ Base case holds

**Inductive Step**: Assume 1 + 2 + ... + k = k(k+1)/2
- Show: 1 + 2 + ... + k + (k+1) = (k+1)(k+2)/2
- Left side: k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2
- ✓ Inductive step holds

**Conclusion**: By induction, the formula holds for all n ∈ ℕ

#### Strong Induction
To prove ∀n ∈ ℕ, P(n):
1. **Base Case**: Prove P(1) is true
2. **Inductive Step**: Prove ∀k ∈ ℕ, (P(1) ∧ P(2) ∧ ... ∧ P(k)) → P(k+1)
3. **Conclusion**: By strong induction, P(n) is true for all n ∈ ℕ

## 4. Common Proof Strategies

### Existence Proofs
- **Constructive**: Show how to construct the object
- **Non-constructive**: Show the object exists without constructing it

### Uniqueness Proofs
1. Show existence
2. Show uniqueness (often by contradiction)

### Proof by Cases
Break the problem into mutually exclusive cases and prove each case separately.

## 5. Practice Problems

### Basic Logic
1. Construct truth tables for:
   - (p → q) ∧ (q → r)
   - ¬(p ∨ q) → (¬p ∧ ¬q)
   - (p ∧ q) ∨ (¬p ∧ ¬q)

2. Prove the following logical equivalences:
   - p → q ≡ ¬p ∨ q
   - p ↔ q ≡ (p → q) ∧ (q → p)

### Proof Techniques
3. Prove by direct proof: "If a and b are odd integers, then a + b is even"

4. Prove by contrapositive: "If n² is even, then n is even"

5. Prove by contradiction: "There are infinitely many prime numbers"

6. Prove by induction: "2ⁿ > n for all n ≥ 1"

### Predicate Logic
7. Translate to English:
   - ∀x (Student(x) → Takes(x, Math))
   - ∃x (Student(x) ∧ Takes(x, CS))

8. Negate the following:
   - ∀x ∃y (x < y)
   - ∃x ∀y (x + y = 0)

## 6. Applications in Computer Science

### Boolean Logic in Programming
- Conditional statements (if-then-else)
- Loop conditions
- Boolean operators (&&, ||, !)

### Formal Verification
- Proving program correctness
- Model checking
- Theorem proving

### Database Query Logic
- SQL WHERE clauses
- Relational algebra
- Query optimization

## Key Takeaways

1. **Logic is fundamental**: All mathematical reasoning is based on logical principles
2. **Proof techniques are tools**: Different problems require different proof strategies
3. **Practice is essential**: Work through many examples to develop intuition
4. **Applications are everywhere**: Logic appears in programming, databases, and AI

## Next Steps
- Master these concepts before moving to Set Theory
- Practice with increasingly complex proofs
- Learn to recognize which proof technique to use
- Connect logic to programming and computer science applications
