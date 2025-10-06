# Set Theory

## Overview
Set theory is the foundation of modern mathematics and provides the language for describing collections of objects. This chapter covers sets, set operations, and their properties, which are essential for understanding functions, relations, and other discrete mathematics concepts.

## Learning Objectives
- Understand sets and set notation
- Master set operations and their properties
- Learn about Venn diagrams and set visualization
- Understand cardinality and infinite sets
- Apply set theory to solve problems

## 1. Basic Concepts

### Definition of a Set
A **set** is a collection of distinct objects called **elements** or **members**.

**Notation:**
- A = {1, 2, 3, 4, 5}
- B = {x | x is an even integer}
- C = {a, b, c}

### Set Membership
- x ∈ A means "x is an element of A"
- x ∉ A means "x is not an element of A"

### Special Sets
- **Empty Set**: ∅ = {} (set with no elements)
- **Universal Set**: U (the set containing all elements under consideration)
- **Natural Numbers**: ℕ = {1, 2, 3, 4, ...}
- **Integers**: ℤ = {..., -2, -1, 0, 1, 2, ...}
- **Rational Numbers**: ℚ = {p/q | p, q ∈ ℤ, q ≠ 0}
- **Real Numbers**: ℝ
- **Complex Numbers**: ℂ

### Set Equality
Two sets A and B are equal (A = B) if they contain exactly the same elements.

## 2. Set Operations

### Union (∪)
A ∪ B = {x | x ∈ A or x ∈ B}

**Properties:**
- A ∪ B = B ∪ A (commutative)
- A ∪ (B ∪ C) = (A ∪ B) ∪ C (associative)
- A ∪ ∅ = A (identity)
- A ∪ A = A (idempotent)

### Intersection (∩)
A ∩ B = {x | x ∈ A and x ∈ B}

**Properties:**
- A ∩ B = B ∩ A (commutative)
- A ∩ (B ∩ C) = (A ∩ B) ∩ C (associative)
- A ∩ U = A (identity)
- A ∩ A = A (idempotent)

### Complement
A' = {x | x ∈ U and x ∉ A}

**Properties:**
- (A')' = A (double complement)
- A ∪ A' = U
- A ∩ A' = ∅

### Difference
A - B = {x | x ∈ A and x ∉ B}

### Symmetric Difference
A △ B = (A - B) ∪ (B - A) = (A ∪ B) - (A ∩ B)

## 3. Set Identities (Laws)

### De Morgan's Laws
- (A ∪ B)' = A' ∩ B'
- (A ∩ B)' = A' ∪ B'

### Distributive Laws
- A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
- A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)

### Absorption Laws
- A ∪ (A ∩ B) = A
- A ∩ (A ∪ B) = A

### Associative Laws
- A ∪ (B ∪ C) = (A ∪ B) ∪ C
- A ∩ (B ∩ C) = (A ∩ B) ∩ C

### Commutative Laws
- A ∪ B = B ∪ A
- A ∩ B = B ∩ A

## 4. Venn Diagrams

Venn diagrams are visual representations of sets and their relationships.

### Basic Venn Diagram
```
    U
    ┌─────────────┐
    │  A      B   │
    │  ┌─┐   ┌─┐  │
    │  │ │   │ │  │
    │  └─┘   └─┘  │
    └─────────────┘
```

### Union A ∪ B
```
    U
    ┌─────────────┐
    │  A      B   │
    │  ┌─────┐    │
    │  │     │    │
    │  └─────┘    │
    └─────────────┘
```

### Intersection A ∩ B
```
    U
    ┌─────────────┐
    │  A      B   │
    │  ┌─┐   ┌─┐  │
    │  │ │   │ │  │
    │  └─┘   └─┘  │
    └─────────────┘
```

## 5. Cartesian Products

### Definition
A × B = {(a, b) | a ∈ A and b ∈ B}

**Example:**
- A = {1, 2}, B = {a, b}
- A × B = {(1, a), (1, b), (2, a), (2, b)}

### Properties
- |A × B| = |A| × |B|
- A × ∅ = ∅
- A × (B ∪ C) = (A × B) ∪ (A × C)
- A × (B ∩ C) = (A × B) ∩ (A × C)

## 6. Power Sets

### Definition
P(A) = {B | B ⊆ A}

**Example:**
- A = {1, 2}
- P(A) = {∅, {1}, {2}, {1, 2}}

### Properties
- |P(A)| = 2^|A|
- ∅ ∈ P(A) and A ∈ P(A)
- If A ⊆ B, then P(A) ⊆ P(B)

## 7. Cardinality

### Finite Sets
For a finite set A, |A| is the number of elements in A.

### Infinite Sets
- **Countably Infinite**: Can be put in one-to-one correspondence with ℕ
- **Uncountably Infinite**: Cannot be put in one-to-one correspondence with ℕ

### Examples
- |ℕ| = ℵ₀ (aleph-null)
- |ℤ| = ℵ₀
- |ℚ| = ℵ₀
- |ℝ| = c (continuum)

### Cantor's Theorem
For any set A, |A| < |P(A)|

## 8. Set Theory Applications

### In Computer Science
- **Data Structures**: Arrays, lists, sets, maps
- **Database Theory**: Relations, queries
- **Algorithm Analysis**: Input/output spaces
- **Formal Languages**: Alphabets, strings

### In Mathematics
- **Functions**: Domain and codomain
- **Relations**: Binary relations
- **Topology**: Open and closed sets
- **Measure Theory**: Measurable sets

## 9. Practice Problems

### Basic Operations
1. Let A = {1, 2, 3, 4, 5}, B = {3, 4, 5, 6, 7}, C = {2, 4, 6, 8}
   - Find A ∪ B
   - Find A ∩ B
   - Find A - B
   - Find A △ B

2. Prove the distributive law: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)

### Venn Diagrams
3. Draw Venn diagrams for:
   - A ∩ (B ∪ C)
   - (A ∪ B) ∩ (A ∪ C)
   - A' ∩ B'

### Cardinality
4. If |A| = 5 and |B| = 3, find:
   - |A × B|
   - |P(A)|
   - |A ∪ B| (assuming A ∩ B = ∅)

### Power Sets
5. Find P({a, b, c})

6. Prove that if A ⊆ B, then P(A) ⊆ P(B)

### Advanced Problems
7. Prove De Morgan's laws using set theory

8. Show that A △ B = (A ∪ B) - (A ∩ B)

9. If A and B are finite sets, prove:
   |A ∪ B| = |A| + |B| - |A ∩ B|

## 10. Common Pitfalls

1. **Confusing ∈ and ⊆**:
   - 1 ∈ {1, 2, 3} ✓
   - {1} ⊆ {1, 2, 3} ✓
   - 1 ⊆ {1, 2, 3} ✗

2. **Empty set properties**:
   - ∅ ⊆ A for any set A
   - ∅ ∈ P(A) for any set A
   - ∅ ≠ {∅}

3. **Set vs. multiset**:
   - Sets don't allow duplicates
   - {1, 1, 2} = {1, 2}

## Key Takeaways

1. **Sets are fundamental**: They provide the language for describing collections
2. **Operations have properties**: Learn the laws and use them in proofs
3. **Visualization helps**: Use Venn diagrams to understand relationships
4. **Cardinality matters**: Understanding size is crucial for applications
5. **Practice with examples**: Work through concrete examples to build intuition

## Next Steps
- Master set operations and their properties
- Practice with Venn diagrams
- Learn about functions and relations (which build on sets)
- Apply set theory to counting problems
