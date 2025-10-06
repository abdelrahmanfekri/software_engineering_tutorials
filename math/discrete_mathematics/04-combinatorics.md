# Combinatorics

## Overview
Combinatorics is the branch of mathematics concerned with counting, arrangement, and selection of objects. It provides powerful tools for solving problems involving discrete structures and is fundamental to computer science, probability, and optimization.

## Learning Objectives
- Master basic counting principles
- Understand permutations and combinations
- Learn the pigeonhole principle
- Apply the inclusion-exclusion principle
- Work with generating functions
- Solve recurrence relations
- Recognize special sequences

## 1. Basic Counting Principles

### Addition Principle
If task A can be performed in m ways and task B can be performed in n ways, and the tasks cannot be performed simultaneously, then either task A or task B can be performed in m + n ways.

**Example**: A student can choose between 3 math courses or 4 science courses. Total choices = 3 + 4 = 7.

### Multiplication Principle
If task A can be performed in m ways and task B can be performed in n ways, then both tasks can be performed in m × n ways.

**Example**: A meal consists of 3 appetizers and 4 main courses. Total meal combinations = 3 × 4 = 12.

### General Counting Principle
If a process can be broken into k steps, where step i can be performed in n_i ways, then the entire process can be performed in n₁ × n₂ × ... × n_k ways.

## 2. Permutations

### Definition
A **permutation** is an arrangement of objects in a specific order.

### Permutations of n distinct objects
The number of ways to arrange n distinct objects is n! (n factorial).

**Formula**: P(n, n) = n! = n × (n-1) × ... × 2 × 1

**Example**: Arranging 3 books on a shelf: 3! = 6 ways.

### Permutations of r objects from n distinct objects
The number of ways to arrange r objects from n distinct objects is:

**Formula**: P(n, r) = n!/(n-r)! = n × (n-1) × ... × (n-r+1)

**Example**: Choosing and arranging 2 books from 5: P(5, 2) = 5!/(5-2)! = 5 × 4 = 20.

### Permutations with repetition
If we have n objects with n₁ of type 1, n₂ of type 2, ..., n_k of type k, then the number of distinct permutations is:

**Formula**: n!/(n₁! × n₂! × ... × n_k!)

**Example**: Arranging the letters in "MISSISSIPPI":
- Total letters: 11
- M: 1, I: 4, S: 4, P: 2
- Number of arrangements: 11!/(1! × 4! × 4! × 2!) = 34,650

## 3. Combinations

### Definition
A **combination** is a selection of objects where order doesn't matter.

### Combinations of r objects from n distinct objects
The number of ways to choose r objects from n distinct objects is:

**Formula**: C(n, r) = n!/(r!(n-r)!) = (n choose r)

**Properties:**
- C(n, r) = C(n, n-r) (symmetry)
- C(n, 0) = C(n, n) = 1
- C(n, 1) = C(n, n-1) = n

**Example**: Choosing 3 books from 5: C(5, 3) = 5!/(3!2!) = 10.

### Pascal's Triangle
```
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```

**Properties:**
- Each entry is the sum of the two entries above it
- C(n, r) appears in row n, position r
- Sum of row n is 2^n

## 4. The Pigeonhole Principle

### Basic Principle
If n + 1 or more objects are placed into n boxes, then at least one box contains more than one object.

### Generalized Principle
If n objects are placed into k boxes, then at least one box contains at least ⌈n/k⌉ objects.

**Examples:**
1. In a group of 13 people, at least 2 have the same birthday month.
2. In a sequence of 10 numbers from 1 to 9, at least one number appears twice.
3. In a group of 6 people, either 3 are mutual friends or 3 are mutual strangers.

## 5. Inclusion-Exclusion Principle

### Two Sets
|A ∪ B| = |A| + |B| - |A ∩ B|

### Three Sets
|A ∪ B ∪ C| = |A| + |B| + |C| - |A ∩ B| - |A ∩ C| - |B ∩ C| + |A ∩ B ∩ C|

### General Form
For sets A₁, A₂, ..., A_n:
|A₁ ∪ A₂ ∪ ... ∪ A_n| = Σ|A_i| - Σ|A_i ∩ A_j| + Σ|A_i ∩ A_j ∩ A_k| - ... + (-1)^(n+1)|A₁ ∩ A₂ ∩ ... ∩ A_n|

**Example**: How many integers from 1 to 100 are divisible by 2, 3, or 5?
- Let A = {divisible by 2}, B = {divisible by 3}, C = {divisible by 5}
- |A| = 50, |B| = 33, |C| = 20
- |A ∩ B| = 16, |A ∩ C| = 10, |B ∩ C| = 6
- |A ∩ B ∩ C| = 3
- |A ∪ B ∪ C| = 50 + 33 + 20 - 16 - 10 - 6 + 3 = 74

## 6. Generating Functions

### Definition
A **generating function** is a formal power series that encodes information about a sequence.

### Ordinary Generating Function
For sequence {a_n}, the ordinary generating function is:
G(x) = Σ(a_n × x^n)

**Examples:**
1. {1, 1, 1, ...} → G(x) = 1 + x + x² + ... = 1/(1-x)
2. {1, 2, 3, 4, ...} → G(x) = 1 + 2x + 3x² + ... = 1/(1-x)²
3. {1, 1, 1, 1, 0, 0, ...} → G(x) = 1 + x + x² + x³ = (1-x⁴)/(1-x)

### Applications
- Counting problems
- Solving recurrence relations
- Finding closed forms for sequences

## 7. Recurrence Relations

### Definition
A **recurrence relation** defines a sequence by relating each term to previous terms.

### Linear Recurrence Relations
A recurrence relation of the form:
a_n = c₁a_{n-1} + c₂a_{n-2} + ... + c_ka_{n-k} + f(n)

### Examples
1. **Fibonacci**: F_n = F_{n-1} + F_{n-2}, F₀ = 0, F₁ = 1
2. **Towers of Hanoi**: T_n = 2T_{n-1} + 1, T₁ = 1
3. **Binary strings without consecutive 1s**: a_n = a_{n-1} + a_{n-2}

### Solving Recurrence Relations
1. **Characteristic equation method** (for linear homogeneous)
2. **Generating functions**
3. **Substitution method**
4. **Master theorem** (for divide-and-conquer recurrences)

## 8. Special Sequences

### Catalan Numbers
C_n = (1/(n+1)) × C(2n, n) = (2n)!/((n+1)!n!)

**First few values**: 1, 1, 2, 5, 14, 42, 132, ...

**Applications**:
- Binary tree counting
- Parentheses matching
- Polygon triangulation
- Lattice paths

### Stirling Numbers
- **First kind**: s(n, k) - permutations with k cycles
- **Second kind**: S(n, k) - partitions of n objects into k non-empty subsets

### Bell Numbers
B_n = Σ S(n, k) for k = 0 to n

**First few values**: 1, 1, 2, 5, 15, 52, 203, ...

**Applications**: Set partitions, equivalence relations

## 9. Practice Problems

### Basic Counting
1. How many ways can 5 people sit in a row?
2. How many ways can 3 people be chosen from 10?
3. How many ways can 4 books be arranged on a shelf if 2 are identical?

### Permutations and Combinations
4. A committee of 5 is to be chosen from 12 people. How many ways if:
   - No restrictions?
   - 2 specific people must be included?
   - 2 specific people cannot both be included?

5. How many ways can the letters in "MATHEMATICS" be arranged?

### Pigeonhole Principle
6. Prove that in any group of 6 people, either 3 are mutual friends or 3 are mutual strangers.

7. Show that in any sequence of 101 integers, there's a subsequence of 11 integers that is either strictly increasing or strictly decreasing.

### Inclusion-Exclusion
8. How many integers from 1 to 1000 are divisible by 2, 3, or 5?

9. In a class of 30 students, 18 take math, 15 take science, and 8 take both. How many take neither?

### Recurrence Relations
10. Solve: a_n = 3a_{n-1} - 2a_{n-2}, a₀ = 1, a₁ = 2

11. Find a closed form for the Fibonacci sequence.

### Advanced Problems
12. How many ways can n identical balls be placed in k distinct boxes?

13. Find the number of ways to partition a set of n elements.

## 10. Applications in Computer Science

### Algorithm Analysis
- Counting operations
- Analyzing complexity
- Comparing algorithms

### Data Structures
- Tree counting
- Graph enumeration
- Hash table analysis

### Cryptography
- Key counting
- Password analysis
- Random number generation

### Optimization
- Resource allocation
- Scheduling problems
- Network design

## Key Takeaways

1. **Counting is fundamental**: Many problems reduce to counting
2. **Order matters for permutations**: Order doesn't matter for combinations
3. **Pigeonhole principle is powerful**: It provides existence proofs
4. **Inclusion-exclusion handles overlaps**: It's essential for complex counting
5. **Generating functions encode sequences**: They're powerful problem-solving tools
6. **Recurrence relations model processes**: They're common in algorithms
7. **Special sequences appear everywhere**: Learn to recognize them

## Next Steps
- Master the basic counting principles
- Practice with permutations and combinations
- Learn to apply the pigeonhole principle
- Work with inclusion-exclusion problems
- Study generating functions and recurrence relations
- Apply combinatorics to graph theory and probability
