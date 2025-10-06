# Functions and Relations

## Overview
Functions and relations are fundamental concepts that build on set theory. Functions describe how elements from one set map to elements in another set, while relations describe relationships between elements of sets. These concepts are essential for understanding discrete mathematics and computer science.

## Learning Objectives
- Understand functions and their properties
- Learn about different types of functions (injective, surjective, bijective)
- Master function composition and inverse functions
- Understand relations and their properties
- Learn about equivalence relations and partial orders
- Apply these concepts to solve problems

## 1. Functions

### Definition
A **function** f from set A to set B (written f: A → B) is a relation that assigns to each element a ∈ A exactly one element b ∈ B.

**Notation:**
- f: A → B (f maps A to B)
- f(a) = b (f maps a to b)
- A is the **domain**
- B is the **codomain**
- The set {f(a) | a ∈ A} is the **range** or **image**

### Examples
1. f: ℝ → ℝ, f(x) = x²
2. g: ℤ → ℤ, g(n) = 2n + 1
3. h: {1, 2, 3} → {a, b, c}, h(1) = a, h(2) = b, h(3) = c

### Function Properties

#### One-to-One (Injective)
A function f: A → B is **injective** if f(a₁) = f(a₂) implies a₁ = a₂.

**Alternative definition**: Different inputs produce different outputs.

**Examples:**
- f(x) = 2x is injective
- g(x) = x² is not injective (g(2) = g(-2) = 4)

#### Onto (Surjective)
A function f: A → B is **surjective** if for every b ∈ B, there exists a ∈ A such that f(a) = b.

**Alternative definition**: Every element in the codomain is mapped to by some element in the domain.

**Examples:**
- f(x) = x + 1 is surjective (for any y, we can find x = y - 1)
- g(x) = x² is not surjective (no real number maps to -1)

#### Bijective (One-to-One and Onto)
A function f: A → B is **bijective** if it is both injective and surjective.

**Examples:**
- f(x) = x + 1 is bijective
- g(x) = x³ is bijective
- h(x) = x² is not bijective

### Function Composition

#### Definition
If f: A → B and g: B → C, then the **composition** g ∘ f: A → C is defined by (g ∘ f)(a) = g(f(a)).

**Properties:**
- Composition is associative: (h ∘ g) ∘ f = h ∘ (g ∘ f)
- Composition is not commutative in general

**Example:**
- f(x) = x + 1, g(x) = x²
- (g ∘ f)(x) = g(f(x)) = g(x + 1) = (x + 1)²
- (f ∘ g)(x) = f(g(x)) = f(x²) = x² + 1

### Inverse Functions

#### Definition
If f: A → B is bijective, then the **inverse function** f⁻¹: B → A is defined by f⁻¹(b) = a if and only if f(a) = b.

**Properties:**
- f⁻¹ ∘ f = id_A (identity function on A)
- f ∘ f⁻¹ = id_B (identity function on B)
- (f⁻¹)⁻¹ = f

**Example:**
- f(x) = 2x + 1 is bijective
- f⁻¹(x) = (x - 1)/2
- Check: f(f⁻¹(x)) = f((x-1)/2) = 2((x-1)/2) + 1 = x

## 2. Relations

### Definition
A **relation** R from set A to set B is a subset of A × B.

**Notation:**
- (a, b) ∈ R or a R b means "a is related to b"
- R ⊆ A × B

### Examples
1. R = {(1, 2), (2, 3), (3, 4)} on A = {1, 2, 3, 4}
2. "Less than" relation: R = {(a, b) | a < b} on ℝ
3. "Divides" relation: R = {(a, b) | a divides b} on ℤ⁺

### Properties of Relations

#### Reflexive
A relation R on set A is **reflexive** if (a, a) ∈ R for all a ∈ A.

**Examples:**
- "≤" on ℝ is reflexive
- "=" on any set is reflexive
- "<" on ℝ is not reflexive

#### Symmetric
A relation R on set A is **symmetric** if (a, b) ∈ R implies (b, a) ∈ R.

**Examples:**
- "=" on any set is symmetric
- "≤" on ℝ is not symmetric
- "is married to" is symmetric

#### Antisymmetric
A relation R on set A is **antisymmetric** if (a, b) ∈ R and (b, a) ∈ R implies a = b.

**Examples:**
- "≤" on ℝ is antisymmetric
- "=" on any set is antisymmetric
- "is married to" is not antisymmetric

#### Transitive
A relation R on set A is **transitive** if (a, b) ∈ R and (b, c) ∈ R implies (a, c) ∈ R.

**Examples:**
- "≤" on ℝ is transitive
- "=" on any set is transitive
- "is parent of" is not transitive

## 3. Special Types of Relations

### Equivalence Relations
A relation R on set A is an **equivalence relation** if it is reflexive, symmetric, and transitive.

**Examples:**
1. "=" on any set
2. "congruent modulo n" on ℤ: a ≡ b (mod n) if n | (a - b)
3. "has the same birthday as" on a set of people

### Equivalence Classes
If R is an equivalence relation on A, then for a ∈ A, the **equivalence class** of a is:
[a] = {b ∈ A | (a, b) ∈ R}

**Properties:**
- a ∈ [a] (reflexive)
- If b ∈ [a], then a ∈ [b] (symmetric)
- If b ∈ [a] and c ∈ [b], then c ∈ [a] (transitive)
- Either [a] = [b] or [a] ∩ [b] = ∅

### Partitions
A **partition** of set A is a collection of non-empty, disjoint subsets whose union is A.

**Theorem**: Every equivalence relation on A corresponds to a partition of A, and vice versa.

### Partial Orders
A relation R on set A is a **partial order** if it is reflexive, antisymmetric, and transitive.

**Examples:**
1. "≤" on ℝ
2. "⊆" on P(A) (power set)
3. "divides" on ℤ⁺

### Total Orders
A **total order** is a partial order where any two elements are comparable.

**Examples:**
1. "≤" on ℝ
2. Lexicographic order on strings

### Hasse Diagrams
A **Hasse diagram** is a visual representation of a partial order where:
- Elements are represented as points
- If a < b and there's no c such that a < c < b, draw a line from a to b
- Higher elements are drawn above lower elements

## 4. Applications

### In Computer Science
- **Functions**: Programming functions, algorithms
- **Relations**: Database relations, graph edges
- **Equivalence**: Object equality, state equivalence
- **Partial Orders**: Task scheduling, dependency graphs

### In Mathematics
- **Functions**: Calculus, analysis
- **Relations**: Graph theory, order theory
- **Equivalence**: Modular arithmetic, quotient spaces
- **Partial Orders**: Lattice theory, topology

## 5. Practice Problems

### Functions
1. Determine if the following functions are injective, surjective, or bijective:
   - f: ℝ → ℝ, f(x) = x²
   - g: ℝ → ℝ, g(x) = x³
   - h: ℤ → ℤ, h(n) = n + 1

2. Find the composition f ∘ g and g ∘ f for:
   - f(x) = 2x + 1, g(x) = x²

3. Find the inverse of f(x) = 3x - 2

### Relations
4. Determine which properties (reflexive, symmetric, antisymmetric, transitive) the following relations have:
   - R = {(1, 1), (2, 2), (3, 3)} on {1, 2, 3}
   - "divides" on ℤ⁺
   - "is perpendicular to" on lines in a plane

5. Show that "congruent modulo 3" is an equivalence relation on ℤ

6. Find the equivalence classes for the relation "has the same remainder when divided by 3" on ℤ

### Advanced Problems
7. Prove that if f: A → B and g: B → C are both bijective, then g ∘ f is bijective

8. Show that the composition of two equivalence relations is not necessarily an equivalence relation

9. Draw the Hasse diagram for the partial order "divides" on {1, 2, 3, 4, 6, 8, 12}

## 6. Common Pitfalls

1. **Confusing codomain and range**:
   - Codomain: target set
   - Range: actual values attained

2. **Function vs. relation**:
   - Functions: each input has exactly one output
   - Relations: inputs can have multiple outputs

3. **Inverse functions**:
   - Only bijective functions have inverses
   - f⁻¹ is not the same as 1/f

4. **Equivalence vs. equality**:
   - Equivalence relations generalize equality
   - Elements in the same equivalence class are equivalent but not necessarily equal

## Key Takeaways

1. **Functions are special relations**: They have unique outputs for each input
2. **Properties matter**: Injective, surjective, and bijective functions have different properties
3. **Composition is powerful**: It allows building complex functions from simple ones
4. **Relations generalize functions**: They can have multiple outputs for one input
5. **Equivalence relations partition sets**: They group elements into equivalence classes
6. **Partial orders model hierarchies**: They represent "less than or equal to" relationships

## Next Steps
- Master function composition and inverse functions
- Practice identifying relation properties
- Learn about equivalence classes and partitions
- Apply these concepts to combinatorics and graph theory
