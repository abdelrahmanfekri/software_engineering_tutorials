# Linear Algebra Tutorial 01: Vectors and Vector Spaces

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand vectors and their geometric representation
- Perform vector operations (addition, scalar multiplication)
- Understand linear combinations and spans
- Determine linear independence and dependence
- Find basis and dimension of vector spaces
- Work with subspaces

## Introduction to Vectors

### Definition
A vector is an ordered list of numbers (components) that represents both magnitude and direction.

**Notation**: 
- Column vector: v = [v₁, v₂, ..., vₙ]ᵀ
- Row vector: v = [v₁, v₂, ..., vₙ]

**Example**: v = [3, -2, 1]ᵀ represents a vector in ℝ³

### Geometric Representation
In ℝ² and ℝ³, vectors can be visualized as arrows from the origin to a point.

**Example**: v = [3, 4]ᵀ represents an arrow from (0,0) to (3,4)

## Vector Operations

### Vector Addition
Add corresponding components: u + v = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]

**Example**: [2, 3] + [1, -1] = [3, 2]

### Scalar Multiplication
Multiply each component by scalar: cv = [cv₁, cv₂, ..., cvₙ]

**Example**: 3[2, 1] = [6, 3]

### Properties of Vector Operations
1. **Commutativity**: u + v = v + u
2. **Associativity**: (u + v) + w = u + (v + w)
3. **Distributivity**: c(u + v) = cu + cv
4. **Identity**: v + 0 = v (where 0 is zero vector)
5. **Inverse**: v + (-v) = 0

## Linear Combinations

### Definition
A linear combination of vectors v₁, v₂, ..., vₖ is:
c₁v₁ + c₂v₂ + ... + cₖvₖ

Where c₁, c₂, ..., cₖ are scalars.

**Example**: If v₁ = [1, 0] and v₂ = [0, 1], then 2v₁ + 3v₂ = [2, 3]

### Span
The span of vectors v₁, v₂, ..., vₖ is the set of all linear combinations:
Span{v₁, v₂, ..., vₖ} = {c₁v₁ + c₂v₂ + ... + cₖvₖ : c₁, c₂, ..., cₖ ∈ ℝ}

**Example**: Span{[1, 0], [0, 1]} = ℝ² (all vectors in 2D space)

## Linear Independence and Dependence

### Definition
Vectors v₁, v₂, ..., vₖ are:
- **Linearly Independent**: If c₁v₁ + c₂v₂ + ... + cₖvₖ = 0 implies c₁ = c₂ = ... = cₖ = 0
- **Linearly Dependent**: If there exist scalars (not all zero) such that c₁v₁ + c₂v₂ + ... + cₖvₖ = 0

### Testing Linear Independence
Set up the equation c₁v₁ + c₂v₂ + ... + cₖvₖ = 0 and solve for c₁, c₂, ..., cₖ.

**Example**: Are [1, 2], [3, 6] linearly independent?
- c₁[1, 2] + c₂[3, 6] = [0, 0]
- c₁ + 3c₂ = 0 and 2c₁ + 6c₂ = 0
- From first equation: c₁ = -3c₂
- Substituting: 2(-3c₂) + 6c₂ = -6c₂ + 6c₂ = 0
- This holds for any c₂, so vectors are linearly dependent

## Vector Spaces

### Definition
A vector space V over a field F is a set with two operations (addition and scalar multiplication) satisfying:
1. **Closure**: u + v ∈ V, cv ∈ V
2. **Associativity**: (u + v) + w = u + (v + w)
3. **Commutativity**: u + v = v + u
4. **Identity**: v + 0 = v
5. **Inverse**: v + (-v) = 0
6. **Distributivity**: c(u + v) = cu + cv, (c + d)v = cv + dv
7. **Associativity**: (cd)v = c(dv)
8. **Identity**: 1v = v

### Examples of Vector Spaces
- **ℝⁿ**: n-dimensional real vectors
- **Polynomials**: Pₙ = {a₀ + a₁x + ... + aₙxⁿ : aᵢ ∈ ℝ}
- **Matrices**: Mₘₙ = m × n matrices
- **Functions**: C[a,b] = continuous functions on [a,b]

## Basis and Dimension

### Basis
A basis for vector space V is a linearly independent set that spans V.

**Example**: {[1, 0], [0, 1]} is a basis for ℝ²

### Standard Basis
The standard basis for ℝⁿ is:
e₁ = [1, 0, 0, ..., 0]ᵀ
e₂ = [0, 1, 0, ..., 0]ᵀ
...
eₙ = [0, 0, 0, ..., 1]ᵀ

### Dimension
The dimension of V is the number of vectors in any basis for V.

**Examples**:
- dim(ℝⁿ) = n
- dim(Pₙ) = n + 1
- dim(Mₘₙ) = mn

## Subspaces

### Definition
A subspace W of vector space V is a subset that is itself a vector space.

### Subspace Test
W is a subspace if:
1. 0 ∈ W
2. u, v ∈ W implies u + v ∈ W
3. u ∈ W, c ∈ F implies cu ∈ W

**Example**: Is W = {(x, y, z) : x + y + z = 0} a subspace of ℝ³?

**Solution**:
1. (0, 0, 0) ∈ W since 0 + 0 + 0 = 0 ✓
2. If (x₁, y₁, z₁), (x₂, y₂, z₂) ∈ W, then (x₁ + x₂) + (y₁ + y₂) + (z₁ + z₂) = 0 + 0 = 0 ✓
3. If (x, y, z) ∈ W, then c(x, y, z) = (cx, cy, cz) and cx + cy + cz = c(x + y + z) = c(0) = 0 ✓
Therefore, W is a subspace.

## Applications

### Computer Graphics
Vectors represent positions, directions, and transformations in 3D graphics.

### Physics
Vectors represent forces, velocities, and accelerations.

### Data Science
Vectors represent data points in high-dimensional spaces.

## Practice Problems

### Problem 1
Find the linear combination 2v₁ - 3v₂ + v₃ where:
v₁ = [1, 0, 2], v₂ = [0, 1, -1], v₃ = [2, 1, 0]

**Solution**:
2[1, 0, 2] - 3[0, 1, -1] + [2, 1, 0] = [2, 0, 4] + [0, -3, 3] + [2, 1, 0] = [4, -2, 7]

### Problem 2
Determine if [1, 2, 3], [2, 1, 0], [1, 1, 1] are linearly independent.

**Solution**:
Set c₁[1, 2, 3] + c₂[2, 1, 0] + c₃[1, 1, 1] = [0, 0, 0]
This gives system:
c₁ + 2c₂ + c₃ = 0
2c₁ + c₂ + c₃ = 0
3c₁ + c₃ = 0

From third equation: c₃ = -3c₁
Substituting into first two equations and solving: c₁ = c₂ = c₃ = 0
Therefore, vectors are linearly independent.

### Problem 3
Find a basis for the subspace W = {(x, y, z) : x - 2y + z = 0}.

**Solution**:
From x - 2y + z = 0, we get z = -x + 2y
So (x, y, z) = (x, y, -x + 2y) = x(1, 0, -1) + y(0, 1, 2)
A basis is {(1, 0, -1), (0, 1, 2)}

## Key Takeaways
- Vectors represent both magnitude and direction
- Vector operations follow specific algebraic rules
- Linear combinations create new vectors from existing ones
- Linear independence is crucial for basis formation
- Vector spaces provide the framework for linear algebra
- Subspaces inherit vector space properties

## Next Steps
In the next tutorial, we'll explore matrices and matrix operations, learning how to perform algebraic operations on matrices and understand their properties.
