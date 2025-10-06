# Linear Algebra Tutorial 11: Vector Spaces over Fields

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand general field theory and its properties
- Work with finite fields (Galois fields)
- Understand complex vector spaces
- Apply field theory to coding theory
- Explore advanced algebraic structures

## Introduction to Fields

### Definition
A field F is a set with two operations (addition + and multiplication ·) satisfying:
1. **Additive group**: (F, +) is an abelian group
2. **Multiplicative group**: (F*, ·) is an abelian group (where F* = F - {0})
3. **Distributivity**: a(b + c) = ab + ac

### Examples of Fields
- **ℝ**: Real numbers
- **ℚ**: Rational numbers
- **ℂ**: Complex numbers
- **ℤₚ**: Integers modulo prime p
- **GF(pⁿ)**: Galois fields

## Finite Fields (Galois Fields)

### Prime Fields ℤₚ
For prime p, ℤₚ = {0, 1, 2, ..., p-1} with operations modulo p.

### Example: ℤ₅
Addition table:
+ | 0 1 2 3 4
--|----------
0 | 0 1 2 3 4
1 | 1 2 3 4 0
2 | 2 3 4 0 1
3 | 3 4 0 1 2
4 | 4 0 1 2 3

Multiplication table:
× | 0 1 2 3 4
--|----------
0 | 0 0 0 0 0
1 | 0 1 2 3 4
2 | 0 2 4 1 3
3 | 0 3 1 4 2
4 | 0 4 3 2 1

### Galois Fields GF(pⁿ)
For prime p and positive integer n, GF(pⁿ) has pⁿ elements.

### Construction of GF(4)
GF(4) = {0, 1, α, α+1} where α² = α + 1

Addition table:
+ | 0 1 α α+1
--|-----------
0 | 0 1 α α+1
1 | 1 0 α+1 α
α | α α+1 0 1
α+1| α+1 α 1 0

Multiplication table:
× | 0 1 α α+1
--|-----------
0 | 0 0 0 0
1 | 0 1 α α+1
α | 0 α α+1 1
α+1| 0 α+1 1 α

## Complex Vector Spaces

### Definition
A complex vector space V over ℂ is a set with operations satisfying the same axioms as real vector spaces, but with scalars from ℂ.

### Examples
- **ℂⁿ**: n-tuples of complex numbers
- **Polynomials**: Pₙ(ℂ) = {a₀ + a₁z + ... + aₙzⁿ : aᵢ ∈ ℂ}
- **Functions**: C[0,1] = continuous functions f: [0,1] → ℂ

### Complex Inner Products
For u, v ∈ ℂⁿ:
⟨u,v⟩ = u₁v̄₁ + u₂v̄₂ + ... + uₙv̄ₙ

Where v̄ᵢ is the complex conjugate of vᵢ.

### Properties
1. **Conjugate symmetry**: ⟨u,v⟩ = ⟨v,u⟩̄
2. **Linearity**: ⟨u + v,w⟩ = ⟨u,w⟩ + ⟨v,w⟩
3. **Scalar multiplication**: ⟨cu,v⟩ = c⟨u,v⟩
4. **Positive definiteness**: ⟨v,v⟩ ≥ 0, with equality iff v = 0

### Example
u = [1+i, 2-i], v = [3, 1+2i]
⟨u,v⟩ = (1+i)(3̄) + (2-i)(1+2ī)
      = (1+i)(3) + (2-i)(1-2i)
      = 3+3i + (2-i)(1-2i)
      = 3+3i + 2-4i-i+2i²
      = 3+3i + 2-5i-2
      = 3-2i

## Vector Spaces over General Fields

### Definition
A vector space V over field F is a set with operations satisfying:
1. **Closure**: u + v ∈ V, cv ∈ V
2. **Associativity**: (u + v) + w = u + (v + w)
3. **Commutativity**: u + v = v + u
4. **Identity**: v + 0 = v
5. **Inverse**: v + (-v) = 0
6. **Distributivity**: c(u + v) = cu + cv, (c + d)v = cv + dv
7. **Associativity**: (cd)v = c(dv)
8. **Identity**: 1v = v

### Examples over Different Fields
- **ℝ³**: 3D real vectors
- **ℂ²**: 2D complex vectors
- **ℤ₃²**: 2D vectors over ℤ₃
- **GF(4)³**: 3D vectors over GF(4)

## Linear Independence over General Fields

### Definition
Vectors v₁, v₂, ..., vₖ are linearly independent over field F if:
c₁v₁ + c₂v₂ + ... + cₖvₖ = 0 implies c₁ = c₂ = ... = cₖ = 0

### Example over ℤ₃
Are [1,2] and [2,1] linearly independent over ℤ₃?

**Solution**:
c₁[1,2] + c₂[2,1] = [0,0]
This gives: c₁ + 2c₂ = 0 and 2c₁ + c₂ = 0

From first equation: c₁ = -2c₂ = c₂ (since -2 ≡ 1 mod 3)
From second equation: 2c₁ + c₂ = 0

Substituting: 2c₂ + c₂ = 3c₂ = 0
Since 3 ≡ 0 mod 3, this holds for any c₂ ∈ ℤ₃

The vectors are linearly dependent.

## Applications in Coding Theory

### Error-Correcting Codes
Vector spaces over finite fields are used to construct error-correcting codes.

### Linear Codes
A linear code C over field F is a subspace of Fⁿ.

### Example: Hamming Code over ℤ₂
The (7,4) Hamming code is a 4-dimensional subspace of ℤ₂⁷.

Generator matrix:
G = [1 0 0 0 1 1 0]
    [0 1 0 0 1 0 1]
    [0 0 1 0 0 1 1]
    [0 0 0 1 1 1 1]

### Syndrome Decoding
For received vector r, compute syndrome s = rHᵀ where H is parity-check matrix.

## Advanced Algebraic Structures

### Modules
A module M over ring R is like a vector space but over a ring instead of a field.

### Tensor Products
For vector spaces V and W over field F:
V ⊗ W = {Σᵢ vᵢ ⊗ wᵢ : vᵢ ∈ V, wᵢ ∈ W}

### Example
If V = span{[1,0], [0,1]} and W = span{[1], [2]}, then:
V ⊗ W = span{[1,0]⊗[1], [1,0]⊗[2], [0,1]⊗[1], [0,1]⊗[2]}

## Practice Problems

### Problem 1
Find all elements of GF(8) and construct addition/multiplication tables.

**Solution**:
GF(8) = {0, 1, α, α², α³, α⁴, α⁵, α⁶} where α³ = α + 1

Using α³ = α + 1:
α⁴ = α·α³ = α(α + 1) = α² + α
α⁵ = α·α⁴ = α(α² + α) = α³ + α² = α + 1 + α² = α² + α + 1
α⁶ = α·α⁵ = α(α² + α + 1) = α³ + α² + α = α + 1 + α² + α = α² + 1

Addition and multiplication tables can be constructed using these relations.

### Problem 2
Find a basis for ℂ³ over ℂ.

**Solution**:
Standard basis: {[1,0,0], [0,1,0], [0,0,1]}

To verify linear independence over ℂ:
c₁[1,0,0] + c₂[0,1,0] + c₃[0,0,1] = [0,0,0]
This gives: c₁ = c₂ = c₃ = 0

The set spans ℂ³ since any [a,b,c] ∈ ℂ³ can be written as:
[a,b,c] = a[1,0,0] + b[0,1,0] + c[0,0,1]

### Problem 3
Determine if [1,2,0] and [2,1,1] are linearly independent over ℤ₅.

**Solution**:
c₁[1,2,0] + c₂[2,1,1] = [0,0,0]
This gives: c₁ + 2c₂ = 0, 2c₁ + c₂ = 0, c₂ = 0

From third equation: c₂ = 0
From first equation: c₁ = 0
From second equation: 2c₁ = 0 → c₁ = 0

The vectors are linearly independent over ℤ₅.

## Field Extensions

### Definition
If F ⊆ E are fields, then E is a field extension of F.

### Degree of Extension
[E : F] = dimension of E as vector space over F

### Example
[ℂ : ℝ] = 2 since {1, i} is a basis for ℂ over ℝ.

### Algebraic Extensions
Element α ∈ E is algebraic over F if it's a root of a polynomial with coefficients in F.

## Key Takeaways
- Fields generalize the concept of real/complex numbers
- Finite fields are essential in coding theory and cryptography
- Complex vector spaces require conjugate symmetry in inner products
- Linear algebra concepts extend naturally to general fields
- Field theory connects to many advanced mathematical areas

## Conclusion

This completes our comprehensive linear algebra tutorial series. We've covered:

1. **Vectors and Vector Spaces**: Foundation concepts
2. **Matrices and Operations**: Computational tools
3. **Systems of Linear Equations**: Solution methods
4. **Determinants**: Matrix properties and applications
5. **Eigenvalues and Eigenvectors**: Matrix analysis
6. **Linear Transformations**: Matrix representations
7. **Inner Product Spaces**: Geometry and projections
8. **Diagonalization and Jordan Form**: Matrix simplification
9. **Singular Value Decomposition**: Data analysis tool
10. **Quadratic Forms**: Optimization and geometry
11. **Vector Spaces over Fields**: Advanced structures

Linear algebra provides the mathematical foundation for many areas of science, engineering, and technology. Mastery of these concepts opens doors to advanced topics in mathematics, computer science, physics, and data science.
