# Linear Algebra Tutorial 07: Inner Product Spaces

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand dot product and general inner products
- Work with orthogonality and orthonormal sets
- Apply the Gram-Schmidt process
- Find orthogonal projections
- Use least squares approximation

## Introduction to Inner Products

### Dot Product (Standard Inner Product)
For vectors u, v ∈ ℝⁿ:
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ

### Properties of Dot Product
1. **Commutativity**: u · v = v · u
2. **Distributivity**: u · (v + w) = u · v + u · w
3. **Scalar multiplication**: (cu) · v = c(u · v)
4. **Positive definiteness**: u · u ≥ 0, with equality iff u = 0

### Example
u = [2, 3, 1], v = [1, -1, 2]
u · v = 2(1) + 3(-1) + 1(2) = 2 - 3 + 2 = 1

## General Inner Products

### Definition
An inner product on vector space V is a function ⟨·,·⟩: V × V → ℝ satisfying:
1. **Positive definiteness**: ⟨v,v⟩ ≥ 0, with equality iff v = 0
2. **Symmetry**: ⟨u,v⟩ = ⟨v,u⟩
3. **Linearity**: ⟨u + v,w⟩ = ⟨u,w⟩ + ⟨v,w⟩
4. **Scalar multiplication**: ⟨cu,v⟩ = c⟨u,v⟩

### Examples of Inner Products
1. **Standard dot product**: ⟨u,v⟩ = u · v
2. **Weighted inner product**: ⟨u,v⟩ = u₁v₁ + 2u₂v₂ + 3u₃v₃
3. **Function inner product**: ⟨f,g⟩ = ∫₀¹ f(x)g(x)dx

## Norm and Distance

### Norm (Length)
||v|| = √⟨v,v⟩

### Properties of Norm
1. **Positive definiteness**: ||v|| ≥ 0, with equality iff v = 0
2. **Scalar multiplication**: ||cv|| = |c|||v||
3. **Triangle inequality**: ||u + v|| ≤ ||u|| + ||v||

### Distance
d(u,v) = ||u - v||

### Example
u = [3, 4]
||u|| = √(3² + 4²) = √25 = 5

## Orthogonality

### Definition
Vectors u and v are orthogonal if ⟨u,v⟩ = 0.

### Orthogonal Set
A set {v₁, v₂, ..., vₖ} is orthogonal if ⟨vᵢ,vⱼ⟩ = 0 for i ≠ j.

### Orthonormal Set
An orthogonal set where ||vᵢ|| = 1 for all i.

### Properties
1. **Orthogonal vectors are linearly independent**
2. **Orthonormal basis simplifies calculations**

### Example
Check if {[1,1,0], [1,-1,0], [0,0,1]} is orthogonal:

[1,1,0] · [1,-1,0] = 1(1) + 1(-1) + 0(0) = 0 ✓
[1,1,0] · [0,0,1] = 1(0) + 1(0) + 0(1) = 0 ✓
[1,-1,0] · [0,0,1] = 1(0) + (-1)(0) + 0(1) = 0 ✓

The set is orthogonal. To make it orthonormal, normalize each vector:
||[1,1,0]|| = √2, ||[1,-1,0]|| = √2, ||[0,0,1]|| = 1

Orthonormal set: {[1/√2, 1/√2, 0], [1/√2, -1/√2, 0], [0,0,1]}

## Gram-Schmidt Process

### Purpose
Convert linearly independent vectors into orthonormal vectors.

### Algorithm
Given linearly independent vectors {v₁, v₂, ..., vₙ}:

1. u₁ = v₁/||v₁||
2. For k = 2 to n:
   - wₖ = vₖ - Σ(i=1 to k-1) ⟨vₖ,uᵢ⟩uᵢ
   - uₖ = wₖ/||wₖ||

### Example
Apply Gram-Schmidt to {[1,1,0], [1,0,1], [0,1,1]}

**Step 1**: u₁ = [1,1,0]/||[1,1,0]|| = [1/√2, 1/√2, 0]

**Step 2**: 
w₂ = [1,0,1] - ⟨[1,0,1], [1/√2, 1/√2, 0]⟩[1/√2, 1/√2, 0]
    = [1,0,1] - (1/√2)[1/√2, 1/√2, 0]
    = [1,0,1] - [1/2, 1/2, 0]
    = [1/2, -1/2, 1]

||w₂|| = √((1/2)² + (-1/2)² + 1²) = √(1/4 + 1/4 + 1) = √(3/2) = √6/2

u₂ = [1/2, -1/2, 1]/(√6/2) = [1/√6, -1/√6, 2/√6]

**Step 3**:
w₃ = [0,1,1] - ⟨[0,1,1], [1/√2, 1/√2, 0]⟩[1/√2, 1/√2, 0] - ⟨[0,1,1], [1/√6, -1/√6, 2/√6]⟩[1/√6, -1/√6, 2/√6]
    = [0,1,1] - (1/√2)[1/√2, 1/√2, 0] - (1/√6)[1/√6, -1/√6, 2/√6]
    = [0,1,1] - [1/2, 1/2, 0] - [1/6, -1/6, 1/3]
    = [-2/3, 2/3, 2/3]

||w₃|| = √((-2/3)² + (2/3)² + (2/3)²) = √(4/9 + 4/9 + 4/9) = √(12/9) = 2/√3

u₃ = [-2/3, 2/3, 2/3]/(2/√3) = [-1/√3, 1/√3, 1/√3]

## Orthogonal Projection

### Definition
The orthogonal projection of vector v onto subspace W is:
proj_W(v) = Σ(i=1 to k) ⟨v,uᵢ⟩uᵢ

Where {u₁, u₂, ..., uₖ} is an orthonormal basis for W.

### Projection onto Line
For projection onto line spanned by unit vector u:
proj_u(v) = ⟨v,u⟩u

### Example
Find projection of v = [3,2,1] onto line through u = [1,1,1]/√3

**Solution**:
⟨v,u⟩ = [3,2,1] · [1/√3, 1/√3, 1/√3] = (3 + 2 + 1)/√3 = 6/√3 = 2√3

proj_u(v) = (2√3)[1/√3, 1/√3, 1/√3] = [2, 2, 2]

## Least Squares Approximation

### Problem
Given inconsistent system Ax = b, find x that minimizes ||Ax - b||.

### Solution
The least squares solution is x = (AᵀA)⁻¹Aᵀb

### Example
Find least squares solution to:
x + y = 3
2x - y = 1
x + 2y = 4

**Solution**:
A = [1  1], b = [3]
    [2 -1]      [1]
    [1  2]      [4]

AᵀA = [1  2  1][1  1] = [6  1]
      [1 -1  2][2 -1]   [1  6]
              [1  2]

Aᵀb = [1  2  1][3] = [13]
      [1 -1  2][1]   [9]
              [4]

(AᵀA)⁻¹ = (1/35)[6  -1]
                [-1  6]

x = (1/35)[6  -1][13] = (1/35)[69] = [69/35]
         [-1  6][9]           [41]   [41/35]

## Applications

### Signal Processing
- Fourier analysis
- Filter design
- Noise reduction

### Data Analysis
- Principal Component Analysis (PCA)
- Linear regression
- Dimensionality reduction

### Computer Graphics
- 3D transformations
- Lighting calculations
- Texture mapping

### Quantum Mechanics
- State vectors
- Observable operators
- Probability amplitudes

## Practice Problems

### Problem 1
Find the angle between vectors u = [1,2,3] and v = [2,-1,1]

**Solution**:
cos θ = (u · v)/(||u|| ||v||)
u · v = 1(2) + 2(-1) + 3(1) = 2 - 2 + 3 = 3
||u|| = √(1² + 2² + 3²) = √14
||v|| = √(2² + (-1)² + 1²) = √6

cos θ = 3/(√14 √6) = 3/√84 = 3/(2√21)
θ = arccos(3/(2√21))

### Problem 2
Apply Gram-Schmidt to {[1,0,1], [1,1,0], [0,1,1]}

**Solution**:
**Step 1**: u₁ = [1,0,1]/√2 = [1/√2, 0, 1/√2]

**Step 2**: 
w₂ = [1,1,0] - ⟨[1,1,0], [1/√2, 0, 1/√2]⟩[1/√2, 0, 1/√2]
    = [1,1,0] - (1/√2)[1/√2, 0, 1/√2]
    = [1,1,0] - [1/2, 0, 1/2]
    = [1/2, 1, -1/2]

||w₂|| = √((1/2)² + 1² + (-1/2)²) = √(1/4 + 1 + 1/4) = √(3/2)

u₂ = [1/2, 1, -1/2]/√(3/2) = [1/√6, 2/√6, -1/√6]

**Step 3**:
w₃ = [0,1,1] - ⟨[0,1,1], [1/√2, 0, 1/√2]⟩[1/√2, 0, 1/√2] - ⟨[0,1,1], [1/√6, 2/√6, -1/√6]⟩[1/√6, 2/√6, -1/√6]
    = [0,1,1] - (1/√2)[1/√2, 0, 1/√2] - (1/√6)[1/√6, 2/√6, -1/√6]
    = [0,1,1] - [1/2, 0, 1/2] - [1/6, 1/3, -1/6]
    = [-2/3, 2/3, 2/3]

||w₃|| = √((-2/3)² + (2/3)² + (2/3)²) = 2/√3

u₃ = [-2/3, 2/3, 2/3]/(2/√3) = [-1/√3, 1/√3, 1/√3]

### Problem 3
Find orthogonal projection of v = [2,1,3] onto plane spanned by u₁ = [1,1,0] and u₂ = [0,1,1]

**Solution**:
First apply Gram-Schmidt to get orthonormal basis:
u₁' = [1,1,0]/√2 = [1/√2, 1/√2, 0]
u₂' = [0,1,1] - ⟨[0,1,1], [1/√2, 1/√2, 0]⟩[1/√2, 1/√2, 0]
     = [0,1,1] - (1/√2)[1/√2, 1/√2, 0]
     = [0,1,1] - [1/2, 1/2, 0]
     = [-1/2, 1/2, 1]

||u₂'|| = √((-1/2)² + (1/2)² + 1²) = √(1/4 + 1/4 + 1) = √(3/2)

u₂'' = [-1/2, 1/2, 1]/√(3/2) = [-1/√6, 1/√6, 2/√6]

Now project:
proj_W(v) = ⟨v,u₁'⟩u₁' + ⟨v,u₂''⟩u₂''
          = ([2,1,3] · [1/√2, 1/√2, 0])[1/√2, 1/√2, 0] + ([2,1,3] · [-1/√6, 1/√6, 2/√6])[-1/√6, 1/√6, 2/√6]
          = (3/√2)[1/√2, 1/√2, 0] + (7/√6)[-1/√6, 1/√6, 2/√6]
          = [3/2, 3/2, 0] + [-7/6, 7/6, 7/3]
          = [1/3, 8/3, 7/3]

## Key Takeaways
- Inner products generalize the dot product concept
- Orthogonality simplifies many calculations
- Gram-Schmidt process creates orthonormal bases
- Orthogonal projections minimize distance to subspaces
- Least squares provides best approximation for inconsistent systems

## Next Steps
In the next tutorial, we'll explore diagonalization and Jordan form, learning about similarity transformations and when matrices can be diagonalized.
