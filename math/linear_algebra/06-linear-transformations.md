# Linear Algebra Tutorial 06: Linear Transformations

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the definition and properties of linear transformations
- Find matrix representations of linear transformations
- Determine kernel and image of transformations
- Compose transformations and understand their properties
- Identify isomorphisms and automorphisms

## Introduction to Linear Transformations

### Definition
A linear transformation T: V → W between vector spaces V and W is a function that preserves:
1. **Vector addition**: T(u + v) = T(u) + T(v)
2. **Scalar multiplication**: T(cv) = cT(v)

### Alternative Definition
T is linear if and only if: T(c₁u + c₂v) = c₁T(u) + c₂T(v)

### Examples of Linear Transformations
1. **Rotation**: T(x,y) = (x cos θ - y sin θ, x sin θ + y cos θ)
2. **Scaling**: T(x,y) = (ax, by)
3. **Projection**: T(x,y,z) = (x, y, 0)
4. **Reflection**: T(x,y) = (x, -y)

### Non-Linear Examples
1. **Translation**: T(x,y) = (x + a, y + b) (fails scalar multiplication)
2. **Quadratic**: T(x) = x² (fails addition)

## Matrix Representation

### Standard Matrix
For T: ℝⁿ → ℝᵐ, the standard matrix A is:
A = [T(e₁) T(e₂) ... T(eₙ)]

Where e₁, e₂, ..., eₙ are standard basis vectors.

### Example
Find matrix for T: ℝ² → ℝ² where T(x,y) = (2x + y, x - 3y)

**Solution**:
T(e₁) = T(1,0) = (2,1)
T(e₂) = T(0,1) = (1,-3)

A = [2  1]
    [1 -3]

### Verification
T(x,y) = A[x] = [2  1][x] = [2x + y]
                [y]   [1 -3][y]   [x - 3y]

## Kernel and Image

### Kernel (Null Space)
ker(T) = {v ∈ V : T(v) = 0}

### Image (Range)
im(T) = {T(v) : v ∈ V}

### Rank-Nullity Theorem
dim(ker(T)) + dim(im(T)) = dim(V)

### Example
For T: ℝ³ → ℝ² defined by T(x,y,z) = (x + y, y + z)

**Kernel**:
T(x,y,z) = (0,0) implies x + y = 0 and y + z = 0
So y = -x and z = -y = x
ker(T) = {(x, -x, x) : x ∈ ℝ} = span{(1, -1, 1)}
dim(ker(T)) = 1

**Image**:
im(T) = {(x + y, y + z) : x,y,z ∈ ℝ}
Since we can choose x,y,z freely, im(T) = ℝ²
dim(im(T)) = 2

**Verification**: 1 + 2 = 3 = dim(ℝ³) ✓

## Properties of Linear Transformations

### Basic Properties
1. **T(0) = 0** (zero vector maps to zero vector)
2. **T(-v) = -T(v)** (additive inverse preserved)
3. **T(c₁v₁ + c₂v₂ + ... + cₙvₙ) = c₁T(v₁) + c₂T(v₂) + ... + cₙT(vₙ)**

### Composition
If T: U → V and S: V → W are linear, then S∘T: U → W is linear.

**Matrix of composition**: [S∘T] = [S][T]

### Inverse
If T: V → W is linear and invertible, then T⁻¹: W → V is linear.

## Special Types of Transformations

### Isomorphism
A linear transformation T: V → W is an isomorphism if:
1. T is one-to-one (injective)
2. T is onto (surjective)

**Equivalent conditions**:
- ker(T) = {0} and im(T) = W
- dim(V) = dim(W) and ker(T) = {0}

### Automorphism
An isomorphism from V to itself (T: V → V)

### Example
T: ℝ² → ℝ² defined by T(x,y) = (2x + y, x - y)

**Check if isomorphism**:
Matrix A = [2  1]
          [1 -1]

det(A) = 2(-1) - 1(1) = -3 ≠ 0
Since det(A) ≠ 0, A is invertible, so T is an isomorphism.

## Geometric Transformations

### Rotation Matrix
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]

### Scaling Matrix
S(a,b) = [a  0]
         [0  b]

### Reflection Matrix
Reflection across x-axis: [1  0]
                         [0 -1]

Reflection across y-axis: [-1  0]
                          [0   1]

### Shear Matrix
Horizontal shear: [1  k]
                 [0  1]

Vertical shear: [1  0]
               [k  1]

## Applications

### Computer Graphics
- 2D and 3D transformations
- Rotation, scaling, translation (using homogeneous coordinates)
- Perspective projections

### Data Analysis
- Principal Component Analysis (PCA)
- Linear regression
- Dimensionality reduction

### Signal Processing
- Fourier transforms
- Filtering operations
- Convolution

### Quantum Mechanics
- State transformations
- Observable operators
- Unitary transformations

## Practice Problems

### Problem 1
Find the matrix representation of T: ℝ³ → ℝ² where T(x,y,z) = (x + 2y - z, 3x - y + 2z)

**Solution**:
T(e₁) = T(1,0,0) = (1,3)
T(e₂) = T(0,1,0) = (2,-1)
T(e₃) = T(0,0,1) = (-1,2)

A = [1   2  -1]
    [3  -1   2]

### Problem 2
Find kernel and image of T: ℝ³ → ℝ³ where T(x,y,z) = (x - y, y - z, z - x)

**Solution**:
**Kernel**: T(x,y,z) = (0,0,0)
x - y = 0, y - z = 0, z - x = 0
So x = y = z
ker(T) = {(x,x,x) : x ∈ ℝ} = span{(1,1,1)}
dim(ker(T)) = 1

**Image**: im(T) = {(x-y, y-z, z-x) : x,y,z ∈ ℝ}
Let u = x-y, v = y-z, w = z-x
Then u + v + w = (x-y) + (y-z) + (z-x) = 0
So im(T) = {(u,v,w) : u + v + w = 0}
This is a plane in ℝ³, so dim(im(T)) = 2

**Verification**: 1 + 2 = 3 = dim(ℝ³) ✓

### Problem 3
Compose transformations T₁(x,y) = (x + y, x - y) and T₂(x,y) = (2x, 3y)

**Solution**:
T₁: ℝ² → ℝ² with matrix A₁ = [1   1]
                              [1  -1]

T₂: ℝ² → ℝ² with matrix A₂ = [2  0]
                              [0  3]

T₂∘T₁ has matrix A₂A₁ = [2  0][1   1] = [2   2]
                        [0  3][1  -1]   [3  -3]

So T₂∘T₁(x,y) = (2x + 2y, 3x - 3y)

## Change of Basis

### Coordinate Vectors
If B = {v₁, v₂, ..., vₙ} is a basis for V, then [v]B denotes coordinates of v in basis B.

### Change of Basis Matrix
If B and C are bases for V, then P = [C]B is the change of basis matrix from B to C.

### Matrix of Transformation in Different Bases
If [T]B is matrix of T in basis B, then [T]C = P⁻¹[T]BP

## Key Takeaways
- Linear transformations preserve vector operations
- Every linear transformation has a matrix representation
- Kernel and image provide important structural information
- Composition of transformations corresponds to matrix multiplication
- Isomorphisms preserve vector space structure

## Next Steps
In the next tutorial, we'll explore inner product spaces, learning about dot products, orthogonality, the Gram-Schmidt process, and orthogonal projections.
