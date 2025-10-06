# Linear Algebra Tutorial 10: Quadratic Forms

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand quadratic forms and their matrix representation
- Classify quadratic forms as positive definite, negative definite, or indefinite
- Apply the principal axis theorem
- Use quadratic forms in optimization problems
- Understand conic sections and quadric surfaces

## Introduction to Quadratic Forms

### Definition
A quadratic form in n variables is a homogeneous polynomial of degree 2:
Q(x₁, x₂, ..., xₙ) = Σ(i=1 to n) Σ(j=1 to n) aᵢⱼxᵢxⱼ

### Matrix Representation
Q(x) = xᵀAx

Where A is a symmetric matrix (aᵢⱼ = aⱼᵢ).

### Example
Q(x,y) = 3x² + 4xy + 2y²

Matrix representation: Q(x,y) = [x y][3 2][x]
                                      [2 2][y]

## Classification of Quadratic Forms

### Positive Definite
Q(x) > 0 for all x ≠ 0
- All eigenvalues of A are positive
- A is invertible
- xᵀAx > 0 for all x ≠ 0

### Negative Definite
Q(x) < 0 for all x ≠ 0
- All eigenvalues of A are negative
- A is invertible
- xᵀAx < 0 for all x ≠ 0

### Positive Semidefinite
Q(x) ≥ 0 for all x
- All eigenvalues of A are non-negative
- xᵀAx ≥ 0 for all x

### Negative Semidefinite
Q(x) ≤ 0 for all x
- All eigenvalues of A are non-positive
- xᵀAx ≤ 0 for all x

### Indefinite
Q(x) takes both positive and negative values
- A has both positive and negative eigenvalues

### Example
Classify Q(x,y) = 2x² + 4xy + 3y²

**Solution**:
A = [2 2]
    [2 3]

Characteristic polynomial: (2-λ)(3-λ) - 4 = λ² - 5λ + 2 = 0
Eigenvalues: λ₁ = (5+√17)/2 > 0, λ₂ = (5-√17)/2 > 0

Both eigenvalues positive → positive definite

## Principal Axis Theorem

### Statement
For symmetric matrix A, there exists orthogonal matrix Q such that:
QᵀAQ = D

Where D is diagonal with eigenvalues on the diagonal.

### Change of Variables
Let y = Qᵀx, then:
Q(x) = xᵀAx = (Qy)ᵀA(Qy) = yᵀ(QᵀAQ)y = yᵀDy

This eliminates cross terms.

### Example
Transform Q(x,y) = 3x² + 4xy + 2y² to diagonal form

**Solution**:
A = [3 2]
    [2 2]

Characteristic polynomial: (3-λ)(2-λ) - 4 = λ² - 5λ + 2 = 0
Eigenvalues: λ₁ = (5+√17)/2, λ₂ = (5-√17)/2

For λ₁: eigenvector v₁ = [1, (1+√17)/4]ᵀ (normalized)
For λ₂: eigenvector v₂ = [1, (1-√17)/4]ᵀ (normalized)

Q = [v₁ v₂]

Diagonal form: Q(u,v) = λ₁u² + λ₂v²

## Conic Sections

### General Form
ax² + 2bxy + cy² + dx + ey + f = 0

### Classification
Using eigenvalues of matrix [a b]:
                        [b c]

- **Ellipse**: Both eigenvalues same sign
- **Hyperbola**: Eigenvalues opposite signs
- **Parabola**: One eigenvalue zero
- **Degenerate cases**: Circle, two lines, point, empty set

### Example
Classify 2x² + 4xy + 3y² - 6x + 2y + 1 = 0

**Solution**:
Matrix [2 2] has eigenvalues λ₁ = (5+√17)/2, λ₂ = (5-√17)/2
       [2 3]

Both positive → ellipse

## Optimization Applications

### Unconstrained Optimization
For quadratic form Q(x) = xᵀAx + bᵀx + c:

- If A is positive definite: unique global minimum
- If A is negative definite: unique global maximum
- If A is indefinite: saddle point

### Critical Points
∇Q(x) = 2Ax + b = 0
Critical point: x* = -½A⁻¹b

### Example
Find extrema of f(x,y) = 2x² + 4xy + 3y² - 6x + 2y

**Solution**:
A = [2 2], b = [-6]
    [2 3]      [2]

Since A is positive definite (from previous example), there's a unique global minimum.

Critical point: x* = -½A⁻¹b = -½[3 -2][-6] = -½[-22] = [11]
                                    [-2  2][2]      [16]   [-8]

## Quadric Surfaces

### General Form
ax² + by² + cz² + 2dxy + 2exz + 2fyz + gx + hy + iz + j = 0

### Classification
Using eigenvalues of matrix [a d e]:
                        [d b f]
                        [e f c]

- **Ellipsoid**: All eigenvalues same sign
- **Hyperboloid**: Mixed signs
- **Paraboloid**: One eigenvalue zero
- **Cylinder**: Two eigenvalues zero

### Example
Classify x² + 2y² + 3z² + 4xy + 2xz + 6yz = 1

**Solution**:
Matrix [1 2 1] has eigenvalues determined by characteristic polynomial
       [2 2 3]
       [1 3 0]

This requires solving cubic equation to classify the surface.

## Practice Problems

### Problem 1
Classify Q(x,y) = x² - 4xy + 4y²

**Solution**:
A = [1  -2]
    [-2  4]

Characteristic polynomial: (1-λ)(4-λ) - 4 = λ² - 5λ = 0
Eigenvalues: λ₁ = 5, λ₂ = 0

One positive, one zero → positive semidefinite

### Problem 2
Transform Q(x,y) = x² + 6xy + y² to diagonal form

**Solution**:
A = [1 3]
    [3 1]

Characteristic polynomial: (1-λ)² - 9 = λ² - 2λ - 8 = 0
Eigenvalues: λ₁ = 4, λ₂ = -2

For λ₁ = 4: (A - 4I)v = 0
[-3  3][x] = [0] → -3x + 3y = 0 → x = y
[3  -3][y]   [0]
Eigenvector: v₁ = [1, 1]ᵀ, normalized: [1/√2, 1/√2]ᵀ

For λ₂ = -2: (A + 2I)v = 0
[3  3][x] = [0] → 3x + 3y = 0 → x = -y
[3  3][y]   [0]
Eigenvector: v₂ = [1, -1]ᵀ, normalized: [1/√2, -1/√2]ᵀ

Q = [1/√2   1/√2]
    [1/√2  -1/√2]

Diagonal form: Q(u,v) = 4u² - 2v²

### Problem 3
Find minimum of f(x,y) = 3x² + 2xy + 2y² - 4x - 6y + 1

**Solution**:
A = [3 1], b = [-4]
    [1 2]      [-6]

Characteristic polynomial: (3-λ)(2-λ) - 1 = λ² - 5λ + 5 = 0
Eigenvalues: λ₁ = (5+√5)/2 > 0, λ₂ = (5-√5)/2 > 0

Both positive → positive definite → unique global minimum

Critical point: x* = -½A⁻¹b = -½[2 -1][-4] = -½[-2] = [1]
                                    [-1  3][-6]      [-14]   [7]

Minimum value: f(1,7) = 3(1)² + 2(1)(7) + 2(7)² - 4(1) - 6(7) + 1 = 3 + 14 + 98 - 4 - 42 + 1 = 70

## Constrained Optimization

### Lagrange Multipliers
For constrained optimization of quadratic forms, use Lagrange multipliers.

### Example
Minimize Q(x,y) = x² + y² subject to x + y = 1

**Solution**:
Lagrangian: L = x² + y² - λ(x + y - 1)

∂L/∂x = 2x - λ = 0 → x = λ/2
∂L/∂y = 2y - λ = 0 → y = λ/2
∂L/∂λ = -(x + y - 1) = 0 → x + y = 1

Substituting: λ/2 + λ/2 = 1 → λ = 1
So: x = y = 1/2

Minimum value: Q(1/2, 1/2) = (1/2)² + (1/2)² = 1/2

## Key Takeaways
- Quadratic forms have symmetric matrix representations
- Classification depends on eigenvalue signs
- Principal axis theorem eliminates cross terms
- Quadratic forms appear in optimization and geometry
- Conic sections and quadric surfaces are classified by eigenvalues

## Next Steps
In the final tutorial, we'll explore vector spaces over general fields, learning about finite fields, complex vector spaces, and advanced algebraic structures.
