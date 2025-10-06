# Systems of Differential Equations

## Overview
Systems of differential equations arise naturally when modeling interconnected systems where multiple variables depend on each other. They are essential for understanding coupled oscillators, population dynamics, chemical reactions, and many other phenomena. This tutorial covers the theory and solution methods for systems of differential equations.

## Learning Objectives
- Understand first-order linear systems
- Apply matrix methods to solve systems
- Use eigenvalue methods for homogeneous systems
- Solve nonhomogeneous systems
- Understand applications to coupled oscillators
- Work with phase plane analysis

## First-Order Linear Systems

### General Form
A system of first-order linear differential equations has the form:
```
dx₁/dt = a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ + f₁(t)
dx₂/dt = a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ + f₂(t)
...
dxₙ/dt = aₙ₁x₁ + aₙ₂x₂ + ... + aₙₙxₙ + fₙ(t)
```

### Matrix Form
```
dX/dt = AX + F(t)
```
where:
- `X = [x₁, x₂, ..., xₙ]ᵀ` is the vector of unknown functions
- `A = [aᵢⱼ]` is the coefficient matrix
- `F(t) = [f₁(t), f₂(t), ..., fₙ(t)]ᵀ` is the forcing vector

## Homogeneous Systems

### Form
```
dX/dt = AX
```

### Solution Method Using Eigenvalues and Eigenvectors

#### Steps:
1. **Find eigenvalues** of matrix A: `det(A - λI) = 0`
2. **Find eigenvectors** corresponding to each eigenvalue
3. **Construct fundamental matrix** Φ(t)
4. **General solution**: `X(t) = Φ(t)C` where C is a constant vector

#### Case 1: Distinct Real Eigenvalues
If A has n distinct real eigenvalues λ₁, λ₂, ..., λₙ with corresponding eigenvectors v₁, v₂, ..., vₙ, then:
```
X(t) = C₁e^(λ₁t)v₁ + C₂e^(λ₂t)v₂ + ... + Cₙe^(λₙt)vₙ
```

#### Case 2: Complex Eigenvalues
If λ = α ± βi is a complex eigenvalue with eigenvector v = a ± bi, then:
```
X(t) = e^(αt)[C₁(a cos(βt) - b sin(βt)) + C₂(a sin(βt) + b cos(βt))]
```

#### Case 3: Repeated Eigenvalues
For repeated eigenvalues, we need generalized eigenvectors.

### Example: 2×2 System
```
dx/dt = 3x - y
dy/dt = x + y
```
Matrix form: `dX/dt = [3 -1; 1 1]X`

Eigenvalues: `det([3-λ -1; 1 1-λ]) = (3-λ)(1-λ) + 1 = λ² - 4λ + 4 = 0`
Roots: `λ = 2, 2` (repeated)

For λ = 2: `[1 -1; 1 -1][v₁; v₂] = [0; 0]`
Eigenvector: `v = [1; 1]`

Generalized eigenvector: `(A - 2I)w = v`
`[1 -1; 1 -1][w₁; w₂] = [1; 1]`
Solution: `w = [0; -1]`

General solution:
```
X(t) = C₁e^(2t)[1; 1] + C₂e^(2t)(t[1; 1] + [0; -1])
```

## Nonhomogeneous Systems

### Form
```
dX/dt = AX + F(t)
```

### Solution Method
**General Solution**: `X(t) = X_c(t) + X_p(t)`
where:
- `X_c(t)` is the complementary solution (solution to homogeneous system)
- `X_p(t)` is a particular solution

### Method of Undetermined Coefficients

For simple forcing functions, make an educated guess for `X_p(t)`.

#### Example:
```
dx/dt = 2x + y + t
dy/dt = x + 2y + 1
```
Homogeneous solution: `X_c = C₁e^(3t)[1; 1] + C₂e^t[1; -1]`

Guess for particular solution: `X_p = [At + B; Ct + D]`
Substitute and solve for coefficients:
`A = 2(At + B) + (Ct + D) + t`
`C = (At + B) + 2(Ct + D) + 1`

This gives: `A = 2A + C + 1`, `B = 2B + D`, `C = A + 2C`, `D = B + 2D + 1`
Solution: `A = -1/3`, `B = -2/9`, `C = -1/3`, `D = -7/9`

### Variation of Parameters

For any forcing function `F(t)`:

1. Find fundamental matrix Φ(t) for homogeneous system
2. Assume: `X_p(t) = Φ(t)U(t)`
3. Substitute: `Φ'(t)U(t) + Φ(t)U'(t) = AΦ(t)U(t) + F(t)`
4. Since `Φ'(t) = AΦ(t)`: `Φ(t)U'(t) = F(t)`
5. Solve: `U'(t) = Φ⁻¹(t)F(t)`
6. Integrate: `U(t) = ∫Φ⁻¹(t)F(t)dt`
7. Particular solution: `X_p(t) = Φ(t)U(t)`

## Phase Plane Analysis

### Definition
The phase plane is the plane with coordinates (x, y) where we plot trajectories of solutions to 2×2 systems.

### Critical Points
Points where `dx/dt = dy/dt = 0` simultaneously.

### Classification of Critical Points

For the system `dX/dt = AX` with eigenvalues λ₁, λ₂:

#### 1. Node (Real, Same Sign)
- **Stable node**: λ₁, λ₂ < 0
- **Unstable node**: λ₁, λ₂ > 0
- All trajectories approach (stable) or leave (unstable) the origin

#### 2. Saddle Point (Real, Opposite Signs)
- λ₁ > 0, λ₂ < 0
- Trajectories approach along one direction, leave along another

#### 3. Spiral Point (Complex)
- **Stable spiral**: Re(λ) < 0
- **Unstable spiral**: Re(λ) > 0
- Trajectories spiral toward (stable) or away from (unstable) origin

#### 4. Center (Pure Imaginary)
- λ = ±βi
- Trajectories are closed curves around the origin

### Example: Phase Plane Analysis
```
dx/dt = x + y
dy/dt = -2x - y
```
Matrix: `A = [1 1; -2 -1]`

Eigenvalues: `det([1-λ 1; -2 -1-λ]) = λ² + 1 = 0`
Roots: `λ = ±i`

Since eigenvalues are pure imaginary, the critical point (0,0) is a center.

## Applications

### Coupled Oscillators
**Two Mass-Spring System:**
```
m₁ẍ₁ = -k₁x₁ + k₂(x₂ - x₁)
m₂ẍ₂ = -k₂(x₂ - x₁) - k₃x₂
```

Convert to first-order system:
```
x₁' = v₁
v₁' = -(k₁+k₂)x₁/m₁ + k₂x₂/m₁
x₂' = v₂
v₂' = k₂x₁/m₂ - (k₂+k₃)x₂/m₂
```

### Population Dynamics
**Predator-Prey Model (Lotka-Volterra):**
```
dx/dt = ax - bxy
dy/dt = -cy + dxy
```
where x is prey population, y is predator population.

### Chemical Reactions
**Consecutive Reactions:**
```
A → B → C
```
Rate equations:
```
d[A]/dt = -k₁[A]
d[B]/dt = k₁[A] - k₂[B]
d[C]/dt = k₂[B]
```

### Electrical Circuits
**RLC Circuit with Two Loops:**
```
L₁(di₁/dt) + R₁i₁ + (1/C₁)∫i₁dt + R₃(i₁ - i₂) = E₁(t)
L₂(di₂/dt) + R₂i₂ + (1/C₂)∫i₂dt + R₃(i₂ - i₁) = E₂(t)
```

## Problem-Solving Strategies

### 1. Classification Steps
1. **Identify the system type**: Linear or nonlinear?
2. **Homogeneous or nonhomogeneous?**
3. **Find eigenvalues** of coefficient matrix
4. **Classify critical points** (for 2×2 systems)
5. **Choose appropriate solution method**

### 2. Method Selection
- **Eigenvalue method**: For homogeneous linear systems
- **Undetermined coefficients**: For simple forcing functions
- **Variation of parameters**: For general forcing functions
- **Phase plane analysis**: For understanding qualitative behavior

### 3. Common Mistakes
- Incorrect eigenvalue calculation
- Missing generalized eigenvectors for repeated eigenvalues
- Wrong sign in complex eigenvalue solutions
- Incorrect classification of critical points
- Algebraic errors in matrix operations

## Practice Problems

### Homogeneous Systems
1. Solve: `dx/dt = 2x + y`, `dy/dt = x + 2y`
2. Solve: `dx/dt = x - 2y`, `dy/dt = 2x + y`
3. Solve: `dx/dt = 3x + y`, `dy/dt = -x + y`

### Nonhomogeneous Systems
1. Solve: `dx/dt = x + y + 1`, `dy/dt = 4x + y + t`
2. Solve: `dx/dt = 2x - y + e^t`, `dy/dt = 3x - 2y + 1`

### Initial Value Problems
1. Solve: `dx/dt = 2x - y`, `dy/dt = x + 2y` with `x(0) = 1`, `y(0) = 0`

### Phase Plane Analysis
1. Classify the critical point (0,0) for: `dx/dt = x + y`, `dy/dt = -2x - y`
2. Sketch the phase portrait for: `dx/dt = 2x + y`, `dy/dt = x + 2y`

### Applications
1. Two masses m₁ = 1, m₂ = 2 are connected by springs with k₁ = 3, k₂ = 2. If initially displaced and released, find the motion.
2. Solve the Lotka-Volterra equations with a = 2, b = 1, c = 1, d = 1, x(0) = 10, y(0) = 5.

## Summary

Systems of differential equations are essential for modeling interconnected systems:

1. **Matrix methods** provide powerful tools for linear systems
2. **Eigenvalues and eigenvectors** determine the qualitative behavior
3. **Phase plane analysis** gives insight into system dynamics
4. **Applications** span physics, biology, chemistry, and engineering
5. **Critical point classification** helps understand long-term behavior

Master these techniques as they are fundamental for understanding complex dynamic systems and their applications across many fields.
