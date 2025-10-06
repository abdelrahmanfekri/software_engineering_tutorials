# Partial Differential Equations

## Overview
Partial Differential Equations (PDEs) involve functions of multiple variables and their partial derivatives. They are fundamental for modeling phenomena in physics, engineering, and other sciences where quantities depend on multiple independent variables like space and time. This tutorial covers the classification of PDEs and the method of separation of variables.

## Learning Objectives
- Classify partial differential equations
- Understand the heat equation, wave equation, and Laplace's equation
- Apply separation of variables to solve PDEs
- Work with boundary conditions and initial conditions
- Understand applications in physics and engineering
- Solve eigenvalue problems arising from separation of variables

## Classification of PDEs

### General Form
A second-order linear PDE in two variables has the form:
```
A ∂²u/∂x² + B ∂²u/∂x∂y + C ∂²u/∂y² + D ∂u/∂x + E ∂u/∂y + Fu = G
```

### Classification
Based on the discriminant Δ = B² - 4AC:

- **Elliptic**: Δ < 0 (e.g., Laplace's equation)
- **Parabolic**: Δ = 0 (e.g., heat equation)
- **Hyperbolic**: Δ > 0 (e.g., wave equation)

### Examples
1. **Laplace's equation**: `∇²u = ∂²u/∂x² + ∂²u/∂y² = 0` (elliptic)
2. **Heat equation**: `∂u/∂t = α²∇²u` (parabolic)
3. **Wave equation**: `∂²u/∂t² = c²∇²u` (hyperbolic)

## The Heat Equation

### One-Dimensional Heat Equation
```
∂u/∂t = α² ∂²u/∂x²
```
where u(x,t) is temperature, α² is thermal diffusivity.

### Physical Interpretation
- Describes heat conduction in a rod
- u(x,t) represents temperature at position x and time t
- Heat flows from regions of high temperature to low temperature

### Boundary Conditions
Common types:
1. **Dirichlet**: u(0,t) = u(L,t) = 0 (fixed temperature)
2. **Neumann**: ∂u/∂x(0,t) = ∂u/∂x(L,t) = 0 (insulated ends)
3. **Mixed**: Combination of the above

### Initial Condition
```
u(x,0) = f(x)
```

### Solution by Separation of Variables

#### Step 1: Assume Separable Solution
```
u(x,t) = X(x)T(t)
```

#### Step 2: Substitute into PDE
```
XT' = α²X''T
```
Divide by XT: `T'/T = α²X''/X`

#### Step 3: Separate Variables
```
T'/T = α²X''/X = -λ (constant)
```

This gives two ODEs:
- `T' + λα²T = 0`
- `X'' + λX = 0`

#### Step 4: Solve ODEs
For T: `T(t) = Ce^(-λα²t)`
For X: `X(x) = A cos(√λ x) + B sin(√λ x)`

#### Step 5: Apply Boundary Conditions
For u(0,t) = u(L,t) = 0:
- X(0) = 0 → A = 0
- X(L) = 0 → B sin(√λ L) = 0

This gives: `√λ L = nπ` or `λ = (nπ/L)²`

#### Step 6: Construct Solution
```
u(x,t) = Σ_{n=1}^∞ B_n sin(nπx/L) e^(-α²n²π²t/L²)
```

#### Step 7: Apply Initial Condition
```
f(x) = Σ_{n=1}^∞ B_n sin(nπx/L)
```
This is a Fourier sine series, so:
```
B_n = (2/L) ∫₀^L f(x) sin(nπx/L) dx
```

### Example: Heat Equation with Specific Initial Condition
Solve: `∂u/∂t = ∂²u/∂x²`, `u(0,t) = u(π,t) = 0`, `u(x,0) = sin(x)`

Solution: `u(x,t) = sin(x) e^(-t)`

## The Wave Equation

### One-Dimensional Wave Equation
```
∂²u/∂t² = c² ∂²u/∂x²
```
where u(x,t) is displacement, c is wave speed.

### Physical Interpretation
- Describes vibrations of a string
- u(x,t) represents displacement at position x and time t
- c is the speed of wave propagation

### Initial Conditions
```
u(x,0) = f(x)    (initial displacement)
∂u/∂t(x,0) = g(x) (initial velocity)
```

### Solution by Separation of Variables

#### Step 1: Assume Separable Solution
```
u(x,t) = X(x)T(t)
```

#### Step 2: Substitute into PDE
```
XT'' = c²X''T
```
Divide by XT: `T''/T = c²X''/X = -λ`

This gives two ODEs:
- `T'' + λc²T = 0`
- `X'' + λX = 0`

#### Step 3: Solve ODEs
For λ > 0: `X(x) = A cos(√λ x) + B sin(√λ x)`
`T(t) = C cos(c√λ t) + D sin(c√λ t)`

#### Step 4: Apply Boundary Conditions
For u(0,t) = u(L,t) = 0: `λ = (nπ/L)²`

#### Step 5: Construct Solution
```
u(x,t) = Σ_{n=1}^∞ sin(nπx/L)[A_n cos(nπct/L) + B_n sin(nπct/L)]
```

#### Step 6: Apply Initial Conditions
```
f(x) = Σ_{n=1}^∞ A_n sin(nπx/L)
g(x) = Σ_{n=1}^∞ (nπc/L) B_n sin(nπx/L)
```

### Example: Wave Equation with Specific Initial Conditions
Solve: `∂²u/∂t² = 4 ∂²u/∂x²`, `u(0,t) = u(π,t) = 0`, `u(x,0) = sin(x)`, `∂u/∂t(x,0) = 0`

Solution: `u(x,t) = sin(x) cos(2t)`

## Laplace's Equation

### Two-Dimensional Laplace's Equation
```
∂²u/∂x² + ∂²u/∂y² = 0
```

### Physical Interpretation
- Describes steady-state temperature distribution
- Electrostatic potential in charge-free regions
- Incompressible, irrotational fluid flow

### Boundary Conditions
For a rectangular region [0,a] × [0,b]:
- `u(0,y) = u(a,y) = 0`
- `u(x,0) = 0`, `u(x,b) = f(x)`

### Solution by Separation of Variables

#### Step 1: Assume Separable Solution
```
u(x,y) = X(x)Y(y)
```

#### Step 2: Substitute into PDE
```
X''Y + XY'' = 0
```
Divide by XY: `X''/X + Y''/Y = 0`

This gives: `X''/X = -Y''/Y = -λ`

#### Step 3: Solve ODEs
- `X'' + λX = 0`
- `Y'' - λY = 0`

#### Step 4: Apply Boundary Conditions
For u(0,y) = u(a,y) = 0: `λ = (nπ/a)²`

#### Step 5: Construct Solution
```
u(x,y) = Σ_{n=1}^∞ A_n sin(nπx/a) sinh(nπy/a)
```

#### Step 6: Apply Remaining Boundary Condition
```
f(x) = Σ_{n=1}^∞ A_n sinh(nπb/a) sin(nπx/a)
```

## Eigenvalue Problems

### Sturm-Liouville Problem
Many separation of variables problems lead to Sturm-Liouville eigenvalue problems:
```
(p(x)y')' + [q(x) + λr(x)]y = 0
```
with boundary conditions.

### Properties
1. **Eigenvalues** are real and can be ordered: λ₁ < λ₂ < λ₃ < ...
2. **Eigenfunctions** are orthogonal with respect to weight function r(x)
3. **Completeness**: Any function can be expanded in terms of eigenfunctions

### Example: Vibrating String
The eigenvalue problem `X'' + λX = 0` with `X(0) = X(L) = 0` has:
- Eigenvalues: `λ_n = (nπ/L)²`
- Eigenfunctions: `X_n(x) = sin(nπx/L)`
- Orthogonality: `∫₀^L sin(mπx/L) sin(nπx/L) dx = 0` for m ≠ n

## Applications

### Heat Conduction
- **Temperature distribution** in solids
- **Cooling of objects**
- **Heat transfer** in engineering systems

### Wave Propagation
- **String vibrations** (musical instruments)
- **Sound waves** in air
- **Electromagnetic waves**
- **Seismic waves**

### Electrostatics
- **Potential distribution** around conductors
- **Capacitance calculations**
- **Electric field determination**

### Fluid Mechanics
- **Potential flow** around obstacles
- **Stream function** calculations
- **Pressure distribution**

## Problem-Solving Strategies

### 1. For Separation of Variables
1. **Identify the PDE type** (heat, wave, Laplace)
2. **Set up boundary and initial conditions**
3. **Assume separable solution**
4. **Separate variables** to get ODEs
5. **Solve eigenvalue problem** for spatial part
6. **Solve time-dependent ODE** (if applicable)
7. **Apply all conditions** to determine constants
8. **Construct general solution** as infinite series

### 2. Common Boundary Conditions
- **Dirichlet**: Fixed values of u
- **Neumann**: Fixed values of ∂u/∂n
- **Mixed**: Combination of the above
- **Periodic**: u(x+L,t) = u(x,t)

### 3. Common Mistakes
- Incorrect separation of variables
- Wrong eigenvalue determination
- Incorrect application of boundary conditions
- Algebraic errors in Fourier coefficients
- Convergence issues with infinite series

## Practice Problems

### Heat Equation
1. Solve: `∂u/∂t = ∂²u/∂x²`, `u(0,t) = u(π,t) = 0`, `u(x,0) = x(π-x)`
2. Solve: `∂u/∂t = ∂²u/∂x²`, `∂u/∂x(0,t) = ∂u/∂x(π,t) = 0`, `u(x,0) = cos(x)`

### Wave Equation
1. Solve: `∂²u/∂t² = ∂²u/∂x²`, `u(0,t) = u(π,t) = 0`, `u(x,0) = sin(x)`, `∂u/∂t(x,0) = 0`
2. Solve: `∂²u/∂t² = 4∂²u/∂x²`, `u(0,t) = u(1,t) = 0`, `u(x,0) = 0`, `∂u/∂t(x,0) = sin(πx)`

### Laplace's Equation
1. Solve: `∂²u/∂x² + ∂²u/∂y² = 0` in [0,π] × [0,π] with `u(0,y) = u(π,y) = 0`, `u(x,0) = 0`, `u(x,π) = sin(x)`
2. Solve: `∂²u/∂x² + ∂²u/∂y² = 0` in [0,1] × [0,1] with `u(0,y) = u(1,y) = 0`, `u(x,0) = x(1-x)`, `u(x,1) = 0`

### Eigenvalue Problems
1. Find eigenvalues and eigenfunctions for: `X'' + λX = 0`, `X(0) = 0`, `X'(π) = 0`
2. Find eigenvalues and eigenfunctions for: `X'' + λX = 0`, `X'(0) = 0`, `X'(π) = 0`

## Summary

Partial differential equations are essential for modeling phenomena in multiple dimensions:

1. **Classification** helps determine appropriate solution methods
2. **Separation of variables** is powerful for linear, homogeneous PDEs
3. **Boundary and initial conditions** determine the specific solution
4. **Eigenvalue problems** arise naturally from separation of variables
5. **Applications** span physics, engineering, and many other fields
6. **Fourier series** are crucial for satisfying initial/boundary conditions

Master these techniques as they provide the foundation for understanding wave propagation, heat conduction, and many other important physical phenomena.
