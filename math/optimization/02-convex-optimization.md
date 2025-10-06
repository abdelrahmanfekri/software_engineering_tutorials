# Optimization Theory Tutorial 02: Convex Optimization

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand convex sets and convex functions
- Formulate convex optimization problems
- Apply duality theory to optimization
- Use KKT conditions for constrained optimization
- Solve linear and quadratic programming problems
- Understand convergence guarantees for convex problems

## Introduction to Convex Optimization

### What is Convex Optimization?
Convex optimization involves minimizing a convex function over a convex set. These problems have special properties that make them easier to solve and analyze.

**Key Advantage**: Any local minimum is also a global minimum.

### Why is Convex Optimization Important?
1. **Global optimality**: Local optima are global optima
2. **Efficient algorithms**: Many polynomial-time algorithms exist
3. **Duality theory**: Provides powerful analysis tools
4. **Wide applicability**: Many ML problems can be formulated as convex optimization

## Convex Sets

### Definition
A set C ⊆ ℝⁿ is convex if for any x₁, x₂ ∈ C and any λ ∈ [0,1]:

λx₁ + (1-λ)x₂ ∈ C

**Intuitive meaning**: The line segment between any two points in the set lies entirely within the set.

### Examples of Convex Sets

**1. Hyperplanes**: {x | aᵀx = b}
**2. Half-spaces**: {x | aᵀx ≤ b}
**3. Balls**: {x | ||x - c|| ≤ r}
**4. Polyhedra**: {x | Ax ≤ b}
**5. Positive semidefinite cone**: {X | X ⪰ 0}

### Operations Preserving Convexity
1. **Intersection**: If C₁ and C₂ are convex, then C₁ ∩ C₂ is convex
2. **Affine transformation**: If C is convex and f(x) = Ax + b, then f(C) is convex
3. **Minkowski sum**: C₁ + C₂ = {x₁ + x₂ | x₁ ∈ C₁, x₂ ∈ C₂}

## Convex Functions

### Definition
A function f: ℝⁿ → ℝ is convex if for any x₁, x₂ ∈ dom(f) and any λ ∈ [0,1]:

f(λx₁ + (1-λ)x₂) ≤ λf(x₁) + (1-λ)f(x₂)

### First-Order Condition
f is convex if and only if:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)

**Interpretation**: The function lies above its tangent hyperplane.

### Second-Order Condition
If f is twice differentiable, then f is convex if and only if:

∇²f(x) ⪰ 0 (positive semidefinite)

### Examples of Convex Functions

**1. Linear functions**: f(x) = aᵀx + b
**2. Quadratic functions**: f(x) = xᵀAx + bᵀx + c (if A ⪰ 0)
**3. Exponential**: f(x) = e^(ax)
**4. Negative logarithm**: f(x) = -log(x)
**5. Norms**: f(x) = ||x||_p for p ≥ 1
**6. Maximum**: f(x) = max{x₁, x₂, ..., xₙ}

## Convex Optimization Problems

### Standard Form
minimize f₀(x)
subject to fᵢ(x) ≤ 0, i = 1, ..., m
           hᵢ(x) = 0, i = 1, ..., p

Where:
- f₀, f₁, ..., fₘ are convex functions
- h₁, ..., hₚ are affine functions

### Examples

**1. Linear Programming (LP)**:
minimize cᵀx
subject to Ax ≤ b
           x ≥ 0

**2. Quadratic Programming (QP)**:
minimize (1/2)xᵀPx + qᵀx
subject to Ax ≤ b
           Cx = d

**3. Semidefinite Programming (SDP)**:
minimize tr(CX)
subject to tr(AᵢX) = bᵢ, i = 1, ..., m
           X ⪰ 0

## Duality Theory

### Lagrangian
For the optimization problem:
minimize f₀(x)
subject to fᵢ(x) ≤ 0, i = 1, ..., m
           hᵢ(x) = 0, i = 1, ..., p

The Lagrangian is:
L(x, λ, ν) = f₀(x) + Σᵢ₌₁ᵐ λᵢfᵢ(x) + Σᵢ₌₁ᵖ νᵢhᵢ(x)

Where λᵢ ≥ 0 are Lagrange multipliers.

### Dual Function
g(λ, ν) = inf_x L(x, λ, ν)

### Dual Problem
maximize g(λ, ν)
subject to λ ≥ 0

### Weak Duality
For any feasible x and (λ, ν) with λ ≥ 0:

f₀(x) ≥ g(λ, ν)

### Strong Duality
If the primal and dual optimal values are equal, we have strong duality.

**Conditions for strong duality**:
1. **Slater's condition**: There exists x such that fᵢ(x) < 0 for all i
2. **Linear constraints**: If all constraints are affine
3. **Convex quadratic**: If f₀ is quadratic and constraints are affine

## KKT Conditions

### Karush-Kuhn-Tucker Conditions
For optimal points x* and (λ*, ν*):

1. **Primal feasibility**: fᵢ(x*) ≤ 0, hᵢ(x*) = 0
2. **Dual feasibility**: λ* ≥ 0
3. **Complementary slackness**: λ*ᵢfᵢ(x*) = 0
4. **Stationarity**: ∇f₀(x*) + Σᵢ λ*ᵢ∇fᵢ(x*) + Σᵢ ν*ᵢ∇hᵢ(x*) = 0

### Interpretation
- **Primal feasibility**: x* satisfies constraints
- **Dual feasibility**: Lagrange multipliers are non-negative
- **Complementary slackness**: Either constraint is tight or multiplier is zero
- **Stationarity**: Gradient of Lagrangian is zero

## Linear Programming

### Standard Form
minimize cᵀx
subject to Ax = b
           x ≥ 0

### Dual Problem
maximize bᵀy
subject to Aᵀy ≤ c

### Example
**Primal**:
minimize 3x₁ + 2x₂
subject to x₁ + x₂ = 4
           x₁, x₂ ≥ 0

**Dual**:
maximize 4y
subject to y ≤ 3
           y ≤ 2

**Solution**: x₁ = 4, x₂ = 0, optimal value = 12

## Quadratic Programming

### Formulation
minimize (1/2)xᵀPx + qᵀx
subject to Ax ≤ b
           Cx = d

Where P ⪰ 0 (positive semidefinite).

### Example: Portfolio Optimization
minimize (1/2)xᵀΣx  (risk)
subject to μᵀx ≥ r  (return constraint)
           Σᵢ xᵢ = 1  (budget constraint)
           x ≥ 0  (no short selling)

Where:
- x is portfolio weights
- Σ is covariance matrix
- μ is expected returns
- r is minimum required return

## Interior Point Methods

### Basic Idea
Instead of staying on the boundary of the feasible region, interior point methods stay in the interior and approach the boundary asymptotically.

### Barrier Method
Transform constrained problem:
minimize f₀(x)
subject to fᵢ(x) ≤ 0

Into unconstrained problem:
minimize f₀(x) - (1/t)Σᵢ log(-fᵢ(x))

As t → ∞, the solution approaches the optimal solution.

### Primal-Dual Interior Point
Solve KKT conditions directly using Newton's method with appropriate step size to maintain feasibility.

## Applications in Machine Learning

### Support Vector Machines
**Primal problem**:
minimize (1/2)||w||² + CΣᵢ ξᵢ
subject to yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
           ξᵢ ≥ 0

**Dual problem**:
maximize Σᵢ αᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
subject to Σᵢ αᵢyᵢ = 0
           0 ≤ αᵢ ≤ C

### Logistic Regression
**Regularized logistic regression**:
minimize Σᵢ log(1 + exp(-yᵢxᵢᵀθ)) + λ||θ||²

This is convex because:
- Log-sum-exp is convex
- L2 norm is convex
- Sum of convex functions is convex

### LASSO
**LASSO problem**:
minimize (1/2)||Xθ - y||² + λ||θ||₁

This is convex because:
- Quadratic term is convex
- L1 norm is convex
- Sum of convex functions is convex

## Practice Problems

### Problem 1
Show that f(x) = x² is convex.

**Solution**:
Using second-order condition: f''(x) = 2 > 0, so f is convex.

Or using definition:
f(λx₁ + (1-λ)x₂) = (λx₁ + (1-λ)x₂)²
= λ²x₁² + 2λ(1-λ)x₁x₂ + (1-λ)²x₂²
≤ λ²x₁² + λ(1-λ)(x₁² + x₂²) + (1-λ)²x₂²
= λx₁² + (1-λ)x₂²
= λf(x₁) + (1-λ)f(x₂)

### Problem 2
Solve the LP:
minimize 2x₁ + 3x₂
subject to x₁ + x₂ ≥ 1
           x₁, x₂ ≥ 0

**Solution**:
Using graphical method or simplex method:
- Corner points: (1,0), (0,1)
- f(1,0) = 2, f(0,1) = 3
- Optimal solution: x₁ = 1, x₂ = 0, optimal value = 2

### Problem 3
Find the dual of:
minimize x₁ + 2x₂
subject to x₁ + x₂ ≥ 3
           x₁ - x₂ = 1
           x₁, x₂ ≥ 0

**Solution**:
Lagrangian: L = x₁ + 2x₂ + λ(3 - x₁ - x₂) + ν(1 - x₁ + x₂)
Dual function: g(λ, ν) = inf_{x≥0} L
Dual problem:
maximize 3λ + ν
subject to λ + ν ≤ 1
           λ - ν ≤ 2
           λ ≥ 0

## Key Takeaways
- Convex optimization problems have global optimality guarantees
- Duality theory provides powerful analysis tools
- KKT conditions characterize optimal solutions
- Many ML problems can be formulated as convex optimization
- Interior point methods provide efficient algorithms
- Understanding convexity is crucial for optimization

## Next Steps
In the next tutorial, we'll explore constrained optimization methods, including penalty methods, barrier methods, and augmented Lagrangian methods.
