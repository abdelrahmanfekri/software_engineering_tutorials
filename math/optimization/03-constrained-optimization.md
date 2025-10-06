# Optimization Theory Tutorial 03: Constrained Optimization

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand different types of constraints and their formulations
- Apply penalty methods for constrained optimization
- Use barrier methods and interior point techniques
- Implement augmented Lagrangian methods
- Solve linear and quadratic programming problems
- Apply constrained optimization to machine learning problems

## Introduction to Constrained Optimization

### What is Constrained Optimization?
Constrained optimization involves finding the minimum of a function subject to equality and inequality constraints:

minimize f₀(x)
subject to fᵢ(x) ≤ 0, i = 1, ..., m
           hᵢ(x) = 0, i = 1, ..., p

### Types of Constraints
1. **Equality constraints**: h(x) = 0
2. **Inequality constraints**: g(x) ≤ 0
3. **Box constraints**: l ≤ x ≤ u
4. **Linear constraints**: Ax ≤ b, Cx = d
5. **Nonlinear constraints**: g(x) ≤ 0, h(x) = 0

## Penalty Methods

### Basic Idea
Transform constrained problem into unconstrained problem by adding penalty terms for constraint violations.

### Quadratic Penalty Method
For equality constraints h(x) = 0:

minimize f₀(x) + (μ/2)||h(x)||²

Where μ > 0 is the penalty parameter.

### Algorithm
```
1. Choose initial μ₀ > 0, μ > 1, x^(0)
2. For k = 0, 1, 2, ...:
   a. Solve: minimize f₀(x) + (μₖ/2)||h(x)||²
   b. Update: μₖ₊₁ = μ·μₖ
   c. Check convergence
3. Return x^(k)
```

### Example
minimize x₁² + x₂²
subject to x₁ + x₂ = 1

**Penalty function**: P(x) = x₁² + x₂² + (μ/2)(x₁ + x₂ - 1)²

**Gradient**: ∇P = [2x₁ + μ(x₁ + x₂ - 1), 2x₂ + μ(x₁ + x₂ - 1)]

**Solution**: x₁ = x₂ = μ/(2 + 2μ) → 1/2 as μ → ∞

## Barrier Methods

### Basic Idea
Keep iterates strictly feasible by using barrier functions that approach infinity at the boundary.

### Logarithmic Barrier Method
For inequality constraints g(x) ≤ 0:

minimize f₀(x) - (1/t)Σᵢ log(-gᵢ(x))

Where t > 0 is the barrier parameter.

### Algorithm
```
1. Choose initial t₀ > 0, μ > 1, x^(0) strictly feasible
2. For k = 0, 1, 2, ...:
   a. Solve: minimize f₀(x) - (1/tₖ)Σᵢ log(-gᵢ(x))
   b. Update: tₖ₊₁ = μ·tₖ
   c. Check convergence
3. Return x^(k)
```

### Example
minimize x₁² + x₂²
subject to x₁ + x₂ ≤ 1

**Barrier function**: B(x) = x₁² + x₂² - (1/t)log(1 - x₁ - x₂)

**Gradient**: ∇B = [2x₁ + 1/(t(1-x₁-x₂)), 2x₂ + 1/(t(1-x₁-x₂))]

## Augmented Lagrangian Method

### Basic Idea
Combine penalty and Lagrangian methods to handle both equality and inequality constraints efficiently.

### Method of Multipliers
For equality constraints h(x) = 0:

L_μ(x, λ) = f₀(x) + λᵀh(x) + (μ/2)||h(x)||²

**Update rule**: λ^(k+1) = λ^(k) + μh(x^(k))

### Algorithm
```
1. Choose μ₀ > 0, μ > 1, λ^(0)
2. For k = 0, 1, 2, ...:
   a. Solve: minimize L_μₖ(x, λ^(k))
   b. Update: λ^(k+1) = λ^(k) + μₖh(x^(k))
   c. Update: μₖ₊₁ = μ·μₖ (if needed)
   d. Check convergence
3. Return x^(k)
```

### Advantages
- Better convergence than penalty methods
- Handles ill-conditioning better
- Can converge with finite penalty parameter

## Linear Programming

### Standard Form
minimize cᵀx
subject to Ax = b
           x ≥ 0

### Simplex Method
1. **Initialization**: Find initial basic feasible solution
2. **Optimality test**: Check if current solution is optimal
3. **Pivoting**: Move to adjacent vertex if not optimal
4. **Termination**: Stop when optimal or unbounded

### Example
minimize 3x₁ + 2x₂
subject to x₁ + x₂ ≤ 4
           x₁ - x₂ ≤ 2
           x₁, x₂ ≥ 0

**Standard form**:
minimize 3x₁ + 2x₂
subject to x₁ + x₂ + x₃ = 4
           x₁ - x₂ + x₄ = 2
           x₁, x₂, x₃, x₄ ≥ 0

**Solution**: x₁ = 3, x₂ = 1, optimal value = 11

## Quadratic Programming

### Formulation
minimize (1/2)xᵀPx + qᵀx
subject to Ax ≤ b
           Cx = d

Where P ⪰ 0 (positive semidefinite).

### Active Set Method
1. **Identify active constraints**: Constraints that are binding at current point
2. **Solve subproblem**: Minimize over active set
3. **Check optimality**: Verify KKT conditions
4. **Update active set**: Add or remove constraints

### Example: Portfolio Optimization
minimize (1/2)xᵀΣx  (portfolio variance)
subject to μᵀx ≥ r  (minimum expected return)
           Σᵢ xᵢ = 1  (budget constraint)
           x ≥ 0  (no short selling)

Where:
- x is portfolio weights
- Σ is covariance matrix of returns
- μ is expected returns vector
- r is minimum required return

## Sequential Quadratic Programming (SQP)

### Basic Idea
Approximate the original problem by a sequence of quadratic programming problems.

### Newton's Method for Constrained Optimization
At each iteration, solve:

minimize (1/2)dᵀ∇²L(x, λ, ν)d + ∇f₀(x)ᵀd
subject to ∇h(x)ᵀd + h(x) = 0
           ∇g(x)ᵀd + g(x) ≤ 0

Where L is the Lagrangian and d is the search direction.

### Algorithm
```
1. Initialize x^(0), λ^(0), ν^(0)
2. For k = 0, 1, 2, ...:
   a. Compute Hessian ∇²L(x^(k), λ^(k), ν^(k))
   b. Solve QP subproblem for search direction d^(k)
   c. Perform line search: x^(k+1) = x^(k) + αₖd^(k)
   d. Update multipliers λ^(k+1), ν^(k+1)
   e. Check convergence
3. Return x^(k)
```

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

### LASSO with Constraints
minimize (1/2)||Xθ - y||² + λ||θ||₁
subject to Aθ ≤ b
           Cθ = d

### Neural Network Training with Constraints
minimize L(θ)  (training loss)
subject to ||θ||₂ ≤ R  (weight decay)
           θᵢ ≥ 0  (non-negative weights)

## Practice Problems

### Problem 1
Solve using penalty method:
minimize x₁² + x₂²
subject to x₁ + x₂ = 1

**Solution**:
Penalty function: P(x) = x₁² + x₂² + (μ/2)(x₁ + x₂ - 1)²

Setting ∇P = 0:
2x₁ + μ(x₁ + x₂ - 1) = 0
2x₂ + μ(x₁ + x₂ - 1) = 0

This gives: x₁ = x₂ = μ/(2 + 2μ)

As μ → ∞: x₁ = x₂ = 1/2

### Problem 2
Solve the LP using simplex method:
minimize 2x₁ + 3x₂
subject to x₁ + x₂ ≥ 1
           x₁, x₂ ≥ 0

**Solution**:
Convert to standard form:
minimize 2x₁ + 3x₂
subject to -x₁ - x₂ ≤ -1
           x₁, x₂ ≥ 0

Add slack variable:
minimize 2x₁ + 3x₂
subject to -x₁ - x₂ + x₃ = -1
           x₁, x₂, x₃ ≥ 0

Initial tableau:
```
x₁  x₂  x₃  RHS
-1  -1   1   -1
```

Basic solution: x₁ = 0, x₂ = 0, x₃ = -1 (infeasible)

Alternative approach: Use two-phase method or big-M method.

### Problem 3
Implement augmented Lagrangian method for:
minimize x₁² + x₂²
subject to x₁ + x₂ = 1

**Solution**:
```python
def augmented_lagrangian(f, h, grad_f, grad_h, x0, lambda0, mu0=1.0, rho=10.0, max_iter=100):
    x = x0.copy()
    lam = lambda0
    mu = mu0
    
    for k in range(max_iter):
        # Solve subproblem: minimize L_mu(x, lambda)
        def L(x_val):
            return f(x_val) + lam * h(x_val) + (mu/2) * h(x_val)**2
        
        def grad_L(x_val):
            return grad_f(x_val) + lam * grad_h(x_val) + mu * h(x_val) * grad_h(x_val)
        
        # Use gradient descent to solve subproblem
        x = gradient_descent(L, grad_L, x, alpha=0.1, max_iter=50)
        
        # Update multiplier
        lam = lam + mu * h(x)
        
        # Check convergence
        if abs(h(x)) < 1e-6:
            break
            
        # Increase penalty parameter if needed
        if abs(h(x)) > 0.1:
            mu *= rho
            
    return x, lam
```

## Convergence Analysis

### Penalty Methods
- **Convergence**: x^(k) → x* as μₖ → ∞
- **Rate**: Linear convergence under suitable conditions
- **Ill-conditioning**: Hessian condition number grows with μ

### Barrier Methods
- **Convergence**: x^(k) → x* as tₖ → ∞
- **Rate**: Superlinear convergence possible
- **Feasibility**: Maintains strict feasibility

### Augmented Lagrangian
- **Convergence**: Can converge with finite penalty parameter
- **Rate**: Quadratic convergence under suitable conditions
- **Robustness**: More robust than penalty methods

## Key Takeaways
- Penalty methods are simple but can be ill-conditioned
- Barrier methods maintain feasibility but require interior starting points
- Augmented Lagrangian methods combine advantages of both
- Linear and quadratic programming have specialized efficient algorithms
- SQP methods are powerful for general nonlinear programming
- Constrained optimization is fundamental to many ML applications

## Next Steps
In the next tutorial, we'll explore non-convex optimization, including methods for handling local minima, saddle points, and global optimization techniques.
