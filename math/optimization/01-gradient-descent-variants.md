# Optimization Theory Tutorial 01: Gradient Descent and Variants

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand gradient descent and its convergence properties
- Implement various gradient descent variants
- Apply momentum and adaptive learning rate methods
- Analyze convergence rates and stability
- Use optimization methods for machine learning problems
- Understand the role of optimization in neural network training

## Introduction to Gradient Descent

### What is Gradient Descent?
Gradient descent is an iterative optimization algorithm used to minimize a function by moving in the direction of steepest descent (negative gradient).

**Intuitive Definition**: Imagine rolling a ball down a hill - it naturally follows the steepest path downward.

### Mathematical Formulation
For a function f(x), gradient descent updates:

x^(t+1) = x^(t) - α∇f(x^(t))

Where:
- α is the learning rate (step size)
- ∇f(x^(t)) is the gradient at iteration t

### Convergence Conditions
For gradient descent to converge to a minimum:
1. **Convexity**: f should be convex (or at least locally convex)
2. **Lipschitz continuity**: ∇f should be Lipschitz continuous
3. **Learning rate**: α should be appropriately chosen

## Basic Gradient Descent

### Algorithm
```
1. Initialize x^(0) randomly
2. For t = 0, 1, 2, ...:
   a. Compute gradient: g^(t) = ∇f(x^(t))
   b. Update: x^(t+1) = x^(t) - αg^(t)
   c. Check convergence
3. Return x^(t)
```

### Example: Quadratic Function
Minimize f(x) = (x - 2)²

**Solution**:
- Gradient: ∇f(x) = 2(x - 2)
- Update rule: x^(t+1) = x^(t) - α·2(x^(t) - 2)
- With α = 0.1, starting from x^(0) = 0:
  - x^(1) = 0 - 0.1·2(0 - 2) = 0.4
  - x^(2) = 0.4 - 0.1·2(0.4 - 2) = 0.72
  - x^(3) = 0.72 - 0.1·2(0.72 - 2) = 0.976
  - ... converges to x* = 2

### Convergence Rate
For strongly convex functions with Lipschitz continuous gradients:
- **Linear convergence**: ||x^(t) - x*|| ≤ ρ^t ||x^(0) - x*||
- Where ρ < 1 depends on condition number

## Stochastic Gradient Descent (SGD)

### Motivation
When minimizing empirical risk:
f(θ) = (1/n) Σᵢ fᵢ(θ)

Instead of computing full gradient ∇f(θ) = (1/n) Σᵢ ∇fᵢ(θ), use stochastic approximation:

∇f(θ) ≈ ∇fᵢ(θ) for random i

### Algorithm
```
1. Initialize θ^(0)
2. For t = 0, 1, 2, ...:
   a. Sample random index i
   b. Compute gradient: g^(t) = ∇fᵢ(θ^(t))
   c. Update: θ^(t+1) = θ^(t) - α_t g^(t)
3. Return θ^(t)
```

### Learning Rate Scheduling
Common schedules:
1. **Constant**: α_t = α
2. **Decay**: α_t = α/(1 + βt)
3. **Square root**: α_t = α/√t
4. **Exponential**: α_t = α·γ^t

### Convergence Properties
- **Sublinear convergence**: O(1/√t) for convex functions
- **Linear convergence**: O(ρ^t) for strongly convex functions
- **Noise**: SGD introduces variance but reduces computational cost

## Mini-batch Gradient Descent

### Algorithm
```
1. Initialize θ^(0)
2. For t = 0, 1, 2, ...:
   a. Sample mini-batch B of size m
   b. Compute gradient: g^(t) = (1/m) Σ_{i∈B} ∇fᵢ(θ^(t))
   c. Update: θ^(t+1) = θ^(t) - α g^(t)
3. Return θ^(t)
```

### Trade-offs
- **Batch size m = 1**: SGD (high variance, low cost)
- **Batch size m = n**: Full gradient descent (low variance, high cost)
- **Mini-batch**: Balance between variance and computational cost

## Momentum Methods

### Momentum
Adds a momentum term to smooth out oscillations:

v^(t+1) = βv^(t) + ∇f(θ^(t))
θ^(t+1) = θ^(t) - αv^(t+1)

Where β ∈ [0,1) is the momentum coefficient.

### Nesterov Accelerated Gradient (NAG)
Predicts the future position:

v^(t+1) = βv^(t) + ∇f(θ^(t) - αβv^(t))
θ^(t+1) = θ^(t) - αv^(t+1)

### Benefits
- **Faster convergence**: Especially for ill-conditioned problems
- **Reduced oscillations**: Smoother optimization path
- **Better generalization**: Often leads to better solutions

## Adaptive Learning Rate Methods

### AdaGrad
Adapts learning rate per parameter:

G^(t) = G^(t-1) + (g^(t))²
θ^(t+1) = θ^(t) - α/√(G^(t) + ε) · g^(t)

**Properties**:
- Learning rate decreases over time
- Good for sparse gradients
- Can stop learning too early

### RMSprop
Addresses AdaGrad's diminishing learning rate:

G^(t) = βG^(t-1) + (1-β)(g^(t))²
θ^(t+1) = θ^(t) - α/√(G^(t) + ε) · g^(t)

### Adam (Adaptive Moment Estimation)
Combines momentum and adaptive learning rate:

m^(t) = β₁m^(t-1) + (1-β₁)g^(t)  (first moment)
v^(t) = β₂v^(t-1) + (1-β₂)(g^(t))²  (second moment)
m̂^(t) = m^(t)/(1-β₁^t)  (bias correction)
v̂^(t) = v^(t)/(1-β₂^t)  (bias correction)
θ^(t+1) = θ^(t) - α·m̂^(t)/√(v̂^(t) + ε)

**Default parameters**: β₁ = 0.9, β₂ = 0.999, α = 0.001

## Convergence Analysis

### Convex Functions
For convex functions with Lipschitz continuous gradients:

**Gradient Descent**: O(1/t) convergence rate
**SGD**: O(1/√t) convergence rate
**Momentum**: O(1/t) with better constants

### Strongly Convex Functions
For μ-strongly convex functions:

**Gradient Descent**: O((1-μα)^t) linear convergence
**SGD**: O(ρ^t) linear convergence with ρ < 1

### Non-convex Functions
- Convergence to stationary points (where ∇f = 0)
- May converge to saddle points or local minima
- No global convergence guarantees

## Practical Considerations

### Learning Rate Selection
1. **Start large**: α = 0.1 or 0.01
2. **Use learning rate scheduling**: Decrease over time
3. **Monitor loss**: Adjust if oscillating or not decreasing
4. **Use adaptive methods**: Adam, RMSprop often work well

### Batch Size Selection
1. **Small datasets**: Use full batch or large mini-batches
2. **Large datasets**: Use small mini-batches (32-256)
3. **Memory constraints**: Choose largest batch that fits
4. **Convergence speed**: Larger batches often converge faster

### Regularization
- **L2 regularization**: Adds λ||θ||² to objective
- **Early stopping**: Stop when validation loss increases
- **Dropout**: Randomly set some parameters to zero

## Applications in Machine Learning

### Linear Regression
Minimize: L(θ) = (1/2n)||Xθ - y||²

**Gradient**: ∇L(θ) = (1/n)Xᵀ(Xθ - y)

### Logistic Regression
Minimize: L(θ) = -(1/n)Σᵢ[yᵢ log σ(xᵢᵀθ) + (1-yᵢ)log(1-σ(xᵢᵀθ))]

**Gradient**: ∇L(θ) = (1/n)Xᵀ(σ(Xθ) - y)

### Neural Networks
Use backpropagation to compute gradients:
1. Forward pass: compute predictions
2. Backward pass: compute gradients using chain rule
3. Update parameters using gradient descent

## Practice Problems

### Problem 1
Implement gradient descent to minimize f(x,y) = x² + 2y² + 2xy + 2x + 4y + 1

**Solution**:
- Gradient: ∇f = [2x + 2y + 2, 4y + 2x + 4]
- Starting point: (0, 0)
- Learning rate: α = 0.1
- Update: (x,y)^(t+1) = (x,y)^(t) - 0.1∇f(x,y)^(t)

### Problem 2
Compare convergence of SGD vs full gradient descent for linear regression with n=1000, d=10

**Solution**:
- Full GD: O(1/t) convergence, high computational cost per iteration
- SGD: O(1/√t) convergence, low computational cost per iteration
- SGD often reaches good solutions faster in wall-clock time

### Problem 3
Implement Adam optimizer for minimizing f(θ) = ||Aθ - b||²

**Solution**:
```python
def adam_optimizer(A, b, theta0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
    m = np.zeros_like(theta0)
    v = np.zeros_like(theta0)
    theta = theta0.copy()
    
    for t in range(max_iter):
        # Compute gradient
        grad = 2 * A.T @ (A @ theta - b)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**(t+1))
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2**(t+1))
        
        # Update parameters
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)
        
        if np.linalg.norm(grad) < 1e-6:
            break
            
    return theta
```

## Key Takeaways
- Gradient descent is fundamental to machine learning optimization
- SGD reduces computational cost but introduces variance
- Momentum methods improve convergence speed
- Adaptive learning rate methods often work better in practice
- Understanding convergence properties helps choose appropriate methods
- Implementation details matter for practical performance

## Next Steps
In the next tutorial, we'll explore convex optimization, including convex sets, convex functions, and duality theory.
