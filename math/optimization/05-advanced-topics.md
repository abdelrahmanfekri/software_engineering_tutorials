# Optimization Theory Tutorial 05: Advanced Topics

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand proximal methods and their applications
- Implement coordinate descent algorithms
- Apply subgradient methods for non-smooth optimization
- Use online optimization for streaming data
- Implement distributed optimization algorithms
- Apply advanced optimization techniques to machine learning
- Understand convergence guarantees for advanced methods

## Introduction to Advanced Optimization

### Overview
This tutorial covers advanced optimization techniques that extend beyond basic gradient descent and convex optimization. These methods are essential for modern machine learning, large-scale optimization, and real-world applications.

### Topics Covered
1. **Proximal Methods**: Handle non-smooth objectives and constraints
2. **Coordinate Descent**: Efficient for high-dimensional problems
3. **Subgradient Methods**: Optimize non-differentiable functions
4. **Online Optimization**: Handle streaming data and changing objectives
5. **Distributed Optimization**: Scale to large datasets and multiple machines

## Proximal Methods

### Proximal Operator
The proximal operator of a function g is defined as:

prox_g(x) = argmin_y {g(y) + (1/2)||y - x||²}

### Proximal Gradient Method
For problems of the form:
minimize f(x) + g(x)

Where f is smooth and g is convex (possibly non-smooth).

**Algorithm**:
```
1. Initialize x^(0)
2. For k = 0, 1, 2, ...:
   a. Compute gradient: g^(k) = ∇f(x^(k))
   b. Update: x^(k+1) = prox_{αₖg}(x^(k) - αₖg^(k))
3. Return x^(k)
```

### Examples of Proximal Operators

**1. L1 Regularization (LASSO)**:
g(x) = λ||x||₁

prox_{λ||·||₁}(x) = sign(x) · max(|x| - λ, 0)

**2. L2 Regularization**:
g(x) = λ||x||₂²

prox_{λ||·||₂²}(x) = x/(1 + 2λ)

**3. Indicator Function**:
g(x) = I_C(x) = {0 if x ∈ C, ∞ otherwise}

prox_{I_C}(x) = proj_C(x) (projection onto set C)

### Fast Proximal Gradient (FISTA)
Accelerated version of proximal gradient method:

```
1. Initialize x^(0), y^(0) = x^(0), t₀ = 1
2. For k = 0, 1, 2, ...:
   a. x^(k+1) = prox_{αg}(y^(k) - α∇f(y^(k)))
   b. t_{k+1} = (1 + √(1 + 4tₖ²))/2
   c. y^(k+1) = x^(k+1) + (tₖ - 1)/t_{k+1} · (x^(k+1) - x^(k))
3. Return x^(k)
```

### Example: LASSO Regression
minimize (1/2)||Xθ - y||² + λ||θ||₁

**Implementation**:
```python
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def fista_lasso(X, y, lambda_reg, alpha=0.01, max_iter=1000):
    n, d = X.shape
    theta = np.zeros(d)
    y_var = theta.copy()
    t = 1.0
    
    for k in range(max_iter):
        # Gradient step
        grad = X.T @ (X @ y_var - y)
        theta_new = y_var - alpha * grad
        
        # Proximal step (soft thresholding)
        theta_new = soft_threshold(theta_new, alpha * lambda_reg)
        
        # FISTA acceleration
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y_var = theta_new + (t - 1) / t_new * (theta_new - theta)
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < 1e-6:
            break
            
        theta = theta_new
        t = t_new
    
    return theta
```

## Coordinate Descent

### Basic Algorithm
Optimize one coordinate at a time while keeping others fixed.

**Algorithm**:
```
1. Initialize x^(0)
2. For k = 0, 1, 2, ...:
   a. Choose coordinate i_k
   b. Update: x_{i_k}^(k+1) = argmin_{x_{i_k}} f(x^(k))
   c. Keep other coordinates fixed: x_j^(k+1) = x_j^(k) for j ≠ i_k
3. Return x^(k)
```

### Coordinate Selection Rules
1. **Cyclic**: i_k = k mod d
2. **Random**: i_k ~ Uniform{1, ..., d}
3. **Greedy**: i_k = argmax_j |∇_j f(x^(k))|
4. **Randomized greedy**: Choose with probability ∝ |∇_j f(x^(k))|

### Convergence Properties
- **Convex functions**: Converges to global minimum
- **Rate**: O(1/k) for convex, O(ρ^k) for strongly convex
- **Coordinate-wise**: Each update is often much cheaper than full gradient

### Example: LASSO with Coordinate Descent
For LASSO: minimize (1/2)||Xθ - y||² + λ||θ||₁

**Coordinate update**:
θ_j^(k+1) = S(θ_j^(k) - α∇_j f(θ^(k)), αλ)

Where S is the soft thresholding function.

**Implementation**:
```python
def coordinate_descent_lasso(X, y, lambda_reg, max_iter=1000):
    n, d = X.shape
    theta = np.zeros(d)
    
    for k in range(max_iter):
        theta_old = theta.copy()
        
        for j in range(d):
            # Compute gradient for coordinate j
            residual = y - X @ theta + X[:, j] * theta[j]
            grad_j = -X[:, j].T @ residual / n
            
            # Soft thresholding update
            theta[j] = soft_threshold(theta[j] - grad_j, lambda_reg)
        
        # Check convergence
        if np.linalg.norm(theta - theta_old) < 1e-6:
            break
    
    return theta
```

## Subgradient Methods

### Subgradient
For a convex function f, g is a subgradient at x if:

f(y) ≥ f(x) + g^T(y - x) for all y

The set of all subgradients at x is called the subdifferential ∂f(x).

### Subgradient Method
**Algorithm**:
```
1. Initialize x^(0)
2. For k = 0, 1, 2, ...:
   a. Choose subgradient: g^(k) ∈ ∂f(x^(k))
   b. Update: x^(k+1) = x^(k) - αₖg^(k)
   c. Choose step size αₖ
3. Return x^(k)
```

### Step Size Rules
1. **Constant**: αₖ = α
2. **Diminishing**: αₖ = α/√k
3. **Square summable**: Σₖ αₖ² < ∞, Σₖ αₖ = ∞
4. **Adaptive**: Based on function values

### Example: L1 Regularized Logistic Regression
minimize Σᵢ log(1 + exp(-yᵢxᵢᵀθ)) + λ||θ||₁

**Subgradient**: ∇_j f(θ) = -Σᵢ yᵢx_{ij}σ(-yᵢxᵢᵀθ) + λ·sign(θ_j)

**Implementation**:
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def subgradient_l1_logistic(X, y, lambda_reg, max_iter=1000):
    n, d = X.shape
    theta = np.zeros(d)
    
    for k in range(max_iter):
        # Compute subgradient
        scores = X @ theta
        probs = sigmoid(-y * scores)
        grad = -X.T @ (y * probs) / n
        
        # Add L1 subgradient
        for j in range(d):
            if theta[j] > 0:
                grad[j] += lambda_reg
            elif theta[j] < 0:
                grad[j] -= lambda_reg
            else:
                grad[j] += lambda_reg * np.sign(grad[j])
        
        # Update with diminishing step size
        alpha = 1.0 / np.sqrt(k + 1)
        theta = theta - alpha * grad
    
    return theta
```

## Online Optimization

### Online Learning Setting
- Data arrives sequentially: (x₁, y₁), (x₂, y₂), ...
- Make prediction ŷₜ based on xₜ
- Observe true label yₜ and suffer loss ℓ(ŷₜ, yₜ)
- Update model parameters

### Online Gradient Descent (OGD)
**Algorithm**:
```
1. Initialize θ₁
2. For t = 1, 2, ...:
   a. Receive xₜ
   b. Predict ŷₜ = f(xₜ; θₜ)
   c. Observe yₜ and suffer loss ℓ(ŷₜ, yₜ)
   d. Compute gradient: gₜ = ∇_θ ℓ(f(xₜ; θₜ), yₜ)
   e. Update: θₜ₊₁ = θₜ - αₜgₜ
3. Return final θ
```

### Regret Analysis
**Regret**: R_T = Σₜ₌₁ᵀ ℓ(ŷₜ, yₜ) - min_θ Σₜ₌₁ᵀ ℓ(f(xₜ; θ), yₜ)

**OGD Regret**: R_T ≤ O(√T) for convex losses with appropriate step size

### Example: Online Linear Regression
```python
def online_linear_regression(X_stream, y_stream, alpha=0.01):
    d = X_stream.shape[1]
    theta = np.zeros(d)
    predictions = []
    
    for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
        # Predict
        y_pred = x_t @ theta
        predictions.append(y_pred)
        
        # Compute gradient
        grad = x_t * (y_pred - y_t)
        
        # Update
        theta = theta - alpha * grad
    
    return theta, predictions
```

### Adaptive Online Methods
**AdaGrad Online**:
```
1. Initialize θ₁, G₀ = 0
2. For t = 1, 2, ...:
   a. Receive xₜ, predict ŷₜ = f(xₜ; θₜ)
   b. Observe yₜ, compute gₜ = ∇ℓ(ŷₜ, yₜ)
   c. Update: Gₜ = G_{t-1} + gₜ²
   d. Update: θₜ₊₁ = θₜ - (α/√Gₜ) · gₜ
```

## Distributed Optimization

### Problem Setup
minimize f(θ) = (1/m) Σᵢ₌₁ᵐ fᵢ(θ)

Where each fᵢ is local to machine i and we have m machines.

### Distributed Gradient Descent
**Algorithm**:
```
1. Initialize θ^(0) on all machines
2. For k = 0, 1, 2, ...:
   a. Each machine i computes: g_i^(k) = ∇f_i(θ^(k))
   b. Average gradients: ḡ^(k) = (1/m) Σᵢ g_i^(k)
   c. Each machine updates: θ^(k+1) = θ^(k) - αḡ^(k)
3. Return θ^(k)
```

### Communication Efficiency
**Challenges**:
- Communication overhead
- Synchronization delays
- Network bandwidth limitations

**Solutions**:
1. **Quantization**: Reduce precision of communicated gradients
2. **Sparsification**: Only communicate large gradient components
3. **Asynchronous updates**: Allow machines to update at different rates

### Federated Learning
**FedAvg Algorithm**:
```
1. Server initializes global model θ^(0)
2. For round t = 0, 1, 2, ...:
   a. Server sends θ^(t) to subset of clients
   b. Each client i updates: θ_i^(t+1) = θ^(t) - α∇f_i(θ^(t))
   c. Server averages: θ^(t+1) = (1/|S|) Σ_{i∈S} θ_i^(t+1)
3. Return final θ
```

**Implementation**:
```python
def federated_averaging(server_model, clients_data, rounds=100, local_epochs=1):
    global_model = server_model.copy()
    
    for round in range(rounds):
        # Select subset of clients
        selected_clients = np.random.choice(len(clients_data), 
                                          size=min(10, len(clients_data)), 
                                          replace=False)
        
        client_updates = []
        
        for client_idx in selected_clients:
            # Local training
            client_model = global_model.copy()
            X_client, y_client = clients_data[client_idx]
            
            for epoch in range(local_epochs):
                # Gradient descent on client data
                grad = compute_gradient(client_model, X_client, y_client)
                client_model = client_model - 0.01 * grad
            
            client_updates.append(client_model)
        
        # Average updates
        global_model = np.mean(client_updates, axis=0)
    
    return global_model
```

## Advanced Applications

### Sparse Optimization
**Problem**: minimize f(θ) + λ||θ||₀

**Approaches**:
1. **L1 relaxation**: Replace ||θ||₀ with ||θ||₁
2. **Greedy methods**: Forward/backward selection
3. **Proximal methods**: Using appropriate proximal operators

### Non-convex Optimization
**Escape from saddle points**:
- Add noise to gradients
- Use second-order information
- Momentum-based methods

### Multi-objective Optimization
**Pareto optimality**: Solution is Pareto optimal if no other solution dominates it in all objectives.

**Methods**:
1. **Weighted sum**: Combine objectives with weights
2. **ε-constraint**: Optimize one objective, constrain others
3. **Genetic algorithms**: NSGA-II, SPEA2

## Practice Problems

### Problem 1
Implement FISTA for LASSO regression and compare with standard gradient descent.

**Solution**:
```python
def fista_vs_gradient_descent(X, y, lambda_reg):
    # FISTA implementation (see earlier code)
    theta_fista = fista_lasso(X, y, lambda_reg)
    
    # Standard gradient descent with soft thresholding
    def gradient_descent_lasso(X, y, lambda_reg, alpha=0.01, max_iter=1000):
        theta = np.zeros(X.shape[1])
        for k in range(max_iter):
            grad = X.T @ (X @ theta - y)
            theta = theta - alpha * grad
            theta = soft_threshold(theta, alpha * lambda_reg)
            if np.linalg.norm(grad) < 1e-6:
                break
        return theta
    
    theta_gd = gradient_descent_lasso(X, y, lambda_reg)
    
    return theta_fista, theta_gd
```

### Problem 2
Implement coordinate descent for elastic net regression:
minimize (1/2)||Xθ - y||² + λ₁||θ||₁ + λ₂||θ||₂²

**Solution**:
```python
def coordinate_descent_elastic_net(X, y, lambda1, lambda2, max_iter=1000):
    n, d = X.shape
    theta = np.zeros(d)
    
    for k in range(max_iter):
        theta_old = theta.copy()
        
        for j in range(d):
            # Compute gradient for coordinate j
            residual = y - X @ theta + X[:, j] * theta[j]
            grad_j = -X[:, j].T @ residual / n
            
            # Elastic net update (soft thresholding with L2 regularization)
            numerator = grad_j + lambda2 * theta[j]
            theta[j] = soft_threshold(numerator, lambda1) / (1 + lambda2)
        
        if np.linalg.norm(theta - theta_old) < 1e-6:
            break
    
    return theta
```

### Problem 3
Implement online learning for logistic regression with adaptive learning rates.

**Solution**:
```python
def online_logistic_regression_adaptive(X_stream, y_stream, alpha=0.01):
    d = X_stream.shape[1]
    theta = np.zeros(d)
    G = np.zeros(d)  # Gradient accumulator
    
    for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
        # Predict
        score = x_t @ theta
        prob = sigmoid(score)
        y_pred = 1 if prob > 0.5 else 0
        
        # Compute gradient
        grad = x_t * (prob - y_t)
        
        # Update gradient accumulator
        G += grad**2
        
        # Update with adaptive learning rate
        theta = theta - alpha / np.sqrt(G + 1e-8) * grad
    
    return theta
```

## Convergence Analysis

### Proximal Methods
- **Convergence rate**: O(1/k) for convex problems
- **Acceleration**: FISTA achieves O(1/k²) rate
- **Conditions**: f smooth, g convex

### Coordinate Descent
- **Convergence**: Guaranteed for convex functions
- **Rate**: O(1/k) for convex, O(ρ^k) for strongly convex
- **Efficiency**: Each iteration is O(n) instead of O(nd)

### Subgradient Methods
- **Convergence**: To optimal set for convex functions
- **Rate**: O(1/√k) with appropriate step sizes
- **No acceleration**: Unlike gradient methods

### Online Methods
- **Regret bound**: O(√T) for convex losses
- **Adaptive methods**: Better regret for sparse gradients
- **No assumptions**: On data distribution

### Distributed Methods
- **Convergence**: Same as centralized with perfect communication
- **Communication complexity**: O(d) per round
- **Fault tolerance**: Asynchronous methods more robust

## Key Takeaways
- Proximal methods handle non-smooth objectives efficiently
- Coordinate descent is effective for high-dimensional problems
- Subgradient methods extend optimization to non-differentiable functions
- Online optimization enables learning from streaming data
- Distributed optimization scales to large datasets and multiple machines
- Advanced methods often combine multiple techniques for better performance
- Understanding convergence properties helps choose appropriate methods

## Conclusion
Advanced optimization methods extend the capabilities of basic gradient descent and convex optimization to handle real-world challenges including non-smooth objectives, high dimensions, streaming data, and distributed computing. These methods are essential for modern machine learning applications and large-scale optimization problems.

The key to success is understanding when and how to apply these advanced techniques, often combining multiple methods to achieve the best performance for specific problem characteristics.
