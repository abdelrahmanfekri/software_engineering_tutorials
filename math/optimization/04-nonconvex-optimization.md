# Optimization Theory Tutorial 04: Non-convex Optimization

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand challenges in non-convex optimization
- Identify local minima, saddle points, and global optima
- Apply global optimization methods
- Use simulated annealing and genetic algorithms
- Implement Bayesian optimization
- Handle non-convex problems in machine learning
- Understand escape mechanisms from poor local minima

## Introduction to Non-convex Optimization

### What is Non-convex Optimization?
Non-convex optimization involves minimizing functions that are not convex, meaning they may have multiple local minima, saddle points, and other complex landscape features.

**Key Challenge**: Local optimization methods may get trapped in poor local minima.

### Why is Non-convex Optimization Important?
1. **Real-world problems**: Many practical problems are non-convex
2. **Neural networks**: Deep learning involves highly non-convex optimization
3. **Feature selection**: Many combinatorial optimization problems
4. **Hyperparameter tuning**: Often involves non-convex objective functions

## Landscape Analysis

### Types of Critical Points

**1. Local Minimum**: f(x*) ≤ f(x) for all x in neighborhood of x*
**2. Global Minimum**: f(x*) ≤ f(x) for all x in domain
**3. Local Maximum**: f(x*) ≥ f(x) for all x in neighborhood of x*
**4. Saddle Point**: Neither minimum nor maximum (∇f(x*) = 0, ∇²f(x*) indefinite)

### Hessian Eigenvalue Analysis
For critical point x* with ∇f(x*) = 0:

- **Local minimum**: All eigenvalues of ∇²f(x*) > 0
- **Local maximum**: All eigenvalues of ∇²f(x*) < 0  
- **Saddle point**: Some eigenvalues > 0, some < 0

### Example: Rosenbrock Function
f(x,y) = 100(y - x²)² + (1 - x)²

**Properties**:
- Non-convex
- Global minimum at (1,1)
- Narrow curved valley makes optimization difficult
- Common test function for optimization algorithms

## Local Optimization Methods

### Gradient Descent Variants
Standard gradient descent can work for non-convex functions but may get trapped in local minima.

**Improvements**:
1. **Momentum**: Helps escape shallow local minima
2. **Adaptive learning rates**: Better navigation of complex landscapes
3. **Restart mechanisms**: Multiple random initializations

### Newton's Method
x^(k+1) = x^(k) - [∇²f(x^(k))]^(-1)∇f(x^(k))

**Issues with non-convex functions**:
- Hessian may not be positive definite
- May converge to saddle points
- Computationally expensive for high dimensions

### Modified Newton Methods
**1. Regularized Newton**: Add regularization term to ensure positive definiteness
**2. Trust region methods**: Limit step size to maintain descent
**3. Line search methods**: Ensure sufficient decrease

## Global Optimization Methods

### Random Search
**Algorithm**:
```
1. Sample points x₁, x₂, ..., xₙ uniformly from domain
2. Evaluate f(xᵢ) for all i
3. Return x* = argminᵢ f(xᵢ)
```

**Pros**: Simple, parallelizable, no assumptions on f
**Cons**: Inefficient, no guarantees for continuous problems

### Multi-start Methods
**Algorithm**:
```
1. For i = 1 to N:
   a. Choose random starting point x₀^(i)
   b. Run local optimizer from x₀^(i) to get x*^(i)
2. Return x* = argminᵢ f(x*^(i))
```

**Advantages**: Simple, can find multiple local minima
**Disadvantages**: No guarantee of finding global optimum

### Simulated Annealing

### Basic Algorithm
```
1. Initialize x^(0), temperature T₀, cooling schedule
2. For k = 0, 1, 2, ...:
   a. Generate candidate x' from neighborhood of x^(k)
   b. Compute Δf = f(x') - f(x^(k))
   c. Accept x' with probability:
      P = min(1, exp(-Δf/Tₖ))
   d. Update temperature: Tₖ₊₁ = α·Tₖ
   e. Check termination
3. Return best solution found
```

### Cooling Schedules
1. **Geometric**: Tₖ = T₀·αᵏ
2. **Linear**: Tₖ = T₀ - k·β
3. **Logarithmic**: Tₖ = T₀/log(k+2)
4. **Adaptive**: Adjust based on acceptance ratio

### Example Implementation
```python
def simulated_annealing(f, x0, bounds, T0=100, alpha=0.95, max_iter=1000):
    x = x0.copy()
    best_x = x.copy()
    best_f = f(x)
    T = T0
    
    for k in range(max_iter):
        # Generate candidate
        x_new = x + np.random.normal(0, 0.1, x.shape)
        x_new = np.clip(x_new, bounds[0], bounds[1])
        
        # Evaluate
        f_new = f(x_new)
        delta_f = f_new - f(x)
        
        # Accept or reject
        if delta_f < 0 or np.random.random() < np.exp(-delta_f/T):
            x = x_new
            if f_new < best_f:
                best_x = x.copy()
                best_f = f_new
        
        # Cool down
        T *= alpha
        
        if T < 1e-6:
            break
            
    return best_x, best_f
```

## Genetic Algorithms

### Basic Concepts
- **Population**: Set of candidate solutions
- **Chromosome**: Representation of a solution
- **Fitness**: Objective function value
- **Selection**: Choose parents for reproduction
- **Crossover**: Combine parent solutions
- **Mutation**: Randomly modify solutions

### Algorithm
```
1. Initialize population P₀ of size N
2. For generation g = 0, 1, 2, ...:
   a. Evaluate fitness f(xᵢ) for all xᵢ ∈ P₍
   b. Select parents for reproduction
   c. Create offspring via crossover and mutation
   d. Replace population: P₍₊₁ = new population
   e. Check termination
3. Return best individual found
```

### Selection Methods
1. **Roulette wheel**: Probability ∝ fitness
2. **Tournament**: Randomly select k individuals, choose best
3. **Rank selection**: Probability based on rank, not fitness
4. **Elitism**: Keep best individuals in next generation

### Crossover Operations
1. **Single-point**: Split at random point
2. **Two-point**: Split at two random points
3. **Uniform**: Randomly choose from each parent
4. **Arithmetic**: Linear combination (for continuous variables)

### Example: Continuous Optimization
```python
def genetic_algorithm(f, bounds, pop_size=50, max_generations=100):
    # Initialize population
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, len(bounds[0])))
    
    for gen in range(max_generations):
        # Evaluate fitness
        fitness = np.array([f(x) for x in pop])
        
        # Selection (tournament)
        new_pop = []
        for _ in range(pop_size):
            # Tournament selection
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent1 = pop[idx1] if fitness[idx1] < fitness[idx2] else pop[idx2]
            
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent2 = pop[idx1] if fitness[idx1] < fitness[idx2] else pop[idx2]
            
            # Crossover (arithmetic)
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # Mutation
            if np.random.random() < 0.1:  # 10% mutation rate
                child += np.random.normal(0, 0.1, child.shape)
                child = np.clip(child, bounds[0], bounds[1])
            
            new_pop.append(child)
        
        pop = np.array(new_pop)
        
        # Check convergence
        if np.std(fitness) < 1e-6:
            break
    
    # Return best solution
    best_idx = np.argmin([f(x) for x in pop])
    return pop[best_idx], f(pop[best_idx])
```

## Bayesian Optimization

### Basic Idea
Use Gaussian processes to model the objective function and guide search toward promising regions.

### Algorithm
```
1. Initialize with random points and evaluations
2. For t = 1, 2, ...:
   a. Fit GP model to observed data
   b. Choose next point using acquisition function
   c. Evaluate objective at chosen point
   d. Add to dataset
3. Return best point found
```

### Gaussian Process Model
For function f(x) with observations {(xᵢ, yᵢ)}:

f(x) ~ GP(μ(x), k(x,x'))

Where:
- μ(x) is the mean function
- k(x,x') is the covariance function (kernel)

### Acquisition Functions
**1. Expected Improvement (EI)**:
EI(x) = E[max(f(x) - f(x⁺), 0)]

**2. Upper Confidence Bound (UCB)**:
UCB(x) = μ(x) + β·σ(x)

**3. Probability of Improvement (PI)**:
PI(x) = P(f(x) < f(x⁺))

### Example Implementation
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def bayesian_optimization(f, bounds, n_init=10, n_iter=50):
    # Initial random sampling
    X = np.random.uniform(bounds[0], bounds[1], (n_init, len(bounds[0])))
    y = np.array([f(x) for x in X])
    
    # GP model
    kernel = RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel)
    
    for i in range(n_iter):
        # Fit GP
        gp.fit(X, y)
        
        # Acquisition function (EI)
        def acquisition(x):
            mu, sigma = gp.predict([x], return_std=True)
            f_best = np.min(y)
            z = (f_best - mu) / sigma
            ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei[0]
        
        # Optimize acquisition function
        from scipy.optimize import minimize
        result = minimize(lambda x: -acquisition(x), 
                         x0=np.random.uniform(bounds[0], bounds[1], len(bounds[0])),
                         bounds=list(zip(bounds[0], bounds[1])))
        
        x_new = result.x
        y_new = f(x_new)
        
        # Update dataset
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
    
    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]
```

## Applications in Machine Learning

### Neural Network Training
**Challenges**:
- Highly non-convex loss landscapes
- Many local minima
- Saddle points in high dimensions
- Vanishing/exploding gradients

**Solutions**:
1. **Good initialization**: Xavier, He initialization
2. **Batch normalization**: Stabilizes training
3. **Residual connections**: Helps with gradient flow
4. **Ensemble methods**: Multiple random initializations

### Hyperparameter Optimization
**Objective**: Minimize validation loss over hyperparameter space

**Methods**:
1. **Grid search**: Exhaustive but expensive
2. **Random search**: Often better than grid search
3. **Bayesian optimization**: Efficient for expensive evaluations
4. **Genetic algorithms**: Good for discrete hyperparameters

### Feature Selection
**Problem**: Select subset of features that maximizes performance

**Formulation**:
maximize f(x) = performance with features x
subject to ||x||₀ ≤ k (sparsity constraint)
           x ∈ {0,1}^d (binary selection)

**Methods**:
1. **Greedy selection**: Forward/backward selection
2. **Genetic algorithms**: For subset optimization
3. **Simulated annealing**: For combinatorial optimization
4. **Relaxation methods**: Continuous relaxation + rounding

## Advanced Techniques

### Escape Mechanisms

**1. Noise Injection**:
Add noise to gradients to escape saddle points:
g^(k) = ∇f(x^(k)) + ε^(k)

**2. Second-order Information**:
Use Hessian information to identify and escape saddle points.

**3. Momentum with Escape**:
Combine momentum with random perturbations.

### Landscape-aware Methods

**1. Gradient-free methods**: When gradients are unavailable or unreliable
**2. Multi-objective optimization**: Handle multiple conflicting objectives
**3. Robust optimization**: Optimize under uncertainty

### Distributed Optimization

**1. Parallel multi-start**: Run multiple local optimizations in parallel
**2. Population-based methods**: Distributed genetic algorithms
**3. Bayesian optimization**: Parallel acquisition function optimization

## Practice Problems

### Problem 1
Minimize the Rosenbrock function using different methods:
f(x,y) = 100(y - x²)² + (1 - x)²

Compare gradient descent, simulated annealing, and genetic algorithm.

**Solution**:
```python
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Gradient descent
def gradient_descent(f, grad_f, x0, alpha=0.001, max_iter=1000):
    x = x0.copy()
    for i in range(max_iter):
        g = grad_f(x)
        x = x - alpha * g
        if np.linalg.norm(g) < 1e-6:
            break
    return x

# Compare methods
x0 = np.array([-1.2, 1.0])
bounds = [(-2, 2), (-2, 2)]

# Results will vary - genetic algorithm and simulated annealing
# are more likely to find global minimum (1,1)
```

### Problem 2
Optimize hyperparameters for a neural network using Bayesian optimization.

**Solution**:
```python
def objective(params):
    # Train neural network with given hyperparameters
    learning_rate = params[0]
    batch_size = int(params[1])
    hidden_units = int(params[2])
    
    # Train and return validation loss
    model = train_model(learning_rate, batch_size, hidden_units)
    return validate_model(model)

bounds = [(0.001, 0.1), (32, 256), (64, 512)]
best_params, best_loss = bayesian_optimization(objective, bounds)
```

### Problem 3
Use genetic algorithm to solve the traveling salesman problem.

**Solution**:
```python
def tsp_fitness(route, distance_matrix):
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i]][route[(i+1) % len(route)]]
    return total_distance

def tsp_genetic_algorithm(distance_matrix, pop_size=100, max_generations=500):
    n_cities = len(distance_matrix)
    
    # Initialize population with random permutations
    pop = [np.random.permutation(n_cities) for _ in range(pop_size)]
    
    for gen in range(max_generations):
        # Evaluate fitness
        fitness = [tsp_fitness(route, distance_matrix) for route in pop]
        
        # Selection and reproduction
        new_pop = []
        for _ in range(pop_size):
            # Tournament selection
            parent1 = tournament_selection(pop, fitness)
            parent2 = tournament_selection(pop, fitness)
            
            # Order crossover
            child = order_crossover(parent1, parent2)
            
            # Mutation (swap two cities)
            if np.random.random() < 0.1:
                child = swap_mutation(child)
            
            new_pop.append(child)
        
        pop = new_pop
    
    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]
```

## Convergence Analysis

### Global Convergence
- **Random search**: Converges to global optimum with probability 1 as n → ∞
- **Simulated annealing**: Converges to global optimum under certain conditions
- **Genetic algorithms**: No global convergence guarantee, but often effective in practice

### Local Convergence
- **Gradient descent**: Converges to local minimum or saddle point
- **Newton's method**: Quadratic convergence near local minimum
- **Trust region**: Global convergence to stationary point

### Computational Complexity
- **Random search**: O(n) evaluations
- **Multi-start**: O(N × T_local) where T_local is cost of local optimization
- **Simulated annealing**: O(T × C_eval) where T is number of iterations
- **Genetic algorithms**: O(G × P × C_eval) where G is generations, P is population size

## Key Takeaways
- Non-convex optimization is challenging due to multiple local optima
- Global optimization methods sacrifice efficiency for global optimality
- Local methods are faster but may get trapped in poor local minima
- Hybrid approaches combining global and local methods are often effective
- Understanding the landscape structure helps choose appropriate methods
- Machine learning applications often require specialized non-convex optimization techniques

## Next Steps
In the next tutorial, we'll explore advanced optimization topics including proximal methods, coordinate descent, subgradient methods, and distributed optimization.
