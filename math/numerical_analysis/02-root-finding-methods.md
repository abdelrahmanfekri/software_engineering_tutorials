# Numerical Analysis Tutorial 02: Root Finding Methods

## Learning Objectives
By the end of this tutorial, you will be able to:
- Implement bisection method for root finding
- Apply Newton's method and understand its convergence properties
- Use secant method as an alternative to Newton's method
- Understand fixed-point iteration and convergence criteria
- Analyze convergence rates and choose appropriate methods
- Apply root finding to machine learning optimization problems

## Introduction to Root Finding

### What is Root Finding?
Root finding is the process of finding solutions to equations of the form f(x) = 0. This is fundamental to:
- **Optimization**: Finding minima/maxima (where f'(x) = 0)
- **Machine Learning**: Training neural networks, parameter estimation
- **Scientific Computing**: Solving differential equations, modeling

### Types of Root Finding Methods
1. **Bracketing Methods**: Require initial interval containing root
   - Bisection method
   - False position method
2. **Open Methods**: Use single or multiple starting points
   - Newton's method
   - Secant method
   - Fixed-point iteration

## Bisection Method

### Algorithm
Given f(x) continuous on [a,b] with f(a)f(b) < 0:

```
1. Set a₀ = a, b₀ = b
2. For k = 0, 1, 2, ...
   a. Compute cₖ = (aₖ + bₖ)/2
   b. If f(cₖ) = 0, return cₖ
   c. If f(aₖ)f(cₖ) < 0, set aₖ₊₁ = aₖ, bₖ₊₁ = cₖ
   d. Else set aₖ₊₁ = cₖ, bₖ₊₁ = bₖ
3. Continue until |bₖ - aₖ| < tolerance
```

### Convergence Analysis
- **Convergence**: Guaranteed if f is continuous and f(a)f(b) < 0
- **Rate**: Linear convergence with rate 1/2
- **Error bound**: |x* - cₖ| ≤ (b₀ - a₀)/2^(k+1)

### Example: Finding √2
Find root of f(x) = x² - 2 = 0 on [1, 2]

```
k | aₖ    | bₖ    | cₖ    | f(cₖ)
--|-------|-------|-------|-------
0 | 1.0   | 2.0   | 1.5   | 0.25
1 | 1.0   | 1.5   | 1.25  | -0.4375
2 | 1.25  | 1.5   | 1.375 | -0.109375
3 | 1.375 | 1.5   | 1.4375| 0.06640625
```

After 10 iterations: x ≈ 1.4140625 (error < 0.001)

### Advantages and Disadvantages
**Advantages**:
- Always converges if root exists in interval
- Simple to implement
- No derivative required

**Disadvantages**:
- Slow convergence (linear rate)
- Requires initial bracketing interval
- May miss multiple roots

## Newton's Method

### Algorithm
Given f(x) differentiable and initial guess x₀:

```
1. For k = 0, 1, 2, ...
   a. Compute f(xₖ) and f'(xₖ)
   b. If f'(xₖ) = 0, stop (method fails)
   c. Set xₖ₊₁ = xₖ - f(xₖ)/f'(xₖ)
   d. If |xₖ₊₁ - xₖ| < tolerance, stop
2. Return xₖ₊₁
```

### Convergence Analysis
- **Convergence**: Not guaranteed (depends on initial guess)
- **Rate**: Quadratic convergence near simple root
- **Error**: |eₖ₊₁| ≈ |f''(x*)/(2f'(x*))| |eₖ|²

### Example: Finding √2 using Newton's Method
f(x) = x² - 2, f'(x) = 2x

Starting with x₀ = 1.5:
```
k | xₖ      | f(xₖ)    | f'(xₖ) | xₖ₊₁
--|---------|----------|--------|--------
0 | 1.5     | 0.25     | 3.0    | 1.416667
1 | 1.416667| 0.006944 | 2.8333 | 1.414216
2 | 1.414216| 0.000006 | 2.8284 | 1.414214
```

After 3 iterations: x ≈ 1.414214 (error < 10^(-6))

### Advantages and Disadvantages
**Advantages**:
- Very fast convergence (quadratic)
- Works well near simple roots
- Extends to systems of equations

**Disadvantages**:
- Requires derivative computation
- May not converge for poor initial guess
- Can diverge or cycle

## Secant Method

### Algorithm
Given f(x) and two initial points x₀, x₁:

```
1. For k = 1, 2, 3, ...
   a. Compute xₖ₊₁ = xₖ - f(xₖ)(xₖ - xₖ₋₁)/(f(xₖ) - f(xₖ₋₁))
   b. If |xₖ₊₁ - xₖ| < tolerance, stop
2. Return xₖ₊₁
```

### Convergence Analysis
- **Convergence**: Superlinear (rate ≈ 1.618, golden ratio)
- **Advantage**: No derivative required
- **Disadvantage**: Slower than Newton's method

### Example: Finding √2 using Secant Method
f(x) = x² - 2

Starting with x₀ = 1.0, x₁ = 2.0:
```
k | xₖ      | f(xₖ) | xₖ₊₁
--|---------|-------|--------
1 | 2.0     | 2.0   | 1.333333
2 | 1.333333| -0.222| 1.400000
3 | 1.400000| -0.040| 1.414634
4 | 1.414634| 0.0012| 1.414211
```

After 4 iterations: x ≈ 1.414211

## Fixed-Point Iteration

### Definition
A fixed point of function g(x) is a value x* such that g(x*) = x*.

To solve f(x) = 0, rewrite as x = g(x) and iterate:
xₖ₊₁ = g(xₖ)

### Convergence Criteria
The fixed-point iteration converges if:
1. g is continuous on [a,b]
2. g maps [a,b] into [a,b]
3. |g'(x)| ≤ k < 1 for all x in [a,b]

### Example: Finding √2 using Fixed-Point Iteration
Rewrite x² - 2 = 0 as x = 2/x or x = (x + 2/x)/2

**Choice 1**: x = 2/x (diverges)
**Choice 2**: x = (x + 2/x)/2 (converges)

Starting with x₀ = 1.5:
```
k | xₖ      | xₖ₊₁
--|---------|--------
0 | 1.5     | 1.416667
1 | 1.416667| 1.414216
2 | 1.414216| 1.414214
```

### Convergence Rate
- **Linear convergence**: |eₖ₊₁| ≤ k|eₖ|
- **Rate depends on**: |g'(x*)| at the fixed point

## Convergence Analysis

### Order of Convergence
For sequence {xₖ} converging to x*:

**Linear**: |eₖ₊₁| ≤ C|eₖ| for some C < 1
**Quadratic**: |eₖ₊₁| ≤ C|eₖ|²
**Superlinear**: lim(k→∞) |eₖ₊₁|/|eₖ| = 0

### Convergence Rates Comparison
1. **Bisection**: Linear, rate = 1/2
2. **Fixed-point**: Linear, rate = |g'(x*)|
3. **Secant**: Superlinear, rate ≈ 1.618
4. **Newton**: Quadratic, rate = |f''(x*)/(2f'(x*))|

## Applications to Machine Learning

### Gradient Descent
Gradient descent for minimizing f(x):
xₖ₊₁ = xₖ - α∇f(xₖ)

This is a fixed-point iteration with g(x) = x - α∇f(x)

**Convergence**: Requires α < 2/L where L is Lipschitz constant

### Newton's Method for Optimization
For minimizing f(x), Newton's method finds critical points:
xₖ₊₁ = xₖ - (∇²f(xₖ))^(-1)∇f(xₖ)

**Advantages**:
- Quadratic convergence near minimum
- Uses second-order information

**Disadvantages**:
- Expensive Hessian computation
- May converge to saddle points

### Logistic Regression
Solving maximum likelihood equations:
∑ᵢ(yᵢ - σ(xᵢᵀβ))xᵢ = 0

Can use Newton's method (IRLS algorithm) or gradient descent.

## Practical Implementation Considerations

### Stopping Criteria
1. **Absolute error**: |xₖ₊₁ - xₖ| < ε
2. **Relative error**: |xₖ₊₁ - xₖ|/|xₖ₊₁| < ε
3. **Function value**: |f(xₖ₊₁)| < ε

### Robustness
1. **Multiple starting points**: Try different initial guesses
2. **Hybrid methods**: Combine bracketing and open methods
3. **Monitoring**: Track convergence behavior
4. **Fallback**: Use slower but more reliable methods

### Common Pitfalls
1. **Poor initial guess**: Can lead to divergence
2. **Multiple roots**: May converge to wrong root
3. **Singular derivatives**: Newton's method fails
4. **Slow convergence**: May need better method

## Practice Problems

### Problem 1
Find the root of f(x) = x³ - x - 1 = 0 using:
a) Bisection method on [1, 2]
b) Newton's method starting at x₀ = 1.5

**Solutions**:
a) Bisection: After 10 iterations, x ≈ 1.3247
b) Newton: After 3 iterations, x ≈ 1.3247

### Problem 2
Show that g(x) = cos(x) has a unique fixed point on [0, π/2].

**Solution**:
- g is continuous and maps [0, π/2] to [0, 1] ⊆ [0, π/2]
- |g'(x)| = |sin(x)| ≤ 1, with strict inequality on (0, π/2)
- By Banach fixed-point theorem, unique fixed point exists

### Problem 3
Implement secant method for f(x) = e^x - 3x = 0.

**Solution**:
Starting with x₀ = 0, x₁ = 1:
```
k | xₖ      | xₖ₊₁
--|---------|--------
1 | 1.0     | 0.6137
2 | 0.6137  | 0.6208
3 | 0.6208  | 0.6191
```

Root ≈ 0.6191

## Key Takeaways
- Choose method based on convergence rate and derivative availability
- Bisection is reliable but slow
- Newton's method is fast but requires derivatives
- Secant method balances speed and simplicity
- Fixed-point iteration is general but may not converge
- Consider hybrid approaches for robustness

## Next Steps
In the next tutorial, we'll explore numerical linear algebra methods including LU decomposition, QR decomposition, and eigenvalue computation.
