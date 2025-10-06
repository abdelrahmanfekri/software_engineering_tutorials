# Numerical Analysis Tutorial 04: Numerical Integration

## Learning Objectives
By the end of this tutorial, you will be able to:
- Implement trapezoidal rule for numerical integration
- Apply Simpson's rule for higher accuracy
- Use Gaussian quadrature for optimal polynomial integration
- Implement Monte Carlo integration for high-dimensional problems
- Apply adaptive methods for automatic error control
- Use numerical integration in machine learning applications

## Introduction to Numerical Integration

### What is Numerical Integration?
Numerical integration (quadrature) approximates definite integrals:
∫ₐᵇ f(x) dx ≈ ∑ᵢ wᵢ f(xᵢ)

Where:
- xᵢ are quadrature points (nodes)
- wᵢ are quadrature weights
- The approximation depends on the choice of points and weights

### Why Numerical Integration?
- **Analytical solutions**: Often don't exist or are too complex
- **Machine Learning**: Computing expectations, marginal distributions
- **Scientific Computing**: Solving differential equations
- **Finance**: Option pricing, risk calculations

## Newton-Cotes Formulas

### Trapezoidal Rule

#### Basic Formula
For n subintervals of [a,b]:
∫ₐᵇ f(x) dx ≈ (b-a)/2n [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]

Where xᵢ = a + ih, h = (b-a)/n

#### Algorithm
```
1. Set h = (b-a)/n
2. Initialize sum = f(a) + f(b)
3. For i = 1 to n-1:
   sum = sum + 2*f(a + i*h)
4. Return (h/2) * sum
```

#### Example: ∫₀¹ x² dx
Exact: 1/3 ≈ 0.3333

For n = 4:
- h = 0.25
- Points: 0, 0.25, 0.5, 0.75, 1.0
- Values: 0, 0.0625, 0.25, 0.5625, 1.0
- Approximation: 0.25/2 * [0 + 2(0.0625 + 0.25 + 0.5625) + 1.0] = 0.34375

#### Error Analysis
Error ≤ (b-a)³|f''(ξ)|/(12n²) for some ξ in [a,b]

### Simpson's Rule

#### Basic Formula
For n even subintervals:
∫ₐᵇ f(x) dx ≈ (b-a)/3n [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]

Pattern: 1, 4, 2, 4, 2, ..., 4, 1

#### Algorithm
```
1. Set h = (b-a)/n (n must be even)
2. Initialize sum = f(a) + f(b)
3. For i = 1 to n-1:
   if i is odd: sum = sum + 4*f(a + i*h)
   else: sum = sum + 2*f(a + i*h)
4. Return (h/3) * sum
```

#### Example: ∫₀¹ x² dx
For n = 4:
- Approximation: 0.25/3 * [0 + 4(0.0625) + 2(0.25) + 4(0.5625) + 1.0] = 0.3333

#### Error Analysis
Error ≤ (b-a)⁵|f⁽⁴⁾(ξ)|/(2880n⁴) for some ξ in [a,b]

### Comparison of Methods
| Method | Order | Error | Best for |
|--------|-------|-------|----------|
| Trapezoidal | 2 | O(h²) | Linear functions |
| Simpson's | 4 | O(h⁴) | Cubic polynomials |

## Gaussian Quadrature

### Theory
Gaussian quadrature uses optimal points and weights to exactly integrate polynomials of degree 2n-1 using n points.

For interval [-1, 1]:
∫₋₁¹ f(x) dx ≈ ∑ᵢ₌₁ⁿ wᵢ f(xᵢ)

Where xᵢ are roots of Legendre polynomials Pₙ(x).

### Legendre Polynomials
Recurrence relation:
P₀(x) = 1
P₁(x) = x
(n+1)Pₙ₊₁(x) = (2n+1)xPₙ(x) - nPₙ₋₁(x)

### Weights
wᵢ = 2/((1-xᵢ²)[P'ₙ(xᵢ)]²)

### Example: 2-Point Gaussian Quadrature
Points: x₁ = -1/√3, x₂ = 1/√3
Weights: w₁ = w₂ = 1

∫₋₁¹ f(x) dx ≈ f(-1/√3) + f(1/√3)

### General Interval [a,b]
Transform: x = (b-a)t/2 + (b+a)/2
∫ₐᵇ f(x) dx = (b-a)/2 ∫₋₁¹ f((b-a)t/2 + (b+a)/2) dt

### Example: ∫₀¹ x² dx using 2-point Gaussian
Transform: x = t/2 + 1/2
∫₀¹ x² dx = 1/2 ∫₋₁¹ ((t/2 + 1/2)²) dt
= 1/2 [(-1/√3/2 + 1/2)² + (1/√3/2 + 1/2)²]
= 1/3 (exact!)

## Monte Carlo Integration

### Basic Idea
For high-dimensional integrals:
∫Ω f(x) dx ≈ |Ω| * (1/N) ∑ᵢ₌₁ᴺ f(xᵢ)

Where xᵢ are random points uniformly distributed in Ω.

### Algorithm
```
1. Generate N random points x₁, ..., xₙ in domain Ω
2. Compute f(x₁), ..., f(xₙ)
3. Estimate integral as |Ω| * (1/N) * ∑f(xᵢ)
```

### Example: ∫₀¹ ∫₀¹ (x² + y²) dx dy
Exact: 2/3 ≈ 0.6667

For N = 1000 random points:
- Generate (xᵢ, yᵢ) uniformly in [0,1]²
- Compute xᵢ² + yᵢ² for each point
- Average: ≈ 0.6667

### Error Analysis
Error decreases as O(1/√N) regardless of dimension.

### Advantages
- Works for any dimension
- No smoothness requirements
- Easy to implement

### Disadvantages
- Slow convergence
- Random error (not deterministic)

## Adaptive Methods

### Adaptive Quadrature
Automatically subdivides intervals to achieve desired accuracy.

### Algorithm
```
1. Estimate integral on [a,b] with two methods (e.g., Simpson's rule)
2. If difference < tolerance, return estimate
3. Otherwise, subdivide into [a,c] and [c,b]
4. Recursively apply to subintervals
```

### Example: Adaptive Simpson's Rule
```
function adaptive_simpson(f, a, b, tol):
    S1 = simpson(f, a, b)
    c = (a + b) / 2
    S2 = simpson(f, a, c) + simpson(f, c, b)
    
    if |S1 - S2| < 15*tol:
        return S2 + (S2 - S1)/15
    else:
        return adaptive_simpson(f, a, c, tol/2) + 
               adaptive_simpson(f, c, b, tol/2)
```

## Applications to Machine Learning

### Computing Expectations
For random variable X with density p(x):
E[f(X)] = ∫ f(x) p(x) dx

**Monte Carlo estimate**:
E[f(X)] ≈ (1/N) ∑ᵢ₌₁ᴺ f(xᵢ)

Where xᵢ ~ p(x).

### Bayesian Inference
Posterior mean:
E[θ|data] = ∫ θ p(θ|data) dθ

Often requires numerical integration or MCMC methods.

### Marginal Distributions
p(x) = ∫ p(x,y) dy

Can use numerical integration or analytical methods.

### Gaussian Processes
Predictive mean involves integrals:
μ(x*) = ∫ f(x*) p(f|x) df

Often requires numerical integration.

## Error Analysis and Convergence

### Convergence Rates
| Method | Error | Convergence |
|--------|-------|-------------|
| Trapezoidal | O(h²) | Linear in n |
| Simpson's | O(h⁴) | Quadratic in n |
| Gaussian | O(h²ⁿ) | Exponential |
| Monte Carlo | O(1/√N) | Independent of dimension |

### Richardson Extrapolation
For methods with known error form, can extrapolate to higher accuracy.

**Example**: Trapezoidal rule with error O(h²)
If T(h) is approximation with step h:
T₀(h) = T(h)
Tₖ₊₁(h) = (4ᵏ⁺¹ Tₖ(h/2) - Tₖ(h))/(4ᵏ⁺¹ - 1)

### Practical Considerations

#### Choosing Step Size
- **Small h**: Better accuracy, more computation
- **Large h**: Faster computation, less accuracy
- **Adaptive**: Automatic step size selection

#### Function Smoothness
- **Smooth functions**: Higher-order methods work well
- **Discontinuous functions**: Lower-order methods more robust
- **Oscillatory functions**: May need special methods

## Practice Problems

### Problem 1
Compute ∫₀¹ e^x dx using:
a) Trapezoidal rule with n = 4
b) Simpson's rule with n = 4
c) 2-point Gaussian quadrature

**Solutions**:
a) Trapezoidal: ≈ 1.7189 (error: 0.0009)
b) Simpson's: ≈ 1.7183 (error: 0.0003)
c) Gaussian: ≈ 1.7183 (error: 0.0003)

### Problem 2
Estimate ∫₀¹ ∫₀¹ sin(xy) dx dy using Monte Carlo with N = 1000.

**Solution**:
Generate random points (xᵢ, yᵢ) in [0,1]²
Estimate ≈ 0.2398 (exact: 0.2398)

### Problem 3
Implement adaptive Simpson's rule for ∫₀¹ √x dx with tolerance 10⁻⁶.

**Solution**:
Function is singular at x = 0, but integrable.
Adaptive method handles this automatically.
Result ≈ 0.6667 (exact: 2/3)

## Key Takeaways
- Choose method based on accuracy requirements and function properties
- Trapezoidal rule is simple but slow
- Simpson's rule is accurate for smooth functions
- Gaussian quadrature is optimal for polynomials
- Monte Carlo is essential for high dimensions
- Adaptive methods provide automatic error control
- Consider function smoothness and computational cost

## Next Steps
In the next tutorial, we'll explore interpolation and approximation methods including polynomial interpolation, splines, and least squares approximation.
