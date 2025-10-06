# Numerical Analysis Tutorial 05: Interpolation and Approximation

## Learning Objectives
By the end of this tutorial, you will be able to:
- Implement polynomial interpolation using Lagrange and Newton methods
- Apply spline interpolation for smooth curve fitting
- Use least squares approximation for noisy data
- Implement Chebyshev approximation for optimal polynomial fitting
- Apply interpolation and approximation to machine learning problems
- Choose appropriate methods based on data characteristics

## Introduction to Interpolation and Approximation

### What is Interpolation?
Given data points (xᵢ, yᵢ), i = 0, 1, ..., n, interpolation finds a function f(x) such that:
f(xᵢ) = yᵢ for all i

### What is Approximation?
Given data points (xᵢ, yᵢ), approximation finds a function f(x) that "best fits" the data, typically minimizing some error measure.

### Applications
- **Data Analysis**: Fitting curves to experimental data
- **Computer Graphics**: Smooth curve generation
- **Machine Learning**: Function approximation, regression
- **Scientific Computing**: Approximating complex functions

## Polynomial Interpolation

### Existence and Uniqueness
For n+1 distinct points, there exists a unique polynomial of degree ≤ n that interpolates the data.

### Lagrange Interpolation

#### Lagrange Basis Polynomials
Lᵢ(x) = ∏ⱼ≠ᵢ (x - xⱼ)/(xᵢ - xⱼ)

Properties:
- Lᵢ(xᵢ) = 1
- Lᵢ(xⱼ) = 0 for j ≠ i

#### Interpolating Polynomial
P(x) = ∑ᵢ₌₀ⁿ yᵢ Lᵢ(x)

#### Example: Interpolate (0,1), (1,2), (2,5)
L₀(x) = (x-1)(x-2)/((0-1)(0-2)) = (x²-3x+2)/2
L₁(x) = (x-0)(x-2)/((1-0)(1-2)) = -(x²-2x)
L₂(x) = (x-0)(x-1)/((2-0)(2-1)) = (x²-x)/2

P(x) = 1·L₀(x) + 2·L₁(x) + 5·L₂(x) = x² + 1

#### Algorithm
```
function lagrange_interpolation(x_data, y_data, x):
    n = length(x_data) - 1
    result = 0
    
    for i = 0 to n:
        L_i = 1
        for j = 0 to n:
            if j != i:
                L_i = L_i * (x - x_data[j]) / (x_data[i] - x_data[j])
        result = result + y_data[i] * L_i
    
    return result
```

### Newton Interpolation

#### Divided Differences
f[xᵢ] = f(xᵢ)
f[xᵢ, xᵢ₊₁] = (f[xᵢ₊₁] - f[xᵢ])/(xᵢ₊₁ - xᵢ)
f[xᵢ, xᵢ₊₁, xᵢ₊₂] = (f[xᵢ₊₁, xᵢ₊₂] - f[xᵢ, xᵢ₊₁])/(xᵢ₊₂ - xᵢ)

#### Newton Form
P(x) = f[x₀] + f[x₀, x₁](x-x₀) + f[x₀, x₁, x₂](x-x₀)(x-x₁) + ...

#### Example: Same data as above
f[0] = 1, f[1] = 2, f[2] = 5
f[0,1] = (2-1)/(1-0) = 1
f[1,2] = (5-2)/(2-1) = 3
f[0,1,2] = (3-1)/(2-0) = 1

P(x) = 1 + 1·x + 1·x(x-1) = 1 + x + x² - x = x² + 1

#### Algorithm
```
function newton_interpolation(x_data, y_data, x):
    n = length(x_data) - 1
    
    // Compute divided differences
    f = y_data
    for j = 1 to n:
        for i = n downto j:
            f[i] = (f[i] - f[i-1]) / (x_data[i] - x_data[i-j])
    
    // Evaluate polynomial
    result = f[n]
    for i = n-1 downto 0:
        result = result * (x - x_data[i]) + f[i]
    
    return result
```

### Error Analysis
For polynomial interpolation:
f(x) - P(x) = f^(n+1)(ξ)/(n+1)! ∏ᵢ₌₀ⁿ (x - xᵢ)

Where ξ is between min(x, x₀, ..., xₙ) and max(x, x₀, ..., xₙ).

### Runge's Phenomenon
High-degree polynomial interpolation can oscillate wildly between data points, especially with equally spaced points.

**Example**: f(x) = 1/(1 + 25x²) on [-1, 1]
- Low-degree interpolation works well
- High-degree interpolation shows oscillations

## Spline Interpolation

### Linear Splines
Piecewise linear interpolation:
S(x) = yᵢ + (yᵢ₊₁ - yᵢ)(x - xᵢ)/(xᵢ₊₁ - xᵢ) for x ∈ [xᵢ, xᵢ₊₁]

### Cubic Splines
Piecewise cubic polynomials with continuous first and second derivatives.

#### Natural Cubic Spline
S''(x₀) = S''(xₙ) = 0

#### Algorithm
1. Set up tridiagonal system for second derivatives
2. Solve for S''(xᵢ)
3. Construct cubic polynomials on each interval

#### Example: Data (0,0), (1,1), (2,4), (3,9)
For natural cubic spline:
- S₀(x) = x + (1/6)x³ on [0,1]
- S₁(x) = 1 + 2(x-1) + (1/2)(x-1)² + (1/6)(x-1)³ on [1,2]
- S₂(x) = 4 + 4(x-2) + (1/2)(x-2)² - (1/6)(x-2)³ on [2,3]

### B-Splines
Basis functions for spline spaces:
Bᵢ,ₖ(x) = (x - tᵢ)/(tᵢ₊ₖ₋₁ - tᵢ) Bᵢ,ₖ₋₁(x) + (tᵢ₊ₖ - x)/(tᵢ₊ₖ - tᵢ₊₁) Bᵢ₊₁,ₖ₋₁(x)

### Advantages of Splines
- Smooth interpolation
- Local control (changing one point affects limited region)
- Stable for large numbers of points
- No Runge's phenomenon

## Least Squares Approximation

### Linear Least Squares
Given data (xᵢ, yᵢ), find line y = ax + b minimizing:
E = ∑ᵢ₌₁ⁿ (yᵢ - axᵢ - b)²

#### Normal Equations
[[∑xᵢ², ∑xᵢ], [∑xᵢ, n]] [[a], [b]] = [[∑xᵢyᵢ], [∑yᵢ]]

#### Example: Data (1,2), (2,3), (3,5), (4,7)
∑xᵢ = 10, ∑xᵢ² = 30, ∑yᵢ = 17, ∑xᵢyᵢ = 49

Normal equations:
[[30, 10], [10, 4]] [[a], [b]] = [[49], [17]]

Solution: a = 1.7, b = 0.5
Line: y = 1.7x + 0.5

### Polynomial Least Squares
Find polynomial P(x) = ∑ⱼ₌₀ᵐ aⱼxʲ minimizing:
E = ∑ᵢ₌₁ⁿ (yᵢ - P(xᵢ))²

#### Normal Equations
AᵀA a = Aᵀy

Where Aᵢⱼ = xᵢʲ

#### Example: Quadratic fit to same data
A = [[1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]]
y = [2, 3, 5, 7]ᵀ

Solving: a₀ = 1.5, a₁ = 0.5, a₂ = 0.2
Polynomial: y = 1.5 + 0.5x + 0.2x²

### Weighted Least Squares
Minimize: E = ∑ᵢ₌₁ⁿ wᵢ(yᵢ - f(xᵢ))²

Where wᵢ are weights reflecting data reliability.

## Chebyshev Approximation

### Chebyshev Polynomials
T₀(x) = 1
T₁(x) = x
Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)

### Chebyshev Points
Optimal interpolation points on [-1, 1]:
xᵢ = cos((2i+1)π/(2n+2)) for i = 0, 1, ..., n

### Minimax Property
Chebyshev approximation minimizes the maximum error:
min max |f(x) - P(x)|

### Example: Approximating e^x on [-1, 1]
Using 4 Chebyshev points: x₀ = cos(π/8), x₁ = cos(3π/8), x₂ = cos(5π/8), x₃ = cos(7π/8)

## Applications to Machine Learning

### Regression Analysis
- **Linear regression**: Least squares line fitting
- **Polynomial regression**: Higher-degree polynomial fitting
- **Regularized regression**: Ridge, Lasso with smoothness constraints

### Function Approximation
- **Neural networks**: Universal approximators
- **Radial basis functions**: Local approximation
- **Support vector regression**: Kernel-based approximation

### Data Smoothing
- **Spline smoothing**: Penalized least squares
- **Kernel smoothing**: Local weighted averages
- **Moving averages**: Simple smoothing techniques

### Feature Engineering
- **Polynomial features**: x, x², x³, ...
- **Spline features**: Piecewise polynomial basis
- **Fourier features**: Trigonometric basis functions

## Error Analysis and Convergence

### Interpolation Error
|f(x) - P(x)| ≤ (1/(n+1)!) |f^(n+1)(ξ)| ∏ᵢ₌₀ⁿ |x - xᵢ|

### Least Squares Error
For polynomial of degree m with n > m data points:
RMS error = √(∑ᵢ₌₁ⁿ (yᵢ - P(xᵢ))²/n)

### Convergence Rates
- **Polynomial interpolation**: Exponential for analytic functions
- **Spline interpolation**: O(h⁴) for cubic splines
- **Least squares**: Depends on model complexity vs. data size

## Practical Considerations

### Choosing Interpolation Method
1. **Polynomial**: Good for smooth functions, few points
2. **Spline**: Good for smooth curves, many points
3. **Least squares**: Good for noisy data
4. **Chebyshev**: Good for minimax approximation

### Data Quality
- **Exact data**: Use interpolation
- **Noisy data**: Use approximation
- **Sparse data**: Consider regularization

### Computational Cost
- **Lagrange**: O(n²) evaluation
- **Newton**: O(n²) setup, O(n) evaluation
- **Spline**: O(n) setup and evaluation
- **Least squares**: O(nm²) for degree m polynomial

## Practice Problems

### Problem 1
Interpolate f(x) = sin(x) at x = 0, π/4, π/2 using:
a) Lagrange interpolation
b) Newton interpolation

**Solutions**:
a) P(x) = sin(0)L₀(x) + sin(π/4)L₁(x) + sin(π/2)L₂(x)
b) Divided differences: f[0] = 0, f[π/4] = √2/2, f[π/2] = 1
   f[0,π/4] = 2√2/π, f[π/4,π/2] = 2(1-√2/2)/π

### Problem 2
Fit a quadratic polynomial to data (0,1), (1,0), (2,3), (3,10) using least squares.

**Solution**:
A = [[1,0,0], [1,1,1], [1,2,4], [1,3,9]]
y = [1,0,3,10]ᵀ
Solving AᵀA a = Aᵀy: a = [1, -2, 1]ᵀ
Polynomial: y = 1 - 2x + x²

### Problem 3
Construct natural cubic spline for data (0,1), (1,2), (2,1), (3,0).

**Solution**:
Solve tridiagonal system for second derivatives, then construct piecewise cubics.

## Key Takeaways
- Choose interpolation method based on data characteristics
- Polynomial interpolation can suffer from Runge's phenomenon
- Splines provide smooth, stable interpolation
- Least squares is robust for noisy data
- Chebyshev approximation minimizes maximum error
- Consider computational cost and accuracy requirements
- Machine learning applications benefit from various approximation methods

## Conclusion
Interpolation and approximation are fundamental tools in numerical analysis with wide applications in machine learning, scientific computing, and data analysis. Understanding the trade-offs between different methods is crucial for choosing appropriate techniques for specific problems.
