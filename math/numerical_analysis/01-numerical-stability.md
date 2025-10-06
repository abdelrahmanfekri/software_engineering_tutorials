# Numerical Analysis Tutorial 01: Numerical Stability and Error Analysis

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand floating-point arithmetic and its limitations
- Analyze rounding errors and their propagation
- Calculate condition numbers for numerical problems
- Identify numerically stable and unstable algorithms
- Apply error analysis to machine learning algorithms
- Choose appropriate numerical methods for given problems

## Introduction to Numerical Analysis

### What is Numerical Analysis?
Numerical analysis is the study of algorithms for solving mathematical problems numerically. It focuses on:
1. **Accuracy**: How close are computed results to exact solutions?
2. **Efficiency**: How fast can problems be solved?
3. **Stability**: How do small errors propagate through computations?

### Why is Numerical Analysis Important for AI?
- **Machine Learning**: Training algorithms involve numerical optimization
- **Neural Networks**: Backpropagation requires stable numerical methods
- **Linear Algebra**: Matrix operations must be numerically stable
- **Optimization**: Gradient descent and other methods need careful implementation

## Floating-Point Arithmetic

### IEEE 754 Standard
Most computers use IEEE 754 floating-point representation:

**Single precision (32-bit)**:
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits
- Range: ±1.4 × 10^(-45) to ±3.4 × 10^38

**Double precision (64-bit)**:
- 1 sign bit
- 11 exponent bits
- 52 mantissa bits
- Range: ±4.9 × 10^(-324) to ±1.8 × 10^308

### Machine Epsilon
Machine epsilon (ε) is the smallest number such that 1 + ε > 1 in floating-point arithmetic.

- **Single precision**: ε ≈ 1.2 × 10^(-7)
- **Double precision**: ε ≈ 2.2 × 10^(-16)

### Rounding Errors
Every floating-point operation introduces rounding errors:

**Example**: Computing 1/3
- Exact: 1/3 = 0.3333...
- Floating-point: 0.3333333 (rounded)
- Error: |0.3333... - 0.3333333| ≈ 3.3 × 10^(-8)

## Error Analysis

### Types of Errors
1. **Truncation Error**: Error from approximating infinite processes
2. **Rounding Error**: Error from finite precision arithmetic
3. **Discretization Error**: Error from approximating continuous problems

### Error Propagation
When performing operations on approximate numbers, errors can grow:

**Addition**: If x = x̂ + εₓ and y = ŷ + εᵧ, then:
x + y = (x̂ + ŷ) + (εₓ + εᵧ)

**Multiplication**: If x = x̂(1 + δₓ) and y = ŷ(1 + δᵧ), then:
xy = x̂ŷ(1 + δₓ + δᵧ + δₓδᵧ) ≈ x̂ŷ(1 + δₓ + δᵧ)

### Relative Error
For an approximation x̂ of exact value x:
Relative error = |x - x̂|/|x|

**Example**: If x = 1.0 and x̂ = 1.01, then relative error = 0.01/1.0 = 0.01 = 1%

## Condition Numbers

### Definition
The condition number measures how sensitive a function is to small changes in its input:

κ(f, x) = |x f'(x)|/|f(x)|

### Interpretation
- **κ ≈ 1**: Well-conditioned (small changes in input → small changes in output)
- **κ >> 1**: Ill-conditioned (small changes in input → large changes in output)
- **κ = ∞**: Singular (function is not differentiable)

### Examples

**Example 1**: f(x) = x²
κ(f, x) = |x · 2x|/|x²| = 2

**Example 2**: f(x) = 1/x
κ(f, x) = |x · (-1/x²)|/|1/x| = 1

**Example 3**: f(x) = e^x
κ(f, x) = |x · e^x|/|e^x| = |x|

### Matrix Condition Number
For a matrix A, the condition number is:
κ(A) = ||A|| ||A^(-1)||

Where ||·|| is a matrix norm.

**Properties**:
- κ(A) ≥ 1
- κ(A) = ∞ if A is singular
- κ(A) measures sensitivity of linear system Ax = b

## Numerical Stability

### Forward Stability
An algorithm is forward stable if the computed result is close to the exact result for the given input.

**Example**: Computing x + y
- Input: x = 1.0, y = 1.0
- Exact: x + y = 2.0
- Computed: 2.0 (forward stable)

### Backward Stability
An algorithm is backward stable if the computed result is the exact result for slightly perturbed input.

**Example**: Computing x + y with rounding
- Input: x = 1.0, y = 1.0
- Computed: 2.0
- This is exact for x = 1.0, y = 1.0 (backward stable)

### Mixed Stability
An algorithm is mixed stable if it combines forward and backward stability.

## Stable and Unstable Algorithms

### Stable Algorithm: Horner's Method
For evaluating polynomial p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ:

```
result = aₙ
for i = n-1 downto 0:
    result = result * x + aᵢ
```

**Advantages**:
- O(n) operations
- Numerically stable
- Minimal rounding errors

### Unstable Algorithm: Direct Evaluation
```
result = a₀
for i = 1 to n:
    result = result + aᵢ * x^i
```

**Problems**:
- Large intermediate values
- Accumulation of rounding errors
- Potential overflow

### Example: Evaluating p(x) = 1 + x + x² + x³ at x = 0.1

**Horner's method**: p(x) = 1 + x(1 + x(1 + x))
- Step 1: 1 + 0.1 = 1.1
- Step 2: 1 + 0.1(1.1) = 1.11
- Step 3: 1 + 0.1(1.11) = 1.111

**Direct evaluation**: p(x) = 1 + 0.1 + 0.01 + 0.001 = 1.111

Both give same result, but Horner's method is more stable for large n.

## Applications to Machine Learning

### Linear Regression
**Problem**: Solve (XᵀX)β = Xᵀy

**Issues**:
- XᵀX may be ill-conditioned
- Direct inversion may be unstable
- Use QR decomposition or SVD instead

**Stable approach**:
1. Compute QR decomposition: X = QR
2. Solve Rβ = Qᵀy (triangular system)

### Neural Network Training
**Problem**: Computing gradients in backpropagation

**Issues**:
- Gradient explosion/vanishing
- Numerical instability in activation functions
- Use stable activation functions (ReLU, tanh)

**Stable techniques**:
- Gradient clipping
- Batch normalization
- Careful initialization

### Matrix Inversion
**Problem**: Computing A^(-1)

**Issues**:
- A may be singular or near-singular
- Direct inversion is unstable
- Use LU decomposition or iterative methods

**Stable approach**:
1. Compute LU decomposition: A = LU
2. Solve Ly = b, then Ux = y

## Error Bounds and Estimates

### Taylor Series Error
For f(x) ≈ f(a) + f'(a)(x-a) + (1/2)f''(a)(x-a)²:

Error ≤ (1/6)|f'''(ξ)|(x-a)³ for some ξ between a and x

### Numerical Integration Error
**Trapezoidal rule**: Error ≤ (b-a)³|f''(ξ)|/(12n²)

**Simpson's rule**: Error ≤ (b-a)⁵|f⁽⁴⁾(ξ)|/(2880n⁴)

### Root Finding Error
**Newton's method**: |xₙ₊₁ - x*| ≤ C|xₙ - x*|² (quadratic convergence)

## Practical Considerations

### Choosing Algorithms
1. **Consider problem size**: Small problems may use less stable but faster methods
2. **Check condition numbers**: Use stable methods for ill-conditioned problems
3. **Monitor errors**: Track error accumulation during computation
4. **Use appropriate precision**: Higher precision for critical calculations

### Implementation Tips
1. **Avoid subtraction of nearly equal numbers**: Can cause catastrophic cancellation
2. **Use appropriate data types**: Double precision for most calculations
3. **Check for overflow/underflow**: Handle extreme values carefully
4. **Validate results**: Compare with known solutions when possible

## Practice Problems

### Problem 1
Calculate the condition number of f(x) = x³ - 2x + 1 at x = 2.

**Solution**:
f'(x) = 3x² - 2
f(2) = 8 - 4 + 1 = 5
f'(2) = 12 - 2 = 10
κ(f, 2) = |2 · 10|/|5| = 20/5 = 4

### Problem 2
Estimate the error in computing 1.0001 - 1.0000 using single precision.

**Solution**:
Exact: 1.0001 - 1.0000 = 0.0001
Single precision: 1.0001 ≈ 1.0001001, 1.0000 = 1.0000000
Computed: 1.0001001 - 1.0000000 = 0.0001001
Error: |0.0001 - 0.0001001| ≈ 1 × 10^(-8)

### Problem 3
Is the algorithm for computing x² + 2x + 1 numerically stable?

**Solution**:
Using Horner's method: x² + 2x + 1 = 1 + x(2 + x)
This is stable because:
- No large intermediate values
- Minimal rounding errors
- O(n) operations

## Key Takeaways
- Floating-point arithmetic has limited precision
- Errors can accumulate during computation
- Condition numbers measure problem sensitivity
- Stable algorithms minimize error propagation
- Understanding numerical stability is crucial for ML
- Choose algorithms based on stability and efficiency

## Next Steps
In the next tutorial, we'll explore root finding methods, including bisection, Newton's method, and their convergence properties.
