# Power Series

## Overview
This tutorial covers power series, which are infinite series of the form ∑[n=0 to ∞] aₙ(x-c)ⁿ. Power series are fundamental tools in calculus, providing a way to represent functions as infinite polynomials and enabling approximation and analysis of complex functions.

## Learning Objectives
- Understand power series and their convergence
- Find radius and interval of convergence
- Work with Taylor and Maclaurin series
- Apply power series to approximate functions
- Use power series to solve problems

## 1. Definition of Power Series

### General Form
A power series centered at c is:
```
∑[n=0 to ∞] aₙ(x-c)ⁿ = a₀ + a₁(x-c) + a₂(x-c)² + a₃(x-c)³ + ...
```

### Special Cases
- **Centered at 0**: ∑[n=0 to ∞] aₙxⁿ
- **Maclaurin series**: Taylor series centered at 0
- **Taylor series**: Power series centered at any point c

## 2. Convergence of Power Series

### Radius of Convergence
Every power series has a radius of convergence R such that:
- The series converges absolutely for |x-c| < R
- The series diverges for |x-c| > R
- At |x-c| = R, the series may converge or diverge

### Finding the Radius of Convergence
Use the ratio test:
```
R = lim[n→∞] |aₙ/aₙ₊₁|
```

Or the root test:
```
R = lim[n→∞] 1/|aₙ|^(1/n)
```

### Examples

#### Example 1: Basic Power Series
```
∑[n=0 to ∞] xⁿ/n!
```

Using ratio test:
```
lim[n→∞] |x^(n+1)/(n+1)!| / |xⁿ/n!| = lim[n→∞] |x|/(n+1) = 0
```

Since the limit is 0 for all x, R = ∞ (converges for all x).

#### Example 2: Geometric Power Series
```
∑[n=0 to ∞] xⁿ
```

Using ratio test:
```
lim[n→∞] |x^(n+1)| / |xⁿ| = |x|
```

The series converges when |x| < 1, so R = 1.

## 3. Interval of Convergence

### Definition
The interval of convergence is the set of all x values for which the power series converges.

### Finding the Interval
1. Find the radius of convergence R
2. Test the endpoints x = c ± R
3. The interval is (c-R, c+R), [c-R, c+R), (c-R, c+R], or [c-R, c+R]

### Examples

#### Example 1: Testing Endpoints
```
∑[n=1 to ∞] xⁿ/n
```

Radius: R = 1
Interval: (-1, 1)

Test endpoints:
- At x = 1: ∑[n=1 to ∞] 1/n diverges (harmonic series)
- At x = -1: ∑[n=1 to ∞] (-1)ⁿ/n converges (alternating series)

Interval of convergence: [-1, 1)

#### Example 2: Both Endpoints Converge
```
∑[n=1 to ∞] xⁿ/n²
```

Radius: R = 1
Test endpoints:
- At x = 1: ∑[n=1 to ∞] 1/n² converges (p-series, p=2)
- At x = -1: ∑[n=1 to ∞] (-1)ⁿ/n² converges absolutely

Interval of convergence: [-1, 1]

## 4. Taylor Series

### Definition
The Taylor series for f(x) centered at c is:
```
∑[n=0 to ∞] f⁽ⁿ⁾(c)/n! · (x-c)ⁿ
```

Where f⁽ⁿ⁾(c) is the nth derivative of f evaluated at c.

### Maclaurin Series
Taylor series centered at 0:
```
∑[n=0 to ∞] f⁽ⁿ⁾(0)/n! · xⁿ
```

### Examples

#### Example 1: Maclaurin Series for eˣ
```
f(x) = eˣ, f⁽ⁿ⁾(x) = eˣ, f⁽ⁿ⁾(0) = 1
```

```
eˣ = ∑[n=0 to ∞] xⁿ/n! = 1 + x + x²/2! + x³/3! + ...
```

#### Example 2: Maclaurin Series for sin(x)
```
f(x) = sin(x), f'(x) = cos(x), f''(x) = -sin(x), f'''(x) = -cos(x), ...
f(0) = 0, f'(0) = 1, f''(0) = 0, f'''(0) = -1, ...
```

```
sin(x) = ∑[n=0 to ∞] (-1)ⁿx^(2n+1)/(2n+1)! = x - x³/3! + x⁵/5! - x⁷/7! + ...
```

#### Example 3: Maclaurin Series for cos(x)
```
f(x) = cos(x), f'(x) = -sin(x), f''(x) = -cos(x), f'''(x) = sin(x), ...
f(0) = 1, f'(0) = 0, f''(0) = -1, f'''(0) = 0, ...
```

```
cos(x) = ∑[n=0 to ∞] (-1)ⁿx^(2n)/(2n)! = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
```

## 5. Common Maclaurin Series

### Exponential and Logarithmic
```
eˣ = ∑[n=0 to ∞] xⁿ/n!
ln(1+x) = ∑[n=1 to ∞] (-1)^(n+1)xⁿ/n
ln(1-x) = -∑[n=1 to ∞] xⁿ/n
```

### Trigonometric
```
sin(x) = ∑[n=0 to ∞] (-1)ⁿx^(2n+1)/(2n+1)!
cos(x) = ∑[n=0 to ∞] (-1)ⁿx^(2n)/(2n)!
tan(x) = ∑[n=1 to ∞] B_(2n)(-4)ⁿ(1-4ⁿ)x^(2n-1)/(2n)!
```

### Hyperbolic
```
sinh(x) = ∑[n=0 to ∞] x^(2n+1)/(2n+1)!
cosh(x) = ∑[n=0 to ∞] x^(2n)/(2n)!
```

### Binomial
```
(1+x)ᵏ = ∑[n=0 to ∞] C(k,n)xⁿ
```

Where C(k,n) = k(k-1)...(k-n+1)/n!

## 6. Operations with Power Series

### Addition and Subtraction
```
∑[n=0 to ∞] aₙxⁿ ± ∑[n=0 to ∞] bₙxⁿ = ∑[n=0 to ∞] (aₙ ± bₙ)xⁿ
```

### Multiplication
```
(∑[n=0 to ∞] aₙxⁿ)(∑[n=0 to ∞] bₙxⁿ) = ∑[n=0 to ∞] (∑[k=0 to n] aₖbₙ₋ₖ)xⁿ
```

### Differentiation
```
d/dx[∑[n=0 to ∞] aₙxⁿ] = ∑[n=1 to ∞] naₙx^(n-1)
```

### Integration
```
∫[∑[n=0 to ∞] aₙxⁿ] dx = C + ∑[n=0 to ∞] aₙx^(n+1)/(n+1)
```

### Examples

#### Example 1: Differentiating a Power Series
```
d/dx[eˣ] = d/dx[∑[n=0 to ∞] xⁿ/n!] = ∑[n=1 to ∞] nx^(n-1)/n! = ∑[n=1 to ∞] x^(n-1)/(n-1)! = ∑[n=0 to ∞] xⁿ/n! = eˣ
```

#### Example 2: Integrating a Power Series
```
∫[0 to x] e^t dt = ∫[0 to x] [∑[n=0 to ∞] tⁿ/n!] dt = ∑[n=0 to ∞] ∫[0 to x] tⁿ/n! dt = ∑[n=0 to ∞] x^(n+1)/(n+1)! = eˣ - 1
```

## 7. Applications of Power Series

### Function Approximation
Power series provide polynomial approximations to functions.

#### Example: Approximating e
```
e = e¹ = ∑[n=0 to ∞] 1ⁿ/n! = 1 + 1 + 1/2! + 1/3! + 1/4! + ... ≈ 2.71828
```

### Solving Differential Equations
Power series can be used to find solutions to differential equations.

#### Example: Solving y' = y
Assume y = ∑[n=0 to ∞] aₙxⁿ
Then y' = ∑[n=1 to ∞] naₙx^(n-1)

Substituting into y' = y:
```
∑[n=1 to ∞] naₙx^(n-1) = ∑[n=0 to ∞] aₙxⁿ
```

Equating coefficients:
```
(n+1)aₙ₊₁ = aₙ → aₙ₊₁ = aₙ/(n+1)
```

If a₀ = 1, then aₙ = 1/n!, giving y = eˣ.

### Evaluating Integrals
Some integrals can be evaluated using power series.

#### Example: ∫[0 to 1] e^(-x²) dx
```
e^(-x²) = ∑[n=0 to ∞] (-x²)ⁿ/n! = ∑[n=0 to ∞] (-1)ⁿx^(2n)/n!
```

```
∫[0 to 1] e^(-x²) dx = ∫[0 to 1] [∑[n=0 to ∞] (-1)ⁿx^(2n)/n!] dx
= ∑[n=0 to ∞] (-1)ⁿ∫[0 to 1] x^(2n)/n! dx = ∑[n=0 to ∞] (-1)ⁿ/((2n+1)n!)
```

## 8. Practice Problems

### Finding Radius and Interval of Convergence
1. ∑[n=0 to ∞] xⁿ/n²
2. ∑[n=1 to ∞] n!xⁿ
3. ∑[n=0 to ∞] (x-2)ⁿ/n
4. ∑[n=0 to ∞] xⁿ/(2n)!

### Finding Maclaurin Series
1. f(x) = 1/(1-x)
2. f(x) = 1/(1+x²)
3. f(x) = arctan(x)
4. f(x) = ln(1+x)

### Operations with Power Series
1. Find the Maclaurin series for x²eˣ
2. Find the Maclaurin series for sin(x²)
3. Find the Maclaurin series for ∫[0 to x] sin(t²) dt
4. Find the sum of ∑[n=1 to ∞] n²xⁿ

### Applications
1. Approximate e^(0.1) using the first 4 terms of its Maclaurin series
2. Approximate sin(0.1) using the first 3 terms of its Maclaurin series
3. Find the Maclaurin series for f(x) = 1/(1+x) and use it to approximate 1/1.1
4. Use power series to evaluate ∫[0 to 0.5] e^(-x²) dx

## 9. Common Mistakes to Avoid

1. **Radius of Convergence**: Don't forget to test the endpoints
2. **Taylor Series**: Make sure to evaluate derivatives at the center point
3. **Convergence**: Remember that power series may not converge at endpoints
4. **Operations**: Be careful with the order of operations when manipulating series
5. **Index Shifts**: Pay attention to index changes when differentiating or integrating

## 10. Advanced Topics

### Remainder Estimation
For Taylor series approximations:
```
|Rₙ(x)| ≤ M|x-c|^(n+1)/(n+1)!
```

Where M is an upper bound for |f⁽ⁿ⁺¹⁾(t)| on the interval between c and x.

### Convergence Acceleration
Techniques to improve convergence of power series:
- Euler's transformation
- Padé approximants
- Continued fractions

## 11. Study Tips

1. **Learn Common Series**: Memorize the basic Maclaurin series
2. **Practice Operations**: Work with differentiation and integration of series
3. **Understand Convergence**: Know when and why series converge
4. **Use Applications**: See how power series solve real problems
5. **Check Your Work**: Verify series by substituting known values

## Next Steps

After mastering power series, proceed to:
- Parametric and polar integration
- Differential equations
- Advanced calculus topics
- Complex analysis

Remember: Power series are powerful tools for representing functions, solving problems, and understanding the behavior of functions. They bridge the gap between algebra and calculus, providing infinite polynomial representations of functions.
