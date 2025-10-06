# Series Solutions

## Overview
Series solutions provide a powerful method for solving differential equations that cannot be solved by elementary methods. They are particularly important for equations with variable coefficients and are fundamental to understanding special functions like Bessel functions and Legendre polynomials. This tutorial covers power series methods and the Frobenius method.

## Learning Objectives
- Understand power series solutions for differential equations
- Identify ordinary and singular points
- Apply the Frobenius method for regular singular points
- Solve equations leading to Bessel functions
- Work with Legendre polynomials
- Understand applications in physics and engineering

## Power Series Solutions

### Basic Concept
A power series solution assumes that the solution can be written as:
```
y(x) = Σ_{n=0}^∞ a_n (x - x₀)^n
```

### Method
1. **Assume power series form** for the solution
2. **Substitute into the differential equation**
3. **Collect like powers** of (x - x₀)
4. **Set coefficients equal to zero** to get recurrence relations
5. **Solve recurrence relations** to find coefficients
6. **Write the general solution**

### Example: Simple Case
```
y'' - y = 0
```
Assume: `y = Σ_{n=0}^∞ a_n x^n`
Then: `y' = Σ_{n=1}^∞ n a_n x^{n-1}`, `y'' = Σ_{n=2}^∞ n(n-1) a_n x^{n-2}`

Substitute: `Σ_{n=2}^∞ n(n-1) a_n x^{n-2} - Σ_{n=0}^∞ a_n x^n = 0`

Shift index in first sum: `Σ_{n=0}^∞ (n+2)(n+1) a_{n+2} x^n - Σ_{n=0}^∞ a_n x^n = 0`

This gives: `Σ_{n=0}^∞ [(n+2)(n+1) a_{n+2} - a_n] x^n = 0`

Recurrence relation: `a_{n+2} = a_n / [(n+2)(n+1)]`

Starting with `a₀` and `a₁` arbitrary:
- `a₂ = a₀/2`
- `a₄ = a₂/12 = a₀/24`
- `a₆ = a₄/30 = a₀/720`
- etc.

- `a₃ = a₁/6`
- `a₅ = a₃/20 = a₁/120`
- `a₇ = a₅/42 = a₁/5040`
- etc.

Solution: `y = a₀(1 + x²/2 + x⁴/24 + ...) + a₁(x + x³/6 + x⁵/120 + ...)`

Recognizing the series: `y = a₀ cosh(x) + a₁ sinh(x)`

## Ordinary and Singular Points

### Definition
For the equation `y'' + p(x)y' + q(x)y = 0`:

- **Ordinary point**: Both p(x) and q(x) are analytic at x₀
- **Singular point**: At least one of p(x) or q(x) is not analytic at x₀

### Regular vs Irregular Singular Points
At a singular point x₀:
- **Regular singular point**: (x-x₀)p(x) and (x-x₀)²q(x) are analytic
- **Irregular singular point**: Otherwise

### Examples
1. **x²y'' + xy' + (x² - ν²)y = 0** (Bessel's equation)
   - x = 0 is a regular singular point
   - x = ∞ is an irregular singular point

2. **y'' + (1/x²)y' + (1/x³)y = 0**
   - x = 0 is an irregular singular point

## Frobenius Method

### When to Use
Use the Frobenius method when x₀ is a regular singular point.

### Method
1. **Assume solution**: `y = (x-x₀)^r Σ_{n=0}^∞ a_n (x-x₀)^n`
2. **Find indicial equation** by substituting the lowest power
3. **Determine roots r₁ and r₂** of the indicial equation
4. **Find recurrence relations** for each root
5. **Construct solutions** based on the nature of roots

### Indicial Equation
For the equation `(x-x₀)²y'' + (x-x₀)p(x)y' + q(x)y = 0`:
The indicial equation is: `r(r-1) + p₀r + q₀ = 0`
where `p₀ = lim_{x→x₀} (x-x₀)p(x)` and `q₀ = lim_{x→x₀} (x-x₀)²q(x)`

### Cases for Roots

#### Case 1: Distinct Roots (r₁ ≠ r₂, r₁ - r₂ not integer)
Two linearly independent solutions:
```
y₁ = (x-x₀)^{r₁} Σ_{n=0}^∞ a_n (x-x₀)^n
y₂ = (x-x₀)^{r₂} Σ_{n=0}^∞ b_n (x-x₀)^n
```

#### Case 2: Repeated Roots (r₁ = r₂)
```
y₁ = (x-x₀)^{r₁} Σ_{n=0}^∞ a_n (x-x₀)^n
y₂ = y₁ ln(x-x₀) + (x-x₀)^{r₁} Σ_{n=0}^∞ b_n (x-x₀)^n
```

#### Case 3: Roots Differ by Integer (r₁ - r₂ = k > 0)
```
y₁ = (x-x₀)^{r₁} Σ_{n=0}^∞ a_n (x-x₀)^n
y₂ = c y₁ ln(x-x₀) + (x-x₀)^{r₂} Σ_{n=0}^∞ b_n (x-x₀)^n
```

### Example: Frobenius Method
```
2xy'' + y' - y = 0
```
Standard form: `y'' + (1/2x)y' - (1/2x)y = 0`
x = 0 is a regular singular point.

Indicial equation: `r(r-1) + (1/2)r - 0 = r(r-1/2) = 0`
Roots: `r = 0, 1/2`

For r = 0: Assume `y = Σ_{n=0}^∞ a_n x^n`
Substitute and get recurrence: `a_n = a_{n-1} / [2n(2n-1)]`
Solution: `y₁ = Σ_{n=0}^∞ x^n / [2^n (2n)!]`

For r = 1/2: Assume `y = x^{1/2} Σ_{n=0}^∞ b_n x^n`
Similar process gives second solution.

## Bessel's Equation and Bessel Functions

### Bessel's Equation
```
x²y'' + xy' + (x² - ν²)y = 0
```
where ν is a parameter (often integer or half-integer).

### Solution Using Frobenius Method
Indicial equation: `r(r-1) + r - ν² = r² - ν² = 0`
Roots: `r = ±ν`

For r = ν: The first solution is the Bessel function of the first kind:
```
J_ν(x) = (x/2)^ν Σ_{n=0}^∞ (-1)^n (x/2)^{2n} / [n! Γ(ν+n+1)]
```

For r = -ν: If ν is not an integer:
```
J_{-ν}(x) = (x/2)^{-ν} Σ_{n=0}^∞ (-1)^n (x/2)^{2n} / [n! Γ(-ν+n+1)]
```

### Bessel Functions of the Second Kind
When ν is an integer, J_ν and J_{-ν} are linearly dependent. The second solution is:
```
Y_ν(x) = [J_ν(x) cos(νπ) - J_{-ν}(x)] / sin(νπ)
```

### Properties of Bessel Functions
- **Orthogonality**: Bessel functions satisfy orthogonality relations
- **Recurrence relations**: Various identities connecting different orders
- **Asymptotic behavior**: Known behavior as x → 0 and x → ∞
- **Zeros**: J_ν(x) has infinitely many positive zeros

### Applications of Bessel Functions
1. **Heat conduction** in circular domains
2. **Vibrations** of circular membranes
3. **Electromagnetic waves** in cylindrical waveguides
4. **Quantum mechanics** in cylindrical coordinates

## Legendre's Equation and Legendre Polynomials

### Legendre's Equation
```
(1-x²)y'' - 2xy' + n(n+1)y = 0
```
where n is typically a non-negative integer.

### Solution Using Power Series
For integer n, one solution terminates (polynomial), and the other is an infinite series.

### Legendre Polynomials
The polynomial solutions are:
```
P₀(x) = 1
P₁(x) = x
P₂(x) = (3x² - 1)/2
P₃(x) = (5x³ - 3x)/2
P₄(x) = (35x⁴ - 30x² + 3)/8
...
```

### Rodrigues' Formula
```
P_n(x) = (1/2^n n!) d^n/dx^n [(x²-1)^n]
```

### Properties of Legendre Polynomials
- **Orthogonality**: `∫_{-1}^1 P_m(x)P_n(x)dx = 0` if m ≠ n
- **Normalization**: `∫_{-1}^1 [P_n(x)]² dx = 2/(2n+1)`
- **Recurrence relation**: `(n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)`

### Applications of Legendre Polynomials
1. **Spherical harmonics** in quantum mechanics
2. **Gravitational potential** expansions
3. **Electrostatics** in spherical coordinates
4. **Numerical analysis** and approximation theory

## Other Special Functions

### Hermite Polynomials
Solutions to: `y'' - 2xy' + 2ny = 0`
Applications in quantum harmonic oscillator.

### Laguerre Polynomials
Solutions to: `xy'' + (1-x)y' + ny = 0`
Applications in quantum mechanics (hydrogen atom).

### Chebyshev Polynomials
Solutions to: `(1-x²)y'' - xy' + n²y = 0`
Applications in numerical analysis and approximation theory.

## Problem-Solving Strategies

### 1. For Power Series Solutions
1. **Identify the point** of expansion (usually x = 0)
2. **Check if it's ordinary** or singular
3. **Assume power series form**
4. **Substitute and collect terms**
5. **Solve recurrence relations**
6. **Write general solution**

### 2. For Frobenius Method
1. **Identify regular singular points**
2. **Write in standard form**
3. **Find indicial equation**
4. **Determine nature of roots**
5. **Construct solutions** according to the case

### 3. Common Mistakes
- Incorrect recurrence relations
- Wrong indicial equation
- Not handling all cases for Frobenius method
- Algebraic errors in series manipulation
- Forgetting convergence considerations

## Practice Problems

### Power Series Solutions
1. Solve: `y'' + xy = 0` using power series
2. Solve: `y'' - xy' + y = 0` using power series
3. Solve: `(1-x²)y'' - 2xy' + 6y = 0` using power series

### Frobenius Method
1. Solve: `xy'' + 2y' + xy = 0` using Frobenius method
2. Solve: `x²y'' + xy' + (x² - 1/4)y = 0` using Frobenius method
3. Solve: `x²y'' - xy' + (1-x)y = 0` using Frobenius method

### Special Functions
1. Find the first few terms of J₀(x) and J₁(x)
2. Verify that P₃(x) = (5x³ - 3x)/2 satisfies Legendre's equation
3. Show that J₀'(x) = -J₁(x)

### Applications
1. Solve the heat equation in a circular domain using Bessel functions
2. Expand f(x) = x² in Legendre polynomials on [-1,1]

## Summary

Series solutions are essential for solving differential equations with variable coefficients:

1. **Power series method** works at ordinary points
2. **Frobenius method** handles regular singular points
3. **Special functions** (Bessel, Legendre, etc.) arise naturally from series solutions
4. **Applications** span physics, engineering, and applied mathematics
5. **Convergence** considerations are important for practical applications

Master these techniques as they provide the foundation for understanding many important special functions and their applications in science and engineering.
