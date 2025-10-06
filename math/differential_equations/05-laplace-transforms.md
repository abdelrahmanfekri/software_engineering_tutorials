# Laplace Transforms

## Overview
The Laplace transform is a powerful integral transform that converts differential equations into algebraic equations, making them easier to solve. It's particularly useful for initial value problems and systems with discontinuous forcing functions. This tutorial covers the theory and applications of Laplace transforms to differential equations.

## Learning Objectives
- Understand the definition and basic properties of Laplace transforms
- Compute Laplace transforms of common functions
- Find inverse Laplace transforms
- Apply Laplace transforms to solve differential equations
- Use the convolution theorem
- Work with step functions and impulse functions

## Definition of Laplace Transform

### Definition
The Laplace transform of a function f(t) is defined as:
```
L{f(t)} = F(s) = ∫₀^∞ e^(-st) f(t) dt
```

### Existence Conditions
The Laplace transform exists if:
1. f(t) is piecewise continuous on [0, ∞)
2. f(t) is of exponential order, i.e., |f(t)| ≤ Me^(at) for some M > 0 and a ≥ 0

## Basic Laplace Transforms

### Elementary Functions
```
L{1} = 1/s
L{t^n} = n!/s^(n+1)
L{e^(at)} = 1/(s-a)
L{sin(bt)} = b/(s² + b²)
L{cos(bt)} = s/(s² + b²)
L{sinh(bt)} = b/(s² - b²)
L{cosh(bt)} = s/(s² - b²)
```

### Derivatives
```
L{f'(t)} = sF(s) - f(0)
L{f''(t)} = s²F(s) - sf(0) - f'(0)
L{f^(n)(t)} = s^n F(s) - s^(n-1)f(0) - s^(n-2)f'(0) - ... - f^(n-1)(0)
```

### Integrals
```
L{∫₀^t f(τ)dτ} = F(s)/s
```

## Properties of Laplace Transforms

### Linearity
```
L{af(t) + bg(t)} = aL{f(t)} + bL{g(t)}
```

### First Shifting Theorem
```
L{e^(at)f(t)} = F(s-a)
```

### Second Shifting Theorem
```
L{f(t-a)u(t-a)} = e^(-as)F(s)
```
where u(t-a) is the unit step function.

### Multiplication by t
```
L{tf(t)} = -F'(s)
L{t^n f(t)} = (-1)^n F^(n)(s)
```

### Division by t
```
L{f(t)/t} = ∫_s^∞ F(u)du
```

### Scaling
```
L{f(at)} = (1/a)F(s/a)
```

## Inverse Laplace Transforms

### Definition
```
L^{-1}{F(s)} = f(t)
```

### Method of Partial Fractions
For rational functions F(s) = P(s)/Q(s):

1. **Factor the denominator** Q(s)
2. **Write partial fraction decomposition**
3. **Use table of transforms** to find inverse

#### Example:
```
F(s) = (s+1)/(s² + 3s + 2)
```
Factor: `s² + 3s + 2 = (s+1)(s+2)`
Partial fractions: `(s+1)/((s+1)(s+2)) = A/(s+1) + B/(s+2)`
Solve: `s+1 = A(s+2) + B(s+1)`
Let s = -1: `0 = A(1) + B(0)` → `A = 0`
Let s = -2: `-1 = A(0) + B(-1)` → `B = 1`
Therefore: `F(s) = 1/(s+2)`
Inverse: `f(t) = e^(-2t)`

### Complex Roots
For quadratic factors with complex roots:
```
1/((s-a)² + b²) ↔ (1/b)e^(at)sin(bt)
(s-a)/((s-a)² + b²) ↔ e^(at)cos(bt)
```

## Applications to Differential Equations

### Solving Initial Value Problems

#### Steps:
1. **Take Laplace transform** of both sides
2. **Use initial conditions** to simplify
3. **Solve for Y(s)**
4. **Take inverse Laplace transform** to find y(t)

#### Example:
```
y'' - 3y' + 2y = 4e^t, y(0) = 1, y'(0) = 0
```
Laplace transform: `s²Y(s) - sy(0) - y'(0) - 3[sY(s) - y(0)] + 2Y(s) = 4/(s-1)`
Substitute initial conditions: `s²Y(s) - s - 3[sY(s) - 1] + 2Y(s) = 4/(s-1)`
Simplify: `(s² - 3s + 2)Y(s) = 4/(s-1) + s - 3`
Factor: `(s-1)(s-2)Y(s) = 4/(s-1) + s - 3`
Solve for Y(s): `Y(s) = 4/[(s-1)²(s-2)] + (s-3)/[(s-1)(s-2)]`

Partial fractions for first term: `4/[(s-1)²(s-2)] = A/(s-1) + B/(s-1)² + C/(s-2)`
Solving: `A = -4`, `B = -4`, `C = 4`
First term: `-4/(s-1) - 4/(s-1)² + 4/(s-2)`

Partial fractions for second term: `(s-3)/[(s-1)(s-2)] = D/(s-1) + E/(s-2)`
Solving: `D = 2`, `E = -1`
Second term: `2/(s-1) - 1/(s-2)`

Combining: `Y(s) = -2/(s-1) - 4/(s-1)² + 3/(s-2)`
Inverse transform: `y(t) = -2e^t - 4te^t + 3e^(2t)`

### Systems of Differential Equations

#### Example:
```
x' = x + 2y + 1, x(0) = 0
y' = 2x + y + t, y(0) = 0
```
Laplace transforms: `sX(s) = X(s) + 2Y(s) + 1/s`
`sY(s) = 2X(s) + Y(s) + 1/s²`

Rearrange: `(s-1)X(s) - 2Y(s) = 1/s`
`-2X(s) + (s-1)Y(s) = 1/s²`

Solve system: `X(s) = (s+1)/(s²(s-3))`, `Y(s) = (2s+1)/(s²(s-3))`
Use partial fractions and inverse transforms to find x(t) and y(t).

## Convolution Theorem

### Definition
The convolution of f(t) and g(t) is:
```
(f * g)(t) = ∫₀^t f(τ)g(t-τ)dτ
```

### Convolution Theorem
```
L{f * g} = L{f}L{g} = F(s)G(s)
L^{-1}{F(s)G(s)} = (f * g)(t)
```

### Applications
1. **Inverse transforms** of products
2. **Integral equations**
3. **Systems with memory**

#### Example:
Find `L^{-1}{1/(s(s²+1))}`:
```
1/(s(s²+1)) = (1/s)(1/(s²+1))
```
Since `L{1} = 1/s` and `L{sin(t)} = 1/(s²+1)`:
```
L^{-1}{1/(s(s²+1))} = 1 * sin(t) = ∫₀^t sin(τ)dτ = 1 - cos(t)
```

## Step Functions and Impulse Functions

### Unit Step Function
```
u(t-a) = {0, t < a
        {1, t ≥ a
```
Laplace transform: `L{u(t-a)} = e^(-as)/s`

### Second Shifting Theorem
```
L{f(t-a)u(t-a)} = e^(-as)F(s)
```

#### Example:
Find the Laplace transform of:
```
f(t) = {0,     t < 2
       {t-2,   t ≥ 2
```
Write as: `f(t) = (t-2)u(t-2)`
Since `L{t} = 1/s²`:
```
L{f(t)} = e^(-2s)/s²
```

### Dirac Delta Function (Impulse Function)
```
δ(t-a) = {∞, t = a
        {0, t ≠ a
```
with the property `∫₀^∞ δ(t-a)dt = 1`

Laplace transform: `L{δ(t-a)} = e^(-as)`

### Applications to Differential Equations
Impulse functions model sudden forces or inputs.

#### Example:
```
y'' + y = δ(t-π), y(0) = 0, y'(0) = 0
```
Laplace transform: `s²Y(s) + Y(s) = e^(-πs)`
Solution: `Y(s) = e^(-πs)/(s²+1)`
Inverse: `y(t) = sin(t-π)u(t-π) = -sin(t)u(t-π)`

## Problem-Solving Strategies

### 1. For Initial Value Problems
1. **Take Laplace transform** of both sides
2. **Apply initial conditions**
3. **Solve for Y(s)**
4. **Use partial fractions** if necessary
5. **Take inverse Laplace transform**

### 2. For Systems
1. **Transform each equation**
2. **Solve the algebraic system**
3. **Find inverse transforms**

### 3. Common Mistakes
- Forgetting initial conditions
- Incorrect partial fraction decomposition
- Wrong signs in shifting theorems
- Algebraic errors in solving for Y(s)

## Practice Problems

### Basic Transforms
1. Find `L{t³e^(2t)}`
2. Find `L{sin(3t)cos(2t)}`
3. Find `L{∫₀^t τe^τ dτ}`

### Inverse Transforms
1. Find `L^{-1}{1/(s²+4s+5)}`
2. Find `L^{-1}{s/(s²+2s+2)}`
3. Find `L^{-1}{e^(-2s)/(s²+1)}`

### Differential Equations
1. Solve: `y'' + 4y' + 3y = 0`, `y(0) = 1`, `y'(0) = 0`
2. Solve: `y'' + y = sin(t)`, `y(0) = 0`, `y'(0) = 1`
3. Solve: `y'' + 2y' + y = te^(-t)`, `y(0) = 0`, `y'(0) = 0`

### Step Functions
1. Find `L{f(t)}` where `f(t) = {0, t < 1; t², t ≥ 1}`
2. Solve: `y'' + y = u(t-1)`, `y(0) = 0`, `y'(0) = 0`

### Convolution
1. Find `L^{-1}{1/(s(s+1))}` using convolution
2. Solve: `y'' + y = ∫₀^t y(τ)dτ`, `y(0) = 1`, `y'(0) = 0`

## Summary

Laplace transforms provide a powerful method for solving differential equations:

1. **Convert differential equations** to algebraic equations
2. **Handle initial conditions** naturally
3. **Work with discontinuous functions** using step functions
4. **Model impulse inputs** using delta functions
5. **Solve systems** of differential equations
6. **Use convolution theorem** for complex inverse transforms

The method is particularly valuable for initial value problems and systems with discontinuous or impulsive forcing functions. Master these techniques as they are essential for advanced applications in engineering and physics.
