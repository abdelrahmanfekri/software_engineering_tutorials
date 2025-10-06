# Numerical Methods for Differential Equations

## Overview
Numerical methods are essential for solving differential equations that cannot be solved analytically. They provide approximate solutions and are crucial for real-world applications where exact solutions are either impossible or impractical to obtain. This tutorial covers the most important numerical methods for solving ordinary differential equations.

## Learning Objectives
- Understand Euler's method and its variants
- Apply Runge-Kutta methods
- Work with multistep methods
- Analyze stability and convergence
- Solve systems of differential equations numerically
- Understand applications to real problems

## Euler's Method

### Basic Euler's Method
For the initial value problem `dy/dt = f(t,y)`, `y(t₀) = y₀`:

**Formula:**
```
y_{n+1} = y_n + hf(t_n, y_n)
```
where h is the step size.

### Derivation
From the definition of derivative:
```
y'(t_n) ≈ (y(t_{n+1}) - y(t_n))/h
```
Rearranging: `y(t_{n+1}) ≈ y(t_n) + hy'(t_n) = y(t_n) + hf(t_n, y_n)`

### Example
Solve: `dy/dt = y - t`, `y(0) = 1` using h = 0.1

Solution:
- `y₁ = y₀ + hf(0, 1) = 1 + 0.1(1-0) = 1.1`
- `y₂ = y₁ + hf(0.1, 1.1) = 1.1 + 0.1(1.1-0.1) = 1.2`
- `y₃ = y₂ + hf(0.2, 1.2) = 1.2 + 0.1(1.2-0.2) = 1.3`
- etc.

### Error Analysis
**Local truncation error**: O(h²)
**Global error**: O(h)

The method is first-order accurate.

## Improved Euler's Method (Heun's Method)

### Formula
```
k₁ = f(t_n, y_n)
k₂ = f(t_{n+1}, y_n + hk₁)
y_{n+1} = y_n + (h/2)(k₁ + k₂)
```

### Interpretation
This is a predictor-corrector method:
1. **Predict**: `y_{n+1}^{(0)} = y_n + hf(t_n, y_n)` (Euler's method)
2. **Correct**: `y_{n+1} = y_n + (h/2)[f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(0)})]`

### Error Analysis
**Local truncation error**: O(h³)
**Global error**: O(h²)

The method is second-order accurate.

## Runge-Kutta Methods

### Fourth-Order Runge-Kutta (RK4)
The most commonly used Runge-Kutta method:

```
k₁ = f(t_n, y_n)
k₂ = f(t_n + h/2, y_n + hk₁/2)
k₃ = f(t_n + h/2, y_n + hk₂/2)
k₄ = f(t_n + h, y_n + hk₃)
y_{n+1} = y_n + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

### Interpretation
- k₁: slope at the beginning of the interval
- k₂: slope at the midpoint using k₁
- k₃: slope at the midpoint using k₂
- k₄: slope at the end using k₃
- Final slope is a weighted average

### Error Analysis
**Local truncation error**: O(h⁵)
**Global error**: O(h⁴)

The method is fourth-order accurate.

### Example
Solve: `dy/dt = y - t`, `y(0) = 1` using RK4 with h = 0.1

For the first step:
- `k₁ = f(0, 1) = 1 - 0 = 1`
- `k₂ = f(0.05, 1 + 0.05×1) = f(0.05, 1.05) = 1.05 - 0.05 = 1`
- `k₃ = f(0.05, 1 + 0.05×1) = f(0.05, 1.05) = 1`
- `k₄ = f(0.1, 1 + 0.1×1) = f(0.1, 1.1) = 1.1 - 0.1 = 1`
- `y₁ = 1 + (0.1/6)(1 + 2×1 + 2×1 + 1) = 1 + 0.1 = 1.1`

## Multistep Methods

### Adams-Bashforth Methods
These methods use information from previous steps:

**Two-step Adams-Bashforth:**
```
y_{n+1} = y_n + (h/2)[3f(t_n, y_n) - f(t_{n-1}, y_{n-1})]
```

**Three-step Adams-Bashforth:**
```
y_{n+1} = y_n + (h/12)[23f(t_n, y_n) - 16f(t_{n-1}, y_{n-1}) + 5f(t_{n-2}, y_{n-2})]
```

### Adams-Moulton Methods
These are implicit methods (predictor-corrector):

**Two-step Adams-Moulton:**
```
y_{n+1} = y_n + (h/2)[f(t_{n+1}, y_{n+1}) + f(t_n, y_n)]
```

### Starting Values
Multistep methods require starting values from single-step methods.

## Systems of Differential Equations

### First-Order Systems
For the system:
```
dx/dt = f(t, x, y)
dy/dt = g(t, x, y)
```
with initial conditions `x(t₀) = x₀`, `y(t₀) = y₀`:

**Euler's Method:**
```
x_{n+1} = x_n + hf(t_n, x_n, y_n)
y_{n+1} = y_n + hg(t_n, x_n, y_n)
```

**RK4 Method:**
```
k₁ = f(t_n, x_n, y_n), l₁ = g(t_n, x_n, y_n)
k₂ = f(t_n + h/2, x_n + hk₁/2, y_n + hl₁/2), l₂ = g(t_n + h/2, x_n + hk₁/2, y_n + hl₁/2)
k₃ = f(t_n + h/2, x_n + hk₂/2, y_n + hl₂/2), l₃ = g(t_n + h/2, x_n + hk₂/2, y_n + hl₂/2)
k₄ = f(t_n + h, x_n + hk₃, y_n + hl₃), l₄ = g(t_n + h, x_n + hk₃, y_n + hl₃)
x_{n+1} = x_n + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
y_{n+1} = y_n + (h/6)(l₁ + 2l₂ + 2l₃ + l₄)
```

### Example: Lotka-Volterra System
```
dx/dt = ax - bxy
dy/dt = -cy + dxy
```

Using RK4 with a = 2, b = 1, c = 1, d = 1, x(0) = 10, y(0) = 5, h = 0.1:

For the first step:
- `k₁ = 2×10 - 1×10×5 = 20 - 50 = -30`
- `l₁ = -1×5 + 1×10×5 = -5 + 50 = 45`
- etc.

## Stability and Convergence

### Stability
A numerical method is **stable** if small errors don't grow exponentially.

**Test equation**: `dy/dt = λy`, `y(0) = 1`

**Euler's method**: `y_{n+1} = y_n + hλy_n = (1 + hλ)y_n`
Solution: `y_n = (1 + hλ)^n`

**Stability condition**: `|1 + hλ| ≤ 1`

For real λ < 0: `h ≤ 2/|λ|`

### Convergence
A method is **convergent** if the numerical solution approaches the exact solution as h → 0.

### Error Control
**Adaptive step size**: Adjust h based on local error estimates.

**Embedded Runge-Kutta methods** (e.g., Runge-Kutta-Fehlberg):
- Provide error estimates
- Allow automatic step size control

## Higher-Order Equations

### Conversion to First-Order Systems
Convert `y'' = f(t, y, y')` to:
```
y₁ = y, y₂ = y'
dy₁/dt = y₂
dy₂/dt = f(t, y₁, y₂)
```

### Example: Second-Order Equation
```
y'' + 2y' + y = 0, y(0) = 1, y'(0) = 0
```

Let `y₁ = y`, `y₂ = y'`:
```
dy₁/dt = y₂
dy₂/dt = -2y₂ - y₁
```

Initial conditions: `y₁(0) = 1`, `y₂(0) = 0`

## Applications

### Population Dynamics
**Logistic growth model**: `dP/dt = rP(1 - P/K)`

### Chemical Reactions
**Rate equations**: `d[A]/dt = -k[A]`

### Mechanical Systems
**Spring-mass-damper**: `mẍ + cẋ + kx = F(t)`

### Electrical Circuits
**RLC circuit**: `L(d²q/dt²) + R(dq/dt) + (1/C)q = E(t)`

## Problem-Solving Strategies

### 1. Method Selection
- **Euler's method**: Simple, first-order accuracy
- **RK4**: Good balance of accuracy and computational cost
- **Multistep methods**: Efficient for smooth solutions
- **Adaptive methods**: For problems with varying solution behavior

### 2. Step Size Selection
- **Too large**: Large errors, possible instability
- **Too small**: Excessive computational cost
- **Adaptive**: Let the algorithm choose

### 3. Common Issues
- **Stability problems**: Use smaller step size or different method
- **Accuracy problems**: Use higher-order method
- **Computational efficiency**: Consider multistep methods

## Practice Problems

### Basic Methods
1. Solve: `dy/dt = y - t`, `y(0) = 1` using Euler's method with h = 0.2 for t ∈ [0,1]
2. Solve: `dy/dt = -2y + 1`, `y(0) = 0` using RK4 with h = 0.1 for t ∈ [0,1]

### Systems
1. Solve: `dx/dt = -x + y`, `dy/dt = x - y` with `x(0) = 1`, `y(0) = 0` using RK4
2. Solve the Lotka-Volterra system with a = 1, b = 0.5, c = 0.5, d = 0.5

### Higher-Order Equations
1. Solve: `y'' + y = 0`, `y(0) = 1`, `y'(0) = 0` by converting to a system and using RK4

### Stability Analysis
1. For `dy/dt = -5y`, `y(0) = 1`, find the maximum step size for Euler's method to be stable
2. Compare the stability of Euler's method and RK4 for `dy/dt = -10y`

## Summary

Numerical methods are essential for solving differential equations in practice:

1. **Euler's method** is simple but has limited accuracy
2. **Runge-Kutta methods** provide good accuracy with reasonable computational cost
3. **Multistep methods** are efficient for smooth solutions
4. **Stability and convergence** are crucial considerations
5. **Systems of equations** are handled by applying methods component-wise
6. **Higher-order equations** are converted to first-order systems

Master these techniques as they are fundamental tools for solving real-world problems in science and engineering where analytical solutions are not available.
