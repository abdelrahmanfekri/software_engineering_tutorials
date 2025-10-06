# Second-Order Linear Differential Equations

## Overview
Second-order linear differential equations are fundamental in physics, engineering, and mathematics. They describe many important physical phenomena including oscillations, vibrations, and wave propagation. This tutorial covers the theory and solution methods for second-order linear equations.

## Learning Objectives
- Understand the general form of second-order linear equations
- Solve homogeneous equations with constant coefficients
- Apply the method of undetermined coefficients
- Use variation of parameters for nonhomogeneous equations
- Solve Cauchy-Euler equations
- Understand the superposition principle

## General Form
A second-order linear differential equation has the form:
```
a(x)y'' + b(x)y' + c(x)y = f(x)
```

If `a(x) ≠ 0`, we can write it in standard form:
```
y'' + p(x)y' + q(x)y = g(x)
```

## Homogeneous Equations with Constant Coefficients

### Form
```
ay'' + by' + cy = 0
```
where a, b, c are constants and a ≠ 0.

### Solution Method
1. **Characteristic Equation**: `ar² + br + c = 0`
2. **Find roots**: r₁ and r₂
3. **General solution depends on the nature of roots**:

#### Case 1: Distinct Real Roots (r₁ ≠ r₂)
**General Solution:**
```
y = C₁e^(r₁x) + C₂e^(r₂x)
```

**Example:**
```
y'' - 5y' + 6y = 0
```
Characteristic equation: `r² - 5r + 6 = 0`
Roots: `r = 2, 3`
Solution: `y = C₁e^(2x) + C₂e^(3x)`

#### Case 2: Repeated Real Roots (r₁ = r₂ = r)
**General Solution:**
```
y = C₁e^(rx) + C₂xe^(rx)
```

**Example:**
```
y'' - 4y' + 4y = 0
```
Characteristic equation: `r² - 4r + 4 = 0`
Roots: `r = 2, 2` (repeated)
Solution: `y = C₁e^(2x) + C₂xe^(2x)`

#### Case 3: Complex Roots (r = α ± βi)
**General Solution:**
```
y = e^(αx)[C₁cos(βx) + C₂sin(βx)]
```

**Example:**
```
y'' - 2y' + 5y = 0
```
Characteristic equation: `r² - 2r + 5 = 0`
Roots: `r = 1 ± 2i`
Solution: `y = e^x[C₁cos(2x) + C₂sin(2x)]`

## Nonhomogeneous Equations

### General Solution Structure
For `y'' + p(x)y' + q(x)y = g(x)`:
```
y = y_c + y_p
```
where:
- `y_c` is the complementary solution (solution to homogeneous equation)
- `y_p` is a particular solution

### Method of Undetermined Coefficients

This method works when `g(x)` is a sum of terms of the form:
- `e^(αx)`
- `sin(βx)` or `cos(βx)`
- `x^n`
- Products of the above

#### Steps:
1. Find the complementary solution `y_c`
2. Make an educated guess for `y_p` based on `g(x)`
3. If any term in `y_p` appears in `y_c`, multiply by `x` (or `x²` if still appears)
4. Substitute `y_p` into the differential equation
5. Solve for the coefficients

#### Example:
```
y'' - 3y' + 2y = 3e^(2x)
```
Complementary solution: `y_c = C₁e^x + C₂e^(2x)`
Guess for particular solution: `y_p = Ae^(2x)`
But `e^(2x)` is in `y_c`, so multiply by `x`: `y_p = Axe^(2x)`
Substitute and solve: `A = -3`
Particular solution: `y_p = -3xe^(2x)`
General solution: `y = C₁e^x + C₂e^(2x) - 3xe^(2x)`

### Variation of Parameters

This method works for any nonhomogeneous equation where we know `y_c`.

#### Steps:
1. Find the complementary solution: `y_c = C₁y₁(x) + C₂y₂(x)`
2. Assume: `y_p = u₁(x)y₁(x) + u₂(x)y₂(x)`
3. Set up the system:
   ```
   u₁'y₁ + u₂'y₂ = 0
   u₁'y₁' + u₂'y₂' = g(x)
   ```
4. Solve for `u₁'` and `u₂'`
5. Integrate to find `u₁` and `u₂`
6. Substitute back to find `y_p`

#### Example:
```
y'' + y = tan(x)
```
Complementary solution: `y_c = C₁cos(x) + C₂sin(x)`
Assume: `y_p = u₁cos(x) + u₂sin(x)`
System:
```
u₁'cos(x) + u₂'sin(x) = 0
-u₁'sin(x) + u₂'cos(x) = tan(x)
```
Solve: `u₁' = -sin²(x)/cos(x)`, `u₂' = sin(x)`
Integrate: `u₁ = sin(x) - ln|sec(x) + tan(x)|`, `u₂ = -cos(x)`
Particular solution: `y_p = -cos(x)ln|sec(x) + tan(x)|`
General solution: `y = C₁cos(x) + C₂sin(x) - cos(x)ln|sec(x) + tan(x)|`

## Cauchy-Euler Equations

### Form
```
ax²y'' + bxy' + cy = 0
```

### Solution Method
1. **Characteristic Equation**: `ar(r-1) + br + c = 0`
2. **Solve for r** and apply the same cases as constant coefficient equations

#### Example:
```
x²y'' - 3xy' + 4y = 0
```
Characteristic equation: `r(r-1) - 3r + 4 = 0`
Simplify: `r² - 4r + 4 = 0`
Roots: `r = 2, 2` (repeated)
Solution: `y = C₁x² + C₂x²ln(x)`

## Applications

### Simple Harmonic Motion
**Undamped Oscillator:**
```
y'' + ω²y = 0
```
Solution: `y = Acos(ωt) + Bsin(ωt)`

**Damped Oscillator:**
```
y'' + 2βy' + ω²y = 0
```
- **Underdamped**: β < ω, oscillatory decay
- **Critically damped**: β = ω, fastest return to equilibrium
- **Overdamped**: β > ω, exponential decay

### Forced Oscillations
```
y'' + 2βy' + ω²y = Fcos(γt)
```
Solution includes resonance effects when γ ≈ ω.

### Electrical Circuits (RLC Circuit)
```
L(d²q/dt²) + R(dq/dt) + (1/C)q = E(t)
```
where q is charge, L is inductance, R is resistance, C is capacitance, E(t) is voltage source.

## Problem-Solving Strategies

### 1. Classification Steps
1. **Identify the type**: Homogeneous or nonhomogeneous?
2. **Constant coefficients?** → Use characteristic equation
3. **Cauchy-Euler form?** → Use substitution method
4. **Nonhomogeneous?** → Choose undetermined coefficients or variation of parameters

### 2. Method Selection
- **Undetermined Coefficients**: Use when `g(x)` is simple (polynomials, exponentials, trig functions)
- **Variation of Parameters**: Use when undetermined coefficients doesn't apply or fails

### 3. Common Mistakes
- Forgetting to find the complementary solution first
- Incorrect guess for particular solution
- Not multiplying by `x` when necessary in undetermined coefficients
- Algebraic errors in solving for coefficients
- Forgetting initial conditions

## Practice Problems

### Homogeneous Equations
1. Solve: `y'' - 4y' + 3y = 0`
2. Solve: `y'' + 6y' + 9y = 0`
3. Solve: `y'' + 2y' + 5y = 0`

### Nonhomogeneous Equations
1. Solve: `y'' - 3y' + 2y = 4e^x`
2. Solve: `y'' + y = 2sin(x)`
3. Solve: `y'' - 4y = x²`

### Initial Value Problems
1. Solve: `y'' + 4y = 0`, `y(0) = 1`, `y'(0) = 2`
2. Solve: `y'' - 2y' + y = e^x`, `y(0) = 1`, `y'(0) = 0`

### Applications
1. A spring-mass system has `m = 1`, `k = 4`, `c = 4`. If initially displaced 1 unit and released, find the position function.
2. An RLC circuit has `L = 1`, `R = 2`, `C = 1/2`, with `E(t) = 10sin(2t)`. Find the charge function.

## Summary

Second-order linear differential equations are crucial for modeling oscillatory and wave phenomena. Key points:

1. **Homogeneous equations** with constant coefficients are solved using characteristic equations
2. **Nonhomogeneous equations** require finding both complementary and particular solutions
3. **Method of undetermined coefficients** works for simple forcing functions
4. **Variation of parameters** is more general but computationally intensive
5. **Applications** span physics, engineering, and many other fields

Master these techniques as they form the foundation for understanding more complex differential equations and their applications.
