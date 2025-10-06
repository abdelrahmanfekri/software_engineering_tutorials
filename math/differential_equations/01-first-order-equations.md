# First-Order Differential Equations

## Overview
First-order differential equations are equations that involve a function and its first derivative. They form the foundation of differential equations and are essential for understanding more complex systems. This tutorial covers the main types and solution methods for first-order differential equations.

## Learning Objectives
- Identify different types of first-order differential equations
- Solve separable differential equations
- Solve linear first-order differential equations
- Understand exact equations and integrating factors
- Apply Bernoulli equation techniques
- Work with homogeneous equations

## Types of First-Order Differential Equations

### 1. Separable Equations
A separable differential equation can be written in the form:
```
dy/dx = f(x)g(y)
```

**Solution Method:**
1. Separate variables: `dy/g(y) = f(x)dx`
2. Integrate both sides: `∫dy/g(y) = ∫f(x)dx + C`
3. Solve for y if possible

**Example:**
```
dy/dx = xy²
```
Separate: `dy/y² = x dx`
Integrate: `-1/y = x²/2 + C`
Solution: `y = -2/(x² + 2C)`

### 2. Linear First-Order Equations
A linear first-order equation has the form:
```
dy/dx + P(x)y = Q(x)
```

**Solution Method - Integrating Factor:**
1. Find integrating factor: `μ(x) = e^(∫P(x)dx)`
2. Multiply both sides by μ(x)
3. Left side becomes: `d/dx[μ(x)y] = μ(x)Q(x)`
4. Integrate: `μ(x)y = ∫μ(x)Q(x)dx + C`
5. Solve for y

**Example:**
```
dy/dx + 2xy = x
```
Integrating factor: `μ(x) = e^(∫2x dx) = e^(x²)`
Multiply: `e^(x²)dy/dx + 2xe^(x²)y = xe^(x²)`
This becomes: `d/dx[e^(x²)y] = xe^(x²)`
Integrate: `e^(x²)y = (1/2)e^(x²) + C`
Solution: `y = 1/2 + Ce^(-x²)`

### 3. Exact Equations
An exact equation has the form:
```
M(x,y)dx + N(x,y)dy = 0
```
where `∂M/∂y = ∂N/∂x`

**Solution Method:**
1. Verify exactness: `∂M/∂y = ∂N/∂x`
2. Find potential function F(x,y) such that:
   - `∂F/∂x = M(x,y)`
   - `∂F/∂y = N(x,y)`
3. Solution: `F(x,y) = C`

**Example:**
```
(2xy + 3x²)dx + (x² + 2y)dy = 0
```
Check: `∂M/∂y = 2x = ∂N/∂x = 2x` ✓ (exact)
Find F: `∂F/∂x = 2xy + 3x²` → `F = x²y + x³ + h(y)`
`∂F/∂y = x² + h'(y) = x² + 2y` → `h'(y) = 2y` → `h(y) = y²`
Solution: `x²y + x³ + y² = C`

### 4. Bernoulli Equations
A Bernoulli equation has the form:
```
dy/dx + P(x)y = Q(x)y^n
```

**Solution Method:**
1. Divide by y^n: `y^(-n)dy/dx + P(x)y^(1-n) = Q(x)`
2. Substitute: `v = y^(1-n)`
3. Then: `dv/dx = (1-n)y^(-n)dy/dx`
4. This gives: `1/(1-n) dv/dx + P(x)v = Q(x)`
5. Solve as linear equation for v
6. Substitute back to find y

**Example:**
```
dy/dx + y = xy²
```
Divide by y²: `y^(-2)dy/dx + y^(-1) = x`
Let v = y^(-1): `dv/dx = -y^(-2)dy/dx`
Substitute: `-dv/dx + v = x`
Rearrange: `dv/dx - v = -x`
Linear equation with solution: `v = x + 1 + Ce^x`
Therefore: `y = 1/(x + 1 + Ce^x)`

### 5. Homogeneous Equations
A homogeneous equation has the form:
```
dy/dx = f(y/x)
```

**Solution Method:**
1. Substitute: `v = y/x`, so `y = vx` and `dy/dx = v + x dv/dx`
2. This gives: `v + x dv/dx = f(v)`
3. Separate: `dv/(f(v) - v) = dx/x`
4. Integrate and substitute back

**Example:**
```
dy/dx = (x² + y²)/(2xy)
```
Let v = y/x: `v + x dv/dx = (1 + v²)/(2v)`
Simplify: `x dv/dx = (1 + v²)/(2v) - v = (1 - v²)/(2v)`
Separate: `2v dv/(1 - v²) = dx/x`
Integrate: `-ln|1 - v²| = ln|x| + C`
This gives: `1 - v² = C/x`
Substitute back: `1 - (y/x)² = C/x`
Solution: `x² - y² = Cx`

## Applications

### Population Growth
**Malthusian Growth Model:**
```
dP/dt = kP
```
Solution: `P(t) = P₀e^(kt)`

**Logistic Growth Model:**
```
dP/dt = kP(1 - P/K)
```
Solution: `P(t) = K/(1 + (K/P₀ - 1)e^(-kt))`

### Radioactive Decay
```
dN/dt = -λN
```
Solution: `N(t) = N₀e^(-λt)`

### Newton's Law of Cooling
```
dT/dt = -k(T - T_a)
```
Solution: `T(t) = T_a + (T₀ - T_a)e^(-kt)`

## Problem-Solving Strategies

### 1. Classification Steps
1. **Check if separable**: Can you write `dy/dx = f(x)g(y)`?
2. **Check if linear**: Is it `dy/dx + P(x)y = Q(x)`?
3. **Check if exact**: Does `∂M/∂y = ∂N/∂x`?
4. **Check if Bernoulli**: Is it `dy/dx + P(x)y = Q(x)y^n`?
5. **Check if homogeneous**: Is it `dy/dx = f(y/x)`?

### 2. Common Mistakes to Avoid
- Forgetting the constant of integration
- Incorrect separation of variables
- Wrong integrating factor calculation
- Not checking exactness before applying exact equation method
- Algebraic errors in substitution methods

### 3. Verification
Always verify your solution by:
1. Substituting back into the original equation
2. Checking initial conditions if given
3. Ensuring the solution makes physical sense

## Practice Problems

### Basic Problems
1. Solve: `dy/dx = x/y`
2. Solve: `dy/dx + 2y = e^(-x)`
3. Solve: `(2xy + 1)dx + (x² + 2y)dy = 0`
4. Solve: `dy/dx + y = xy³`
5. Solve: `dy/dx = (x + y)/(x - y)`

### Application Problems
1. A population grows according to `dP/dt = 0.05P`. If P(0) = 1000, find P(10).
2. A radioactive substance decays at rate `dN/dt = -0.1N`. If N(0) = 100, find when N = 50.
3. An object cools from 100°C to 80°C in 10 minutes in a room at 20°C. Find the temperature after 20 minutes.

## Summary

First-order differential equations provide the foundation for understanding dynamic systems. The key to success is:

1. **Proper classification** of the equation type
2. **Correct application** of the appropriate solution method
3. **Careful verification** of solutions
4. **Understanding applications** to real-world problems

Master these techniques before moving on to higher-order equations, as they build upon these fundamental concepts.
