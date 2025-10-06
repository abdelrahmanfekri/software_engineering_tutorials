# Differential Equations

## Overview
This tutorial covers differential equations, which are equations involving derivatives. Differential equations are fundamental tools in mathematics, physics, engineering, and many other fields, describing how quantities change over time or space.

## Learning Objectives
- Understand different types of differential equations
- Solve separable differential equations
- Work with linear differential equations
- Apply differential equations to growth and decay problems
- Use slope fields to visualize solutions

## 1. Types of Differential Equations

### Ordinary Differential Equations (ODEs)
Equations involving derivatives of a function of one variable.

### Partial Differential Equations (PDEs)
Equations involving partial derivatives of a function of multiple variables.

### Order
The order of a differential equation is the highest derivative present.

### Examples

#### First Order
```
dy/dx = 2x
```

#### Second Order
```
d²y/dx² + 3dy/dx + 2y = 0
```

#### Partial Differential Equation
```
∂u/∂t = k ∂²u/∂x² (heat equation)
```

## 2. Separable Differential Equations

### Definition
A separable differential equation can be written as:
```
dy/dx = f(x)g(y)
```

### Solution Method
1. Separate variables: dy/g(y) = f(x)dx
2. Integrate both sides
3. Solve for y

### Examples

#### Example 1: Basic Separable Equation
```
dy/dx = 2xy
```

Separate variables:
```
dy/y = 2x dx
```

Integrate:
```
ln|y| = x² + C
y = Ce^(x²)
```

#### Example 2: Population Growth
```
dP/dt = kP
```

Separate variables:
```
dP/P = k dt
```

Integrate:
```
ln|P| = kt + C
P = Ce^(kt)
```

If P(0) = P₀, then C = P₀, so P = P₀e^(kt)

#### Example 3: Newton's Law of Cooling
```
dT/dt = -k(T - Tₐ)
```

Where T is temperature, Tₐ is ambient temperature, k > 0.

Separate variables:
```
dT/(T - Tₐ) = -k dt
```

Integrate:
```
ln|T - Tₐ| = -kt + C
T - Tₐ = Ce^(-kt)
T = Tₐ + Ce^(-kt)
```

## 3. Linear Differential Equations

### First Order Linear
A first order linear differential equation has the form:
```
dy/dx + P(x)y = Q(x)
```

### Solution Method
1. Find the integrating factor: μ(x) = e^(∫P(x)dx)
2. Multiply both sides by μ(x)
3. The left side becomes d/dx[μ(x)y]
4. Integrate both sides
5. Solve for y

### Examples

#### Example 1: Basic Linear Equation
```
dy/dx + 2y = 4x
```

Here P(x) = 2, Q(x) = 4x

Integrating factor:
```
μ(x) = e^(∫2dx) = e^(2x)
```

Multiply by μ(x):
```
e^(2x) dy/dx + 2e^(2x) y = 4xe^(2x)
d/dx[e^(2x) y] = 4xe^(2x)
```

Integrate:
```
e^(2x) y = ∫4xe^(2x) dx = 2xe^(2x) - e^(2x) + C
y = 2x - 1 + Ce^(-2x)
```

#### Example 2: Mixing Problem
A tank contains 100 L of water with 10 kg of salt. Pure water flows in at 2 L/min and the mixture flows out at 2 L/min. Find the amount of salt at time t.

Let S(t) be the amount of salt at time t.

Rate of change of salt:
```
dS/dt = rate in - rate out = 0 - (S/100) · 2 = -S/50
```

This is separable:
```
dS/S = -dt/50
ln|S| = -t/50 + C
S = Ce^(-t/50)
```

Since S(0) = 10, C = 10, so S = 10e^(-t/50)

## 4. Homogeneous Linear Differential Equations

### Second Order Homogeneous
A second order homogeneous linear differential equation has the form:
```
a d²y/dx² + b dy/dx + c y = 0
```

### Characteristic Equation
For the equation above, the characteristic equation is:
```
ar² + br + c = 0
```

### Solutions Based on Roots

#### Case 1: Distinct Real Roots (r₁ ≠ r₂)
```
y = C₁e^(r₁x) + C₂e^(r₂x)
```

#### Case 2: Repeated Real Root (r₁ = r₂ = r)
```
y = C₁e^(rx) + C₂xe^(rx)
```

#### Case 3: Complex Roots (r = α ± βi)
```
y = e^(αx)(C₁cos(βx) + C₂sin(βx))
```

### Examples

#### Example 1: Distinct Real Roots
```
d²y/dx² - 5dy/dx + 6y = 0
```

Characteristic equation: r² - 5r + 6 = 0
Roots: r = 2, 3

Solution: y = C₁e^(2x) + C₂e^(3x)

#### Example 2: Repeated Root
```
d²y/dx² - 4dy/dx + 4y = 0
```

Characteristic equation: r² - 4r + 4 = 0
Roots: r = 2, 2

Solution: y = C₁e^(2x) + C₂xe^(2x)

#### Example 3: Complex Roots
```
d²y/dx² + 4y = 0
```

Characteristic equation: r² + 4 = 0
Roots: r = ±2i

Solution: y = C₁cos(2x) + C₂sin(2x)

## 5. Nonhomogeneous Linear Differential Equations

### Second Order Nonhomogeneous
A second order nonhomogeneous linear differential equation has the form:
```
a d²y/dx² + b dy/dx + c y = f(x)
```

### Solution Method
1. Find the general solution to the homogeneous equation (yₕ)
2. Find a particular solution to the nonhomogeneous equation (yₚ)
3. The general solution is y = yₕ + yₚ

### Method of Undetermined Coefficients
For certain forms of f(x), guess a particular solution.

#### Example: Nonhomogeneous Equation
```
d²y/dx² - 3dy/dx + 2y = 2e^x
```

Homogeneous solution: yₕ = C₁e^x + C₂e^(2x)

For the particular solution, try yₚ = Axe^x (since e^x is already in yₕ):
```
yₚ' = Ae^x + Axe^x
yₚ'' = 2Ae^x + Axe^x
```

Substituting:
```
(2Ae^x + Axe^x) - 3(Ae^x + Axe^x) + 2(Axe^x) = 2e^x
2Ae^x + Axe^x - 3Ae^x - 3Axe^x + 2Axe^x = 2e^x
-Ae^x = 2e^x
```

So A = -2, giving yₚ = -2xe^x

General solution: y = C₁e^x + C₂e^(2x) - 2xe^x

## 6. Applications

### Growth and Decay Models

#### Exponential Growth
```
dP/dt = kP
```
Solution: P = P₀e^(kt)

#### Logistic Growth
```
dP/dt = kP(1 - P/K)
```
Where K is the carrying capacity.

#### Radioactive Decay
```
dN/dt = -λN
```
Solution: N = N₀e^(-λt)

### Physics Applications

#### Simple Harmonic Motion
```
d²x/dt² + ω²x = 0
```
Solution: x = A cos(ωt + φ)

#### Damped Harmonic Motion
```
d²x/dt² + 2γ dx/dt + ω₀²x = 0
```

### Economics Applications

#### Compound Interest
```
dA/dt = rA
```
Solution: A = A₀e^(rt)

#### Continuous Growth Models
```
dS/dt = rS - C
```
Where S is savings, r is interest rate, C is consumption.

## 7. Slope Fields

### Definition
A slope field (direction field) shows the slope of the solution curve at various points.

### Construction
For dy/dx = f(x,y), draw short line segments with slope f(x,y) at each point (x,y).

### Examples

#### Example 1: dy/dx = x
The slope field shows lines with slope equal to the x-coordinate.

#### Example 2: dy/dx = y
The slope field shows lines with slope equal to the y-coordinate.

### Using Slope Fields
1. Sketch solution curves that follow the direction field
2. Estimate values of solutions
3. Understand the behavior of solutions

## 8. Practice Problems

### Separable Equations
1. dy/dx = xy²
2. dy/dx = (x + 1)/y
3. dy/dx = e^(x-y)
4. dy/dx = y(1 - y)

### Linear Equations
1. dy/dx + y = e^x
2. dy/dx + 2y = 4x
3. dy/dx + y/x = x²
4. dy/dx + y = sin(x)

### Second Order Homogeneous
1. d²y/dx² - 6dy/dx + 9y = 0
2. d²y/dx² + 4dy/dx + 5y = 0
3. d²y/dx² - y = 0
4. d²y/dx² + 9y = 0

### Applications
1. A population grows according to dP/dt = 0.02P. If P(0) = 1000, find P(t).
2. A radioactive substance decays according to dN/dt = -0.05N. If N(0) = 100, find N(t).
3. A spring-mass system satisfies d²x/dt² + 4x = 0. If x(0) = 1, x'(0) = 0, find x(t).
4. A mixing problem: A tank contains 200 L of water with 20 kg of salt. Salt water (0.1 kg/L) flows in at 3 L/min and the mixture flows out at 3 L/min. Find the amount of salt at time t.

## 9. Common Mistakes to Avoid

1. **Separation**: Make sure to separate variables correctly
2. **Integration**: Don't forget the constant of integration
3. **Initial Conditions**: Use initial conditions to find specific solutions
4. **Linear vs. Nonlinear**: Distinguish between linear and nonlinear equations
5. **Particular Solutions**: Choose appropriate forms for particular solutions

## 10. Advanced Topics

### Laplace Transforms
Laplace transforms can be used to solve differential equations:
```
L{f'(t)} = sF(s) - f(0)
L{f''(t)} = s²F(s) - sf(0) - f'(0)
```

### Systems of Differential Equations
```
dx/dt = f(x,y)
dy/dt = g(x,y)
```

### Partial Differential Equations
- Heat equation: ∂u/∂t = k ∂²u/∂x²
- Wave equation: ∂²u/∂t² = c² ∂²u/∂x²
- Laplace's equation: ∂²u/∂x² + ∂²u/∂y² = 0

## 11. Study Tips

1. **Learn Types**: Understand different types of differential equations
2. **Practice Methods**: Master the solution methods for each type
3. **Use Applications**: See how differential equations model real phenomena
4. **Check Solutions**: Verify solutions by substituting back
5. **Understand Behavior**: Use slope fields to understand solution behavior

## Next Steps

After mastering differential equations, proceed to:
- Advanced calculus topics
- Vector calculus
- Complex analysis
- Partial differential equations

Remember: Differential equations are powerful tools for modeling change and understanding the behavior of dynamic systems. They connect mathematics to the real world, describing how quantities evolve over time or space.
