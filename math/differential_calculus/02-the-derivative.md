# Differential Calculus Tutorial 02: The Derivative

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the definition of the derivative
- Calculate derivatives using the limit definition
- Interpret derivatives geometrically and physically
- Understand differentiability and continuity
- Apply derivatives to find tangent lines
- Use derivatives in real-world applications

## Definition of the Derivative

### Limit Definition
The derivative of f(x) at point a is:
f'(a) = lim(h→0) [f(a + h) - f(a)]/h

Alternative form:
f'(a) = lim(x→a) [f(x) - f(a)]/(x - a)

### Derivative as a Function
If the limit exists for all x in the domain, we get the derivative function:
f'(x) = lim(h→0) [f(x + h) - f(x)]/h

**Notation**: f'(x), dy/dx, d/dx[f(x)]

## Calculating Derivatives

### Using the Limit Definition
**Example**: Find f'(x) for f(x) = x²

f'(x) = lim(h→0) [(x + h)² - x²]/h
= lim(h→0) [x² + 2xh + h² - x²]/h
= lim(h→0) [2xh + h²]/h
= lim(h→0) [h(2x + h)]/h
= lim(h→0) (2x + h)
= 2x

**Example**: Find f'(x) for f(x) = √x

f'(x) = lim(h→0) [√(x + h) - √x]/h
= lim(h→0) [√(x + h) - √x]/h × [√(x + h) + √x]/[√(x + h) + √x]
= lim(h→0) [(x + h) - x]/[h(√(x + h) + √x)]
= lim(h→0) h/[h(√(x + h) + √x)]
= lim(h→0) 1/[√(x + h) + √x]
= 1/(2√x)

## Geometric Interpretation

### Tangent Line
The derivative f'(a) gives the slope of the tangent line to the curve y = f(x) at point (a, f(a)).

**Equation of tangent line**:
y - f(a) = f'(a)(x - a)

**Example**: Find tangent line to f(x) = x² at x = 3
- f(3) = 9
- f'(x) = 2x, so f'(3) = 6
- Tangent line: y - 9 = 6(x - 3)
- Simplifying: y = 6x - 9

### Secant Line Approximation
As h approaches 0, the secant line approaches the tangent line.

## Physical Interpretation

### Velocity
If s(t) represents position at time t, then:
- v(t) = s'(t) = velocity
- a(t) = v'(t) = s''(t) = acceleration

**Example**: s(t) = t³ - 6t² + 9t (position in meters, time in seconds)
- v(t) = s'(t) = 3t² - 12t + 9
- a(t) = v'(t) = 6t - 12

At t = 2:
- s(2) = 8 - 24 + 18 = 2 meters
- v(2) = 12 - 24 + 9 = -3 m/s (moving backward)
- a(2) = 12 - 12 = 0 m/s² (constant velocity)

### Rate of Change
The derivative represents the instantaneous rate of change.

**Example**: Population P(t) = 1000e^(0.02t)
- P'(t) = 1000(0.02)e^(0.02t) = 20e^(0.02t)
- At t = 10: P'(10) = 20e^0.2 ≈ 24.4 people per year

## Differentiability and Continuity

### Differentiability Implies Continuity
If f is differentiable at a, then f is continuous at a.

**Proof**: lim(x→a) f(x) = f(a) + lim(x→a) [f(x) - f(a)]
= f(a) + lim(x→a) [f(x) - f(a)]/(x - a) × (x - a)
= f(a) + f'(a) × 0 = f(a)

### Continuity Does Not Imply Differentiability
**Example**: f(x) = |x| is continuous at x = 0 but not differentiable.

At x = 0:
- Left derivative: lim(h→0⁻) [|0 + h| - |0|]/h = lim(h→0⁻) |h|/h = -1
- Right derivative: lim(h→0⁺) [|0 + h| - |0|]/h = lim(h→0⁺) |h|/h = 1
- Since left ≠ right derivative, f'(0) doesn't exist

### Points of Non-Differentiability
1. **Corners**: |x| at x = 0
2. **Cusps**: x^(2/3) at x = 0
3. **Vertical tangents**: x^(1/3) at x = 0
4. **Discontinuities**: Any discontinuity

## Applications

### Optimization Preview
Critical points occur where f'(x) = 0 or f'(x) doesn't exist.

**Example**: f(x) = x³ - 3x² + 1
- f'(x) = 3x² - 6x = 3x(x - 2)
- Critical points: x = 0, x = 2

### Marginal Analysis
In economics, marginal cost is the derivative of total cost.

**Example**: C(x) = 1000 + 50x + 0.1x² (total cost)
- C'(x) = 50 + 0.2x (marginal cost)
- At x = 100: C'(100) = 50 + 20 = $70 per unit

## Practice Problems

### Problem 1
Use the limit definition to find f'(x) for f(x) = 3x + 2

**Solution**:
f'(x) = lim(h→0) [3(x + h) + 2 - (3x + 2)]/h
= lim(h→0) [3x + 3h + 2 - 3x - 2]/h
= lim(h→0) 3h/h = 3

### Problem 2
Find the equation of the tangent line to f(x) = x³ at x = 2

**Solution**:
- f(2) = 8
- f'(x) = 3x², so f'(2) = 12
- Tangent line: y - 8 = 12(x - 2)
- Simplifying: y = 12x - 16

### Problem 3
A ball is thrown upward with position s(t) = -16t² + 64t + 80. Find velocity and acceleration.

**Solution**:
- v(t) = s'(t) = -32t + 64
- a(t) = v'(t) = -32
- At t = 1: v(1) = 32 ft/s (upward), a(1) = -32 ft/s² (downward)

### Problem 4
Determine if f(x) = x^(1/3) is differentiable at x = 0

**Solution**:
f'(x) = (1/3)x^(-2/3) = 1/(3x^(2/3))
At x = 0, f'(0) is undefined (vertical tangent)
Not differentiable at x = 0

## Key Takeaways
- Derivative is the limit of difference quotient
- Derivative gives slope of tangent line
- Derivative represents instantaneous rate of change
- Differentiability implies continuity
- Derivatives have many practical applications
- Understanding the definition is crucial for advanced topics

## Next Steps
In the next tutorial, we'll explore differentiation rules, learning efficient methods for calculating derivatives without using the limit definition every time.
