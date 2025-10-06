# Differential Calculus Tutorial 04: Implicit Differentiation

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the concept of implicit functions
- Apply implicit differentiation technique
- Find derivatives of implicitly defined functions
- Solve related rates problems
- Apply implicit differentiation to geometric problems
- Use implicit differentiation in optimization problems

## Introduction to Implicit Differentiation

### Explicit vs. Implicit Functions

**Explicit Function**: y = f(x) (y is expressed directly in terms of x)
- Examples: y = x² + 3x, y = sin(x), y = e^x

**Implicit Function**: F(x, y) = 0 (relationship between x and y is given implicitly)
- Examples: x² + y² = 25, xy + y² = 1, x³ + y³ = 6xy

### Why Use Implicit Differentiation?

Sometimes it's difficult or impossible to solve for y explicitly, but we still need to find dy/dx. Implicit differentiation allows us to find the derivative without solving for y first.

## Basic Implicit Differentiation

### The Process
1. Differentiate both sides of the equation with respect to x
2. Treat y as a function of x (use chain rule when differentiating y)
3. Solve for dy/dx

### Examples

**Example 1**: Find dy/dx for x² + y² = 25

**Solution**:
- Differentiate both sides: d/dx[x² + y²] = d/dx[25]
- 2x + 2y(dy/dx) = 0
- Solve for dy/dx: 2y(dy/dx) = -2x
- dy/dx = -2x/(2y) = -x/y

**Example 2**: Find dy/dx for xy + y² = 1

**Solution**:
- Differentiate both sides: d/dx[xy + y²] = d/dx[1]
- Use product rule for xy: y + x(dy/dx) + 2y(dy/dx) = 0
- Factor out dy/dx: y + dy/dx(x + 2y) = 0
- Solve for dy/dx: dy/dx = -y/(x + 2y)

**Example 3**: Find dy/dx for x³ + y³ = 6xy

**Solution**:
- Differentiate both sides: d/dx[x³ + y³] = d/dx[6xy]
- 3x² + 3y²(dy/dx) = 6y + 6x(dy/dx)
- Collect terms with dy/dx: 3y²(dy/dx) - 6x(dy/dx) = 6y - 3x²
- Factor: dy/dx(3y² - 6x) = 6y - 3x²
- dy/dx = (6y - 3x²)/(3y² - 6x) = (2y - x²)/(y² - 2x)

## Finding Tangent Lines

### Example: Find the tangent line to x² + y² = 25 at (3, 4)

**Solution**:
- From Example 1: dy/dx = -x/y
- At (3, 4): dy/dx = -3/4
- Tangent line: y - 4 = (-3/4)(x - 3)
- Simplifying: y = (-3/4)x + 9/4 + 4 = (-3/4)x + 25/4

### Example: Find the tangent line to xy + y² = 1 at (1, 1/2)

**Solution**:
- From Example 2: dy/dx = -y/(x + 2y)
- At (1, 1/2): dy/dx = -(1/2)/(1 + 2(1/2)) = -(1/2)/2 = -1/4
- Tangent line: y - 1/2 = (-1/4)(x - 1)
- Simplifying: y = (-1/4)x + 1/4 + 1/2 = (-1/4)x + 3/4

## Higher-Order Derivatives

### Finding Second Derivatives Implicitly

**Example**: Find d²y/dx² for x² + y² = 25

**Solution**:
- First derivative: dy/dx = -x/y
- Differentiate again: d/dx[dy/dx] = d/dx[-x/y]
- d²y/dx² = d/dx[-x/y] = -[y - x(dy/dx)]/y²
- Substitute dy/dx = -x/y: d²y/dx² = -[y - x(-x/y)]/y²
- = -[y + x²/y]/y² = -[y² + x²]/y³
- Since x² + y² = 25: d²y/dx² = -25/y³

## Related Rates Problems

### Strategy for Related Rates
1. Identify the given rate and the rate to find
2. Find a relationship between the variables
3. Differentiate with respect to time
4. Substitute known values and solve

### Examples

**Example 1**: A balloon is being inflated. The radius is increasing at 2 cm/s. How fast is the volume increasing when the radius is 5 cm?

**Solution**:
- Given: dr/dt = 2 cm/s, r = 5 cm
- Find: dV/dt
- Relationship: V = (4/3)πr³
- Differentiate: dV/dt = 4πr²(dr/dt)
- Substitute: dV/dt = 4π(5)²(2) = 4π(25)(2) = 200π cm³/s

**Example 2**: A ladder 13 ft long leans against a wall. The bottom is pulled away at 2 ft/s. How fast is the top sliding down when the bottom is 5 ft from the wall?

**Solution**:
- Given: dx/dt = 2 ft/s, x = 5 ft
- Find: dy/dt
- Relationship: x² + y² = 13² = 169
- Differentiate: 2x(dx/dt) + 2y(dy/dt) = 0
- When x = 5: y² = 169 - 25 = 144, so y = 12
- Substitute: 2(5)(2) + 2(12)(dy/dt) = 0
- 20 + 24(dy/dt) = 0
- dy/dt = -20/24 = -5/6 ft/s (negative means sliding down)

**Example 3**: A conical tank has radius 3 ft and height 10 ft. Water is flowing in at 2 ft³/min. How fast is the water level rising when the water is 6 ft deep?

**Solution**:
- Given: dV/dt = 2 ft³/min, h = 6 ft
- Find: dh/dt
- Relationship: V = (1/3)πr²h
- Similar triangles: r/h = 3/10, so r = 3h/10
- V = (1/3)π(3h/10)²h = (1/3)π(9h²/100)h = 3πh³/100
- Differentiate: dV/dt = 9πh²(dh/dt)/100
- Substitute: 2 = 9π(6)²(dh/dt)/100 = 324π(dh/dt)/100
- dh/dt = 200/(324π) = 50/(81π) ft/min

## Applications to Optimization

### Example: Find the point on the circle x² + y² = 25 closest to (1, 1)

**Solution**:
- Distance squared: D² = (x - 1)² + (y - 1)²
- Constraint: x² + y² = 25
- Using Lagrange multipliers or substitution:
- From constraint: y² = 25 - x²
- D² = (x - 1)² + (√(25 - x²) - 1)²
- Differentiate and set equal to 0:
- dD²/dx = 2(x - 1) + 2(√(25 - x²) - 1)(-x/√(25 - x²)) = 0
- Solving gives x = 5/√2, y = 5/√2
- Closest point: (5/√2, 5/√2)

## Practice Problems

### Problem 1
Find dy/dx for x²y + xy² = 6

**Solution**:
- Differentiate: 2xy + x²(dy/dx) + y² + 2xy(dy/dx) = 0
- Factor: dy/dx(x² + 2xy) = -2xy - y²
- dy/dx = -(2xy + y²)/(x² + 2xy) = -y(2x + y)/[x(x + 2y)]

### Problem 2
Find the tangent line to x² + 2xy + y² = 4 at (1, 1)

**Solution**:
- Differentiate: 2x + 2y + 2x(dy/dx) + 2y(dy/dx) = 0
- Factor: dy/dx(2x + 2y) = -2x - 2y
- dy/dx = -1
- Tangent line: y - 1 = -1(x - 1), so y = -x + 2

### Problem 3
A spherical balloon is being deflated. The radius decreases at 3 cm/s. How fast is the surface area decreasing when the radius is 8 cm?

**Solution**:
- Given: dr/dt = -3 cm/s, r = 8 cm
- Find: dS/dt
- Relationship: S = 4πr²
- Differentiate: dS/dt = 8πr(dr/dt)
- Substitute: dS/dt = 8π(8)(-3) = -192π cm²/s

### Problem 4
Find dy/dx for sin(xy) = x + y

**Solution**:
- Differentiate: cos(xy)[y + x(dy/dx)] = 1 + dy/dx
- Expand: y cos(xy) + x cos(xy)(dy/dx) = 1 + dy/dx
- Collect terms: dy/dx[x cos(xy) - 1] = 1 - y cos(xy)
- dy/dx = (1 - y cos(xy))/[x cos(xy) - 1]

## Key Takeaways
- Implicit differentiation finds derivatives without solving for y
- Always use chain rule when differentiating y
- Related rates problems connect derivatives to real-world situations
- Implicit differentiation is essential for many optimization problems
- Practice with various equation types builds confidence
- Always check your work by verifying the result makes sense

## Next Steps
In the next tutorial, we'll explore higher-order derivatives, learning about second derivatives, concavity, and inflection points.
