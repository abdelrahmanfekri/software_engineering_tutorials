# Differential Calculus Tutorial 08: Related Rates

## Learning Objectives
By the end of this tutorial, you will be able to:
- Set up related rates problems systematically
- Solve related rates problems using implicit differentiation
- Apply related rates to geometric problems
- Solve related rates problems in physics contexts
- Handle complex multi-variable related rates problems
- Use related rates in optimization contexts

## Introduction to Related Rates

### What are Related Rates?
Related rates problems involve finding the rate of change of one quantity with respect to time when the rate of change of another related quantity is known.

### General Strategy
1. **Identify** the given rate and the rate to find
2. **Find** a relationship between the variables
3. **Differentiate** both sides with respect to time
4. **Substitute** known values and solve for the unknown rate

### Key Steps
1. Draw a diagram if helpful
2. Assign variables to quantities
3. Write down what's given and what's needed
4. Find an equation relating the variables
5. Differentiate with respect to time
6. Substitute and solve

## Basic Related Rates Problems

### Example 1: Expanding Circle
A circle's radius is increasing at 3 cm/s. How fast is the area increasing when the radius is 5 cm?

**Solution**:
- Given: dr/dt = 3 cm/s, r = 5 cm
- Find: dA/dt
- Relationship: A = πr²
- Differentiate: dA/dt = 2πr(dr/dt)
- Substitute: dA/dt = 2π(5)(3) = 30π cm²/s

### Example 2: Melting Ice Cube
An ice cube is melting. Each edge decreases at 2 cm/min. How fast is the volume decreasing when each edge is 3 cm?

**Solution**:
- Given: ds/dt = -2 cm/min, s = 3 cm
- Find: dV/dt
- Relationship: V = s³
- Differentiate: dV/dt = 3s²(ds/dt)
- Substitute: dV/dt = 3(3)²(-2) = 3(9)(-2) = -54 cm³/min

## Geometric Related Rates

### Example 3: Ladder Problem
A ladder 13 ft long leans against a wall. The bottom is pulled away at 2 ft/s. How fast is the top sliding down when the bottom is 5 ft from the wall?

**Solution**:
- Given: dx/dt = 2 ft/s, x = 5 ft
- Find: dy/dt
- Relationship: x² + y² = 13² = 169
- Differentiate: 2x(dx/dt) + 2y(dy/dt) = 0
- When x = 5: y² = 169 - 25 = 144, so y = 12
- Substitute: 2(5)(2) + 2(12)(dy/dt) = 0
- 20 + 24(dy/dt) = 0
- dy/dt = -20/24 = -5/6 ft/s (negative means sliding down)

### Example 4: Expanding Rectangle
A rectangle's length increases at 3 cm/s and width decreases at 2 cm/s. How fast is the area changing when length is 8 cm and width is 6 cm?

**Solution**:
- Given: dl/dt = 3 cm/s, dw/dt = -2 cm/s, l = 8 cm, w = 6 cm
- Find: dA/dt
- Relationship: A = lw
- Differentiate: dA/dt = w(dl/dt) + l(dw/dt)
- Substitute: dA/dt = 6(3) + 8(-2) = 18 - 16 = 2 cm²/s

## Volume and Surface Area Problems

### Example 5: Conical Tank
A conical tank has radius 3 ft and height 10 ft. Water flows in at 2 ft³/min. How fast is the water level rising when the water is 6 ft deep?

**Solution**:
- Given: dV/dt = 2 ft³/min, h = 6 ft
- Find: dh/dt
- Relationship: V = (1/3)πr²h
- Similar triangles: r/h = 3/10, so r = 3h/10
- V = (1/3)π(3h/10)²h = (1/3)π(9h²/100)h = 3πh³/100
- Differentiate: dV/dt = 9πh²(dh/dt)/100
- Substitute: 2 = 9π(6)²(dh/dt)/100 = 324π(dh/dt)/100
- dh/dt = 200/(324π) = 50/(81π) ft/min

### Example 6: Spherical Balloon
A spherical balloon is being inflated. The radius increases at 4 cm/s. How fast is the surface area increasing when the radius is 6 cm?

**Solution**:
- Given: dr/dt = 4 cm/s, r = 6 cm
- Find: dS/dt
- Relationship: S = 4πr²
- Differentiate: dS/dt = 8πr(dr/dt)
- Substitute: dS/dt = 8π(6)(4) = 192π cm²/s

## Physics Applications

### Example 7: Projectile Motion
A projectile is launched with initial velocity 100 m/s at angle 30°. How fast is the horizontal distance changing when the projectile is at its maximum height?

**Solution**:
- Given: v₀ = 100 m/s, θ = 30°
- Find: dx/dt at maximum height
- At maximum height: vy = 0
- vy = v₀ sin(θ) - gt = 0
- t = v₀ sin(θ)/g = 100 sin(30°)/9.8 = 50/9.8 ≈ 5.1 s
- x = v₀ cos(θ)t = 100 cos(30°)(5.1) ≈ 100(0.866)(5.1) ≈ 442 m
- dx/dt = v₀ cos(θ) = 100 cos(30°) = 100(0.866) = 86.6 m/s

### Example 8: Shadow Problem
A man 6 ft tall walks away from a streetlight 15 ft tall at 5 ft/s. How fast is his shadow lengthening?

**Solution**:
- Given: dh/dt = 5 ft/s (distance from light)
- Find: ds/dt (shadow length)
- Similar triangles: 6/s = 15/(s + h)
- Cross multiply: 6(s + h) = 15s
- 6s + 6h = 15s
- 6h = 9s, so h = 1.5s
- Differentiate: dh/dt = 1.5(ds/dt)
- Substitute: 5 = 1.5(ds/dt)
- ds/dt = 5/1.5 = 10/3 ft/s

## Complex Related Rates

### Example 9: Multiple Variables
A right triangle has legs of length x and y. If x increases at 2 cm/s and y decreases at 3 cm/s, how fast is the hypotenuse changing when x = 3 cm and y = 4 cm?

**Solution**:
- Given: dx/dt = 2 cm/s, dy/dt = -3 cm/s, x = 3 cm, y = 4 cm
- Find: dz/dt
- Relationship: z² = x² + y²
- Differentiate: 2z(dz/dt) = 2x(dx/dt) + 2y(dy/dt)
- When x = 3, y = 4: z² = 9 + 16 = 25, so z = 5
- Substitute: 2(5)(dz/dt) = 2(3)(2) + 2(4)(-3)
- 10(dz/dt) = 12 - 24 = -12
- dz/dt = -12/10 = -1.2 cm/s

### Example 10: Optimization with Related Rates
A rectangle is inscribed in a semicircle of radius 5. If the width increases at 1 cm/s, how fast is the area changing when the width is 6 cm?

**Solution**:
- Given: dw/dt = 1 cm/s, w = 6 cm
- Find: dA/dt
- Let width = 2x, height = y
- Constraint: x² + y² = 25, so y = √(25 - x²)
- Area: A = 2xy = 2x√(25 - x²)
- Differentiate: dA/dt = 2√(25 - x²)(dx/dt) + 2x(-x/√(25 - x²))(dx/dt)
- dA/dt = 2(√(25 - x²) - x²/√(25 - x²))(dx/dt)
- When w = 6: x = 3, y = √(25 - 9) = √16 = 4
- dA/dt = 2(4 - 9/4)(1) = 2(16/4 - 9/4) = 2(7/4) = 7/2 cm²/s

## Practice Problems

### Problem 1
A snowball melts so that its surface area decreases at 1 cm²/min. How fast is the radius decreasing when the radius is 4 cm?

**Solution**:
- Given: dS/dt = -1 cm²/min, r = 4 cm
- Find: dr/dt
- Relationship: S = 4πr²
- Differentiate: dS/dt = 8πr(dr/dt)
- Substitute: -1 = 8π(4)(dr/dt)
- dr/dt = -1/(32π) cm/min

### Problem 2
A kite 100 ft above the ground moves horizontally at 10 ft/s. How fast is the string being let out when the kite is 200 ft away horizontally?

**Solution**:
- Given: dx/dt = 10 ft/s, x = 200 ft
- Find: ds/dt
- Relationship: s² = x² + 100² = x² + 10,000
- Differentiate: 2s(ds/dt) = 2x(dx/dt)
- When x = 200: s² = 40,000 + 10,000 = 50,000, so s = 100√5
- ds/dt = x(dx/dt)/s = 200(10)/(100√5) = 2000/(100√5) = 20/√5 = 4√5 ft/s

### Problem 3
A particle moves along the curve y = x². The x-coordinate increases at 2 units/s. How fast is the distance from the origin changing when x = 3?

**Solution**:
- Given: dx/dt = 2 units/s, x = 3
- Find: dD/dt
- When x = 3: y = 9
- Distance: D² = x² + y² = x² + x⁴
- Differentiate: 2D(dD/dt) = 2x(dx/dt) + 4x³(dx/dt)
- dD/dt = (x + 2x³)(dx/dt)/D
- When x = 3: D² = 9 + 81 = 90, so D = 3√10
- dD/dt = (3 + 2(27))(2)/(3√10) = (3 + 54)(2)/(3√10) = 114/(3√10) = 38/√10 units/s

### Problem 4
A conical pile of sand has height equal to diameter. Sand is added at 10 ft³/min. How fast is the height increasing when the height is 6 ft?

**Solution**:
- Given: dV/dt = 10 ft³/min, h = 6 ft
- Find: dh/dt
- Since h = d = 2r: r = h/2
- V = (1/3)πr²h = (1/3)π(h/2)²h = πh³/12
- Differentiate: dV/dt = πh²(dh/dt)/4
- Substitute: 10 = π(6)²(dh/dt)/4 = 36π(dh/dt)/4 = 9π(dh/dt)
- dh/dt = 10/(9π) ft/min

## Key Takeaways
- Always identify given and unknown rates clearly
- Draw diagrams for geometric problems
- Find relationships between variables before differentiating
- Differentiate with respect to time
- Substitute known values at the end
- Check that units make sense
- Practice with various problem types builds confidence
- Related rates connect calculus to real-world applications

## Next Steps
This completes our comprehensive coverage of differential calculus fundamentals. You now have the tools to understand rates of change, find derivatives using various methods, apply derivatives to optimization and curve analysis, and solve complex related rates problems. These concepts form the foundation for advanced calculus topics and have wide applications in science, engineering, and other fields.
