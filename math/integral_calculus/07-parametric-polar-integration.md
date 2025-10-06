# Parametric and Polar Integration

## Overview
This tutorial covers integration techniques for parametric curves and polar coordinates. These coordinate systems provide alternative ways to describe curves and regions, often simplifying integration problems that would be difficult in Cartesian coordinates.

## Learning Objectives
- Understand parametric curves and their properties
- Learn to integrate parametric functions
- Work with polar coordinates and curves
- Calculate areas and arc lengths in polar coordinates
- Apply parametric and polar integration to solve problems

## 1. Parametric Curves

### Definition
A parametric curve is defined by:
```
x = f(t), y = g(t), α ≤ t ≤ β
```

Where t is the parameter and f(t), g(t) are functions of t.

### Examples

#### Example 1: Circle
```
x = r cos(t), y = r sin(t), 0 ≤ t ≤ 2π
```

#### Example 2: Ellipse
```
x = a cos(t), y = b sin(t), 0 ≤ t ≤ 2π
```

#### Example 3: Cycloid
```
x = t - sin(t), y = 1 - cos(t), 0 ≤ t ≤ 2π
```

## 2. Derivatives of Parametric Curves

### First Derivative
```
dy/dx = (dy/dt)/(dx/dt) = g'(t)/f'(t)
```

### Second Derivative
```
d²y/dx² = d/dx[dy/dx] = d/dt[dy/dx] / (dx/dt)
```

### Examples

#### Example 1: Circle
```
x = r cos(t), y = r sin(t)
dx/dt = -r sin(t), dy/dt = r cos(t)
dy/dx = (r cos(t))/(-r sin(t)) = -cot(t)
```

#### Example 2: Parabola
```
x = t², y = t³
dx/dt = 2t, dy/dt = 3t²
dy/dx = (3t²)/(2t) = 3t/2
```

## 3. Arc Length of Parametric Curves

### Formula
The length of a parametric curve from t = α to t = β is:

```
L = ∫[α to β] √([f'(t)]² + [g'(t)]²) dt
```

### Examples

#### Example 1: Circle
```
x = r cos(t), y = r sin(t), 0 ≤ t ≤ 2π
dx/dt = -r sin(t), dy/dt = r cos(t)
```

```
L = ∫[0 to 2π] √([-r sin(t)]² + [r cos(t)]²) dt
= ∫[0 to 2π] √(r² sin²(t) + r² cos²(t)) dt
= ∫[0 to 2π] r√(sin²(t) + cos²(t)) dt
= ∫[0 to 2π] r dt = 2πr
```

#### Example 2: Cycloid
```
x = t - sin(t), y = 1 - cos(t), 0 ≤ t ≤ 2π
dx/dt = 1 - cos(t), dy/dt = sin(t)
```

```
L = ∫[0 to 2π] √([1 - cos(t)]² + [sin(t)]²) dt
= ∫[0 to 2π] √(1 - 2cos(t) + cos²(t) + sin²(t)) dt
= ∫[0 to 2π] √(2 - 2cos(t)) dt
= ∫[0 to 2π] √(4 sin²(t/2)) dt
= ∫[0 to 2π] 2|sin(t/2)| dt = 8
```

## 4. Area Under Parametric Curves

### Formula
The area under a parametric curve from t = α to t = β is:

```
A = ∫[α to β] y(t) · x'(t) dt
```

### Examples

#### Example 1: Ellipse
```
x = a cos(t), y = b sin(t), 0 ≤ t ≤ 2π
```

Area of ellipse:
```
A = ∫[0 to 2π] b sin(t) · (-a sin(t)) dt
= -ab ∫[0 to 2π] sin²(t) dt
= -ab ∫[0 to 2π] (1 - cos(2t))/2 dt
= -ab/2 [t - sin(2t)/2][0 to 2π] = -ab/2 · 2π = -πab
```

Since area is positive: A = πab

## 5. Polar Coordinates

### Definition
In polar coordinates, a point is described by (r, θ) where:
- r is the distance from the origin
- θ is the angle from the positive x-axis

### Conversion Formulas
```
x = r cos(θ), y = r sin(θ)
r = √(x² + y²), θ = arctan(y/x)
```

### Examples

#### Example 1: Circle
```
r = a (constant radius)
```

#### Example 2: Cardioid
```
r = 1 + cos(θ)
```

#### Example 3: Rose Curve
```
r = a sin(nθ) or r = a cos(nθ)
```

## 6. Polar Curves and Their Properties

### Common Polar Curves

#### Circle
```
r = a
```

#### Cardioid
```
r = a(1 + cos(θ)) or r = a(1 + sin(θ))
```

#### Limaçon
```
r = a + b cos(θ) or r = a + b sin(θ)
```

#### Rose Curves
```
r = a sin(nθ) or r = a cos(nθ)
```

#### Spiral
```
r = aθ
```

### Examples

#### Example 1: Cardioid
```
r = 1 + cos(θ)
```

This curve has a cusp at θ = π and is symmetric about the x-axis.

#### Example 2: Rose Curve
```
r = 2 sin(3θ)
```

This is a 3-petaled rose curve.

## 7. Area in Polar Coordinates

### Formula
The area bounded by the polar curve r = f(θ) from θ = α to θ = β is:

```
A = (1/2) ∫[α to β] [f(θ)]² dθ
```

### Examples

#### Example 1: Circle
```
r = a, 0 ≤ θ ≤ 2π
```

```
A = (1/2) ∫[0 to 2π] a² dθ = (1/2) a² [θ][0 to 2π] = πa²
```

#### Example 2: Cardioid
```
r = 1 + cos(θ), 0 ≤ θ ≤ 2π
```

```
A = (1/2) ∫[0 to 2π] (1 + cos(θ))² dθ
= (1/2) ∫[0 to 2π] (1 + 2cos(θ) + cos²(θ)) dθ
= (1/2) ∫[0 to 2π] (1 + 2cos(θ) + (1 + cos(2θ))/2) dθ
= (1/2) ∫[0 to 2π] (3/2 + 2cos(θ) + cos(2θ)/2) dθ
= (1/2) [3θ/2 + 2sin(θ) + sin(2θ)/4][0 to 2π] = 3π/2
```

#### Example 3: Area Between Two Curves
Find the area inside r = 2 + cos(θ) and outside r = 2.

Points of intersection: 2 + cos(θ) = 2 → cos(θ) = 0 → θ = ±π/2

```
A = (1/2) ∫[-π/2 to π/2] [(2 + cos(θ))² - 2²] dθ
= (1/2) ∫[-π/2 to π/2] [4 + 4cos(θ) + cos²(θ) - 4] dθ
= (1/2) ∫[-π/2 to π/2] [4cos(θ) + cos²(θ)] dθ
= (1/2) ∫[-π/2 to π/2] [4cos(θ) + (1 + cos(2θ))/2] dθ
= (1/2) [4sin(θ) + θ/2 + sin(2θ)/4][-π/2 to π/2] = π/2
```

## 8. Arc Length in Polar Coordinates

### Formula
The length of a polar curve r = f(θ) from θ = α to θ = β is:

```
L = ∫[α to β] √([f(θ)]² + [f'(θ)]²) dθ
```

### Examples

#### Example 1: Circle
```
r = a, 0 ≤ θ ≤ 2π
```

```
L = ∫[0 to 2π] √(a² + 0²) dθ = ∫[0 to 2π] a dθ = 2πa
```

#### Example 2: Cardioid
```
r = 1 + cos(θ), 0 ≤ θ ≤ 2π
r' = -sin(θ)
```

```
L = ∫[0 to 2π] √((1 + cos(θ))² + (-sin(θ))²) dθ
= ∫[0 to 2π] √(1 + 2cos(θ) + cos²(θ) + sin²(θ)) dθ
= ∫[0 to 2π] √(2 + 2cos(θ)) dθ
= ∫[0 to 2π] √(4 cos²(θ/2)) dθ
= ∫[0 to 2π] 2|cos(θ/2)| dθ = 8
```

## 9. Practice Problems

### Parametric Curves
1. Find dy/dx for x = t², y = t³
2. Find the arc length of x = cos(t), y = sin(t) from t = 0 to t = π
3. Find the area under x = t², y = t³ from t = 0 to t = 1
4. Find d²y/dx² for x = e^t, y = e^(-t)

### Polar Coordinates
1. Convert (3, π/4) to Cartesian coordinates
2. Convert (4, -3) to polar coordinates
3. Find the area inside r = 2 sin(θ)
4. Find the area between r = 1 and r = 2 cos(θ)

### Arc Lengths
1. Find the arc length of r = θ from θ = 0 to θ = 2π
2. Find the arc length of r = 1 + cos(θ) from θ = 0 to θ = π
3. Find the arc length of x = t - sin(t), y = 1 - cos(t) from t = 0 to t = π
4. Find the arc length of r = e^θ from θ = 0 to θ = 1

### Applications
1. Find the area of the region bounded by r = 2 + cos(θ)
2. Find the area inside r = 3 sin(θ) and outside r = 1 + sin(θ)
3. Find the arc length of the spiral r = θ from θ = 0 to θ = 4π
4. Find the area of the region bounded by the cardioid r = 1 + cos(θ)

## 10. Common Mistakes to Avoid

1. **Parametric Derivatives**: Remember to use the chain rule for dy/dx
2. **Polar Area**: Don't forget the factor of 1/2 in the area formula
3. **Arc Length**: Use the correct formula for parametric vs. polar curves
4. **Limits**: Pay attention to the parameter ranges
5. **Symmetry**: Use symmetry to simplify calculations when possible

## 11. Advanced Topics

### Surface Area of Revolution
For parametric curves rotated around an axis:
```
S = 2π ∫[α to β] y(t)√([x'(t)]² + [y'(t)]²) dt
```

### Polar Surface Area
For polar curves rotated around an axis:
```
S = 2π ∫[α to β] r sin(θ)√(r² + (dr/dθ)²) dθ
```

### Applications in Physics
- Planetary motion (elliptical orbits)
- Wave motion (cycloids)
- Engineering design (gear teeth)

## 12. Study Tips

1. **Visualize Curves**: Draw parametric and polar curves to understand their shapes
2. **Practice Conversions**: Master the conversion between coordinate systems
3. **Use Symmetry**: Look for symmetry to simplify calculations
4. **Check Limits**: Verify parameter ranges and limits of integration
5. **Understand Applications**: See how these concepts apply to real problems

## Next Steps

After mastering parametric and polar integration, proceed to:
- Differential equations
- Advanced calculus topics
- Vector calculus
- Complex analysis

Remember: Parametric and polar coordinates provide powerful tools for describing curves and regions that would be difficult to handle in Cartesian coordinates. They are essential for many applications in physics, engineering, and mathematics.
