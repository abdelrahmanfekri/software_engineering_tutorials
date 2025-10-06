# Applications of Integration

## Overview
This tutorial covers the practical applications of integration, including calculating areas, volumes, arc lengths, and solving work and force problems. These applications demonstrate the power of integration in solving real-world problems across various fields.

## Learning Objectives
- Calculate areas between curves
- Find volumes of revolution using disk and shell methods
- Compute arc lengths of curves
- Calculate surface areas of revolution
- Solve work and force problems
- Apply integration to physics and engineering problems

## 1. Area Between Curves

### Basic Concept
The area between two curves y = f(x) and y = g(x) from x = a to x = b is:

```
A = ∫[a to b] |f(x) - g(x)| dx
```

### Method
1. Find points of intersection
2. Determine which function is on top
3. Set up the integral
4. Evaluate

### Examples

#### Example 1: Simple Area Between Curves
Find the area between y = x² and y = x from x = 0 to x = 1.

Points of intersection: x² = x → x = 0, 1
For 0 ≤ x ≤ 1: x ≥ x² (since x - x² = x(1-x) ≥ 0)

```
A = ∫[0 to 1] (x - x²) dx = [x²/2 - x³/3][0 to 1] = 1/2 - 1/3 = 1/6
```

#### Example 2: Area Between Intersecting Curves
Find the area between y = x² and y = 2x - x².

Points of intersection: x² = 2x - x² → 2x² - 2x = 0 → x = 0, 1
For 0 ≤ x ≤ 1: 2x - x² ≥ x²

```
A = ∫[0 to 1] (2x - x² - x²) dx = ∫[0 to 1] (2x - 2x²) dx
= [x² - 2x³/3][0 to 1] = 1 - 2/3 = 1/3
```

### Area Between Curves with Respect to y
Sometimes it's easier to integrate with respect to y:

```
A = ∫[c to d] |f(y) - g(y)| dy
```

## 2. Volume of Revolution

### Disk Method
When rotating a region around a horizontal or vertical axis:

```
V = π ∫[a to b] [R(x)]² dx  (rotation around x-axis)
V = π ∫[c to d] [R(y)]² dy  (rotation around y-axis)
```

Where R(x) or R(y) is the radius of the disk.

### Examples

#### Example 1: Disk Method - Rotation around x-axis
Find the volume when the region bounded by y = x², y = 0, and x = 1 is rotated around the x-axis.

```
V = π ∫[0 to 1] (x²)² dx = π ∫[0 to 1] x⁴ dx = π[x⁵/5][0 to 1] = π/5
```

#### Example 2: Disk Method - Rotation around y-axis
Find the volume when the region bounded by y = x², x = 0, and y = 1 is rotated around the y-axis.

Since y = x², we have x = √y
```
V = π ∫[0 to 1] (√y)² dy = π ∫[0 to 1] y dy = π[y²/2][0 to 1] = π/2
```

### Washer Method
When there's a hole in the middle (region between two curves):

```
V = π ∫[a to b] ([R(x)]² - [r(x)]²) dx
```

Where R(x) is the outer radius and r(x) is the inner radius.

#### Example: Washer Method
Find the volume when the region bounded by y = x² and y = x is rotated around the x-axis.

```
V = π ∫[0 to 1] (x² - (x²)²) dx = π ∫[0 to 1] (x² - x⁴) dx
= π[x³/3 - x⁵/5][0 to 1] = π(1/3 - 1/5) = 2π/15
```

### Shell Method
Alternative method for volumes of revolution:

```
V = 2π ∫[a to b] x · f(x) dx  (rotation around y-axis)
V = 2π ∫[c to d] y · f(y) dy  (rotation around x-axis)
```

#### Example: Shell Method
Find the volume when the region bounded by y = x², y = 0, and x = 1 is rotated around the y-axis.

```
V = 2π ∫[0 to 1] x · x² dx = 2π ∫[0 to 1] x³ dx = 2π[x⁴/4][0 to 1] = π/2
```

## 3. Arc Length

### Formula
The length of a curve y = f(x) from x = a to x = b is:

```
L = ∫[a to b] √(1 + [f'(x)]²) dx
```

### Examples

#### Example 1: Arc Length of a Parabola
Find the length of y = x² from x = 0 to x = 1.

f'(x) = 2x
```
L = ∫[0 to 1] √(1 + (2x)²) dx = ∫[0 to 1] √(1 + 4x²) dx
```

Let u = 2x, then du = 2dx, dx = du/2
When x = 0, u = 0; when x = 1, u = 2
```
L = ∫[0 to 2] √(1 + u²) (du/2) = (1/2)∫[0 to 2] √(1 + u²) du
```

Using the formula ∫√(1 + u²) du = (1/2)[u√(1 + u²) + ln|u + √(1 + u²)|] + C:
```
L = (1/4)[u√(1 + u²) + ln|u + √(1 + u²)|][0 to 2]
= (1/4)[2√5 + ln(2 + √5)]
```

### Parametric Arc Length
For parametric curves x = f(t), y = g(t):

```
L = ∫[α to β] √([f'(t)]² + [g'(t)]²) dt
```

## 4. Surface Area of Revolution

### Formula
The surface area when rotating y = f(x) around the x-axis is:

```
S = 2π ∫[a to b] f(x)√(1 + [f'(x)]²) dx
```

### Examples

#### Example: Surface Area of a Sphere
Find the surface area of a sphere of radius R.

The upper semicircle is y = √(R² - x²) from x = -R to x = R
y' = -x/√(R² - x²)

```
S = 2π ∫[-R to R] √(R² - x²)√(1 + x²/(R² - x²)) dx
= 2π ∫[-R to R] √(R² - x²)√(R²/(R² - x²)) dx
= 2π ∫[-R to R] √(R² - x²) · R/√(R² - x²) dx
= 2π ∫[-R to R] R dx = 2πR · 2R = 4πR²
```

## 5. Work and Force Problems

### Work Done by a Variable Force
If a force F(x) acts along the x-axis from x = a to x = b:

```
W = ∫[a to b] F(x) dx
```

### Examples

#### Example 1: Spring Problem
A spring has natural length 10 cm. A force of 40 N is required to stretch it to 15 cm. How much work is done stretching it from 15 cm to 20 cm?

First, find the spring constant k:
F = kx → 40 = k(0.05) → k = 800 N/m

Work to stretch from 15 cm to 20 cm:
```
W = ∫[0.05 to 0.10] 800x dx = 800[x²/2][0.05 to 0.10]
= 400[(0.10)² - (0.05)²] = 400[0.01 - 0.0025] = 3 J
```

#### Example 2: Pumping Water
A cylindrical tank of radius 3 m and height 10 m is full of water. How much work is required to pump all the water to the top?

Consider a thin horizontal slice at height y with thickness dy.
Volume of slice: π(3)² dy = 9π dy
Weight of slice: 9π dy · 1000 · 9.8 = 88200π dy N
Distance to pump: (10 - y) m

```
W = ∫[0 to 10] 88200π(10 - y) dy = 88200π ∫[0 to 10] (10 - y) dy
= 88200π[10y - y²/2][0 to 10] = 88200π[100 - 50] = 4,410,000π J
```

## 6. Center of Mass

### Center of Mass of a Region
For a region bounded by y = f(x) and y = g(x):

```
x̄ = (1/A) ∫[a to b] x[f(x) - g(x)] dx
ȳ = (1/A) ∫[a to b] (1/2)[f(x) + g(x)][f(x) - g(x)] dx
```

Where A is the area of the region.

### Example: Center of Mass
Find the center of mass of the region bounded by y = x² and y = x.

Area: A = ∫[0 to 1] (x - x²) dx = 1/6

```
x̄ = (1/(1/6)) ∫[0 to 1] x(x - x²) dx = 6 ∫[0 to 1] (x² - x³) dx
= 6[x³/3 - x⁴/4][0 to 1] = 6(1/3 - 1/4) = 1/2
```

```
ȳ = (1/(1/6)) ∫[0 to 1] (1/2)(x + x²)(x - x²) dx = 3 ∫[0 to 1] (x² - x⁴) dx
= 3[x³/3 - x⁵/5][0 to 1] = 3(1/3 - 1/5) = 2/5
```

## 7. Practice Problems

### Area Between Curves
1. Find the area between y = x³ and y = x from x = -1 to x = 1
2. Find the area between y = sin(x) and y = cos(x) from x = 0 to x = π/2
3. Find the area between y = x² and y = 2x - x²
4. Find the area between y = x² and y = x³

### Volume of Revolution
1. Find the volume when y = x² from x = 0 to x = 2 is rotated around the x-axis
2. Find the volume when y = √x from x = 0 to x = 4 is rotated around the y-axis
3. Find the volume when the region between y = x² and y = x is rotated around the x-axis
4. Find the volume when y = x² from x = 0 to x = 1 is rotated around the line y = 1

### Arc Length
1. Find the length of y = x^(3/2) from x = 0 to x = 4
2. Find the length of y = ln(cos(x)) from x = 0 to x = π/4
3. Find the length of the parametric curve x = t², y = t³ from t = 0 to t = 1

### Work Problems
1. A spring requires 20 N to stretch 5 cm. How much work is needed to stretch it 10 cm?
2. A tank is shaped like a cone with height 6 m and radius 3 m. How much work is needed to pump water to the top if the tank is full?
3. A chain 10 m long weighs 2 kg/m. How much work is needed to lift the chain to a height of 5 m?

## 8. Common Mistakes to Avoid

1. **Wrong Setup**: Make sure you're using the correct formula for the application
2. **Sign Errors**: Be careful with signs, especially in area calculations
3. **Units**: Pay attention to units in work and force problems
4. **Limits**: Double-check your limits of integration
5. **Geometry**: Visualize the problem to ensure correct setup

## 9. Advanced Applications

### Fluid Pressure
Pressure at depth h: P = ρgh
Force on a vertical surface: F = ∫[a to b] ρgh · width(h) dh

### Electric Fields
Electric field due to a line charge: E = ∫[a to b] (kλ/r²) dr

### Economics
Consumer surplus: CS = ∫[0 to q*] [D(q) - p*] dq
Producer surplus: PS = ∫[0 to q*] [p* - S(q)] dq

## 10. Study Tips

1. **Visualize Problems**: Draw diagrams to understand the geometry
2. **Practice Setup**: Focus on setting up integrals correctly
3. **Check Units**: Always verify units make sense
4. **Use Symmetry**: Look for symmetry to simplify calculations
5. **Understand Applications**: See how these concepts apply to real problems

## Next Steps

After mastering applications of integration, proceed to:
- Sequences and series
- Power series
- Parametric and polar integration
- Differential equations

Remember: Integration applications connect abstract mathematics to real-world problems. Understanding these applications helps develop intuition and problem-solving skills.
