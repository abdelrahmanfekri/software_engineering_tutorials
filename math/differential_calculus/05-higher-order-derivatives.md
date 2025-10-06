# Differential Calculus Tutorial 05: Higher-Order Derivatives

## Learning Objectives
By the end of this tutorial, you will be able to:
- Calculate second and higher-order derivatives
- Understand the geometric meaning of second derivatives
- Determine concavity and inflection points
- Apply second derivative test for extrema
- Interpret acceleration and jerk in physical contexts
- Use higher-order derivatives in Taylor series

## Introduction to Higher-Order Derivatives

### Definition
The second derivative of f(x) is the derivative of the first derivative:
f''(x) = d/dx[f'(x)]

Similarly, the nth derivative is:
f^(n)(x) = d/dx[f^(n-1)(x)]

### Notation
- Second derivative: f''(x), d²y/dx², d²/dx²[f(x)]
- Third derivative: f'''(x), d³y/dx³, d³/dx³[f(x)]
- nth derivative: f^(n)(x), d^n y/dx^n, d^n/dx^n[f(x)]

## Calculating Higher-Order Derivatives

### Examples

**Example 1**: Find all derivatives of f(x) = x⁴ + 3x³ - 2x² + x - 1

**Solution**:
- f'(x) = 4x³ + 9x² - 4x + 1
- f''(x) = 12x² + 18x - 4
- f'''(x) = 24x + 18
- f⁽⁴⁾(x) = 24
- f⁽ⁿ⁾(x) = 0 for n ≥ 5

**Example 2**: Find f''(x) for f(x) = sin(x)

**Solution**:
- f'(x) = cos(x)
- f''(x) = -sin(x)

**Example 3**: Find f''(x) for f(x) = e^(2x)

**Solution**:
- f'(x) = 2e^(2x)
- f''(x) = 4e^(2x)

**Example 4**: Find f''(x) for f(x) = ln(x)

**Solution**:
- f'(x) = 1/x = x^(-1)
- f''(x) = -x^(-2) = -1/x²

## Geometric Interpretation of Second Derivatives

### Concavity

**Definition**: A function f is concave up on an interval if f''(x) > 0 for all x in that interval.
A function f is concave down on an interval if f''(x) < 0 for all x in that interval.

### Visual Interpretation
- **Concave up**: The graph curves upward like a cup (∪)
- **Concave down**: The graph curves downward like a cap (∩)

### Examples

**Example 1**: Determine concavity of f(x) = x³ - 3x² + 2

**Solution**:
- f'(x) = 3x² - 6x
- f''(x) = 6x - 6 = 6(x - 1)
- f''(x) > 0 when x > 1 (concave up)
- f''(x) < 0 when x < 1 (concave down)

**Example 2**: Determine concavity of f(x) = x⁴ - 4x²

**Solution**:
- f'(x) = 4x³ - 8x
- f''(x) = 12x² - 8 = 4(3x² - 2)
- f''(x) > 0 when |x| > √(2/3) (concave up)
- f''(x) < 0 when |x| < √(2/3) (concave down)

## Inflection Points

### Definition
An inflection point is a point where the concavity changes. At an inflection point, f''(x) = 0 or f''(x) is undefined.

### Finding Inflection Points
1. Find f''(x)
2. Set f''(x) = 0 and solve for x
3. Check where f''(x) is undefined
4. Test the sign of f''(x) on both sides of each candidate point

### Examples

**Example 1**: Find inflection points of f(x) = x³ - 3x² + 2

**Solution**:
- f''(x) = 6x - 6 = 6(x - 1)
- f''(x) = 0 when x = 1
- Test: f''(0) = -6 < 0, f''(2) = 6 > 0
- Inflection point at x = 1, point (1, 0)

**Example 2**: Find inflection points of f(x) = x⁴ - 4x²

**Solution**:
- f''(x) = 12x² - 8
- f''(x) = 0 when 12x² - 8 = 0, so x² = 2/3, x = ±√(2/3)
- Test around x = √(2/3): f''(0) = -8 < 0, f''(1) = 4 > 0
- Inflection points at x = ±√(2/3)

## Second Derivative Test

### Statement
If f'(c) = 0 and f''(c) > 0, then f has a local minimum at x = c.
If f'(c) = 0 and f''(c) < 0, then f has a local maximum at x = c.
If f'(c) = 0 and f''(c) = 0, the test is inconclusive.

### Examples

**Example 1**: Use second derivative test for f(x) = x³ - 3x² + 2

**Solution**:
- f'(x) = 3x² - 6x = 3x(x - 2)
- Critical points: x = 0, x = 2
- f''(x) = 6x - 6
- f''(0) = -6 < 0 → local maximum at x = 0
- f''(2) = 6 > 0 → local minimum at x = 2

**Example 2**: Use second derivative test for f(x) = x⁴ - 4x²

**Solution**:
- f'(x) = 4x³ - 8x = 4x(x² - 2) = 4x(x - √2)(x + √2)
- Critical points: x = 0, x = ±√2
- f''(x) = 12x² - 8
- f''(0) = -8 < 0 → local maximum at x = 0
- f''(√2) = 12(2) - 8 = 16 > 0 → local minimum at x = √2
- f''(-√2) = 16 > 0 → local minimum at x = -√2

## Physical Interpretation

### Acceleration
If s(t) represents position, then:
- v(t) = s'(t) = velocity
- a(t) = v'(t) = s''(t) = acceleration

### Jerk
The third derivative represents jerk (rate of change of acceleration):
j(t) = a'(t) = s'''(t)

### Examples

**Example 1**: A particle moves with position s(t) = t³ - 6t² + 9t + 1

**Solution**:
- v(t) = s'(t) = 3t² - 12t + 9 = 3(t² - 4t + 3) = 3(t - 1)(t - 3)
- a(t) = v'(t) = 6t - 12 = 6(t - 2)
- j(t) = a'(t) = 6

At t = 2:
- s(2) = 8 - 24 + 18 + 1 = 3
- v(2) = 12 - 24 + 9 = -3 (moving backward)
- a(2) = 0 (constant velocity)
- j(2) = 6 (acceleration increasing)

**Example 2**: A ball is thrown upward with position s(t) = -16t² + 64t + 80

**Solution**:
- v(t) = s'(t) = -32t + 64
- a(t) = v'(t) = -32 (constant acceleration due to gravity)

## Curve Sketching with Second Derivatives

### Complete Analysis Process
1. Find domain and intercepts
2. Find f'(x) and critical points
3. Find f''(x) and inflection points
4. Determine intervals of increase/decrease
5. Determine intervals of concavity
6. Sketch the curve

### Example: Sketch f(x) = x³ - 3x² + 2

**Solution**:
- Domain: All real numbers
- Intercepts: f(0) = 2, f(x) = 0 when x³ - 3x² + 2 = 0
- f'(x) = 3x² - 6x = 3x(x - 2)
- Critical points: x = 0, x = 2
- f''(x) = 6x - 6 = 6(x - 1)
- Inflection point: x = 1

**Intervals**:
- x < 0: f'(x) > 0 (increasing), f''(x) < 0 (concave down)
- 0 < x < 1: f'(x) < 0 (decreasing), f''(x) < 0 (concave down)
- 1 < x < 2: f'(x) < 0 (decreasing), f''(x) > 0 (concave up)
- x > 2: f'(x) > 0 (increasing), f''(x) > 0 (concave up)

## Practice Problems

### Problem 1
Find f''(x) for f(x) = (x² + 1)³

**Solution**:
- f'(x) = 3(x² + 1)²(2x) = 6x(x² + 1)²
- f''(x) = 6(x² + 1)² + 6x(2)(x² + 1)(2x) = 6(x² + 1)² + 24x²(x² + 1)
- = 6(x² + 1)[(x² + 1) + 4x²] = 6(x² + 1)(5x² + 1)

### Problem 2
Find inflection points of f(x) = x⁴ - 8x²

**Solution**:
- f'(x) = 4x³ - 16x = 4x(x² - 4)
- f''(x) = 12x² - 16 = 4(3x² - 4)
- f''(x) = 0 when 3x² - 4 = 0, so x² = 4/3, x = ±2/√3
- Inflection points at x = ±2/√3

### Problem 3
Use second derivative test for f(x) = x³ - 12x + 1

**Solution**:
- f'(x) = 3x² - 12 = 3(x² - 4) = 3(x - 2)(x + 2)
- Critical points: x = ±2
- f''(x) = 6x
- f''(2) = 12 > 0 → local minimum at x = 2
- f''(-2) = -12 < 0 → local maximum at x = -2

### Problem 4
A particle moves with position s(t) = t⁴ - 4t³ + 6t². Find velocity, acceleration, and jerk.

**Solution**:
- v(t) = s'(t) = 4t³ - 12t² + 12t = 4t(t² - 3t + 3)
- a(t) = v'(t) = 12t² - 24t + 12 = 12(t² - 2t + 1) = 12(t - 1)²
- j(t) = a'(t) = 24t - 24 = 24(t - 1)

## Key Takeaways
- Second derivatives determine concavity
- Inflection points occur where concavity changes
- Second derivative test helps classify critical points
- Higher-order derivatives have physical interpretations
- Second derivatives are essential for complete curve analysis
- Practice with various function types builds understanding

## Next Steps
In the next tutorial, we'll explore applications of derivatives, focusing on optimization problems and advanced curve sketching techniques.
