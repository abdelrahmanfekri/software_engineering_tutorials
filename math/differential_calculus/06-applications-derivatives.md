# Differential Calculus Tutorial 06: Applications of Derivatives

## Learning Objectives
By the end of this tutorial, you will be able to:
- Find critical points and extrema using derivatives
- Apply the first and second derivative tests
- Solve optimization problems
- Sketch curves using calculus
- Apply L'Hôpital's rule for indeterminate forms
- Use derivatives in real-world applications

## Critical Points and Extrema

### Definitions
- **Critical Point**: A point where f'(x) = 0 or f'(x) is undefined
- **Local Maximum**: f(c) ≥ f(x) for all x near c
- **Local Minimum**: f(c) ≤ f(x) for all x near c
- **Absolute Maximum**: f(c) ≥ f(x) for all x in the domain
- **Absolute Minimum**: f(c) ≤ f(x) for all x in the domain

### Finding Critical Points
1. Find f'(x)
2. Set f'(x) = 0 and solve for x
3. Find where f'(x) is undefined
4. These x-values are critical points

### Examples

**Example 1**: Find critical points of f(x) = x³ - 3x² + 2

**Solution**:
- f'(x) = 3x² - 6x = 3x(x - 2)
- f'(x) = 0 when x = 0 or x = 2
- f'(x) is always defined
- Critical points: x = 0, x = 2

**Example 2**: Find critical points of f(x) = x^(2/3)

**Solution**:
- f'(x) = (2/3)x^(-1/3) = 2/(3x^(1/3))
- f'(x) = 0 has no solution
- f'(x) is undefined when x = 0
- Critical point: x = 0

## First Derivative Test

### Process
1. Find critical points
2. Test the sign of f'(x) on both sides of each critical point
3. If f'(x) changes from + to -, then local maximum
4. If f'(x) changes from - to +, then local minimum
5. If f'(x) doesn't change sign, then neither

### Examples

**Example 1**: Use first derivative test for f(x) = x³ - 3x² + 2

**Solution**:
- Critical points: x = 0, x = 2
- Test intervals:
  - x < 0: f'(x) = 3x(x - 2) > 0 (increasing)
  - 0 < x < 2: f'(x) < 0 (decreasing)
  - x > 2: f'(x) > 0 (increasing)
- Local maximum at x = 0
- Local minimum at x = 2

**Example 2**: Use first derivative test for f(x) = x⁴ - 4x²

**Solution**:
- f'(x) = 4x³ - 8x = 4x(x² - 2) = 4x(x - √2)(x + √2)
- Critical points: x = 0, x = ±√2
- Test intervals:
  - x < -√2: f'(x) < 0 (decreasing)
  - -√2 < x < 0: f'(x) > 0 (increasing)
  - 0 < x < √2: f'(x) < 0 (decreasing)
  - x > √2: f'(x) > 0 (increasing)
- Local minimum at x = -√2
- Local maximum at x = 0
- Local minimum at x = √2

## Optimization Problems

### Strategy
1. Identify the quantity to optimize
2. Express it as a function of one variable
3. Find critical points
4. Use derivative tests or evaluate at critical points and endpoints
5. Answer the question

### Examples

**Example 1**: Find the rectangle with maximum area that can be inscribed in a circle of radius 5.

**Solution**:
- Let the rectangle have width 2x and height 2y
- Constraint: x² + y² = 25, so y = √(25 - x²)
- Area: A = (2x)(2y) = 4xy = 4x√(25 - x²)
- A'(x) = 4√(25 - x²) + 4x(-x/√(25 - x²)) = 4(25 - x² - x²)/√(25 - x²)
- A'(x) = 4(25 - 2x²)/√(25 - x²) = 0 when 25 - 2x² = 0
- x² = 25/2, x = 5/√2
- Maximum area when x = y = 5/√2 (square)
- Maximum area = 4(5/√2)² = 4(25/2) = 50

**Example 2**: A farmer wants to fence a rectangular area of 1000 m² using the least amount of fencing. One side is along a river and doesn't need fencing.

**Solution**:
- Let x = length parallel to river, y = length perpendicular to river
- Constraint: xy = 1000, so y = 1000/x
- Fencing needed: F = x + 2y = x + 2(1000/x) = x + 2000/x
- F'(x) = 1 - 2000/x² = 0 when x² = 2000, x = √2000 = 20√5
- y = 1000/(20√5) = 50/√5 = 10√5
- Minimum fencing: F = 20√5 + 2(10√5) = 40√5 m

**Example 3**: Find the point on the parabola y = x² closest to (0, 3).

**Solution**:
- Distance squared: D² = (x - 0)² + (x² - 3)² = x² + (x² - 3)²
- D² = x² + x⁴ - 6x² + 9 = x⁴ - 5x² + 9
- dD²/dx = 4x³ - 10x = 2x(2x² - 5) = 0
- x = 0 or x² = 5/2, x = ±√(5/2)
- Test: D²(0) = 9, D²(±√(5/2)) = 5/2 - 5(5/2) + 9 = 5/2 - 25/2 + 9 = -10 + 9 = -1
- Wait, this gives negative distance squared, which is impossible
- Let me recalculate: D²(±√(5/2)) = (5/2)² - 5(5/2) + 9 = 25/4 - 25/2 + 9 = 25/4 - 50/4 + 36/4 = 11/4
- Closest points: (±√(5/2), 5/2)

## Curve Sketching

### Complete Process
1. **Domain and Intercepts**
2. **Symmetry** (even/odd functions)
3. **Asymptotes** (vertical, horizontal, oblique)
4. **First Derivative Analysis**
   - Find f'(x) and critical points
   - Determine intervals of increase/decrease
5. **Second Derivative Analysis**
   - Find f''(x) and inflection points
   - Determine concavity
6. **Sketch the Curve**

### Example: Sketch f(x) = (x² - 1)/(x² + 1)

**Solution**:
1. **Domain**: All real numbers
2. **Intercepts**: f(0) = -1, f(x) = 0 when x² - 1 = 0, so x = ±1
3. **Symmetry**: f(-x) = f(x), so even function
4. **Asymptotes**: 
   - Horizontal: lim(x→±∞) f(x) = 1
   - No vertical asymptotes
5. **First Derivative**:
   - f'(x) = [2x(x² + 1) - (x² - 1)(2x)]/(x² + 1)² = [2x³ + 2x - 2x³ + 2x]/(x² + 1)² = 4x/(x² + 1)²
   - f'(x) = 0 when x = 0
   - x < 0: f'(x) < 0 (decreasing)
   - x > 0: f'(x) > 0 (increasing)
6. **Second Derivative**:
   - f''(x) = [4(x² + 1)² - 4x(2)(x² + 1)(2x)]/(x² + 1)⁴ = 4(x² + 1)[(x² + 1) - 4x²]/(x² + 1)⁴
   - f''(x) = 4(1 - 3x²)/(x² + 1)³
   - f''(x) = 0 when 1 - 3x² = 0, so x = ±1/√3
   - Concavity changes at x = ±1/√3

## L'Hôpital's Rule

### Statement
If lim(x→a) f(x)/g(x) is of the form 0/0 or ∞/∞, then:
lim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x)

### Examples

**Example 1**: Find lim(x→0) sin(x)/x

**Solution**:
- Direct substitution gives 0/0
- Apply L'Hôpital's rule: lim(x→0) cos(x)/1 = cos(0) = 1

**Example 2**: Find lim(x→∞) x/e^x

**Solution**:
- Direct substitution gives ∞/∞
- Apply L'Hôpital's rule: lim(x→∞) 1/e^x = 0

**Example 3**: Find lim(x→0) (1 - cos(x))/x²

**Solution**:
- Direct substitution gives 0/0
- Apply L'Hôpital's rule: lim(x→0) sin(x)/(2x) = 0/0
- Apply again: lim(x→0) cos(x)/2 = 1/2

## Practice Problems

### Problem 1
Find the maximum value of f(x) = x³ - 3x² + 2 on [0, 3]

**Solution**:
- f'(x) = 3x² - 6x = 3x(x - 2)
- Critical points: x = 0, x = 2
- Evaluate: f(0) = 2, f(2) = 8 - 12 + 2 = -2, f(3) = 27 - 27 + 2 = 2
- Maximum value: 2

### Problem 2
Find the dimensions of the rectangle with maximum area that can be inscribed in a semicircle of radius 5.

**Solution**:
- Let rectangle have width 2x and height y
- Constraint: x² + y² = 25, so y = √(25 - x²)
- Area: A = 2xy = 2x√(25 - x²)
- A'(x) = 2√(25 - x²) + 2x(-x/√(25 - x²)) = 2(25 - 2x²)/√(25 - x²) = 0
- 25 - 2x² = 0, x² = 25/2, x = 5/√2
- y = √(25 - 25/2) = √(25/2) = 5/√2
- Dimensions: width = 10/√2, height = 5/√2

### Problem 3
Find lim(x→0) (e^x - 1 - x)/x²

**Solution**:
- Direct substitution gives 0/0
- Apply L'Hôpital's rule: lim(x→0) (e^x - 1)/(2x) = 0/0
- Apply again: lim(x→0) e^x/2 = 1/2

### Problem 4
Sketch f(x) = x³ - 3x + 1

**Solution**:
- Domain: All real numbers
- Intercepts: f(0) = 1
- f'(x) = 3x² - 3 = 3(x² - 1) = 3(x - 1)(x + 1)
- Critical points: x = ±1
- f''(x) = 6x
- Inflection point: x = 0
- Intervals:
  - x < -1: f'(x) > 0 (increasing), f''(x) < 0 (concave down)
  - -1 < x < 0: f'(x) < 0 (decreasing), f''(x) < 0 (concave down)
  - 0 < x < 1: f'(x) < 0 (decreasing), f''(x) > 0 (concave up)
  - x > 1: f'(x) > 0 (increasing), f''(x) > 0 (concave up)

## Key Takeaways
- Critical points occur where f'(x) = 0 or f'(x) is undefined
- First derivative test determines local extrema
- Second derivative test provides alternative method
- Optimization problems require careful setup and analysis
- Curve sketching combines multiple derivative concepts
- L'Hôpital's rule handles indeterminate forms
- Practice with various problem types builds confidence

## Next Steps
In the next tutorial, we'll explore the Mean Value Theorem and its applications, including Rolle's theorem and consequences for function behavior.
