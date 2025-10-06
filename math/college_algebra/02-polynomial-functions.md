# College Algebra Tutorial 02: Polynomial Functions

## Learning Objectives
By the end of this tutorial, you will be able to:
- Identify and classify polynomial functions
- Graph linear and quadratic functions
- Find zeros and factors of polynomials
- Perform polynomial division
- Use synthetic division
- Understand rational functions and asymptotes

## Introduction to Polynomial Functions

### Definition
A polynomial function is a function of the form:
f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀

Where:
- n is a non-negative integer (degree)
- aₙ, aₙ₋₁, ..., a₁, a₀ are real numbers (coefficients)
- aₙ ≠ 0 (leading coefficient)

### Classification by Degree
- **Constant**: f(x) = c (degree 0)
- **Linear**: f(x) = ax + b (degree 1)
- **Quadratic**: f(x) = ax² + bx + c (degree 2)
- **Cubic**: f(x) = ax³ + bx² + cx + d (degree 3)
- **Quartic**: f(x) = ax⁴ + bx³ + cx² + dx + e (degree 4)

## Linear Functions

### Standard Form
f(x) = mx + b
- m = slope
- b = y-intercept

### Graphing Linear Functions
1. Plot the y-intercept (0, b)
2. Use the slope m = rise/run to find another point
3. Draw a line through both points

**Example**: Graph f(x) = 2x - 3
- y-intercept: (0, -3)
- slope: 2 = 2/1, so from (0, -3), go up 2 and right 1 to (1, -1)

### Finding Linear Functions
Given two points (x₁, y₁) and (x₂, y₂):
1. Find slope: m = (y₂ - y₁)/(x₂ - x₁)
2. Use point-slope form: y - y₁ = m(x - x₁)
3. Convert to slope-intercept form: y = mx + b

## Quadratic Functions

### Standard Form
f(x) = ax² + bx + c

### Vertex Form
f(x) = a(x - h)² + k
- Vertex: (h, k)
- Axis of symmetry: x = h

### Converting Between Forms
**Standard to Vertex** (Completing the square):
1. Factor out a from x² and x terms
2. Complete the square
3. Simplify

**Example**: Convert f(x) = 2x² - 8x + 5 to vertex form
1. f(x) = 2(x² - 4x) + 5
2. f(x) = 2(x² - 4x + 4 - 4) + 5
3. f(x) = 2((x - 2)² - 4) + 5
4. f(x) = 2(x - 2)² - 8 + 5
5. f(x) = 2(x - 2)² - 3

### Graphing Quadratic Functions
1. Find the vertex
2. Find the y-intercept
3. Find x-intercepts (if they exist)
4. Plot additional points if needed
5. Draw the parabola

### Finding Zeros
Use the quadratic formula: x = (-b ± √(b² - 4ac))/(2a)

**Example**: Find zeros of f(x) = x² - 5x + 6
- a = 1, b = -5, c = 6
- x = (5 ± √(25 - 24))/2 = (5 ± 1)/2
- x = 3 or x = 2

## Higher-Degree Polynomials

### End Behavior
- If degree is even and leading coefficient > 0: both ends go up
- If degree is even and leading coefficient < 0: both ends go down
- If degree is odd and leading coefficient > 0: left end down, right end up
- If degree is odd and leading coefficient < 0: left end up, right end down

### Finding Zeros
1. **Rational Root Theorem**: If p/q is a rational zero, then p divides the constant term and q divides the leading coefficient
2. **Factor Theorem**: If f(a) = 0, then (x - a) is a factor
3. **Remainder Theorem**: When f(x) is divided by (x - a), the remainder is f(a)

## Polynomial Division

### Long Division
Divide polynomials using the same process as numerical long division.

**Example**: Divide x³ - 2x² + 3x - 1 by x - 1

```
        x² - x + 2
x - 1 | x³ - 2x² + 3x - 1
        x³ - x²
        --------
           -x² + 3x
           -x² + x
           --------
               2x - 1
               2x - 2
               ------
                  1
```

Result: x² - x + 2 + 1/(x - 1)

### Synthetic Division
A shortcut for dividing by (x - a).

**Example**: Divide x³ - 2x² + 3x - 1 by x - 1

```
1 | 1  -2   3  -1
  |    1  -1   2
  ----------------
    1  -1   2   1
```

Result: x² - x + 2 + 1/(x - 1)

## Rational Functions

### Definition
A rational function is the ratio of two polynomials: f(x) = P(x)/Q(x)

### Asymptotes
1. **Vertical Asymptotes**: Occur where Q(x) = 0 (but P(x) ≠ 0)
2. **Horizontal Asymptotes**: 
   - If degree of P < degree of Q: y = 0
   - If degree of P = degree of Q: y = leading coefficient of P / leading coefficient of Q
   - If degree of P > degree of Q: no horizontal asymptote (but may have oblique)

**Example**: Find asymptotes of f(x) = (x² - 1)/(x² - 4)
- Vertical: x = ±2 (where x² - 4 = 0)
- Horizontal: y = 1 (same degree, ratio of leading coefficients)

## Practice Problems

### Problem 1
Find the vertex and axis of symmetry of f(x) = -2x² + 8x - 5.

**Solution**:
- Complete the square: f(x) = -2(x² - 4x) - 5 = -2((x - 2)² - 4) - 5 = -2(x - 2)² + 3
- Vertex: (2, 3)
- Axis of symmetry: x = 2

### Problem 2
Use synthetic division to divide x⁴ - 3x³ + 2x² - x + 1 by x - 2.

**Solution**:
```
2 | 1  -3   2  -1   1
  |    2  -2   0  -2
  --------------------
    1  -1   0  -1  -1
```

Result: x³ - x² - 1/(x - 2)

### Problem 3
Find all rational zeros of f(x) = 2x³ - 3x² - 8x + 3.

**Solution**:
Possible rational zeros: ±1, ±3, ±1/2, ±3/2
Test f(1) = 2 - 3 - 8 + 3 = -6 ≠ 0
Test f(-1) = -2 - 3 + 8 + 3 = 6 ≠ 0
Test f(3) = 54 - 27 - 24 + 3 = 6 ≠ 0
Test f(-3) = -54 - 27 + 24 + 3 = -54 ≠ 0
Test f(1/2) = 1/4 - 3/4 - 4 + 3 = 0 ✓

So x = 1/2 is a zero. Factor out (x - 1/2) to find other zeros.

## Key Takeaways
- Polynomial functions are fundamental in algebra
- Linear functions have constant rate of change
- Quadratic functions have parabolic graphs
- Higher-degree polynomials have more complex behavior
- Division helps factor polynomials and find zeros
- Rational functions have asymptotes where denominators are zero

## Next Steps
In the next tutorial, we'll explore exponential and logarithmic functions, which are crucial for modeling growth, decay, and many real-world phenomena.
