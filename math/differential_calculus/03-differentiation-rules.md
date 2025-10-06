# Differential Calculus Tutorial 03: Differentiation Rules

## Learning Objectives
By the end of this tutorial, you will be able to:
- Apply the power rule for derivatives
- Use the product rule for derivatives
- Apply the quotient rule for derivatives
- Master the chain rule for composite functions
- Find derivatives of trigonometric functions
- Calculate derivatives of exponential and logarithmic functions
- Combine multiple rules in complex problems

## Introduction to Differentiation Rules

While the limit definition of the derivative is fundamental, it's often tedious to use for every derivative calculation. Differentiation rules provide efficient shortcuts for finding derivatives of common function types.

## Basic Differentiation Rules

### Power Rule
If f(x) = x^n, then f'(x) = nx^(n-1)

**Examples**:
- f(x) = x³ → f'(x) = 3x²
- f(x) = x⁵ → f'(x) = 5x⁴
- f(x) = x^(-2) → f'(x) = -2x^(-3) = -2/x³
- f(x) = √x = x^(1/2) → f'(x) = (1/2)x^(-1/2) = 1/(2√x)

### Constant Multiple Rule
If f(x) = c·g(x), then f'(x) = c·g'(x)

**Examples**:
- f(x) = 5x³ → f'(x) = 5(3x²) = 15x²
- f(x) = -3x⁴ → f'(x) = -3(4x³) = -12x³

### Sum and Difference Rules
If f(x) = g(x) ± h(x), then f'(x) = g'(x) ± h'(x)

**Example**: f(x) = x³ + 2x² - 5x + 1
- f'(x) = 3x² + 4x - 5

## Product Rule

### Statement
If f(x) = g(x) · h(x), then f'(x) = g'(x) · h(x) + g(x) · h'(x)

**Memory aid**: "First times derivative of second, plus second times derivative of first"

### Examples

**Example 1**: f(x) = x² · sin(x)
- g(x) = x², h(x) = sin(x)
- g'(x) = 2x, h'(x) = cos(x)
- f'(x) = 2x · sin(x) + x² · cos(x)

**Example 2**: f(x) = (x + 1)(x² - 3x)
- g(x) = x + 1, h(x) = x² - 3x
- g'(x) = 1, h'(x) = 2x - 3
- f'(x) = 1 · (x² - 3x) + (x + 1) · (2x - 3)
- = x² - 3x + (x + 1)(2x - 3)
- = x² - 3x + 2x² - 3x + 2x - 3
- = 3x² - 4x - 3

## Quotient Rule

### Statement
If f(x) = g(x)/h(x), then f'(x) = [g'(x) · h(x) - g(x) · h'(x)]/[h(x)]²

**Memory aid**: "Bottom times derivative of top, minus top times derivative of bottom, all over bottom squared"

### Examples

**Example 1**: f(x) = x²/(x + 1)
- g(x) = x², h(x) = x + 1
- g'(x) = 2x, h'(x) = 1
- f'(x) = [2x · (x + 1) - x² · 1]/(x + 1)²
- = [2x² + 2x - x²]/(x + 1)²
- = (x² + 2x)/(x + 1)²

**Example 2**: f(x) = sin(x)/x
- g(x) = sin(x), h(x) = x
- g'(x) = cos(x), h'(x) = 1
- f'(x) = [cos(x) · x - sin(x) · 1]/x²
- = (x cos(x) - sin(x))/x²

## Chain Rule

### Statement
If f(x) = g(h(x)), then f'(x) = g'(h(x)) · h'(x)

**Memory aid**: "Derivative of outside function times derivative of inside function"

### Examples

**Example 1**: f(x) = (x² + 1)³
- Outside function: g(u) = u³, inside function: h(x) = x² + 1
- g'(u) = 3u², h'(x) = 2x
- f'(x) = 3(x² + 1)² · 2x = 6x(x² + 1)²

**Example 2**: f(x) = sin(x²)
- Outside function: g(u) = sin(u), inside function: h(x) = x²
- g'(u) = cos(u), h'(x) = 2x
- f'(x) = cos(x²) · 2x = 2x cos(x²)

**Example 3**: f(x) = e^(3x + 2)
- Outside function: g(u) = e^u, inside function: h(x) = 3x + 2
- g'(u) = e^u, h'(x) = 3
- f'(x) = e^(3x + 2) · 3 = 3e^(3x + 2)

## Trigonometric Functions

### Basic Trigonometric Derivatives
- d/dx[sin(x)] = cos(x)
- d/dx[cos(x)] = -sin(x)
- d/dx[tan(x)] = sec²(x)
- d/dx[cot(x)] = -csc²(x)
- d/dx[sec(x)] = sec(x)tan(x)
- d/dx[csc(x)] = -csc(x)cot(x)

### Examples

**Example 1**: f(x) = sin(2x)
- f'(x) = cos(2x) · 2 = 2cos(2x)

**Example 2**: f(x) = tan(x²)
- f'(x) = sec²(x²) · 2x = 2x sec²(x²)

**Example 3**: f(x) = cos³(x)
- f(x) = [cos(x)]³
- f'(x) = 3[cos(x)]² · (-sin(x)) = -3cos²(x)sin(x)

## Exponential and Logarithmic Functions

### Basic Derivatives
- d/dx[e^x] = e^x
- d/dx[a^x] = a^x ln(a)
- d/dx[ln(x)] = 1/x
- d/dx[log_a(x)] = 1/(x ln(a))

### Examples

**Example 1**: f(x) = e^(x²)
- f'(x) = e^(x²) · 2x = 2xe^(x²)

**Example 2**: f(x) = ln(x² + 1)
- f'(x) = 1/(x² + 1) · 2x = 2x/(x² + 1)

**Example 3**: f(x) = 2^x
- f'(x) = 2^x ln(2)

## Combining Multiple Rules

### Complex Examples

**Example 1**: f(x) = x²e^x sin(x)
Using product rule twice:
- Let g(x) = x²e^x, h(x) = sin(x)
- g'(x) = 2xe^x + x²e^x = e^x(2x + x²)
- h'(x) = cos(x)
- f'(x) = e^x(2x + x²)sin(x) + x²e^x cos(x)
- = e^x[(2x + x²)sin(x) + x² cos(x)]

**Example 2**: f(x) = ln(x² + 1)/(x + 1)
Using quotient rule:
- g(x) = ln(x² + 1), h(x) = x + 1
- g'(x) = 2x/(x² + 1), h'(x) = 1
- f'(x) = [2x/(x² + 1) · (x + 1) - ln(x² + 1) · 1]/(x + 1)²
- = [2x(x + 1) - (x² + 1)ln(x² + 1)]/[(x² + 1)(x + 1)²]

## Practice Problems

### Problem 1
Find f'(x) for f(x) = (x³ + 2x)⁴

**Solution**:
Using chain rule:
- f'(x) = 4(x³ + 2x)³ · (3x² + 2)
- = 4(x³ + 2x)³(3x² + 2)

### Problem 2
Find f'(x) for f(x) = x² sin(x) cos(x)

**Solution**:
Using product rule:
- Let g(x) = x², h(x) = sin(x)cos(x)
- g'(x) = 2x
- h'(x) = cos(x) · cos(x) + sin(x) · (-sin(x)) = cos²(x) - sin²(x) = cos(2x)
- f'(x) = 2x · sin(x)cos(x) + x² · cos(2x)
- = 2x sin(x)cos(x) + x² cos(2x)

### Problem 3
Find f'(x) for f(x) = e^(x²) ln(x)

**Solution**:
Using product rule:
- g(x) = e^(x²), h(x) = ln(x)
- g'(x) = e^(x²) · 2x = 2xe^(x²)
- h'(x) = 1/x
- f'(x) = 2xe^(x²) · ln(x) + e^(x²) · 1/x
- = e^(x²)[2x ln(x) + 1/x]

### Problem 4
Find f'(x) for f(x) = (x² + 1)/(x³ - 2x)

**Solution**:
Using quotient rule:
- g(x) = x² + 1, h(x) = x³ - 2x
- g'(x) = 2x, h'(x) = 3x² - 2
- f'(x) = [2x · (x³ - 2x) - (x² + 1) · (3x² - 2)]/(x³ - 2x)²
- = [2x⁴ - 4x² - (x² + 1)(3x² - 2)]/(x³ - 2x)²
- = [2x⁴ - 4x² - 3x⁴ + 2x² - 3x² + 2]/(x³ - 2x)²
- = (-x⁴ - 5x² + 2)/(x³ - 2x)²

## Key Takeaways
- Power rule: d/dx[x^n] = nx^(n-1)
- Product rule: (fg)' = f'g + fg'
- Quotient rule: (f/g)' = (f'g - fg')/g²
- Chain rule: (f(g(x)))' = f'(g(x)) · g'(x)
- Trigonometric derivatives follow specific patterns
- Exponential and logarithmic derivatives have special forms
- Complex problems often require combining multiple rules

## Next Steps
In the next tutorial, we'll explore implicit differentiation, learning how to find derivatives of functions defined implicitly and solve related rates problems.
