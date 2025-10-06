# College Algebra Tutorial 01: Functions and Their Properties

## Learning Objectives
By the end of this tutorial, you will be able to:
- Define and identify functions
- Determine domain and range of functions
- Perform function operations (addition, subtraction, multiplication, division)
- Find composition of functions
- Determine inverse functions
- Identify even and odd functions

## Introduction to Functions

### What is a Function?
A function is a relation between two sets where each input (domain element) corresponds to exactly one output (range element). We write f(x) = y, where f is the function name, x is the input, and y is the output.

**Notation**: f: A → B means f is a function from set A to set B.

### Function Notation
- f(x) = 2x + 3
- g(x) = x² - 1
- h(x) = √(x + 2)

### Domain and Range
- **Domain**: The set of all possible input values (x-values)
- **Range**: The set of all possible output values (y-values)

**Example**: For f(x) = √(x - 1)
- Domain: x ≥ 1 (because √(x - 1) is defined only when x - 1 ≥ 0)
- Range: y ≥ 0 (because √(x - 1) ≥ 0 for all x in the domain)

## Function Operations

### Arithmetic Operations on Functions
Given functions f(x) and g(x):

1. **Addition**: (f + g)(x) = f(x) + g(x)
2. **Subtraction**: (f - g)(x) = f(x) - g(x)
3. **Multiplication**: (f · g)(x) = f(x) · g(x)
4. **Division**: (f/g)(x) = f(x)/g(x), where g(x) ≠ 0

**Example**: If f(x) = x + 2 and g(x) = x - 1
- (f + g)(x) = (x + 2) + (x - 1) = 2x + 1
- (f · g)(x) = (x + 2)(x - 1) = x² + x - 2

### Function Composition
The composition of f and g is written as (f ∘ g)(x) = f(g(x)).

**Example**: If f(x) = x² and g(x) = x + 3
- (f ∘ g)(x) = f(g(x)) = f(x + 3) = (x + 3)²
- (g ∘ f)(x) = g(f(x)) = g(x²) = x² + 3

## Inverse Functions

### Definition
The inverse function f⁻¹(x) of f(x) satisfies:
- f⁻¹(f(x)) = x for all x in the domain of f
- f(f⁻¹(x)) = x for all x in the domain of f⁻¹

### Finding Inverse Functions
1. Replace f(x) with y
2. Swap x and y
3. Solve for y
4. Replace y with f⁻¹(x)

**Example**: Find the inverse of f(x) = 2x + 3
1. y = 2x + 3
2. x = 2y + 3
3. x - 3 = 2y
4. y = (x - 3)/2
5. f⁻¹(x) = (x - 3)/2

## Even and Odd Functions

### Even Functions
A function f is even if f(-x) = f(x) for all x in the domain.
- Graph is symmetric about the y-axis
- Examples: f(x) = x², f(x) = |x|, f(x) = cos(x)

### Odd Functions
A function f is odd if f(-x) = -f(x) for all x in the domain.
- Graph is symmetric about the origin
- Examples: f(x) = x³, f(x) = sin(x), f(x) = x

## Practice Problems

### Problem 1
Find the domain and range of f(x) = 1/(x - 2).

**Solution**:
- Domain: x ≠ 2 (all real numbers except 2)
- Range: y ≠ 0 (all real numbers except 0)

### Problem 2
If f(x) = x² and g(x) = x + 1, find (f ∘ g)(x) and (g ∘ f)(x).

**Solution**:
- (f ∘ g)(x) = f(g(x)) = f(x + 1) = (x + 1)² = x² + 2x + 1
- (g ∘ f)(x) = g(f(x)) = g(x²) = x² + 1

### Problem 3
Determine if f(x) = x³ - x is even, odd, or neither.

**Solution**:
f(-x) = (-x)³ - (-x) = -x³ + x = -(x³ - x) = -f(x)
Since f(-x) = -f(x), the function is odd.

## Key Takeaways
- Functions are relations where each input has exactly one output
- Domain and range are fundamental properties of functions
- Function operations allow us to combine functions algebraically
- Composition creates new functions by substituting one function into another
- Inverse functions "undo" the original function
- Even and odd functions have special symmetry properties

## Next Steps
In the next tutorial, we'll explore polynomial functions, including linear, quadratic, and higher-degree polynomials, and learn about their properties and applications.
