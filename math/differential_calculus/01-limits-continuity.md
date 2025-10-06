# Differential Calculus Tutorial 01: Limits and Continuity

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the concept of limits
- Evaluate limits using various techniques
- Understand one-sided limits
- Determine continuity of functions
- Use the intermediate value theorem
- Apply the squeeze theorem

## Introduction to Limits

### What is a Limit?
A limit describes the behavior of a function as the input approaches a particular value. It answers: "What value does f(x) approach as x approaches a?"

**Notation**: lim(x→a) f(x) = L

**Intuitive Definition**: As x gets closer to a, f(x) gets closer to L.

### Examples
1. lim(x→2) (x + 3) = 5
2. lim(x→0) sin(x)/x = 1
3. lim(x→∞) 1/x = 0

## Evaluating Limits

### Direct Substitution
If f(a) is defined, then lim(x→a) f(x) = f(a).

**Example**: lim(x→3) (x² - 2x + 1) = 3² - 2(3) + 1 = 9 - 6 + 1 = 4

### Factoring and Canceling
When direct substitution gives 0/0, try factoring.

**Example**: lim(x→2) (x² - 4)/(x - 2)
- Direct substitution: (4 - 4)/(2 - 2) = 0/0 (indeterminate)
- Factor: (x - 2)(x + 2)/(x - 2) = x + 2
- lim(x→2) (x + 2) = 2 + 2 = 4

### Rationalizing
For limits involving square roots, rationalize the numerator or denominator.

**Example**: lim(x→0) (√(x + 4) - 2)/x
- Multiply by conjugate: (√(x + 4) - 2)/x × (√(x + 4) + 2)/(√(x + 4) + 2)
- = (x + 4 - 4)/(x(√(x + 4) + 2)) = x/(x(√(x + 4) + 2)) = 1/(√(x + 4) + 2)
- lim(x→0) 1/(√(x + 4) + 2) = 1/(√4 + 2) = 1/4

## One-Sided Limits

### Left-Hand Limit
lim(x→a⁻) f(x) = L means f(x) approaches L as x approaches a from the left.

### Right-Hand Limit
lim(x→a⁺) f(x) = L means f(x) approaches L as x approaches a from the right.

### Relationship to Two-Sided Limits
lim(x→a) f(x) = L if and only if lim(x→a⁻) f(x) = lim(x→a⁺) f(x) = L

**Example**: For f(x) = |x|/x:
- lim(x→0⁻) f(x) = lim(x→0⁻) (-x)/x = -1
- lim(x→0⁺) f(x) = lim(x→0⁺) x/x = 1
- Since left and right limits differ, lim(x→0) f(x) does not exist

## Limits at Infinity

### Definition
lim(x→∞) f(x) = L means f(x) approaches L as x becomes arbitrarily large.

### Techniques
1. **Divide by highest power**: For rational functions
2. **Use known limits**: lim(x→∞) 1/x = 0
3. **L'Hôpital's Rule**: For indeterminate forms

**Example**: lim(x→∞) (3x² + 2x - 1)/(2x² - x + 3)
- Divide by x²: (3 + 2/x - 1/x²)/(2 - 1/x + 3/x²)
- As x → ∞: (3 + 0 - 0)/(2 - 0 + 0) = 3/2

## Continuity

### Definition
A function f is continuous at x = a if:
1. f(a) is defined
2. lim(x→a) f(x) exists
3. lim(x→a) f(x) = f(a)

### Types of Discontinuities
1. **Removable**: Limit exists but ≠ f(a)
2. **Jump**: Left and right limits exist but differ
3. **Infinite**: Limit is ±∞

**Example**: f(x) = (x² - 1)/(x - 1) has removable discontinuity at x = 1
- f(1) is undefined
- lim(x→1) f(x) = lim(x→1) (x + 1) = 2
- Redefining f(1) = 2 makes function continuous

## Important Theorems

### Intermediate Value Theorem
If f is continuous on [a, b] and k is between f(a) and f(b), then there exists c ∈ (a, b) such that f(c) = k.

**Example**: Show that x³ - x - 1 = 0 has a solution between 1 and 2.
- f(x) = x³ - x - 1
- f(1) = 1 - 1 - 1 = -1
- f(2) = 8 - 2 - 1 = 5
- Since f is continuous and 0 is between -1 and 5, there exists c ∈ (1, 2) with f(c) = 0

### Squeeze Theorem
If f(x) ≤ g(x) ≤ h(x) near a and lim(x→a) f(x) = lim(x→a) h(x) = L, then lim(x→a) g(x) = L.

**Example**: Find lim(x→0) x² sin(1/x)
- -1 ≤ sin(1/x) ≤ 1
- -x² ≤ x² sin(1/x) ≤ x²
- lim(x→0) (-x²) = lim(x→0) x² = 0
- Therefore, lim(x→0) x² sin(1/x) = 0

## Special Limits

### Trigonometric Limits
1. lim(x→0) sin(x)/x = 1
2. lim(x→0) (1 - cos(x))/x = 0
3. lim(x→0) tan(x)/x = 1

### Exponential and Logarithmic Limits
1. lim(x→0) (eˣ - 1)/x = 1
2. lim(x→0) ln(1 + x)/x = 1
3. lim(x→∞) (1 + 1/x)ˣ = e

## Practice Problems

### Problem 1
Find lim(x→3) (x² - 9)/(x - 3)

**Solution**:
- Factor: (x - 3)(x + 3)/(x - 3) = x + 3
- lim(x→3) (x + 3) = 6

### Problem 2
Find lim(x→0) (√(x + 1) - 1)/x

**Solution**:
- Rationalize: (√(x + 1) - 1)/x × (√(x + 1) + 1)/(√(x + 1) + 1)
- = (x + 1 - 1)/(x(√(x + 1) + 1)) = x/(x(√(x + 1) + 1)) = 1/(√(x + 1) + 1)
- lim(x→0) 1/(√(x + 1) + 1) = 1/2

### Problem 3
Determine if f(x) = (x² - 4)/(x - 2) is continuous at x = 2

**Solution**:
- f(2) is undefined (removable discontinuity)
- lim(x→2) f(x) = lim(x→2) (x + 2) = 4
- Function is not continuous at x = 2, but discontinuity is removable

### Problem 4
Find lim(x→∞) (2x³ - x + 1)/(3x³ + 2x² - 5)

**Solution**:
- Divide by x³: (2 - 1/x² + 1/x³)/(3 + 2/x - 5/x³)
- As x → ∞: (2 - 0 + 0)/(3 + 0 - 0) = 2/3

## Key Takeaways
- Limits describe function behavior near a point
- Direct substitution works when function is defined
- Factoring and rationalizing help with indeterminate forms
- One-sided limits check behavior from each direction
- Continuity requires function value to equal limit
- Special limits provide shortcuts for common cases

## Next Steps
In the next tutorial, we'll explore the derivative, learning how to find rates of change and slopes of curves using the definition and basic rules.
