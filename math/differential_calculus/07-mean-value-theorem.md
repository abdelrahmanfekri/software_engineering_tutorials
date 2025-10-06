# Differential Calculus Tutorial 07: Mean Value Theorem

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand and apply Rolle's theorem
- Master the Mean Value Theorem (MVT)
- Use MVT to prove function properties
- Apply MVT to optimization problems
- Understand consequences and corollaries of MVT
- Use MVT in proofs and applications

## Rolle's Theorem

### Statement
If f is continuous on [a, b], differentiable on (a, b), and f(a) = f(b), then there exists at least one c ∈ (a, b) such that f'(c) = 0.

### Geometric Interpretation
Rolle's theorem guarantees that if a smooth curve starts and ends at the same height, there's at least one point where the tangent line is horizontal.

### Examples

**Example 1**: Verify Rolle's theorem for f(x) = x² - 4x + 3 on [1, 3]

**Solution**:
- f is continuous and differentiable everywhere
- f(1) = 1 - 4 + 3 = 0
- f(3) = 9 - 12 + 3 = 0
- f'(x) = 2x - 4 = 2(x - 2)
- f'(x) = 0 when x = 2
- Since 2 ∈ (1, 3), Rolle's theorem is satisfied

**Example 2**: Show that f(x) = x³ - 3x + 1 has exactly one root in [0, 1]

**Solution**:
- f(0) = 1 > 0, f(1) = 1 - 3 + 1 = -1 < 0
- By Intermediate Value Theorem, f has at least one root in (0, 1)
- f'(x) = 3x² - 3 = 3(x² - 1) = 3(x - 1)(x + 1)
- f'(x) < 0 for all x ∈ (0, 1), so f is strictly decreasing
- Therefore, f has exactly one root in [0, 1]

## Mean Value Theorem (MVT)

### Statement
If f is continuous on [a, b] and differentiable on (a, b), then there exists at least one c ∈ (a, b) such that:
f'(c) = [f(b) - f(a)]/(b - a)

### Geometric Interpretation
MVT guarantees that there's at least one point where the tangent line is parallel to the secant line connecting the endpoints.

### Examples

**Example 1**: Verify MVT for f(x) = x² on [1, 3]

**Solution**:
- f is continuous and differentiable everywhere
- f(1) = 1, f(3) = 9
- [f(3) - f(1)]/(3 - 1) = (9 - 1)/2 = 4
- f'(x) = 2x
- f'(c) = 4 when 2c = 4, so c = 2
- Since 2 ∈ (1, 3), MVT is satisfied

**Example 2**: Find all values of c that satisfy MVT for f(x) = x³ - x on [-1, 1]

**Solution**:
- f(-1) = -1 - (-1) = 0, f(1) = 1 - 1 = 0
- [f(1) - f(-1)]/(1 - (-1)) = 0/2 = 0
- f'(x) = 3x² - 1
- f'(c) = 0 when 3c² - 1 = 0, so c² = 1/3, c = ±1/√3
- Both values are in (-1, 1)

## Applications of MVT

### Proving Function Properties

**Example 1**: Prove that if f'(x) = 0 for all x in an interval, then f is constant on that interval.

**Proof**:
- Let a, b be any two points in the interval with a < b
- By MVT: f'(c) = [f(b) - f(a)]/(b - a) for some c ∈ (a, b)
- Since f'(c) = 0: 0 = [f(b) - f(a)]/(b - a)
- Therefore: f(b) = f(a)
- Since a, b were arbitrary, f is constant

**Example 2**: Prove that if f'(x) > 0 for all x in an interval, then f is increasing on that interval.

**Proof**:
- Let a, b be any two points in the interval with a < b
- By MVT: f'(c) = [f(b) - f(a)]/(b - a) for some c ∈ (a, b)
- Since f'(c) > 0 and b - a > 0: [f(b) - f(a)]/(b - a) > 0
- Therefore: f(b) - f(a) > 0, so f(b) > f(a)
- Since a, b were arbitrary with a < b, f is increasing

### Bounds on Function Values

**Example**: Show that |sin(x) - sin(y)| ≤ |x - y| for all x, y

**Solution**:
- If x = y, the inequality is trivially true
- If x ≠ y, apply MVT to f(t) = sin(t) on the interval between x and y
- f'(c) = cos(c) for some c between x and y
- MVT gives: sin(x) - sin(y) = cos(c)(x - y)
- Since |cos(c)| ≤ 1: |sin(x) - sin(y)| = |cos(c)||x - y| ≤ |x - y|

## Consequences of MVT

### Corollary 1: Constant Difference
If f'(x) = g'(x) for all x in an interval, then f(x) = g(x) + C for some constant C.

**Proof**:
- Let h(x) = f(x) - g(x)
- h'(x) = f'(x) - g'(x) = 0
- By the previous result, h(x) is constant
- Therefore: f(x) = g(x) + C

### Corollary 2: Monotonicity
- If f'(x) > 0 on (a, b), then f is increasing on [a, b]
- If f'(x) < 0 on (a, b), then f is decreasing on [a, b]

### Example: Prove that e^x > 1 + x for x > 0

**Solution**:
- Let f(x) = e^x - (1 + x) = e^x - 1 - x
- f(0) = e^0 - 1 - 0 = 0
- f'(x) = e^x - 1
- For x > 0: f'(x) = e^x - 1 > e^0 - 1 = 0
- Since f'(x) > 0 for x > 0 and f(0) = 0: f(x) > 0 for x > 0
- Therefore: e^x - 1 - x > 0, so e^x > 1 + x

## MVT in Optimization

### Example: Find the maximum value of f(x) = x³ - 3x² + 2 on [0, 3]

**Solution**:
- f'(x) = 3x² - 6x = 3x(x - 2)
- Critical points: x = 0, x = 2
- Evaluate at critical points and endpoints:
  - f(0) = 2
  - f(2) = 8 - 12 + 2 = -2
  - f(3) = 27 - 27 + 2 = 2
- Maximum value: 2

### Using MVT to Estimate Values

**Example**: Use MVT to estimate √101

**Solution**:
- Let f(x) = √x
- f'(x) = 1/(2√x)
- Apply MVT on [100, 101]:
- f'(c) = [f(101) - f(100)]/(101 - 100) = [√101 - 10]/1 = √101 - 10
- f'(c) = 1/(2√c) for some c ∈ (100, 101)
- Since c ≈ 100: f'(c) ≈ 1/(2√100) = 1/20 = 0.05
- Therefore: √101 - 10 ≈ 0.05, so √101 ≈ 10.05

## Practice Problems

### Problem 1
Verify Rolle's theorem for f(x) = x³ - 6x² + 11x - 6 on [1, 3]

**Solution**:
- f(1) = 1 - 6 + 11 - 6 = 0
- f(3) = 27 - 54 + 33 - 6 = 0
- f'(x) = 3x² - 12x + 11
- f'(x) = 0 when 3x² - 12x + 11 = 0
- Using quadratic formula: x = (12 ± √(144 - 132))/6 = (12 ± √12)/6 = (12 ± 2√3)/6 = 2 ± √3/3
- Both roots are in (1, 3)

### Problem 2
Find all values of c that satisfy MVT for f(x) = x³ on [0, 2]

**Solution**:
- f(0) = 0, f(2) = 8
- [f(2) - f(0)]/(2 - 0) = 8/2 = 4
- f'(x) = 3x²
- f'(c) = 4 when 3c² = 4, so c² = 4/3, c = ±2/√3
- Only c = 2/√3 is in (0, 2)

### Problem 3
Prove that if f'(x) ≥ 0 for all x in an interval, then f is non-decreasing on that interval.

**Solution**:
- Let a, b be any two points in the interval with a < b
- By MVT: f'(c) = [f(b) - f(a)]/(b - a) for some c ∈ (a, b)
- Since f'(c) ≥ 0 and b - a > 0: [f(b) - f(a)]/(b - a) ≥ 0
- Therefore: f(b) - f(a) ≥ 0, so f(b) ≥ f(a)
- Since a, b were arbitrary with a < b, f is non-decreasing

### Problem 4
Show that |cos(x) - cos(y)| ≤ |x - y| for all x, y

**Solution**:
- If x = y, the inequality is trivially true
- If x ≠ y, apply MVT to f(t) = cos(t) on the interval between x and y
- f'(c) = -sin(c) for some c between x and y
- MVT gives: cos(x) - cos(y) = -sin(c)(x - y)
- Since |sin(c)| ≤ 1: |cos(x) - cos(y)| = |sin(c)||x - y| ≤ |x - y|

## Key Takeaways
- Rolle's theorem is a special case of MVT
- MVT connects average and instantaneous rates of change
- MVT is powerful for proving function properties
- MVT provides estimates and bounds
- Understanding MVT deepens understanding of derivatives
- MVT is essential for advanced calculus topics

## Next Steps
In the next tutorial, we'll explore related rates problems in detail, learning systematic approaches to solve complex rate of change problems.
