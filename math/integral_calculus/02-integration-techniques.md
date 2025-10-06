# Integration Techniques

## Overview
This tutorial covers advanced integration techniques beyond basic substitution. These methods are essential for integrating more complex functions that cannot be solved using elementary rules alone.

## Learning Objectives
- Master integration by parts
- Learn trigonometric integration techniques
- Apply trigonometric substitution
- Use partial fractions for rational functions
- Integrate various types of functions using appropriate techniques

## 1. Integration by Parts

### The Formula
Integration by parts is based on the product rule for differentiation:

```
∫u dv = uv - ∫v du
```

### Choosing u and dv
Use the LIATE rule to choose u (in order of preference):
- **L**ogarithmic functions
- **I**nverse trigonometric functions
- **A**lgebraic functions
- **T**rigonometric functions
- **E**xponential functions

### Examples

#### Example 1: Basic Integration by Parts
```
∫x e^x dx
```

Let u = x, dv = e^x dx
Then du = dx, v = e^x
```
∫x e^x dx = x e^x - ∫e^x dx = x e^x - e^x + C = e^x(x - 1) + C
```

#### Example 2: Repeated Integration by Parts
```
∫x² e^x dx
```

Let u = x², dv = e^x dx
Then du = 2x dx, v = e^x
```
∫x² e^x dx = x² e^x - ∫2x e^x dx = x² e^x - 2∫x e^x dx
```

Now apply integration by parts again to ∫x e^x dx:
```
∫x e^x dx = x e^x - e^x + C
```

Therefore:
```
∫x² e^x dx = x² e^x - 2(x e^x - e^x) + C = e^x(x² - 2x + 2) + C
```

## 2. Trigonometric Integrals

### Powers of Sine and Cosine

#### Case 1: Odd Power of Sine
```
∫sin^(2n+1)(x) cos^m(x) dx
```

Strategy: Factor out one sin(x) and use sin²(x) = 1 - cos²(x)

#### Case 2: Odd Power of Cosine
```
∫sin^n(x) cos^(2m+1)(x) dx
```

Strategy: Factor out one cos(x) and use cos²(x) = 1 - sin²(x)

#### Case 3: Even Powers of Both
```
∫sin^(2n)(x) cos^(2m)(x) dx
```

Strategy: Use power-reducing formulas:
- sin²(x) = (1 - cos(2x))/2
- cos²(x) = (1 + cos(2x))/2

### Examples

#### Example 1: Odd Power of Sine
```
∫sin³(x) cos²(x) dx
```

```
∫sin³(x) cos²(x) dx = ∫sin(x) sin²(x) cos²(x) dx
= ∫sin(x)(1 - cos²(x)) cos²(x) dx
= ∫sin(x)(cos²(x) - cos⁴(x)) dx
```

Let u = cos(x), then du = -sin(x) dx
```
= ∫(u⁴ - u²) du = u⁵/5 - u³/3 + C = cos⁵(x)/5 - cos³(x)/3 + C
```

### Powers of Tangent and Secant

#### Case 1: Even Power of Secant
```
∫tan^n(x) sec^(2m)(x) dx
```

Strategy: Factor out sec²(x) and use sec²(x) = 1 + tan²(x)

#### Case 2: Odd Power of Tangent
```
∫tan^(2n+1)(x) sec^m(x) dx
```

Strategy: Factor out sec(x)tan(x) and use tan²(x) = sec²(x) - 1

## 3. Trigonometric Substitution

### When to Use
Use trigonometric substitution for integrals containing:
- √(a² - x²) → use x = a sin(θ)
- √(a² + x²) → use x = a tan(θ)
- √(x² - a²) → use x = a sec(θ)

### The Three Cases

#### Case 1: √(a² - x²)
Substitution: x = a sin(θ), dx = a cos(θ) dθ
Identity: √(a² - x²) = a cos(θ)

#### Case 2: √(a² + x²)
Substitution: x = a tan(θ), dx = a sec²(θ) dθ
Identity: √(a² + x²) = a sec(θ)

#### Case 3: √(x² - a²)
Substitution: x = a sec(θ), dx = a sec(θ)tan(θ) dθ
Identity: √(x² - a²) = a tan(θ)

### Examples

#### Example 1: √(a² - x²)
```
∫√(4 - x²) dx
```

Let x = 2 sin(θ), then dx = 2 cos(θ) dθ
```
∫√(4 - x²) dx = ∫2 cos(θ) · 2 cos(θ) dθ = 4∫cos²(θ) dθ
= 4∫(1 + cos(2θ))/2 dθ = 2θ + sin(2θ) + C
```

Convert back to x:
```
θ = arcsin(x/2)
sin(2θ) = 2 sin(θ)cos(θ) = 2(x/2)(√(4-x²)/2) = x√(4-x²)/2
```

Therefore:
```
∫√(4 - x²) dx = 2 arcsin(x/2) + x√(4-x²)/2 + C
```

## 4. Partial Fractions

### When to Use
Use partial fractions for rational functions where the degree of the numerator is less than the degree of the denominator.

### The Method
1. Factor the denominator completely
2. Write the partial fraction decomposition
3. Solve for the unknown coefficients
4. Integrate each term separately

### Types of Partial Fractions

#### Case 1: Distinct Linear Factors
```
P(x)/((x-a)(x-b)) = A/(x-a) + B/(x-b)
```

#### Case 2: Repeated Linear Factors
```
P(x)/(x-a)² = A/(x-a) + B/(x-a)²
```

#### Case 3: Distinct Quadratic Factors
```
P(x)/((x²+ax+b)(x²+cx+d)) = (Ax+B)/(x²+ax+b) + (Cx+D)/(x²+cx+d)
```

### Examples

#### Example 1: Distinct Linear Factors
```
∫(x+1)/(x²-1) dx
```

Factor: x² - 1 = (x-1)(x+1)
```
(x+1)/(x²-1) = A/(x-1) + B/(x+1)
```

Multiply by (x-1)(x+1):
```
x + 1 = A(x+1) + B(x-1)
```

Solve: A = 1/2, B = 1/2
```
∫(x+1)/(x²-1) dx = ∫(1/2)/(x-1) + (1/2)/(x+1) dx
= (1/2)ln|x-1| + (1/2)ln|x+1| + C
= (1/2)ln|(x-1)(x+1)| + C
```

## 5. Rationalizing Substitutions

### When to Use
Use for integrals containing expressions like:
- √(ax + b)
- (ax + b)^(1/n)

### Method
Let u = √(ax + b) or u = (ax + b)^(1/n), then solve for x and dx in terms of u.

### Example
```
∫x√(x+1) dx
```

Let u = √(x+1), then u² = x + 1, so x = u² - 1, dx = 2u du
```
∫x√(x+1) dx = ∫(u² - 1)u · 2u du = 2∫(u⁴ - u²) du
= 2(u⁵/5 - u³/3) + C = 2u³(u²/5 - 1/3) + C
= 2(x+1)^(3/2)((x+1)/5 - 1/3) + C
```

## 6. Integration Strategy

### Decision Tree
1. **Is it a basic form?** → Use basic rules
2. **Can you substitute?** → Try substitution first
3. **Is it a product?** → Try integration by parts
4. **Is it rational?** → Try partial fractions
5. **Contains radicals?** → Try trigonometric substitution or rationalizing substitution
6. **Contains trig functions?** → Use trigonometric integration techniques

## 7. Practice Problems

### Integration by Parts
1. ∫x ln(x) dx
2. ∫x² sin(x) dx
3. ∫e^x cos(x) dx
4. ∫arcsin(x) dx

### Trigonometric Integrals
1. ∫sin⁴(x) dx
2. ∫cos³(x) sin²(x) dx
3. ∫tan³(x) sec(x) dx
4. ∫sin(2x) cos(3x) dx

### Trigonometric Substitution
1. ∫1/√(9-x²) dx
2. ∫1/(x²+4) dx
3. ∫√(x²-1)/x dx
4. ∫x²/√(1-x²) dx

### Partial Fractions
1. ∫(x+2)/(x²-4) dx
2. ∫1/(x²-5x+6) dx
3. ∫(x²+1)/(x³-x) dx
4. ∫(2x+1)/(x²+2x+1) dx

## 8. Common Mistakes to Avoid

1. **Wrong u and dv**: Use LIATE rule for integration by parts
2. **Incomplete substitution**: Make sure to substitute back completely
3. **Missing cases**: Consider all cases in partial fractions
4. **Algebraic errors**: Double-check your algebra
5. **Forgetting constants**: Remember the constant of integration

## 9. Advanced Techniques

### Tabular Integration
For repeated integration by parts, use a table:
- Differentiate u repeatedly
- Integrate dv repeatedly
- Alternate signs
- Multiply diagonally

### Weierstrass Substitution
For integrals of rational functions of sin(x) and cos(x):
Let t = tan(x/2), then:
- sin(x) = 2t/(1+t²)
- cos(x) = (1-t²)/(1+t²)
- dx = 2dt/(1+t²)

## 10. Study Tips

1. **Practice Pattern Recognition**: Learn to identify which technique to use
2. **Work Systematically**: Follow the decision tree approach
3. **Check Your Work**: Differentiate your answer to verify
4. **Build Intuition**: Understand why each technique works
5. **Use Technology**: Verify complex integrals with software

## Next Steps

After mastering these integration techniques, proceed to:
- Improper integrals
- Applications of integration
- Sequences and series
- Differential equations

Remember: The key to success is recognizing which technique to apply and practicing until the recognition becomes automatic.
