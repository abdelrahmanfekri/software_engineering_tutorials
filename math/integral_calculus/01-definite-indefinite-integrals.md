# Definite and Indefinite Integrals

## Overview
This tutorial covers the fundamental concepts of definite and indefinite integrals, including their definitions, properties, and basic integration rules. Understanding these concepts is essential for all subsequent topics in integral calculus.

## Learning Objectives
- Understand the definition of the integral as the inverse of differentiation
- Learn the fundamental theorem of calculus
- Master basic integration rules and properties
- Apply integration by substitution
- Solve problems involving definite and indefinite integrals

## 1. Definition of the Integral

### Indefinite Integral
The indefinite integral of a function f(x) is the collection of all antiderivatives of f(x):

```
∫f(x) dx = F(x) + C
```

Where:
- F(x) is any antiderivative of f(x)
- C is the constant of integration
- The symbol ∫ is the integral sign
- f(x) is the integrand
- dx indicates the variable of integration

### Definite Integral
The definite integral of f(x) from a to b is defined as:

```
∫[a to b] f(x) dx = F(b) - F(a)
```

Where F(x) is any antiderivative of f(x).

## 2. Fundamental Theorem of Calculus

### Part I: Differentiation of Integrals
If f is continuous on [a,b] and F(x) = ∫[a to x] f(t) dt, then:
```
F'(x) = f(x)
```

### Part II: Integration of Derivatives
If f is continuous on [a,b] and F is any antiderivative of f, then:
```
∫[a to b] f(x) dx = F(b) - F(a)
```

## 3. Properties of Integrals

### Basic Properties
1. **Constant Multiple Rule**: ∫cf(x) dx = c∫f(x) dx
2. **Sum Rule**: ∫[f(x) + g(x)] dx = ∫f(x) dx + ∫g(x) dx
3. **Difference Rule**: ∫[f(x) - g(x)] dx = ∫f(x) dx - ∫g(x) dx

### Definite Integral Properties
1. **Additivity**: ∫[a to b] f(x) dx + ∫[b to c] f(x) dx = ∫[a to c] f(x) dx
2. **Reversing Limits**: ∫[a to b] f(x) dx = -∫[b to a] f(x) dx
3. **Zero Width**: ∫[a to a] f(x) dx = 0
4. **Constant Function**: ∫[a to b] c dx = c(b - a)

## 4. Basic Integration Rules

### Power Rule
```
∫x^n dx = x^(n+1)/(n+1) + C, where n ≠ -1
```

### Exponential Functions
```
∫e^x dx = e^x + C
∫a^x dx = a^x/ln(a) + C, where a > 0, a ≠ 1
```

### Logarithmic Functions
```
∫1/x dx = ln|x| + C
∫1/(x ln(a)) dx = log_a|x| + C
```

### Trigonometric Functions
```
∫sin(x) dx = -cos(x) + C
∫cos(x) dx = sin(x) + C
∫sec²(x) dx = tan(x) + C
∫csc²(x) dx = -cot(x) + C
∫sec(x)tan(x) dx = sec(x) + C
∫csc(x)cot(x) dx = -csc(x) + C
```

### Inverse Trigonometric Functions
```
∫1/√(1-x²) dx = arcsin(x) + C
∫1/(1+x²) dx = arctan(x) + C
∫1/(x√(x²-1)) dx = arcsec(x) + C
```

## 5. Integration by Substitution

### The Substitution Method
Integration by substitution is the reverse of the chain rule:

1. Identify a function g(x) inside f(x) whose derivative g'(x) is also present
2. Let u = g(x), then du = g'(x)dx
3. Rewrite the integral in terms of u
4. Integrate with respect to u
5. Substitute back to get the answer in terms of x

### Examples

#### Example 1: Basic Substitution
```
∫(2x + 1)³ dx
```

Let u = 2x + 1, then du = 2dx, so dx = du/2
```
∫(2x + 1)³ dx = ∫u³ (du/2) = (1/2)∫u³ du = (1/2)(u⁴/4) + C = (2x + 1)⁴/8 + C
```

#### Example 2: Trigonometric Substitution
```
∫sin(3x) dx
```

Let u = 3x, then du = 3dx, so dx = du/3
```
∫sin(3x) dx = ∫sin(u) (du/3) = (1/3)∫sin(u) du = (1/3)(-cos(u)) + C = -cos(3x)/3 + C
```

## 6. Definite Integrals with Substitution

When using substitution with definite integrals, you can either:
1. Change the limits of integration to match the new variable
2. Substitute back to the original variable before evaluating

### Method 1: Change Limits
```
∫[0 to 2] (2x + 1)³ dx
```

Let u = 2x + 1, then du = 2dx
When x = 0, u = 1; when x = 2, u = 5
```
∫[0 to 2] (2x + 1)³ dx = ∫[1 to 5] u³ (du/2) = (1/2)[u⁴/4][1 to 5] = (1/8)[5⁴ - 1⁴] = 624/8 = 78
```

## 7. Common Integration Patterns

### Pattern Recognition
Learn to recognize these common patterns:

1. **Chain Rule Pattern**: f'(x)[f(x)]^n
2. **Product Pattern**: f(x)g'(x) + f'(x)g(x)
3. **Quotient Pattern**: f'(x)/f(x)
4. **Trigonometric Patterns**: sin^n(x)cos^m(x)

## 8. Practice Problems

### Basic Integration
1. ∫x⁴ dx
2. ∫(3x² - 2x + 5) dx
3. ∫e^(2x) dx
4. ∫sin(2x) dx

### Substitution Problems
1. ∫(x² + 1)³(2x) dx
2. ∫cos(5x) dx
3. ∫e^(x²)(2x) dx
4. ∫1/(2x + 3) dx

### Definite Integrals
1. ∫[0 to 1] x³ dx
2. ∫[0 to π] sin(x) dx
3. ∫[1 to 2] (x² + 1) dx

## 9. Common Mistakes to Avoid

1. **Forgetting the Constant**: Always add +C for indefinite integrals
2. **Incorrect Substitution**: Make sure du matches the derivative of u
3. **Wrong Limits**: When changing variables, update the limits accordingly
4. **Algebraic Errors**: Be careful with signs and arithmetic

## 10. Applications Preview

Understanding definite and indefinite integrals is crucial for:
- Finding areas under curves
- Calculating volumes of revolution
- Solving differential equations
- Computing work and energy
- Analyzing accumulation problems

## Study Tips

1. **Practice Regularly**: Work through many examples to build fluency
2. **Understand Patterns**: Learn to recognize common integration patterns
3. **Check Your Work**: Differentiate your answer to verify correctness
4. **Use Technology**: Verify answers with calculators or software
5. **Build Intuition**: Understand what integrals represent geometrically

## Next Steps

After mastering definite and indefinite integrals, proceed to:
- Advanced integration techniques (integration by parts, partial fractions)
- Applications of integration (area, volume, work)
- Improper integrals
- Sequences and series

Remember: Integration is about accumulation and finding antiderivatives. Build a strong foundation with these basic concepts before moving to more advanced topics.
