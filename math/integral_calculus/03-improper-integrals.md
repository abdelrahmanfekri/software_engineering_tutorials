# Improper Integrals

## Overview
Improper integrals extend the concept of definite integrals to cases where either the interval of integration is infinite or the integrand has infinite discontinuities. These integrals are essential for many applications in mathematics, physics, and engineering.

## Learning Objectives
- Understand the definition of improper integrals
- Learn to evaluate integrals with infinite limits
- Handle integrals with infinite discontinuities
- Apply comparison tests for convergence
- Solve real-world problems using improper integrals

## 1. Types of Improper Integrals

### Type 1: Infinite Intervals
Integrals where one or both limits of integration are infinite:

```
∫[a to ∞] f(x) dx = lim[t→∞] ∫[a to t] f(x) dx
∫[-∞ to b] f(x) dx = lim[t→-∞] ∫[t to b] f(x) dx
∫[-∞ to ∞] f(x) dx = ∫[-∞ to c] f(x) dx + ∫[c to ∞] f(x) dx
```

### Type 2: Infinite Discontinuities
Integrals where the integrand has infinite discontinuities:

```
∫[a to b] f(x) dx where f(x) → ±∞ as x → a⁺
∫[a to b] f(x) dx where f(x) → ±∞ as x → b⁻
∫[a to b] f(x) dx where f(x) → ±∞ as x → c (a < c < b)
```

## 2. Type 1 Improper Integrals

### Definition
For Type 1 improper integrals, we define:

```
∫[a to ∞] f(x) dx = lim[t→∞] ∫[a to t] f(x) dx
```

If the limit exists and is finite, the integral converges; otherwise, it diverges.

### Examples

#### Example 1: Convergent Integral
```
∫[1 to ∞] 1/x² dx
```

```
∫[1 to ∞] 1/x² dx = lim[t→∞] ∫[1 to t] 1/x² dx
= lim[t→∞] [-1/x][1 to t] = lim[t→∞] [-1/t + 1] = 1
```

Since the limit exists and is finite, the integral converges to 1.

#### Example 2: Divergent Integral
```
∫[1 to ∞] 1/x dx
```

```
∫[1 to ∞] 1/x dx = lim[t→∞] ∫[1 to t] 1/x dx
= lim[t→∞] [ln|x|][1 to t] = lim[t→∞] [ln(t) - ln(1)] = ∞
```

Since the limit is infinite, the integral diverges.

#### Example 3: Integral from -∞ to ∞
```
∫[-∞ to ∞] 1/(1+x²) dx
```

```
∫[-∞ to ∞] 1/(1+x²) dx = ∫[-∞ to 0] 1/(1+x²) dx + ∫[0 to ∞] 1/(1+x²) dx
```

Evaluate each part:
```
∫[0 to ∞] 1/(1+x²) dx = lim[t→∞] ∫[0 to t] 1/(1+x²) dx
= lim[t→∞] [arctan(x)][0 to t] = lim[t→∞] [arctan(t) - arctan(0)] = π/2
```

Similarly:
```
∫[-∞ to 0] 1/(1+x²) dx = π/2
```

Therefore:
```
∫[-∞ to ∞] 1/(1+x²) dx = π/2 + π/2 = π
```

## 3. Type 2 Improper Integrals

### Definition
For Type 2 improper integrals with infinite discontinuities:

```
∫[a to b] f(x) dx = lim[t→a⁺] ∫[t to b] f(x) dx  (if discontinuity at x = a)
∫[a to b] f(x) dx = lim[t→b⁻] ∫[a to t] f(x) dx  (if discontinuity at x = b)
```

### Examples

#### Example 1: Discontinuity at Lower Limit
```
∫[0 to 1] 1/√x dx
```

The integrand has an infinite discontinuity at x = 0.
```
∫[0 to 1] 1/√x dx = lim[t→0⁺] ∫[t to 1] 1/√x dx
= lim[t→0⁺] [2√x][t to 1] = lim[t→0⁺] [2√1 - 2√t] = 2
```

#### Example 2: Discontinuity at Upper Limit
```
∫[0 to 1] 1/(1-x) dx
```

The integrand has an infinite discontinuity at x = 1.
```
∫[0 to 1] 1/(1-x) dx = lim[t→1⁻] ∫[0 to t] 1/(1-x) dx
= lim[t→1⁻] [-ln|1-x|][0 to t] = lim[t→1⁻] [-ln|1-t| + ln|1|] = ∞
```

Since the limit is infinite, the integral diverges.

#### Example 3: Discontinuity in the Middle
```
∫[-1 to 1] 1/x² dx
```

The integrand has an infinite discontinuity at x = 0.
```
∫[-1 to 1] 1/x² dx = ∫[-1 to 0] 1/x² dx + ∫[0 to 1] 1/x² dx
```

Both integrals diverge:
```
∫[-1 to 0] 1/x² dx = lim[t→0⁻] ∫[-1 to t] 1/x² dx = ∞
∫[0 to 1] 1/x² dx = lim[t→0⁺] ∫[t to 1] 1/x² dx = ∞
```

Therefore, the original integral diverges.

## 4. Comparison Tests for Convergence

### Direct Comparison Test
If 0 ≤ f(x) ≤ g(x) for x ≥ a, then:
- If ∫[a to ∞] g(x) dx converges, then ∫[a to ∞] f(x) dx converges
- If ∫[a to ∞] f(x) dx diverges, then ∫[a to ∞] g(x) dx diverges

### Limit Comparison Test
If f(x) ≥ 0 and g(x) > 0 for x ≥ a, and lim[x→∞] f(x)/g(x) = L (0 < L < ∞), then:
- ∫[a to ∞] f(x) dx and ∫[a to ∞] g(x) dx both converge or both diverge

### Examples

#### Example 1: Direct Comparison
```
∫[1 to ∞] 1/(x² + 1) dx
```

Since 1/(x² + 1) ≤ 1/x² for x ≥ 1, and ∫[1 to ∞] 1/x² dx converges, the integral converges.

#### Example 2: Limit Comparison
```
∫[1 to ∞] (x + 1)/(x³ + 2x + 1) dx
```

Compare with g(x) = 1/x²:
```
lim[x→∞] [(x + 1)/(x³ + 2x + 1)] / [1/x²] = lim[x→∞] [x³ + x²]/[x³ + 2x + 1] = 1
```

Since ∫[1 to ∞] 1/x² dx converges, the original integral converges.

## 5. Special Cases and Techniques

### P-Integrals
The integral ∫[1 to ∞] 1/x^p dx:
- Converges if p > 1
- Diverges if p ≤ 1

### Exponential Decay
Integrals of the form ∫[0 to ∞] e^(-ax) dx converge to 1/a for a > 0.

### Trigonometric Integrals
```
∫[0 to ∞] sin(x)/x dx = π/2
∫[0 to ∞] cos(x)/x dx diverges
```

## 6. Applications of Improper Integrals

### Probability Theory
The normal distribution involves:
```
∫[-∞ to ∞] e^(-x²/2) dx = √(2π)
```

### Physics Applications
- Escape velocity calculations
- Potential energy in gravitational fields
- Electric field calculations

### Economics
- Present value of infinite income streams
- Consumer surplus calculations

## 7. Practice Problems

### Type 1 Improper Integrals
1. ∫[0 to ∞] e^(-x) dx
2. ∫[1 to ∞] 1/x³ dx
3. ∫[-∞ to ∞] 1/(x² + 4) dx
4. ∫[0 to ∞] x e^(-x²) dx

### Type 2 Improper Integrals
1. ∫[0 to 1] 1/√(1-x²) dx
2. ∫[0 to 1] ln(x) dx
3. ∫[0 to 2] 1/(x-1) dx
4. ∫[0 to 1] 1/x^(1/3) dx

### Comparison Tests
1. ∫[1 to ∞] 1/(x² + x) dx
2. ∫[1 to ∞] (x + 1)/(x³ + 1) dx
3. ∫[1 to ∞] sin²(x)/x² dx
4. ∫[1 to ∞] e^(-x²) dx

## 8. Common Mistakes to Avoid

1. **Forgetting Limits**: Always use limits for improper integrals
2. **Wrong Limit Direction**: Pay attention to which limit approaches infinity
3. **Splitting Incorrectly**: Be careful when splitting integrals with discontinuities
4. **Comparison Test Errors**: Ensure functions are positive for comparison tests
5. **Convergence vs. Divergence**: Don't confuse convergence with divergence

## 9. Advanced Topics

### Cauchy Principal Value
For integrals that don't converge in the usual sense:
```
PV ∫[-∞ to ∞] f(x) dx = lim[R→∞] ∫[-R to R] f(x) dx
```

### Laplace Transforms
Many Laplace transforms involve improper integrals:
```
L{f(t)} = ∫[0 to ∞] e^(-st) f(t) dt
```

## 10. Study Tips

1. **Understand Definitions**: Know the precise definitions of convergence and divergence
2. **Practice Limits**: Master limit calculations for improper integrals
3. **Use Comparison Tests**: Learn when and how to apply comparison tests
4. **Check Your Work**: Verify convergence/divergence with multiple methods
5. **Understand Applications**: See how improper integrals apply to real problems

## Next Steps

After mastering improper integrals, proceed to:
- Applications of integration
- Sequences and series
- Power series
- Differential equations

Remember: Improper integrals extend the power of integration to handle infinite intervals and discontinuities, making them essential tools for advanced mathematics and applications.
