# Sequences and Series

## Overview
This tutorial covers sequences and infinite series, which are fundamental concepts in calculus and analysis. Understanding convergence and divergence of series is essential for advanced mathematics and many applications in science and engineering.

## Learning Objectives
- Understand sequences and their limits
- Learn about infinite series and convergence
- Master convergence tests for series
- Work with geometric and p-series
- Apply series to solve problems

## 1. Sequences

### Definition
A sequence is an ordered list of numbers: {a₁, a₂, a₃, ...} or {aₙ}

### Limit of a Sequence
A sequence {aₙ} converges to L if:
```
lim[n→∞] aₙ = L
```

If the limit exists and is finite, the sequence converges; otherwise, it diverges.

### Examples

#### Example 1: Convergent Sequence
```
aₙ = 1/n
lim[n→∞] 1/n = 0
```
The sequence converges to 0.

#### Example 2: Divergent Sequence
```
aₙ = n²
lim[n→∞] n² = ∞
```
The sequence diverges to infinity.

#### Example 3: Oscillating Sequence
```
aₙ = (-1)ⁿ
```
The sequence oscillates between -1 and 1, so it diverges.

### Properties of Sequence Limits
1. **Sum Rule**: lim[n→∞] (aₙ + bₙ) = lim[n→∞] aₙ + lim[n→∞] bₙ
2. **Product Rule**: lim[n→∞] (aₙ · bₙ) = lim[n→∞] aₙ · lim[n→∞] bₙ
3. **Quotient Rule**: lim[n→∞] (aₙ/bₙ) = lim[n→∞] aₙ / lim[n→∞] bₙ (if bₙ ≠ 0)

## 2. Infinite Series

### Definition
An infinite series is the sum of infinitely many terms:
```
∑[n=1 to ∞] aₙ = a₁ + a₂ + a₃ + ...
```

### Partial Sums
The nth partial sum is:
```
Sₙ = ∑[k=1 to n] aₖ = a₁ + a₂ + ... + aₙ
```

### Convergence of Series
A series converges if the sequence of partial sums converges:
```
∑[n=1 to ∞] aₙ converges if lim[n→∞] Sₙ exists and is finite
```

### Examples

#### Example 1: Geometric Series
```
∑[n=0 to ∞] arⁿ = a + ar + ar² + ar³ + ...
```

Converges if |r| < 1, diverges if |r| ≥ 1.
If convergent: S = a/(1-r)

#### Example 2: Harmonic Series
```
∑[n=1 to ∞] 1/n = 1 + 1/2 + 1/3 + 1/4 + ...
```
This series diverges (though it grows very slowly).

## 3. Geometric Series

### Formula
```
∑[n=0 to ∞] arⁿ = a/(1-r)  if |r| < 1
```

### Examples

#### Example 1: Basic Geometric Series
```
∑[n=0 to ∞] (1/2)ⁿ = 1 + 1/2 + 1/4 + 1/8 + ... = 1/(1-1/2) = 2
```

#### Example 2: Starting from Different Index
```
∑[n=2 to ∞] 3ⁿ/4ⁿ = ∑[n=2 to ∞] (3/4)ⁿ = (3/4)²/(1-3/4) = (9/16)/(1/4) = 9/4
```

#### Example 3: Alternating Geometric Series
```
∑[n=0 to ∞] (-1)ⁿ/2ⁿ = ∑[n=0 to ∞] (-1/2)ⁿ = 1/(1-(-1/2)) = 1/(3/2) = 2/3
```

## 4. P-Series

### Definition
A p-series is:
```
∑[n=1 to ∞] 1/nᵖ
```

### Convergence
- Converges if p > 1
- Diverges if p ≤ 1

### Examples

#### Example 1: Convergent P-Series
```
∑[n=1 to ∞] 1/n²
```
Since p = 2 > 1, this series converges.

#### Example 2: Divergent P-Series
```
∑[n=1 to ∞] 1/√n = ∑[n=1 to ∞] 1/n^(1/2)
```
Since p = 1/2 ≤ 1, this series diverges.

## 5. Convergence Tests

### Divergence Test
If lim[n→∞] aₙ ≠ 0, then ∑[n=1 to ∞] aₙ diverges.

**Note**: If lim[n→∞] aₙ = 0, the test is inconclusive.

### Integral Test
If f(x) is positive, continuous, and decreasing for x ≥ 1, then:
- ∑[n=1 to ∞] f(n) converges if and only if ∫[1 to ∞] f(x) dx converges

#### Example: Using Integral Test
```
∑[n=1 to ∞] 1/n²
```

Let f(x) = 1/x²
```
∫[1 to ∞] 1/x² dx = lim[t→∞] ∫[1 to t] 1/x² dx = lim[t→∞] [-1/x][1 to t] = 1
```

Since the integral converges, the series converges.

### Comparison Test
If 0 ≤ aₙ ≤ bₙ for all n:
- If ∑[n=1 to ∞] bₙ converges, then ∑[n=1 to ∞] aₙ converges
- If ∑[n=1 to ∞] aₙ diverges, then ∑[n=1 to ∞] bₙ diverges

#### Example: Using Comparison Test
```
∑[n=1 to ∞] 1/(n² + 1)
```

Since 1/(n² + 1) ≤ 1/n² and ∑[n=1 to ∞] 1/n² converges, the series converges.

### Limit Comparison Test
If aₙ ≥ 0 and bₙ > 0 for all n, and lim[n→∞] aₙ/bₙ = L (0 < L < ∞):
- ∑[n=1 to ∞] aₙ and ∑[n=1 to ∞] bₙ both converge or both diverge

#### Example: Using Limit Comparison Test
```
∑[n=1 to ∞] (n + 1)/(n³ + 1)
```

Compare with ∑[n=1 to ∞] 1/n²:
```
lim[n→∞] [(n + 1)/(n³ + 1)] / [1/n²] = lim[n→∞] [n³ + n²]/[n³ + 1] = 1
```

Since ∑[n=1 to ∞] 1/n² converges, the original series converges.

### Ratio Test
If lim[n→∞] |aₙ₊₁/aₙ| = L:
- If L < 1, the series converges absolutely
- If L > 1, the series diverges
- If L = 1, the test is inconclusive

#### Example: Using Ratio Test
```
∑[n=1 to ∞] n!/nⁿ
```

```
lim[n→∞] |(n+1)!/(n+1)^(n+1)| / |n!/nⁿ| = lim[n→∞] [(n+1)!nⁿ]/[n!(n+1)^(n+1)]
= lim[n→∞] (n+1)nⁿ/(n+1)^(n+1) = lim[n→∞] nⁿ/(n+1)ⁿ = lim[n→∞] (n/(n+1))ⁿ = 1/e < 1
```

Since L = 1/e < 1, the series converges.

### Root Test
If lim[n→∞] |aₙ|^(1/n) = L:
- If L < 1, the series converges absolutely
- If L > 1, the series diverges
- If L = 1, the test is inconclusive

### Alternating Series Test
If aₙ > 0, aₙ₊₁ ≤ aₙ for all n, and lim[n→∞] aₙ = 0:
- ∑[n=1 to ∞] (-1)ⁿaₙ converges

#### Example: Alternating Harmonic Series
```
∑[n=1 to ∞] (-1)ⁿ/n
```

Since 1/n > 1/(n+1) and lim[n→∞] 1/n = 0, the series converges.

## 6. Absolute and Conditional Convergence

### Absolute Convergence
A series ∑[n=1 to ∞] aₙ converges absolutely if ∑[n=1 to ∞] |aₙ| converges.

### Conditional Convergence
A series converges conditionally if it converges but does not converge absolutely.

### Examples

#### Example 1: Absolutely Convergent
```
∑[n=1 to ∞] (-1)ⁿ/n²
```

Since ∑[n=1 to ∞] 1/n² converges, the series converges absolutely.

#### Example 2: Conditionally Convergent
```
∑[n=1 to ∞] (-1)ⁿ/n
```

The series converges (alternating series test), but ∑[n=1 to ∞] 1/n diverges, so it converges conditionally.

## 7. Practice Problems

### Sequence Limits
1. lim[n→∞] (n² + 1)/(2n² - 3)
2. lim[n→∞] (1 + 1/n)ⁿ
3. lim[n→∞] sin(n)/n
4. lim[n→∞] n!/nⁿ

### Series Convergence
1. ∑[n=1 to ∞] 1/(n² + 2n)
2. ∑[n=1 to ∞] n/(n² + 1)
3. ∑[n=1 to ∞] (-1)ⁿ/√n
4. ∑[n=1 to ∞] n²/2ⁿ

### Geometric Series
1. ∑[n=0 to ∞] (2/3)ⁿ
2. ∑[n=2 to ∞] (1/4)ⁿ
3. ∑[n=0 to ∞] (-1)ⁿ/3ⁿ
4. ∑[n=1 to ∞] 5ⁿ/6ⁿ

### Convergence Tests
1. Use integral test for ∑[n=1 to ∞] 1/n³
2. Use comparison test for ∑[n=1 to ∞] 1/(n² + n)
3. Use ratio test for ∑[n=1 to ∞] n!/nⁿ
4. Use alternating series test for ∑[n=1 to ∞] (-1)ⁿ/(n + 1)

## 8. Common Mistakes to Avoid

1. **Divergence Test**: Remember that lim[n→∞] aₙ = 0 doesn't guarantee convergence
2. **Test Selection**: Choose the most appropriate test for each series
3. **Absolute vs. Conditional**: Distinguish between absolute and conditional convergence
4. **Geometric Series**: Check the condition |r| < 1 for geometric series
5. **P-Series**: Remember p > 1 for convergence

## 9. Applications

### Decimal Representations
Every decimal number can be written as a geometric series:
```
0.333... = 3/10 + 3/100 + 3/1000 + ... = (3/10)/(1-1/10) = 1/3
```

### Probability
Geometric series appear in probability calculations, especially in expected value problems.

### Physics
Series are used in:
- Fourier analysis
- Quantum mechanics
- Signal processing
- Approximation methods

## 10. Study Tips

1. **Learn Tests**: Master the convergence tests and when to use each
2. **Practice Recognition**: Learn to quickly identify series types
3. **Check Conditions**: Always verify the conditions for each test
4. **Use Multiple Tests**: Sometimes more than one test applies
5. **Understand Convergence**: Distinguish between different types of convergence

## Next Steps

After mastering sequences and series, proceed to:
- Power series
- Taylor and Maclaurin series
- Parametric and polar integration
- Differential equations

Remember: Series are fundamental tools in calculus and analysis. Understanding convergence is crucial for advanced mathematics and many applications in science and engineering.
