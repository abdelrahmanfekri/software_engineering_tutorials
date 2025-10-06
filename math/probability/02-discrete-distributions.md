# Probability Tutorial 02: Discrete Probability Distributions

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand probability mass functions
- Work with Bernoulli and binomial distributions
- Apply geometric and negative binomial distributions
- Use the Poisson distribution
- Calculate probabilities and expected values
- Apply distributions to real-world problems

## Introduction to Discrete Distributions

### Probability Mass Function (PMF)
For a discrete random variable X, the PMF is:
p(x) = P(X = x)

**Properties**:
1. p(x) ≥ 0 for all x
2. Σ p(x) = 1 (sum over all possible values)

### Cumulative Distribution Function (CDF)
F(x) = P(X ≤ x) = Σ(t≤x) p(t)

## Bernoulli Distribution

### Definition
A Bernoulli trial has two outcomes: success (1) or failure (0).

**PMF**: p(x) = p^x(1-p)^(1-x) for x ∈ {0, 1}

**Parameters**:
- p = probability of success

**Properties**:
- E[X] = p
- Var(X) = p(1-p)

**Example**: Coin flip (p = 0.5)
- P(X = 1) = 0.5
- P(X = 0) = 0.5
- E[X] = 0.5

## Binomial Distribution

### Definition
Number of successes in n independent Bernoulli trials.

**PMF**: P(X = k) = C(n,k) p^k(1-p)^(n-k)

Where C(n,k) = n!/(k!(n-k)!)

**Parameters**:
- n = number of trials
- p = probability of success per trial

**Properties**:
- E[X] = np
- Var(X) = np(1-p)

**Example**: 10 coin flips, probability of exactly 3 heads
P(X = 3) = C(10,3) (0.5)³(0.5)⁷ = 120 × (1/8) × (1/128) = 120/1024 ≈ 0.117

### Applications
- Quality control (defective items)
- Survey responses
- Medical trials
- Sports statistics

## Geometric Distribution

### Definition
Number of trials until first success.

**PMF**: P(X = k) = (1-p)^(k-1) p for k = 1, 2, 3, ...

**Parameters**:
- p = probability of success per trial

**Properties**:
- E[X] = 1/p
- Var(X) = (1-p)/p²

**Example**: Rolling die until first 6
- p = 1/6
- E[X] = 6 (expect 6 rolls on average)
- P(X = 3) = (5/6)²(1/6) = 25/216 ≈ 0.116

### Memoryless Property
P(X > m + n | X > m) = P(X > n)

## Negative Binomial Distribution

### Definition
Number of trials until rth success.

**PMF**: P(X = k) = C(k-1, r-1) p^r(1-p)^(k-r) for k = r, r+1, ...

**Parameters**:
- r = number of successes needed
- p = probability of success per trial

**Properties**:
- E[X] = r/p
- Var(X) = r(1-p)/p²

**Example**: Rolling die until third 6
- r = 3, p = 1/6
- E[X] = 3/(1/6) = 18
- P(X = 5) = C(4,2) (1/6)³(5/6)² = 6 × (1/216) × (25/36) = 150/7776 ≈ 0.019

## Poisson Distribution

### Definition
Number of events in fixed interval of time or space.

**PMF**: P(X = k) = (λ^k e^(-λ))/k! for k = 0, 1, 2, ...

**Parameters**:
- λ = average rate of occurrence

**Properties**:
- E[X] = λ
- Var(X) = λ

**Example**: Calls to call center (λ = 5 per hour)
- P(X = 3) = (5³ e^(-5))/3! = 125e^(-5)/6 ≈ 0.140
- P(X ≤ 2) = e^(-5) + 5e^(-5) + 25e^(-5)/2 ≈ 0.125

### Poisson Approximation to Binomial
When n is large and p is small, Binomial(n,p) ≈ Poisson(np)

**Rule of thumb**: n ≥ 20, p ≤ 0.05, np ≤ 5

## Hypergeometric Distribution

### Definition
Number of successes in n draws without replacement from finite population.

**PMF**: P(X = k) = C(K,k) C(N-K, n-k) / C(N,n)

**Parameters**:
- N = population size
- K = number of successes in population
- n = sample size

**Properties**:
- E[X] = nK/N
- Var(X) = n(K/N)(1-K/N)(N-n)/(N-1)

**Example**: Drawing 5 cards from deck of 52, probability of 2 aces
- N = 52, K = 4, n = 5, k = 2
- P(X = 2) = C(4,2) C(48,3) / C(52,5) = 6 × 17296 / 2598960 ≈ 0.040

## Applications

### Quality Control
**Binomial**: Number of defective items in sample
**Hypergeometric**: Sampling without replacement

### Reliability
**Geometric**: Time to first failure
**Negative Binomial**: Time to rth failure

### Service Systems
**Poisson**: Arrivals at service facility
**Binomial**: Customers choosing service

## Practice Problems

### Problem 1
A fair coin is flipped 8 times. Find the probability of getting exactly 5 heads.

**Solution**:
Binomial distribution: n = 8, p = 0.5, k = 5
P(X = 5) = C(8,5) (0.5)⁵(0.5)³ = 56 × (1/32) × (1/8) = 56/256 = 7/32 ≈ 0.219

### Problem 2
A manufacturing process produces 5% defective items. In a sample of 20 items, what's the probability of at most 2 defectives?

**Solution**:
Binomial distribution: n = 20, p = 0.05
P(X ≤ 2) = P(X = 0) + P(X = 1) + P(X = 2)
= C(20,0)(0.05)⁰(0.95)²⁰ + C(20,1)(0.05)¹(0.95)¹⁹ + C(20,2)(0.05)²(0.95)¹⁸
= (0.95)²⁰ + 20(0.05)(0.95)¹⁹ + 190(0.05)²(0.95)¹⁸
≈ 0.358 + 0.377 + 0.189 = 0.924

### Problem 3
Calls arrive at a call center at rate 3 per minute. What's the probability of exactly 2 calls in the next minute?

**Solution**:
Poisson distribution: λ = 3, k = 2
P(X = 2) = (3² e^(-3))/2! = 9e^(-3)/2 ≈ 0.224

### Problem 4
A student takes a multiple-choice test with 20 questions, each with 4 choices. If guessing randomly, what's the expected number of correct answers?

**Solution**:
Binomial distribution: n = 20, p = 0.25
E[X] = np = 20 × 0.25 = 5

## Key Takeaways
- Discrete distributions model countable outcomes
- Each distribution has specific applications
- Parameters determine shape and properties
- Expected values and variances are important characteristics
- Real-world problems often fit specific distributions

## Next Steps
In the next tutorial, we'll explore continuous probability distributions, learning about probability density functions and important continuous distributions like normal, exponential, and gamma distributions.
