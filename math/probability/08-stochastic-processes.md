# Probability Tutorial 08: Stochastic Processes

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand Markov chains and their properties
- Work with Poisson processes
- Apply Brownian motion concepts
- Analyze random walks
- Use stochastic processes in applications
- Calculate transition probabilities and steady states

## Introduction to Stochastic Processes

### Definition
A stochastic process {X(t), t ∈ T} is a collection of random variables indexed by time (or space).

**Examples**:
- Stock prices over time
- Number of customers in a queue
- Temperature measurements
- Random walk positions

### Types of Stochastic Processes
1. **Discrete time, discrete state**: Markov chains
2. **Continuous time, discrete state**: Poisson processes
3. **Continuous time, continuous state**: Brownian motion
4. **Discrete time, continuous state**: ARMA processes

## Markov Chains

### Definition
A Markov chain is a stochastic process where the future depends only on the present, not the past:
P(Xₙ₊₁ = j | Xₙ = i, Xₙ₋₁ = iₙ₋₁, ..., X₁ = i₁) = P(Xₙ₊₁ = j | Xₙ = i)

### Transition Matrix
P = [pᵢⱼ] where pᵢⱼ = P(Xₙ₊₁ = j | Xₙ = i)

**Properties**:
- pᵢⱼ ≥ 0 for all i, j
- Σⱼ pᵢⱼ = 1 for all i

**Example**: Weather model
```
        Sunny  Rainy
Sunny    0.7    0.3
Rainy    0.4    0.6
```

### n-Step Transition Probabilities
P^(n) = P^n (matrix power)

**Chapman-Kolmogorov Equations**:
pᵢⱼ^(n+m) = Σₖ pᵢₖ^(n) pₖⱼ^(m)

### Classification of States

**Communicating States**: i ↔ j if i is reachable from j and vice versa

**Recurrent vs Transient**:
- Recurrent: P(return to state) = 1
- Transient: P(return to state) < 1

**Periodicity**:
- Period d: gcd{n: pᵢᵢ^(n) > 0}
- Aperiodic: d = 1

### Stationary Distribution
A probability vector π is stationary if π = πP

**Finding Stationary Distribution**:
Solve the system: πP = π with Σᵢ πᵢ = 1

**Example**: For the weather model above
π₁(0.7) + π₂(0.4) = π₁
π₁(0.3) + π₂(0.6) = π₂
π₁ + π₂ = 1

Solution: π₁ = 4/7, π₂ = 3/7

### Ergodic Theorem
For irreducible, aperiodic Markov chains:
lim(n→∞) pᵢⱼ^(n) = πⱼ

## Poisson Processes

### Definition
A Poisson process {N(t), t ≥ 0} counts events occurring randomly in time.

**Properties**:
1. N(0) = 0
2. Independent increments
3. Stationary increments
4. P(N(t+h) - N(t) = 1) = λh + o(h)
5. P(N(t+h) - N(t) ≥ 2) = o(h)

### Poisson Distribution
N(t) ~ Poisson(λt)
P(N(t) = k) = (λt)^k e^(-λt) / k!

**Properties**:
- E[N(t)] = λt
- Var(N(t)) = λt
- Interarrival times ~ Exponential(λ)

### Interarrival Times
Let Tᵢ be the time between the (i-1)th and ith events.
T₁, T₂, ... ~ Exponential(λ) independent

### Arrival Times
Sₙ = T₁ + T₂ + ... + Tₙ (time of nth event)
Sₙ ~ Gamma(n, λ)

### Superposition
If N₁(t) ~ Poisson(λ₁t) and N₂(t) ~ Poisson(λ₂t) are independent:
N₁(t) + N₂(t) ~ Poisson((λ₁ + λ₂)t)

### Thinning
If events are Poisson(λt) and each is kept with probability p:
Kept events ~ Poisson(λpt)

## Brownian Motion

### Definition
Brownian motion {B(t), t ≥ 0} is a continuous-time stochastic process with:

**Properties**:
1. B(0) = 0
2. Independent increments
3. Stationary increments
4. B(t) ~ Normal(0, t)
5. Continuous sample paths

### Standard Brownian Motion
- E[B(t)] = 0
- Var(B(t)) = t
- Cov(B(s), B(t)) = min(s, t)

### Geometric Brownian Motion
S(t) = S(0) exp((μ - σ²/2)t + σB(t))

Used in finance for stock prices.

### Properties
- B(t) - B(s) ~ Normal(0, t-s)
- B(t) is nowhere differentiable
- Quadratic variation: [B, B](t) = t

## Random Walks

### Simple Random Walk
Sₙ = X₁ + X₂ + ... + Xₙ where Xᵢ = ±1 with equal probability

**Properties**:
- E[Sₙ] = 0
- Var(Sₙ) = n
- Sₙ/√n → N(0, 1) by CLT

### Gambler's Ruin
Starting with $a, betting $1 each time with probability p of winning:
P(ruin) = ((1-p)/p)^a / (1 + ((1-p)/p)^a) if p ≠ 1/2
P(ruin) = 1/2 if p = 1/2

### First Passage Time
Time to first reach level b:
E[T] = ∞ (for symmetric random walk)

## Applications

### Queueing Theory
- M/M/1 queue: Poisson arrivals, exponential service
- Little's Law: L = λW
- Steady-state probabilities

### Finance
- Black-Scholes model
- Option pricing
- Risk management
- Portfolio optimization

### Biology
- Population genetics
- Epidemiology
- Molecular motion
- Evolution

### Engineering
- Reliability analysis
- Signal processing
- Control systems
- Network analysis

## Practice Problems

### Problem 1
A Markov chain has transition matrix:
```
     A    B    C
A   0.5  0.3  0.2
B   0.2  0.6  0.2
C   0.1  0.4  0.5
```
Find the stationary distribution.

**Solution**:
Solve πP = π:
π₁(0.5) + π₂(0.2) + π₃(0.1) = π₁
π₁(0.3) + π₂(0.6) + π₃(0.4) = π₂
π₁(0.2) + π₂(0.2) + π₃(0.5) = π₃
π₁ + π₂ + π₃ = 1

Solution: π₁ = 0.2, π₂ = 0.4, π₃ = 0.4

### Problem 2
Customers arrive at a bank according to a Poisson process with rate 10 per hour. What's the probability of exactly 3 customers in 30 minutes?

**Solution**:
N(0.5) ~ Poisson(10 × 0.5) = Poisson(5)
P(N(0.5) = 3) = 5³ e^(-5) / 3! = 125e^(-5) / 6 ≈ 0.140

### Problem 3
For standard Brownian motion B(t), find P(B(1) > 1).

**Solution**:
B(1) ~ Normal(0, 1)
P(B(1) > 1) = P(Z > 1) ≈ 0.159

### Problem 4
A symmetric random walk starts at 0. What's the probability of reaching 5 before reaching -3?

**Solution**:
Using gambler's ruin with a = 3, b = 5, p = 1/2:
P(reach 5 first) = a/(a+b) = 3/8

## Key Takeaways
- Stochastic processes model random phenomena over time
- Markov chains have memoryless property
- Poisson processes model random event occurrences
- Brownian motion provides continuous random motion
- Random walks are discrete analogs of Brownian motion
- Each process has specific applications
- Stationary distributions characterize long-term behavior
- Limit theorems connect discrete and continuous processes

## Conclusion
Stochastic processes provide powerful tools for modeling uncertainty in time-dependent systems. They form the foundation for many applications in finance, biology, engineering, and other fields. Understanding their properties and applications enables better decision-making under uncertainty.

The key is to recognize which type of process best models your situation and then apply the appropriate mathematical tools to analyze and make predictions about the system's behavior.
