# Probability Basics

## Introduction
Probability is the mathematical framework for reasoning about uncertainty. It provides tools to quantify and manipulate uncertainty in a rigorous way.

## Core Concepts

### Sample Space & Events
- **Sample Space (Ω)**: Set of all possible outcomes
- **Event**: Subset of sample space
- **Probability Function P()**: Maps events to [0,1] with:
  - P(Ω) = 1
  - P(∅) = 0
  - P(A∪B) = P(A) + P(B) for disjoint events

### Conditional Probability
Probability of event A given that event B has occurred:

P(A|B) = P(A∩B) / P(B)

**Key Insight**: Conditioning reduces uncertainty by incorporating new information.

### Independence
Events A and B are independent if:
P(A∩B) = P(A) × P(B)

This means: P(A|B) = P(A) and P(B|A) = P(B)

## Bayes' Theorem

### Basic Form
P(A|B) = P(B|A) × P(A) / P(B)

### Extended Form
P(A|B) = P(B|A) × P(A) / Σ P(B|Aᵢ) × P(Aᵢ)

### Interpretation
- **Prior P(A)**: Initial belief about A
- **Likelihood P(B|A)**: How well A explains B
- **Posterior P(A|B)**: Updated belief after observing B
- **Evidence P(B)**: Total probability of observing B

## Key Properties

### Law of Total Probability
P(B) = Σ P(B|Aᵢ) × P(Aᵢ)

### Chain Rule
P(A₁∩A₂∩...∩Aₙ) = P(A₁) × P(A₂|A₁) × P(A₃|A₁∩A₂) × ... × P(Aₙ|A₁∩...∩Aₙ₋₁)

### Bayes' Rule for Multiple Hypotheses
P(Hᵢ|E) = P(E|Hᵢ) × P(Hᵢ) / Σ P(E|Hⱼ) × P(Hⱼ)

## Examples

### Medical Testing
- Disease prevalence: P(D) = 0.01
- Test sensitivity: P(T+|D) = 0.95
- Test specificity: P(T-|¬D) = 0.90

**Question**: What's P(D|T+)?
**Answer**: Use Bayes' theorem to find posterior probability

### Monty Hall Problem
- 3 doors, 1 prize
- You pick door 1
- Host opens door 3 (no prize)
- Should you switch to door 2?

**Solution**: P(Prize behind door 2 | Host opens door 3) = 2/3

## Common Fallacies

### Prosecutor's Fallacy
Confusing P(E|H) with P(H|E)

### Base Rate Neglect
Ignoring prior probabilities when interpreting evidence

### Gambler's Fallacy
Believing independent events are dependent

## Practice Problems

1. **Coin Flips**: 3 fair coins. What's P(exactly 2 heads | first flip is heads)?

2. **Disease Testing**: Given sensitivity 0.9, specificity 0.8, prevalence 0.1, find P(disease | positive test)

3. **Conditional Independence**: Show that if A⊥B|C, then P(A∩B|C) = P(A|C) × P(B|C)

## Next Steps
- Probability distributions
- Random variables
- Expectation and variance
