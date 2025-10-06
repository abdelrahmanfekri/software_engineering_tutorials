# Probability Tutorial 01: Basic Probability Concepts

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand sample spaces and events
- Apply the axioms of probability
- Calculate conditional probability
- Determine independence of events
- Use Bayes' theorem
- Solve probability problems systematically

## Introduction to Probability

### What is Probability?
Probability is a measure of the likelihood that an event will occur. It quantifies uncertainty and ranges from 0 (impossible) to 1 (certain).

**Example**: The probability of rolling a 6 on a fair die is 1/6 ≈ 0.167

### Sample Spaces
The sample space S is the set of all possible outcomes of an experiment.

**Examples**:
- Rolling a die: S = {1, 2, 3, 4, 5, 6}
- Flipping two coins: S = {HH, HT, TH, TT}
- Measuring temperature: S = [0, 100] (continuous)

### Events
An event is a subset of the sample space.

**Examples**:
- Rolling an even number: E = {2, 4, 6}
- Getting at least one head: E = {HH, HT, TH}
- Temperature above 80°F: E = (80, 100]

## Axioms of Probability

### Axiom 1: Non-negativity
P(A) ≥ 0 for any event A

### Axiom 2: Normalization
P(S) = 1

### Axiom 3: Additivity
For mutually exclusive events A₁, A₂, ..., Aₙ:
P(A₁ ∪ A₂ ∪ ... ∪ Aₙ) = P(A₁) + P(A₂) + ... + P(Aₙ)

### Consequences
1. P(∅) = 0
2. P(Aᶜ) = 1 - P(A)
3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

## Basic Probability Rules

### Complement Rule
P(Aᶜ) = 1 - P(A)

**Example**: If P(rain) = 0.3, then P(no rain) = 1 - 0.3 = 0.7

### Addition Rule
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

**Example**: In a class of 30 students, 15 like math, 12 like science, and 8 like both.
- P(math ∪ science) = P(math) + P(science) - P(math ∩ science)
- P(math ∪ science) = 15/30 + 12/30 - 8/30 = 19/30

### Multiplication Rule
P(A ∩ B) = P(A) · P(B|A) = P(B) · P(A|B)

## Conditional Probability

### Definition
The conditional probability of A given B is:
P(A|B) = P(A ∩ B)/P(B), provided P(B) > 0

**Example**: In a deck of cards, what's the probability of drawing a king given that you drew a red card?
- P(king|red) = P(king ∩ red)/P(red) = (2/52)/(26/52) = 2/26 = 1/13

### Interpreting Conditional Probability
P(A|B) represents the probability of A occurring when we know B has occurred.

## Independence

### Definition
Events A and B are independent if:
P(A ∩ B) = P(A) · P(B)

This is equivalent to:
P(A|B) = P(A) or P(B|A) = P(B)

**Example**: Are "rolling a 6" and "rolling an even number" independent?
- P(6) = 1/6
- P(even) = 3/6 = 1/2
- P(6 ∩ even) = P(6) = 1/6
- P(6) · P(even) = (1/6)(1/2) = 1/12 ≠ 1/6
- Therefore, they are NOT independent

## Bayes' Theorem

### Statement
P(A|B) = P(B|A) · P(A)/P(B)

### Extended Form
P(Aᵢ|B) = P(B|Aᵢ) · P(Aᵢ)/Σⱼ P(B|Aⱼ) · P(Aⱼ)

**Example**: A test is 95% accurate for disease detection. The disease affects 1% of the population. If someone tests positive, what's the probability they have the disease?

**Solution**:
- P(disease) = 0.01
- P(positive|disease) = 0.95
- P(positive|no disease) = 0.05
- P(positive) = P(positive|disease)P(disease) + P(positive|no disease)P(no disease)
- P(positive) = 0.95(0.01) + 0.05(0.99) = 0.0095 + 0.0495 = 0.059
- P(disease|positive) = P(positive|disease)P(disease)/P(positive)
- P(disease|positive) = 0.95(0.01)/0.059 ≈ 0.161

## Counting Principles

### Multiplication Principle
If there are m ways to do one thing and n ways to do another, there are m·n ways to do both.

**Example**: 3 shirts × 4 pants = 12 outfits

### Permutations
Number of ways to arrange r objects from n distinct objects:
P(n,r) = n!/(n-r)!

**Example**: How many ways to arrange 3 books from 5?
P(5,3) = 5!/(5-3)! = 5!/2! = 60

### Combinations
Number of ways to choose r objects from n distinct objects:
C(n,r) = n!/(r!(n-r)!)

**Example**: How many ways to choose 3 books from 5?
C(5,3) = 5!/(3!2!) = 10

## Practice Problems

### Problem 1
A bag contains 5 red, 3 blue, and 2 green marbles. What's the probability of drawing a red marble?

**Solution**:
P(red) = 5/(5+3+2) = 5/10 = 1/2

### Problem 2
Two dice are rolled. What's the probability of getting a sum of 7?

**Solution**:
Sample space has 36 outcomes. Sum of 7 occurs for: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
P(sum = 7) = 6/36 = 1/6

### Problem 3
In a group of 100 people, 60 like coffee, 40 like tea, and 20 like both. What's the probability that a randomly selected person likes coffee or tea?

**Solution**:
P(coffee ∪ tea) = P(coffee) + P(tea) - P(coffee ∩ tea)
P(coffee ∪ tea) = 60/100 + 40/100 - 20/100 = 80/100 = 4/5

### Problem 4
A fair coin is flipped 3 times. What's the probability of getting exactly 2 heads?

**Solution**:
Sample space: {HHH, HHT, HTH, THH, HTT, THT, TTH, TTT}
Favorable outcomes: {HHT, HTH, THH}
P(exactly 2 heads) = 3/8

## Key Takeaways
- Probability measures likelihood from 0 to 1
- Sample spaces contain all possible outcomes
- Events are subsets of sample spaces
- Conditional probability updates beliefs given new information
- Independence means events don't affect each other
- Bayes' theorem relates conditional probabilities
- Counting principles help calculate probabilities

## Next Steps
In the next tutorial, we'll explore discrete probability distributions, learning about probability mass functions and common distributions like binomial and Poisson.
