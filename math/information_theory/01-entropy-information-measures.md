# Information Theory Tutorial 01: Entropy and Information Measures

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the concept of entropy and its intuitive meaning
- Calculate entropy for discrete and continuous distributions
- Apply entropy properties and inequalities
- Understand conditional entropy and chain rules
- Calculate mutual information between variables
- Apply information measures to machine learning problems

## Introduction to Entropy

### What is Entropy?
Entropy measures the "uncertainty" or "surprise" in a random variable. It quantifies how much information we expect to gain when we observe the outcome of a random process.

**Intuitive Definition**: Entropy measures how "spread out" or "unpredictable" a probability distribution is.

### Shannon Entropy (Discrete Case)
For a discrete random variable X with probability mass function p(x):

H(X) = -Σ p(x) log p(x)

Where the sum is over all possible values of x.

**Properties**:
- H(X) ≥ 0 (non-negative)
- H(X) = 0 if and only if X is deterministic (one outcome has probability 1)
- H(X) is maximized when all outcomes are equally likely

### Examples

**Example 1**: Fair coin flip
- p(Heads) = 0.5, p(Tails) = 0.5
- H(X) = -(0.5 log 0.5 + 0.5 log 0.5) = -(-0.5 - 0.5) = 1 bit

**Example 2**: Biased coin
- p(Heads) = 0.8, p(Tails) = 0.2
- H(X) = -(0.8 log 0.8 + 0.2 log 0.2) = -(-0.258 - 0.464) = 0.722 bits

**Example 3**: Deterministic case
- p(Heads) = 1, p(Tails) = 0
- H(X) = -(1 log 1 + 0 log 0) = -(0 + 0) = 0 bits

## Entropy Properties

### Maximum Entropy
For a discrete random variable with n possible outcomes, entropy is maximized when all outcomes are equally likely:

H(X) ≤ log n

**Example**: For a 6-sided die, maximum entropy = log 6 ≈ 2.585 bits

### Entropy Inequalities

1. **Jensen's Inequality**: H(X) ≤ log n (for n outcomes)
2. **Subadditivity**: H(X,Y) ≤ H(X) + H(Y)
3. **Monotonicity**: H(X|Y) ≤ H(X)

## Conditional Entropy

### Definition
The conditional entropy of X given Y is:

H(X|Y) = -Σ p(x,y) log p(x|y)

This measures the remaining uncertainty in X after observing Y.

### Chain Rule for Entropy
H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)

**Interpretation**: Total uncertainty = uncertainty of first variable + remaining uncertainty of second variable given the first.

### Examples

**Example**: Weather and temperature
- H(Weather) = 1.5 bits
- H(Temperature|Weather) = 0.8 bits
- H(Weather, Temperature) = 1.5 + 0.8 = 2.3 bits

## Mutual Information

### Definition
Mutual information measures the amount of information that one random variable provides about another:

I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)

### Properties
- I(X;Y) ≥ 0 (non-negative)
- I(X;Y) = 0 if and only if X and Y are independent
- I(X;Y) = I(Y;X) (symmetric)
- I(X;Y) ≤ min(H(X), H(Y))

### Examples

**Example 1**: Independent variables
- If X and Y are independent: I(X;Y) = 0

**Example 2**: Perfect correlation
- If Y = X: I(X;Y) = H(X) = H(Y)

**Example 3**: Partial correlation
- Weather and Temperature: I(Weather; Temperature) = 0.7 bits

## Joint Entropy

### Definition
For two random variables X and Y:

H(X,Y) = -Σ p(x,y) log p(x,y)

### Properties
- H(X,Y) ≥ max(H(X), H(Y))
- H(X,Y) ≤ H(X) + H(Y)
- H(X,Y) = H(X) + H(Y) if and only if X and Y are independent

## Applications in Machine Learning

### Feature Selection
Mutual information can be used to select features that are most informative about the target variable:

**Algorithm**:
1. Calculate I(X_i; Y) for each feature X_i
2. Select features with highest mutual information
3. Use threshold or top-k selection

**Example**: Text classification
- I(word_frequency; class) measures how informative word frequency is for classification
- Select words with highest mutual information

### Information Bottleneck
The information bottleneck principle states that good representations should:
1. Maximize mutual information with the target: I(Z; Y)
2. Minimize mutual information with the input: I(Z; X)

This leads to compressed representations that retain relevant information.

### Model Comparison
KL divergence can be used to compare probability distributions:

D(P||Q) = Σ p(x) log(p(x)/q(x))

**Applications**:
- Comparing model predictions to true distributions
- Measuring distribution shift
- Regularization in neural networks

## Continuous Entropy

### Differential Entropy
For continuous random variables:

h(X) = -∫ f(x) log f(x) dx

**Note**: Differential entropy can be negative and doesn't have the same interpretation as discrete entropy.

### Examples

**Example 1**: Uniform distribution on [a,b]
- f(x) = 1/(b-a) for x ∈ [a,b]
- h(X) = log(b-a)

**Example 2**: Gaussian distribution N(μ, σ²)
- h(X) = (1/2) log(2πeσ²)

## Practice Problems

### Problem 1
Calculate the entropy of a random variable X with distribution:
- P(X=1) = 0.3
- P(X=2) = 0.2
- P(X=3) = 0.5

**Solution**:
H(X) = -(0.3 log 0.3 + 0.2 log 0.2 + 0.5 log 0.5)
= -(-0.521 - 0.464 - 0.5)
= 1.485 bits

### Problem 2
Given joint distribution:
- P(X=0, Y=0) = 0.2
- P(X=0, Y=1) = 0.3
- P(X=1, Y=0) = 0.1
- P(X=1, Y=1) = 0.4

Find H(X), H(Y), H(X|Y), and I(X;Y).

**Solution**:
- P(X=0) = 0.5, P(X=1) = 0.5
- P(Y=0) = 0.3, P(Y=1) = 0.7
- H(X) = 1 bit, H(Y) = 0.881 bits
- H(X|Y) = 0.875 bits
- I(X;Y) = 1 - 0.875 = 0.125 bits

### Problem 3
In a binary classification problem, calculate the mutual information between a feature and the class label if:
- P(feature=1|class=1) = 0.8
- P(feature=1|class=0) = 0.2
- P(class=1) = 0.6

**Solution**:
- P(feature=1) = 0.8×0.6 + 0.2×0.4 = 0.56
- P(feature=0) = 0.44
- H(feature) = 0.99 bits
- H(feature|class) = 0.72 bits
- I(feature; class) = 0.99 - 0.72 = 0.27 bits

## Key Takeaways
- Entropy measures uncertainty in random variables
- Conditional entropy measures remaining uncertainty
- Mutual information measures shared information
- These measures are fundamental for feature selection and model comparison
- Information theory provides principled approaches to machine learning problems

## Next Steps
In the next tutorial, we'll explore divergence measures, including Kullback-Leibler divergence and its applications in machine learning and variational inference.
