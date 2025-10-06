# Information Theory Tutorial 02: Divergence Measures and Applications

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand Kullback-Leibler (KL) divergence and its properties
- Calculate KL divergence for various distributions
- Apply Jensen-Shannon divergence
- Use cross-entropy in machine learning
- Understand information geometry concepts
- Apply divergence measures to model comparison and optimization

## Kullback-Leibler Divergence

### Definition
The Kullback-Leibler divergence (also called relative entropy) between two probability distributions P and Q is:

D(P||Q) = Σ p(x) log(p(x)/q(x))

For continuous distributions:
D(P||Q) = ∫ p(x) log(p(x)/q(x)) dx

### Properties
1. **Non-negativity**: D(P||Q) ≥ 0
2. **Identity**: D(P||Q) = 0 if and only if P = Q
3. **Asymmetry**: D(P||Q) ≠ D(Q||P) in general
4. **Not a metric**: Doesn't satisfy triangle inequality

### Intuitive Interpretation
KL divergence measures how much "surprise" we experience when we expect distribution Q but observe distribution P.

### Examples

**Example 1**: KL divergence between two Bernoulli distributions
- P: p = 0.3, Q: q = 0.7
- D(P||Q) = 0.3 log(0.3/0.7) + 0.7 log(0.7/0.3)
- = 0.3 log(3/7) + 0.7 log(7/3)
- ≈ 0.3(-0.847) + 0.7(0.847) = 0.339

**Example 2**: KL divergence between Gaussian distributions
- P: N(μ₁, σ₁²), Q: N(μ₂, σ₂²)
- D(P||Q) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

## Jensen-Shannon Divergence

### Definition
The Jensen-Shannon divergence is a symmetric version of KL divergence:

JS(P||Q) = (1/2)D(P||M) + (1/2)D(Q||M)

Where M = (P + Q)/2 is the average distribution.

### Properties
1. **Symmetry**: JS(P||Q) = JS(Q||P)
2. **Bounded**: 0 ≤ JS(P||Q) ≤ log 2
3. **Metric-like**: Satisfies triangle inequality
4. **Smooth**: Continuous and differentiable

### Applications
- Measuring distribution similarity
- Clustering and classification
- Generative model evaluation

## Cross-Entropy

### Definition
Cross-entropy between distributions P and Q:

H(P,Q) = -Σ p(x) log q(x)

### Relationship to KL Divergence
H(P,Q) = H(P) + D(P||Q)

Where H(P) is the entropy of P.

### Applications in Machine Learning

**Classification Loss**:
- True distribution: one-hot vector
- Predicted distribution: softmax output
- Cross-entropy loss: H(true, predicted)

**Example**: Binary classification
- True label: y ∈ {0,1}
- Predicted probability: ŷ ∈ [0,1]
- Cross-entropy loss: -y log ŷ - (1-y) log(1-ŷ)

## Information Geometry

### Fisher Information Matrix
For a parametric family of distributions p(x|θ):

I(θ) = E[∇θ log p(x|θ) ∇θ log p(x|θ)ᵀ]

### Fisher-Rao Metric
The Fisher information matrix defines a Riemannian metric on the parameter space:

ds² = dθᵀ I(θ) dθ

### Applications
- Natural gradient descent
- Bayesian inference
- Optimization on manifolds

## Applications in Machine Learning

### Variational Inference
In variational inference, we approximate a complex posterior p(z|x) with a simpler distribution q(z):

**Evidence Lower Bound (ELBO)**:
ELBO = E_q[log p(x,z)] - E_q[log q(z)]
= E_q[log p(x|z)] - D(q(z)||p(z))

**Interpretation**: Maximizing ELBO minimizes KL divergence between approximate and true posterior.

### Generative Models

**Variational Autoencoders (VAEs)**:
- Encoder: q(z|x) ≈ p(z|x)
- Decoder: p(x|z)
- Loss: -ELBO = -E_q[log p(x|z)] + D(q(z|x)||p(z))

**Generative Adversarial Networks (GANs)**:
- Generator: learns to minimize JS divergence
- Discriminator: learns to distinguish real from fake

### Model Comparison

**Akaike Information Criterion (AIC)**:
AIC = 2k - 2 log L

Where k is the number of parameters and L is the likelihood.

**Bayesian Information Criterion (BIC)**:
BIC = k log n - 2 log L

Where n is the sample size.

## Advanced Divergence Measures

### f-Divergences
General family of divergences:

D_f(P||Q) = ∫ q(x) f(p(x)/q(x)) dx

Where f is a convex function with f(1) = 0.

**Examples**:
- KL divergence: f(t) = t log t
- Total variation: f(t) = |t - 1|/2
- Chi-squared: f(t) = (t - 1)²

### Wasserstein Distance
Also called Earth Mover's Distance:

W_p(P,Q) = (inf E[|X - Y|^p])^(1/p)

Where the infimum is over all joint distributions with marginals P and Q.

**Properties**:
- True metric
- Handles distributions with different supports
- Used in optimal transport

## Practical Considerations

### Numerical Stability
When computing KL divergence:

1. **Avoid log(0)**: Add small epsilon to probabilities
2. **Use log-sum-exp trick**: For numerical stability
3. **Check for zero probabilities**: Handle carefully

**Example**:
```python
def kl_divergence(p, q, eps=1e-8):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))
```

### Computational Complexity
- KL divergence: O(n) for discrete distributions
- JS divergence: O(n) 
- Wasserstein distance: O(n³) for exact computation

## Practice Problems

### Problem 1
Calculate KL divergence between:
- P: [0.4, 0.3, 0.2, 0.1]
- Q: [0.25, 0.25, 0.25, 0.25]

**Solution**:
D(P||Q) = 0.4 log(0.4/0.25) + 0.3 log(0.3/0.25) + 0.2 log(0.2/0.25) + 0.1 log(0.1/0.25)
= 0.4 log(1.6) + 0.3 log(1.2) + 0.2 log(0.8) + 0.1 log(0.4)
= 0.4(0.470) + 0.3(0.182) + 0.2(-0.223) + 0.1(-0.916)
= 0.188 + 0.055 - 0.045 - 0.092 = 0.106

### Problem 2
In a binary classification problem, compare two models:
- Model A: predicts [0.8, 0.2] for class 1
- Model B: predicts [0.6, 0.4] for class 1
- True distribution: [0.7, 0.3]

Which model is better according to KL divergence?

**Solution**:
D(true||A) = 0.7 log(0.7/0.8) + 0.3 log(0.3/0.2) = 0.7(-0.133) + 0.3(0.405) = -0.093 + 0.122 = 0.029

D(true||B) = 0.7 log(0.7/0.6) + 0.3 log(0.3/0.4) = 0.7(0.154) + 0.3(-0.288) = 0.108 - 0.086 = 0.022

Model B is better (lower KL divergence).

### Problem 3
Calculate Jensen-Shannon divergence between:
- P: [0.6, 0.4]
- Q: [0.3, 0.7]

**Solution**:
M = [0.45, 0.55]
JS(P||Q) = 0.5 × D(P||M) + 0.5 × D(Q||M)

D(P||M) = 0.6 log(0.6/0.45) + 0.4 log(0.4/0.55) = 0.6(0.288) + 0.4(-0.318) = 0.173 - 0.127 = 0.046

D(Q||M) = 0.3 log(0.3/0.45) + 0.7 log(0.7/0.55) = 0.3(-0.405) + 0.7(0.241) = -0.122 + 0.169 = 0.047

JS(P||Q) = 0.5(0.046) + 0.5(0.047) = 0.0465

## Key Takeaways
- KL divergence measures difference between distributions
- JS divergence provides symmetric alternative
- Cross-entropy is fundamental loss function in ML
- Divergence measures enable principled model comparison
- Information geometry connects probability and optimization

## Next Steps
In the next tutorial, we'll explore channel capacity and coding theory, including error-correcting codes and their applications to machine learning.
