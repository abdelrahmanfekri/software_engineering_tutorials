# Information Theory Tutorial 04: Applications in Machine Learning

## Learning Objectives
By the end of this tutorial, you will be able to:
- Apply mutual information for feature selection
- Understand the information bottleneck principle
- Use variational inference with ELBO
- Apply information-theoretic clustering methods
- Understand regularization from information perspective
- Apply information theory to neural network analysis

## Feature Selection Using Mutual Information

### Motivation
In machine learning, we often have many features but want to select the most informative ones. Mutual information provides a principled way to measure feature relevance.

### Mutual Information Feature Selection

**Algorithm**:
1. Calculate I(X_i; Y) for each feature X_i
2. Select features with highest mutual information
3. Use threshold or top-k selection

**Advantages**:
- Captures non-linear relationships
- Handles both discrete and continuous variables
- Provides principled ranking

### Example: Text Classification

**Problem**: Classify documents into topics
**Features**: Word frequencies
**Target**: Document class

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_20newsgroups

# Load data
newsgroups = fetch_20newsgroups(subset='train')
X = newsgroups.data
y = newsgroups.target

# Calculate mutual information for each word
mi_scores = mutual_info_classif(X, y, discrete_features=True)

# Select top features
top_features = np.argsort(mi_scores)[-100:]  # Top 100 words
```

### Conditional Mutual Information
For selecting features given already selected features:

I(X_i; Y | X_S) = H(X_i | X_S) - H(X_i | X_S, Y)

**Forward Selection Algorithm**:
1. Start with empty set S = ∅
2. For each feature X_i not in S:
   - Calculate I(X_i; Y | X_S)
3. Add feature with highest conditional mutual information
4. Repeat until desired number of features

## Information Bottleneck Principle

### Theory
The information bottleneck principle states that good representations should:
1. **Maximize** mutual information with the target: I(Z; Y)
2. **Minimize** mutual information with the input: I(Z; X)

**Objective**: Find representation Z that maximizes:
L = I(Z; Y) - β I(Z; X)

Where β controls the compression-accuracy trade-off.

### Applications

**Neural Networks**:
- Hidden layers act as compressed representations
- Information bottleneck explains generalization
- β controls overfitting vs. underfitting

**Deep Learning**:
- Early layers extract relevant features
- Later layers compress information
- Bottleneck layers force compression

### Variational Information Bottleneck

**Practical Implementation**:
Instead of computing exact mutual information, use variational bounds:

L_VIB = E[log q(y|z)] - β D(q(z|x) || p(z))

Where:
- q(y|z) is the decoder
- q(z|x) is the encoder
- p(z) is the prior

## Variational Inference and ELBO

### Problem Setup
We want to approximate a complex posterior p(z|x) with a simpler distribution q(z).

**Evidence Lower Bound (ELBO)**:
ELBO = E_q[log p(x,z)] - E_q[log q(z)]
     = E_q[log p(x|z)] - D(q(z)||p(z))

### Interpretation
- First term: Reconstruction quality
- Second term: Regularization (KL divergence)
- Maximizing ELBO minimizes KL divergence to true posterior

### Variational Autoencoders (VAEs)

**Architecture**:
- Encoder: q(z|x) ≈ N(μ(x), σ²(x))
- Decoder: p(x|z) ≈ N(f(z), I)
- Prior: p(z) = N(0, I)

**Loss Function**:
L_VAE = -ELBO = -E_q[log p(x|z)] + D(q(z|x)||p(z))

**Implementation**:
```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
```

## Information-Theoretic Clustering

### Mutual Information Clustering
Cluster data points to maximize mutual information between clusters and features.

**Objective**: Maximize I(C; X) where C is cluster assignment.

### Information-Theoretic K-means
Modify K-means to use information-theoretic distance measures.

**Distance Measure**: KL divergence between distributions
D(p||q) = Σ p(x) log(p(x)/q(x))

### Agglomerative Information Clustering
Merge clusters that minimize information loss.

**Criterion**: Minimize I(X; C) - I(X; C') where C' is merged clustering.

## Regularization from Information Perspective

### Dropout as Information Regularization
Dropout can be viewed as adding noise to reduce mutual information:

I(X; Y) ≥ I(X; Y|Dropout)

This prevents overfitting by reducing information flow.

### Batch Normalization
Batch normalization reduces internal covariate shift, which can be understood as stabilizing information flow through the network.

### Weight Decay
Weight decay can be interpreted as minimizing information content of weights, promoting simpler models.

## Neural Network Analysis

### Information Flow in Deep Networks
Analyze how information flows through neural network layers:

**Information Plane**:
- X-axis: I(X; T) (input information)
- Y-axis: I(T; Y) (target information)
- T represents hidden layer representations

**Observations**:
- Fitting phase: I(X; T) increases, I(T; Y) increases
- Compression phase: I(X; T) decreases, I(T; Y) maintained

### Layer-wise Information Analysis

**Method**:
1. Train neural network
2. For each layer, estimate mutual information
3. Plot information plane trajectory
4. Analyze compression and fitting phases

```python
def estimate_mutual_information(X, Y, bins=30):
    """Estimate mutual information between X and Y"""
    # Discretize continuous variables
    X_discrete = np.digitize(X, np.linspace(X.min(), X.max(), bins))
    Y_discrete = np.digitize(Y, np.linspace(Y.min(), Y.max(), bins))
    
    # Calculate joint and marginal distributions
    joint_dist = np.histogram2d(X_discrete, Y_discrete, bins=bins)[0]
    joint_dist = joint_dist / joint_dist.sum()
    
    x_dist = joint_dist.sum(axis=1)
    y_dist = joint_dist.sum(axis=0)
    
    # Calculate mutual information
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if joint_dist[i, j] > 0:
                mi += joint_dist[i, j] * np.log2(
                    joint_dist[i, j] / (x_dist[i] * y_dist[j])
                )
    
    return mi
```

## Advanced Applications

### Information-Theoretic Exploration
In reinforcement learning, use information theory to guide exploration:

**Objective**: Maximize information gain about environment
I(S_{t+1}; A_t | S_t)

### Natural Language Processing
Information theory applications in NLP:

**Language Modeling**:
- Perplexity: 2^H(X) where H(X) is entropy
- Cross-entropy loss for language models

**Word Embeddings**:
- Mutual information between words and contexts
- Information-theoretic word similarity

### Computer Vision
Information theory in computer vision:

**Image Compression**:
- Rate-distortion optimization
- Information-theoretic image quality metrics

**Feature Learning**:
- Information bottleneck in convolutional networks
- Mutual information for feature selection

## Practical Implementation Tips

### Numerical Stability
When computing mutual information:

1. **Handle Zero Probabilities**:
```python
def safe_log(p, eps=1e-10):
    return np.log(p + eps)
```

2. **Use Log-Sum-Exp Trick**:
```python
def log_sum_exp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))
```

### Efficient Computation
For large datasets:

1. **Sampling**: Use random samples for MI estimation
2. **Discretization**: Bin continuous variables
3. **Approximation**: Use neural estimators for MI

### Common Pitfalls
1. **Bias in MI Estimation**: Use bias-corrected estimators
2. **Discretization Effects**: Choose appropriate bin sizes
3. **Sample Size**: Ensure sufficient data for reliable estimates

## Practice Problems

### Problem 1
In a binary classification problem with features X₁, X₂, X₃ and target Y:
- I(X₁; Y) = 0.3 bits
- I(X₂; Y) = 0.5 bits  
- I(X₃; Y) = 0.2 bits
- I(X₁; X₂) = 0.1 bits
- I(X₁; X₃) = 0.05 bits
- I(X₂; X₃) = 0.15 bits

Which features should you select for a 2-feature model?

**Solution**:
Using forward selection:
1. First feature: X₂ (highest MI = 0.5)
2. Second feature: Calculate conditional MI
   - I(X₁; Y | X₂) = I(X₁; Y) - I(X₁; X₂) = 0.3 - 0.1 = 0.2
   - I(X₃; Y | X₂) = I(X₃; Y) - I(X₃; X₂) = 0.2 - 0.15 = 0.05
   
Select X₁ as second feature.

### Problem 2
In a VAE with latent dimension d=2, if the encoder outputs:
- μ = [0.1, -0.2]
- σ² = [0.5, 0.3]

Calculate the KL divergence term D(q(z|x)||p(z)).

**Solution**:
D(q(z|x)||p(z)) = (1/2) Σᵢ (μᵢ² + σᵢ² - log σᵢ² - 1)
= (1/2) [(0.1² + 0.5 - log(0.5) - 1) + ((-0.2)² + 0.3 - log(0.3) - 1)]
= (1/2) [(0.01 + 0.5 - (-0.693) - 1) + (0.04 + 0.3 - (-1.204) - 1)]
= (1/2) [(0.51 + 0.693 - 1) + (0.34 + 1.204 - 1)]
= (1/2) [0.203 + 0.544] = 0.374

### Problem 3
For a neural network with information plane trajectory:
- Layer 1: I(X; T₁) = 0.8, I(T₁; Y) = 0.3
- Layer 2: I(X; T₂) = 0.6, I(T₂; Y) = 0.5
- Layer 3: I(X; T₃) = 0.4, I(T₃; Y) = 0.7

Identify the fitting and compression phases.

**Solution**:
- **Fitting Phase**: Layers 1-2 (I(T; Y) increases from 0.3 to 0.5)
- **Compression Phase**: Layers 2-3 (I(X; T) decreases from 0.6 to 0.4 while maintaining I(T; Y))

## Key Takeaways
- Mutual information provides principled feature selection
- Information bottleneck explains neural network behavior
- Variational inference uses information-theoretic bounds
- Information theory guides regularization strategies
- Neural networks exhibit fitting and compression phases

## Next Steps
In the next tutorial, we'll explore advanced topics in information theory, including maximum entropy principle, information geometry, quantum information theory, and information-theoretic security.
