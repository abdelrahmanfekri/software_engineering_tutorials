# 6. ML Applications: Bringing It All Together

## Introduction

This final tutorial demonstrates how all the linear algebra concepts we've learned come together in real-world machine learning applications. We'll build practical examples that showcase the power of linear algebra in ML.

## Neural Networks: Weights as Matrices

### Forward Pass

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            # Linear transformation: z = XW + b
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # Non-linear activation (ReLU)
            if i < len(self.weights) - 1:  # Not output layer
                activation = np.maximum(0, z)
            else:
                activation = z  # Linear output for regression
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward pass (gradient descent)"""
        m = X.shape[0]
        
        # Compute gradients
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient of weights
            dW = (self.activations[i].T @ delta) / m
            # Gradient of biases
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Propagate error backward
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self.z_values[i-1] > 0)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR input
y = np.array([[0], [1], [1], [0]])  # XOR output

# Create network: 2 inputs -> 4 hidden -> 1 output
nn = SimpleNeuralNetwork([2, 4, 1])

# Train
for epoch in range(1000):
    # Forward pass
    output = nn.forward(X)
    
    # Backward pass
    nn.backward(X, y, learning_rate=0.1)
    
    if epoch % 100 == 0:
        loss = np.mean((output - y)**2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test
predictions = nn.forward(X)
print("\nPredictions:")
print(predictions)
```

### Weight Analysis

```python
def analyze_network_weights(nn):
    """Analyze neural network weights using linear algebra"""
    for i, w in enumerate(nn.weights):
        print(f"\nLayer {i+1} weights:")
        print(f"Shape: {w.shape}")
        
        # Eigenvalue analysis
        if w.shape[0] == w.shape[1]:  # Square matrix
            eigenvals = np.linalg.eigvals(w)
            print(f"Eigenvalues: {eigenvals}")
            print(f"Condition number: {np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals)):.2f}")
        
        # Singular value analysis
        U, S, Vt = np.linalg.svd(w, full_matrices=False)
        print(f"Singular values: {S[:5]}...")  # Top 5
        print(f"Rank: {np.sum(S > 1e-10)}")

analyze_network_weights(nn)
```

## Word Embeddings and Semantic Space

### Building Word Vectors

```python
class WordEmbeddings:
    def __init__(self, vocabulary_size, embedding_dim=100):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        
        # Initialize random embeddings
        self.embeddings = np.random.randn(vocabulary_size, embedding_dim) * 0.1
        
        # Normalize embeddings
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
    
    def cosine_similarity(self, word1_idx, word2_idx):
        """Compute cosine similarity between two words"""
        v1 = self.embeddings[word1_idx]
        v2 = self.embeddings[word2_idx]
        return v1 @ v2
    
    def find_similar_words(self, word_idx, top_k=5):
        """Find most similar words"""
        similarities = self.embeddings @ self.embeddings[word_idx]
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return similar_indices, similarities[similar_indices]
    
    def update_embeddings(self, context_pairs, learning_rate=0.01):
        """Update embeddings using context pairs"""
        for target_idx, context_idx in context_pairs:
            # Simple update rule
            target_vec = self.embeddings[target_idx]
            context_vec = self.embeddings[context_idx]
            
            # Update both vectors
            self.embeddings[target_idx] += learning_rate * context_vec
            self.embeddings[context_idx] += learning_rate * target_vec
            
            # Renormalize
            self.embeddings[target_idx] /= np.linalg.norm(self.embeddings[target_idx])
            self.embeddings[context_idx] /= np.linalg.norm(self.embeddings[context_idx])

# Example usage
vocab_size = 1000
embeddings = WordEmbeddings(vocab_size, embedding_dim=50)

# Simulate some context pairs (word, context_word)
context_pairs = [(0, 1), (0, 2), (1, 0), (2, 0), (3, 4), (4, 3)]

# Train embeddings
for epoch in range(100):
    embeddings.update_embeddings(context_pairs, learning_rate=0.01)

# Analyze embeddings
print("Embedding matrix shape:", embeddings.embeddings.shape)
print("Embedding norms:", np.linalg.norm(embeddings.embeddings, axis=1)[:5])

# Find similar words
similar_words, similarities = embeddings.find_similar_words(0, top_k=3)
print(f"Words similar to word 0: {similar_words}")
print(f"Similarities: {similarities}")
```

### Dimensionality Reduction for Visualization

```python
def visualize_embeddings(embeddings, word_indices, labels):
    """Reduce embeddings to 2D for visualization"""
    # Extract embeddings for selected words
    word_vectors = embeddings.embeddings[word_indices]
    
    # PCA to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
    
    for i, label in enumerate(labels):
        plt.annotate(label, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
    
    plt.title("Word Embeddings (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()
    
    return word_vectors_2d

# Visualize some embeddings
word_indices = [0, 1, 2, 3, 4, 5]
word_labels = [f"word_{i}" for i in word_indices]
word_vectors_2d = visualize_embeddings(embeddings, word_indices, word_labels)
```

## Advanced Dimensionality Reduction

### t-SNE Implementation

```python
def tsne_implementation(X, perplexity=30, n_iter=1000, learning_rate=200):
    """Simple t-SNE implementation"""
    n_samples = X.shape[0]
    
    # Initialize low-dimensional representation
    Y = np.random.randn(n_samples, 2) * 0.0001
    
    # Compute pairwise distances in high-dimensional space
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            D[i, j] = np.sum((X[i] - X[j])**2)
    
    # Compute similarities (Gaussian kernel)
    P = np.exp(-D / (2 * perplexity**2))
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)  # Avoid numerical issues
    
    # Gradient descent
    for iteration in range(n_iter):
        # Compute pairwise distances in low-dimensional space
        d = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                d[i, j] = np.sum((Y[i] - Y[j])**2)
        
        # Compute low-dimensional similarities (t-distribution)
        Q = 1 / (1 + d)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradient
        grad = np.zeros_like(Y)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    grad[i] += 4 * (P[i, j] - Q[i, j]) * (Y[i] - Y[j]) * Q[i, j]
        
        # Update
        Y = Y - learning_rate * grad
        
        if iteration % 100 == 0:
            kl_divergence = np.sum(P * np.log(P / Q))
            print(f"Iteration {iteration}, KL divergence: {kl_divergence:.4f}")
    
    return Y

# Example: reduce 3D data to 2D
X_3d = np.random.rand(50, 3)
Y_2d = tsne_implementation(X_3d, perplexity=10, n_iter=500)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ax = plt.axes(projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])
ax.set_title("Original 3D Data")

plt.subplot(1, 2, 2)
plt.scatter(Y_2d[:, 0], Y_2d[:, 1])
plt.title("t-SNE 2D Projection")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.tight_layout()
plt.show()
```

## Matrix Factorization for Recommendations

### Non-negative Matrix Factorization (NMF)

```python
def nmf_recommendation(ratings_matrix, n_factors=10, max_iter=100, tol=1e-6):
    """Non-negative Matrix Factorization for recommendations"""
    m, n = ratings_matrix.shape
    
    # Initialize non-negative matrices
    W = np.random.rand(m, n_factors)
    H = np.random.rand(n_factors, n)
    
    # Mask for observed ratings
    mask = ~np.isnan(ratings_matrix)
    
    for iteration in range(max_iter):
        W_old = W.copy()
        H_old = H.copy()
        
        # Update W
        numerator = np.zeros((m, n_factors))
        denominator = np.zeros((m, n_factors))
        
        for i in range(m):
            for j in range(n):
                if mask[i, j]:
                    numerator[i] += ratings_matrix[i, j] * H[:, j]
                    denominator[i] += (W[i] @ H[:, j:j+1]) * H[:, j]
        
        W = W * (numerator / (denominator + 1e-10))
        
        # Update H
        numerator = np.zeros((n_factors, n))
        denominator = np.zeros((n_factors, n))
        
        for i in range(m):
            for j in range(n):
                if mask[i, j]:
                    numerator[:, j] += ratings_matrix[i, j] * W[i]
                    denominator[:, j] += (W[i:i+1] @ H) * W[i]
        
        H = H * (numerator / (denominator + 1e-10))
        
        # Check convergence
        if (np.linalg.norm(W - W_old) < tol and 
            np.linalg.norm(H - H_old) < tol):
            break
    
    return W, H

# Example usage
ratings = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 5],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 0],
    [0, 1, 5, 4, 0]
])

# Add some missing values
ratings[0, 2] = np.nan
ratings[1, 1] = np.nan
ratings[2, 4] = np.nan

# Factorize
W, H = nmf_recommendation(ratings, n_factors=3, max_iter=200)

# Reconstruct ratings
predicted_ratings = W @ H

print("Original ratings:")
print(ratings)
print("\nPredicted ratings:")
print(predicted_ratings.round(2))
print("\nReconstruction error:")
print(np.nanmean((ratings - predicted_ratings)**2))
```

## Performance Optimization

### Vectorized Operations

```python
def vectorized_operations_example():
    """Demonstrate vectorized vs loop-based operations"""
    import time
    
    # Large matrix
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    
    # Vectorized matrix multiplication
    start_time = time.time()
    C_vectorized = A @ B
    vectorized_time = time.time() - start_time
    
    # Loop-based matrix multiplication
    start_time = time.time()
    C_loop = np.zeros((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            for k in range(1000):
                C_loop[i, j] += A[i, k] * B[k, j]
    loop_time = time.time() - start_time
    
    print(f"Vectorized time: {vectorized_time:.4f}s")
    print(f"Loop time: {loop_time:.4f}s")
    print(f"Speedup: {loop_time / vectorized_time:.1f}x")
    
    # Verify results are the same
    print(f"Results match: {np.allclose(C_vectorized, C_loop)}")

vectorized_operations_example()
```

## Practice Problems

1. **Neural Network Analysis**: Analyze the weight matrices of a trained neural network using SVD
2. **Embedding Visualization**: Create word embeddings for a small vocabulary and visualize them using PCA and t-SNE
3. **Recommendation System**: Build a complete recommendation system using matrix factorization
4. **Performance Comparison**: Compare different dimensionality reduction techniques on the same dataset

## Key Takeaways

1. **Linear algebra is the foundation** of modern machine learning
2. **Matrix operations** enable efficient computation on large datasets
3. **Eigenvalues and SVD** reveal the structure of data and models
4. **Orthogonality and projections** provide optimal representations
5. **Vectorization** is crucial for performance in ML systems

## Next Steps

You now have a solid foundation in linear algebra for machine learning! Consider exploring:
- **Advanced optimization** techniques (gradient descent variants)
- **Deep learning** architectures and their mathematical foundations
- **Probabilistic models** and their linear algebra connections
- **Signal processing** and Fourier analysis
- **Computer vision** and image processing algorithms

The concepts you've learned here will serve as the mathematical foundation for understanding and implementing advanced machine learning algorithms.
