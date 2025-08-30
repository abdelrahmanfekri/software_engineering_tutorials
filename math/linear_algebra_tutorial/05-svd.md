# 5. Singular Value Decomposition (SVD)

## Introduction

Singular Value Decomposition (SVD) is one of the most powerful tools in linear algebra and machine learning. It generalizes eigendecomposition to non-square matrices and provides insights into data structure, dimensionality reduction, and matrix approximation.

## What is SVD?

### Definition

For any matrix A (m×n), SVD decomposes it into:
```
A = UΣVᵀ
```

Where:
- **U**: m×m orthogonal matrix (left singular vectors)
- **Σ**: m×n diagonal matrix (singular values)
- **Vᵀ**: n×n orthogonal matrix (right singular vectors)

### Geometric Interpretation

SVD represents A as:
1. **Rotation/Reflection** (Vᵀ)
2. **Scaling** (Σ)
3. **Rotation/Reflection** (U)

## Computing SVD

### Python Implementation

```python
import numpy as np

def svd_example():
    # Example matrix
    A = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    
    print("Original matrix A:")
    print(A)
    print("\nU (left singular vectors):")
    print(U)
    print("\nS (singular values):")
    print(S)
    print("\nVt (right singular vectors, transposed):")
    print(Vt)
    
    # Reconstruct A
    # Note: S is 1D, need to create diagonal matrix
    Sigma = np.zeros_like(A, dtype=float)
    Sigma[:len(S), :len(S)] = np.diag(S)
    
    A_reconstructed = U @ Sigma @ Vt
    print("\nReconstructed A:")
    print(A_reconstructed)
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

svd_example()
```

### Compact SVD

For m×n matrix with rank r ≤ min(m,n):

```python
def compact_svd(A):
    """Compute compact SVD (only non-zero singular values)"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # U: m×r, S: r×1, Vt: r×n
    return U, S, Vt

# Example
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = compact_svd(A)

print(f"U shape: {U.shape}")  # (3, 2)
print(f"S length: {len(S)}")  # 2
print(f"Vt shape: {Vt.shape}")  # (2, 2)
```

## Properties and Insights

### 1. Singular Values

- **Magnitude**: Indicates importance of corresponding singular vectors
- **Rank**: Number of non-zero singular values = rank of matrix
- **Condition Number**: σ₁/σᵣ (ratio of largest to smallest singular value)

### 2. Singular Vectors

- **U**: Basis for column space of A
- **V**: Basis for row space of A
- **Orthogonal**: Columns of U and V are orthonormal

### 3. Matrix Approximation

```python
def svd_approximation(A, k):
    """Approximate A using k largest singular values"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only k largest singular values
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct approximation
    A_approx = U_k @ np.diag(S_k) @ Vt_k
    
    return A_approx, U_k, S_k, Vt_k

# Example: image compression
def compress_image(image_matrix, k):
    """Compress image using SVD approximation"""
    compressed, U_k, S_k, Vt_k = svd_approximation(image_matrix, k)
    
    # Calculate compression ratio
    original_size = image_matrix.size
    compressed_size = U_k.size + S_k.size + Vt_k.size
    compression_ratio = original_size / compressed_size
    
    return compressed, compression_ratio

# Simulate grayscale image (8×8)
image = np.random.rand(8, 8)
compressed_image, ratio = compress_image(image, 4)

print(f"Compression ratio: {ratio:.2f}x")
print(f"Original size: {image.size}")
print(f"Compressed size: {compressed_image.size}")
```

## ML Applications

### 1. Principal Component Analysis (PCA)

SVD is the computational foundation of PCA:

```python
def pca_svd(X, n_components=2):
    """PCA using SVD"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # SVD of centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components are right singular vectors
    principal_components = Vt[:n_components, :].T
    
    # Project data
    X_pca = X_centered @ principal_components
    
    return X_pca, principal_components, S

# Example
X = np.random.rand(100, 5)  # 100 samples, 5 features
X_reduced, pc, singular_values = pca_svd(X, n_components=2)

print(f"Explained variance ratio: {singular_values[:2]**2 / np.sum(singular_values**2)}")
```

### 2. Recommendation Systems

SVD for collaborative filtering:

```python
def svd_recommendation(ratings_matrix, n_factors=10):
    """SVD-based recommendation system"""
    # Fill missing values with mean
    ratings_filled = ratings_matrix.copy()
    mask = np.isnan(ratings_matrix)
    ratings_filled[mask] = np.nanmean(ratings_matrix)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(ratings_filled, full_matrices=False)
    
    # Keep top factors
    U_k = U[:, :n_factors]
    S_k = S[:n_factors]
    Vt_k = Vt[:n_factors, :]
    
    # Predict ratings
    predicted_ratings = U_k @ np.diag(S_k) @ Vt_k
    
    return predicted_ratings

# Example: user-item ratings matrix
ratings = np.array([
    [5, 3, 0, 1],  # user 1 ratings
    [4, 0, 0, 1],  # user 2 ratings
    [1, 1, 0, 5],  # user 3 ratings
    [1, 0, 0, 4],  # user 4 ratings
    [0, 1, 5, 4]   # user 5 ratings
])

# Add some missing values
ratings[0, 2] = np.nan
ratings[1, 1] = np.nan

predicted = svd_recommendation(ratings, n_factors=2)
print("Predicted ratings:")
print(predicted)
```

### 3. Dimensionality Reduction

SVD for feature reduction:

```python
def svd_dimension_reduction(X, n_components):
    """Reduce dimensions using SVD"""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Project onto top singular vectors
    X_reduced = U[:, :n_components] @ np.diag(S[:n_components])
    
    return X_reduced, U[:, :n_components], S[:n_components]

# Example: text embeddings
# Simulate document-term matrix (documents × terms)
doc_term_matrix = np.random.rand(100, 1000)  # 100 docs, 1000 terms

# Reduce to 50 dimensions
docs_reduced, U_50, S_50 = svd_dimension_reduction(doc_term_matrix, 50)

print(f"Original shape: {doc_term_matrix.shape}")
print(f"Reduced shape: {docs_reduced.shape}")
print(f"Compression: {doc_term_matrix.size / docs_reduced.size:.1f}x")
```

### 4. Matrix Completion

SVD for filling missing values:

```python
def matrix_completion_svd(A, mask, max_iter=100, tol=1e-6):
    """Complete matrix using iterative SVD"""
    A_completed = A.copy()
    A_completed[mask] = 0  # Initialize missing values to 0
    
    for iteration in range(max_iter):
        A_old = A_completed.copy()
        
        # SVD of current completion
        U, S, Vt = np.linalg.svd(A_completed, full_matrices=False)
        
        # Keep top singular values (truncate)
        k = min(len(S), 10)  # Keep top 10 components
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct
        A_completed = U_k @ np.diag(S_k) @ Vt_k
        
        # Keep known values
        A_completed[~mask] = A[~mask]
        
        # Check convergence
        if np.linalg.norm(A_completed - A_old) < tol:
            break
    
    return A_completed

# Example
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = np.array([[False, False, True], [False, True, False], [True, False, False]])

A_completed = matrix_completion_svd(A, mask)
print("Original matrix:")
print(A)
print("\nCompleted matrix:")
print(A_completed)
```

## Computational Considerations

### 1. Large Matrices

```python
# For very large matrices, use randomized SVD
from sklearn.decomposition import TruncatedSVD

# Efficient for large, sparse matrices
svd = TruncatedSVD(n_components=10)
X_reduced = svd.fit_transform(X)
```

### 2. Memory Efficiency

```python
# Compute SVD incrementally for streaming data
def incremental_svd_update(U, S, Vt, new_data):
    """Update SVD with new data"""
    # This is a simplified version
    # In practice, use specialized algorithms
    pass
```

## Practice Problems

1. Implement SVD from scratch for 2×2 matrices
2. Use SVD to compress a simple image
3. Build a basic recommendation system with SVD
4. Compare SVD-based PCA with eigendecomposition PCA

## Next Steps

In the final tutorial, we'll explore real-world ML applications that bring together all the concepts we've learned: neural networks, embeddings, and advanced dimensionality reduction techniques.
