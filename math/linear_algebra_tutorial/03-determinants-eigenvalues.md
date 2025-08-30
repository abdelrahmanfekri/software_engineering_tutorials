# 3. Determinants, Eigenvalues, and Eigenvectors

## Introduction

Determinants, eigenvalues, and eigenvectors reveal the fundamental structure and behavior of matrices. They're crucial for understanding matrix properties, solving systems of equations, and implementing key ML algorithms.

## Determinants

### Definition

The determinant is a scalar value that provides information about a square matrix:
- **Invertibility**: det(A) ≠ 0 means A is invertible
- **Volume**: |det(A)| represents volume scaling factor
- **Orientation**: sign indicates if transformation preserves orientation

### 2×2 Matrix Determinant

```
det([a  b]) = ad - bc
     [c  d]
```

### 3×3 Matrix Determinant

```
det([a  b  c]) = a(ei - fh) - b(di - fg) + c(dh - eg)
     [d  e  f]
     [g  h  i]
```

### Python Implementation

```python
import numpy as np

# 2×2 matrix
A = np.array([[1, 2], [3, 4]])
det_2x2 = np.linalg.det(A)

# 3×3 matrix
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
det_3x3 = np.linalg.det(B)

print(f"2×2 determinant: {det_2x2}")
print(f"3×3 determinant: {det_3x3}")
```

### Properties

1. **det(AB) = det(A) × det(B)**
2. **det(Aᵀ) = det(A)**
3. **det(cA) = cⁿ × det(A)** for n×n matrix
4. **det(A⁻¹) = 1/det(A)**

## Eigenvalues and Eigenvectors

### Definition

For a square matrix A, if there exists a non-zero vector v and scalar λ such that:
```
Av = λv
```

Then:
- **λ** is an eigenvalue of A
- **v** is the corresponding eigenvector

### Geometric Interpretation

Eigenvectors represent directions that are preserved under the transformation A:
- **Eigenvalue**: scaling factor in that direction
- **Eigenvector**: direction that doesn't change (only scales)

### Finding Eigenvalues

Solve the characteristic equation:
```
det(A - λI) = 0
```

### Python Implementation

```python
import numpy as np

A = np.array([[4, -1], [2, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Verify: Av = λv
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = A @ v
    λv = λ * v
    print(f"Eigenvalue {i+1}: Av = {Av}, λv = {λv}")
```

## ML Applications

### 1. Principal Component Analysis (PCA)

PCA finds the principal components (eigenvectors) of the covariance matrix:

```python
def pca(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Project data onto principal components
    X_pca = X_centered @ eigenvectors[:, :n_components]
    
    return X_pca, eigenvalues, eigenvectors

# Example usage
X = np.random.rand(100, 3)  # 100 samples, 3 features
X_reduced, eigenvals, eigenvecs = pca(X, n_components=2)
```

### 2. Neural Network Weight Initialization

Eigenvalues help understand weight matrix properties:

```python
def analyze_weights(weight_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(weight_matrix)
    
    # Check for vanishing/exploding gradients
    max_eigenvalue = np.max(np.abs(eigenvalues))
    min_eigenvalue = np.min(np.abs(eigenvalues))
    
    print(f"Max eigenvalue: {max_eigenvalue}")
    print(f"Min eigenvalue: {min_eigenvalue}")
    print(f"Condition number: {max_eigenvalue / min_eigenvalue}")
    
    return eigenvalues, eigenvectors

# Example: analyze neural network layer
weights = np.random.randn(100, 100) * 0.01
eigenvals, eigenvecs = analyze_weights(weights)
```

### 3. Stability Analysis

Eigenvalues determine system stability:

```python
def check_stability(A):
    eigenvalues = np.linalg.eigvals(A)
    
    # Check if all eigenvalues have negative real parts
    is_stable = np.all(np.real(eigenvalues) < 0)
    
    print(f"Eigenvalues: {eigenvalues}")
    print(f"System is stable: {is_stable}")
    
    return is_stable

# Example: linear system dynamics
A = np.array([[-1, 2], [0, -3]])
stability = check_stability(A)
```

## Key Properties

### 1. Eigenvalue Properties

- **Sum of eigenvalues = trace of matrix**
- **Product of eigenvalues = determinant**
- **Eigenvalues of Aᵀ = eigenvalues of A**

### 2. Eigenvector Properties

- **Eigenvectors corresponding to distinct eigenvalues are linearly independent**
- **Symmetric matrices have real eigenvalues and orthogonal eigenvectors**

### 3. Diagonalization

If A has n linearly independent eigenvectors, then:
```
A = PDP⁻¹
```
where P contains eigenvectors and D is diagonal with eigenvalues.

## Computational Considerations

### 1. Numerical Stability

```python
# Use more stable methods for large matrices
from scipy.linalg import eig

# More stable than np.linalg.eig for large matrices
eigenvalues, eigenvectors = eig(A)
```

### 2. Sparse Matrices

```python
from scipy.sparse.linalg import eigs

# For large sparse matrices, compute only k largest eigenvalues
eigenvalues, eigenvectors = eigs(A, k=5)
```

## Practice Problems

1. Calculate the determinant of a 3×3 matrix by hand
2. Find eigenvalues and eigenvectors of a 2×2 matrix
3. Implement a simple PCA function
4. Analyze the stability of a given matrix

## Next Steps

In the next tutorial, we'll explore orthogonality and projections - geometric concepts that are fundamental to understanding how data is transformed and represented in different coordinate systems.
