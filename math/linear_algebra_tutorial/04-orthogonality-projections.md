# 4. Orthogonality and Projections

## Introduction

Orthogonality and projections are geometric concepts that are fundamental to understanding how data is transformed, how dimensions are reduced, and how optimal solutions are found in machine learning.

## Orthogonality

### Definition

Two vectors are **orthogonal** if their dot product is zero:
```
u ⊥ v ⟺ u·v = 0
```

### Geometric Interpretation

Orthogonal vectors are perpendicular to each other in space:
- They form a 90° angle
- They represent independent directions
- They provide a natural coordinate system

### Examples

```python
import numpy as np

# Orthogonal vectors in 2D
v1 = np.array([1, 0])  # x-axis
v2 = np.array([0, 1])  # y-axis

# Check orthogonality
dot_product = v1 @ v2  # 0
print(f"v1 and v2 are orthogonal: {dot_product == 0}")

# Orthogonal vectors in 3D
u = np.array([1, 1, 0])
v = np.array([1, -1, 0])
w = np.array([0, 0, 1])

print(f"u ⊥ v: {(u @ v) == 0}")
print(f"u ⊥ w: {(u @ w) == 0}")
print(f"v ⊥ w: {(v @ w) == 0}")
```

## Orthogonal Matrices

### Definition

A matrix Q is **orthogonal** if:
```
QᵀQ = QQᵀ = I
```

### Properties

1. **Q⁻¹ = Qᵀ** (inverse equals transpose)
2. **det(Q) = ±1** (determinant is ±1)
3. **Preserves lengths and angles**
4. **Columns/rows form orthonormal basis**

### Examples

```python
# Rotation matrix (orthogonal)
def rotation_matrix(angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

# 45-degree rotation
R = rotation_matrix(np.pi/4)

# Verify orthogonality
I = np.eye(2)
print("RᵀR = I:", np.allclose(R.T @ R, I))
print("RRᵀ = I:", np.allclose(R @ R.T, I))

# Identity matrix (orthogonal)
I = np.eye(3)
print("Identity is orthogonal:", np.allclose(I.T @ I, I))
```

## Projections

### Vector Projection

The projection of vector u onto vector v is:
```
proj_v(u) = (u·v / ||v||²) × v
```

### Matrix Projection

The projection matrix onto a subspace spanned by columns of A is:
```
P = A(AᵀA)⁻¹Aᵀ
```

### Python Implementation

```python
def vector_projection(u, v):
    """Project vector u onto vector v"""
    return ((u @ v) / (v @ v)) * v

def projection_matrix(A):
    """Projection matrix onto column space of A"""
    return A @ np.linalg.inv(A.T @ A) @ A.T

# Example
u = np.array([3, 4])  # vector to project
v = np.array([1, 0])  # direction to project onto

proj = vector_projection(u, v)
print(f"Projection of u onto v: {proj}")

# Verify projection properties
residual = u - proj
print(f"Residual (u - proj) is orthogonal to v: {(residual @ v) == 0}")
```

## Gram-Schmidt Orthogonalization

### Process

Convert a set of linearly independent vectors into an orthogonal set:

```python
def gram_schmidt(vectors):
    """Convert vectors to orthogonal basis using Gram-Schmidt"""
    n = len(vectors)
    orthogonal_vectors = []
    
    for i in range(n):
        v = vectors[i].copy()
        
        # Subtract projections onto previous vectors
        for j in range(i):
            proj = vector_projection(vectors[i], orthogonal_vectors[j])
            v = v - proj
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm > 1e-10:  # Check if vector is not zero
            orthogonal_vectors.append(v / norm)
    
    return orthogonal_vectors

# Example
vectors = [
    np.array([1, 1, 0]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1])
]

orthogonal_basis = gram_schmidt(vectors)
print("Orthogonal basis:")
for i, v in enumerate(orthogonal_basis):
    print(f"v{i+1}: {v}")
```

## ML Applications

### 1. Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance:

```python
def pca_orthogonal(X, n_components=2):
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvectors (orthogonal by construction)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices[:n_components]]
    
    # Project data onto principal components
    X_pca = X_centered @ principal_components
    
    return X_pca, principal_components

# Verify orthogonality
X = np.random.rand(100, 3)
X_reduced, pc = pca_orthogonal(X, n_components=2)

# Check if principal components are orthogonal
orthogonality_check = pc.T @ pc
print("Principal components orthogonality:")
print(orthogonality_check)
```

### 2. QR Decomposition

QR decomposition factors a matrix into orthogonal and upper triangular:

```python
def qr_decomposition(A):
    """QR decomposition using Gram-Schmidt"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j]
        
        # Orthogonalize against previous columns
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        
        # Normalize
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

# Example
A = np.array([[1, 2], [3, 4], [5, 6]])
Q, R = qr_decomposition(A)

print("Q (orthogonal):")
print(Q)
print("\nR (upper triangular):")
print(R)
print("\nVerify: A = QR")
print(np.allclose(A, Q @ R))
```

### 3. Least Squares Solutions

Orthogonal projections solve least squares problems:

```python
def least_squares_solution(A, b):
    """Solve Ax = b using orthogonal projection"""
    # Project b onto column space of A
    P = projection_matrix(A)
    b_projected = P @ b
    
    # Solve Ax = b_projected
    x = np.linalg.lstsq(A, b_projected, rcond=None)[0]
    
    return x

# Example: linear regression
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([2, 4, 6, 8])

# Add bias term
X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

# Solve least squares
coefficients = least_squares_solution(X_with_bias, y)
print(f"Regression coefficients: {coefficients}")
```

## Geometric Intuition

### 1. Projection as Closest Point

The projection of u onto v gives the closest point to u in the direction of v.

### 2. Residual Minimization

The residual vector (u - proj) is orthogonal to the projection direction, minimizing the distance.

### 3. Coordinate Systems

Orthogonal vectors provide natural coordinate systems where each direction is independent.

## Practice Problems

1. Find the projection of vector [3, 4] onto [1, 1]
2. Implement QR decomposition for a 2×2 matrix
3. Verify that Gram-Schmidt produces orthogonal vectors
4. Use projections to solve a simple least squares problem

## Next Steps

In the next tutorial, we'll explore Singular Value Decomposition (SVD) - a powerful matrix factorization that generalizes many concepts we've learned and is fundamental to modern machine learning algorithms.
