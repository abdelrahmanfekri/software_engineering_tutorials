# 7. Advanced Matrix Decompositions: QR, LU, Cholesky, and Jordan Forms

## Introduction

Matrix decompositions are fundamental tools that break down complex matrices into simpler, more manageable components. This advanced tutorial covers the most important decompositions used in machine learning, scientific computing, and numerical analysis.

## QR Decomposition

### Definition and Theory

**QR decomposition** factors any matrix A (m×n) into:
```
A = QR
```

Where:
- **Q**: m×m orthogonal matrix (QᵀQ = I)
- **R**: m×n upper triangular matrix

### Mathematical Properties

1. **Uniqueness**: If A has full column rank, R has positive diagonal elements and Q is unique
2. **Stability**: QR is numerically stable for solving least squares problems
3. **Orthogonality**: Q preserves lengths and angles

### Implementation Methods

#### 1. Gram-Schmidt Process

```python
import numpy as np
from typing import Tuple
import scipy.linalg as la

def gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition using Gram-Schmidt orthogonalization"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        # Orthogonalize against previous columns
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        
        # Normalize
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-12:  # Avoid division by zero
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v  # Zero vector
    
    return Q, R

def modified_gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Modified Gram-Schmidt for better numerical stability"""
    m, n = A.shape
    Q = A.copy().astype(float)
    R = np.zeros((n, n))
    
    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        if R[i, i] > 1e-12:
            Q[:, i] = Q[:, i] / R[i, i]
            
            # Orthogonalize remaining columns
            for j in range(i + 1, n):
                R[i, j] = Q[:, i] @ Q[:, j]
                Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
    
    return Q, R
```

#### 2. Householder Reflections

```python
def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition using Householder reflections"""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    
    for k in range(min(m, n)):
        # Extract column k from row k onwards
        x = R[k:, k]
        
        # Compute Householder vector
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        v = x.copy()
        v[0] = v[0] - alpha
        
        if np.linalg.norm(v) > 1e-12:
            v = v / np.linalg.norm(v)
            
            # Apply Householder reflection
            # R[k:, k:] = R[k:, k:] - 2 * v @ (v.T @ R[k:, k:])
            R[k:, k:] = R[k:, k:] - 2 * np.outer(v, v.T @ R[k:, k:])
            
            # Update Q
            Q[k:, :] = Q[k:, :] - 2 * np.outer(v, v.T @ Q[k:, :])
    
    return Q, R

def householder_vector(x: np.ndarray) -> np.ndarray:
    """Compute Householder vector for reflection"""
    n = len(x)
    v = x.copy()
    
    # Choose sign to avoid cancellation
    if v[0] >= 0:
        v[0] = v[0] + np.linalg.norm(v)
    else:
        v[0] = v[0] - np.linalg.norm(v)
    
    return v / np.linalg.norm(v)
```

#### 3. Givens Rotations

```python
def givens_rotation(i: int, j: int, angle: float, n: int) -> np.ndarray:
    """Create Givens rotation matrix"""
    G = np.eye(n)
    c = np.cos(angle)
    s = np.sin(angle)
    
    G[i, i] = c
    G[i, j] = -s
    G[j, i] = s
    G[j, j] = c
    
    return G

def givens_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition using Givens rotations"""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    
    for j in range(n):
        for i in range(m - 1, j, -1):
            if abs(R[i, j]) > 1e-12:
                # Compute rotation angle
                r = np.sqrt(R[i-1, j]**2 + R[i, j]**2)
                c = R[i-1, j] / r
                s = R[i, j] / r
                
                # Create Givens rotation
                G = givens_rotation(i-1, i, np.arctan2(s, c), m)
                
                # Apply rotation
                R = G @ R
                Q = G @ Q
    
    return Q.T, R
```

### Applications in Machine Learning

#### 1. Least Squares Solutions

```python
def qr_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve least squares problem using QR decomposition"""
    Q, R = np.linalg.qr(A)
    
    # Solve Rx = Q^T b
    b_projected = Q.T @ b
    x = la.solve_triangular(R, b_projected)
    
    return x

def qr_regularized_least_squares(A: np.ndarray, b: np.ndarray, 
                                lambda_reg: float) -> np.ndarray:
    """Ridge regression using QR decomposition"""
    m, n = A.shape
    
    # Augment system: [A; λI] x = [b; 0]
    A_aug = np.vstack([A, np.sqrt(lambda_reg) * np.eye(n)])
    b_aug = np.hstack([b, np.zeros(n)])
    
    return qr_least_squares(A_aug, b_aug)

# Example: polynomial regression
def polynomial_regression_qr(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial regression using QR decomposition"""
    # Create Vandermonde matrix
    A = np.column_stack([x**i for i in range(degree + 1)])
    
    # Solve using QR
    coefficients = qr_least_squares(A, y)
    
    return coefficients

# Example usage
x = np.linspace(0, 1, 20)
y = 2 * x**2 + 3 * x + 1 + 0.1 * np.random.randn(20)

coeffs = polynomial_regression_qr(x, y, degree=2)
print(f"Polynomial coefficients: {coeffs}")
```

#### 2. Eigenvalue Computation (QR Algorithm)

```python
def qr_algorithm(A: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """QR algorithm for computing eigenvalues"""
    n = A.shape[0]
    A_k = A.copy()
    Q_total = np.eye(n)
    
    for k in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q
        
        # Check convergence (off-diagonal elements)
        off_diag = np.sum(np.abs(A_k - np.diag(np.diag(A_k))))
        if off_diag < tol:
            break
    
    eigenvalues = np.diag(A_k)
    eigenvectors = Q_total
    
    return eigenvalues, eigenvectors

# Example
A = np.array([[4, 1, 2], [1, 3, 1], [2, 1, 5]])
eigenvals, eigenvecs = qr_algorithm(A)
print(f"Eigenvalues: {eigenvals}")
```

## LU Decomposition

### Definition and Theory

**LU decomposition** factors a square matrix A into:
```
A = LU
```

Where:
- **L**: lower triangular matrix with ones on diagonal
- **U**: upper triangular matrix

### Existence and Uniqueness

**Theorem**: A square matrix A has an LU decomposition if and only if all leading principal minors are non-zero.

### Implementation

#### 1. Doolittle Algorithm

```python
def doolittle_lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LU decomposition using Doolittle algorithm"""
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)
    
    for i in range(n):
        # Compute U[i, j]
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        
        # Compute L[j, i] for j > i
        for j in range(i + 1, n):
            if abs(U[i, i]) < 1e-12:
                raise ValueError("Matrix is singular or near-singular")
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]
    
    return L, U

def lu_with_pivoting(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU decomposition with partial pivoting: PA = LU"""
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    P = np.eye(n)
    
    for k in range(n - 1):
        # Find pivot
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        
        if pivot_row != k:
            # Swap rows in U and P
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            
            # Swap rows in L (only below diagonal)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Check for singularity
        if abs(U[k, k]) < 1e-12:
            raise ValueError("Matrix is singular")
        
        # Eliminate column k
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
    
    return L, U, P
```

#### 2. Crout Algorithm

```python
def crout_lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LU decomposition using Crout algorithm"""
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.eye(n)
    
    for j in range(n):
        # Compute L[i, j] for i >= j
        for i in range(j, n):
            L[i, j] = A[i, j] - np.sum(L[i, :j] * U[:j, j])
        
        # Compute U[j, i] for i > j
        for i in range(j + 1, n):
            if abs(L[j, j]) < 1e-12:
                raise ValueError("Matrix is singular")
            U[j, i] = (A[j, i] - np.sum(L[j, :j] * U[:j, i])) / L[j, j]
    
    return L, U
```

### Applications

#### 1. Solving Linear Systems

```python
def lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using LU decomposition"""
    n = len(b)
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])
    
    # Backward substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i+1:] * x[i+1:])) / U[i, i]
    
    return x

def lu_solve_with_pivoting(L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using LU decomposition with pivoting"""
    # Solve PAx = Pb, so LUx = Pb
    Pb = P @ b
    return lu_solve(L, U, Pb)

# Example
A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
b = np.array([1, 2, 3])

L, U, P = lu_with_pivoting(A)
x = lu_solve_with_pivoting(L, U, P, b)

print(f"Solution: {x}")
print(f"Verification: A @ x = {A @ x}")
```

#### 2. Matrix Inversion

```python
def lu_inverse(L: np.ndarray, U: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute matrix inverse using LU decomposition"""
    n = L.shape[0]
    A_inv = np.zeros_like(L)
    
    # Solve AX = I column by column
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        A_inv[:, i] = lu_solve_with_pivoting(L, U, P, e_i)
    
    return A_inv

# Example
A_inv = lu_inverse(L, U, P)
print(f"Matrix inverse:")
print(A_inv)
print(f"Verification: A @ A_inv = I? {np.allclose(A @ A_inv, np.eye(3))}")
```

## Cholesky Decomposition

### Definition and Theory

**Cholesky decomposition** factors a positive definite matrix A into:
```
A = LLᵀ
```

Where L is lower triangular with positive diagonal elements.

### Properties

1. **Uniqueness**: If A is positive definite, L is unique
2. **Stability**: More stable than LU for positive definite matrices
3. **Efficiency**: Requires half the storage and computation of LU

### Implementation

```python
def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """Cholesky decomposition A = LL^T"""
    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric")
    
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal element
                sum_sq = np.sum(L[i, :i]**2)
                if A[i, i] - sum_sq < 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, i] = np.sqrt(A[i, i] - sum_sq)
            else:
                # Off-diagonal element
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    
    return L

def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using Cholesky decomposition"""
    n = len(b)
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]
    
    # Backward substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(L[i+1:, i] * x[i+1:])) / L[i, i]
    
    return x

def modified_cholesky(A: np.ndarray, delta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Modified Cholesky for near-singular matrices"""
    n = A.shape[0]
    L = np.zeros_like(A)
    D = np.zeros(n)
    
    for i in range(n):
        # Compute diagonal element
        sum_sq = np.sum(L[i, :i]**2 * D[:i])
        d_i = A[i, i] - sum_sq
        
        # Add regularization if necessary
        if d_i < delta:
            d_i = delta
        
        D[i] = d_i
        L[i, i] = 1
        
        # Compute off-diagonal elements
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * L[i, :i] * D[:i])) / D[i]
    
    return L, D

# Example: positive definite matrix
A = np.array([[4, 2, 1], [2, 5, 2], [1, 2, 6]])
b = np.array([1, 2, 3])

L = cholesky_decomposition(A)
x = cholesky_solve(L, b)

print(f"Cholesky factor L:")
print(L)
print(f"Solution: {x}")
print(f"Verification: A @ x = {A @ x}")
```

### Applications in Machine Learning

#### 1. Gaussian Process Regression

```python
def gaussian_process_regression(X_train: np.ndarray, y_train: np.ndarray, 
                               X_test: np.ndarray, kernel_func, noise_var: float = 0.01):
    """Gaussian process regression using Cholesky decomposition"""
    n_train = X_train.shape[0]
    
    # Compute kernel matrix
    K = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            K[i, j] = kernel_func(X_train[i], X_train[j])
    
    # Add noise to diagonal
    K += noise_var * np.eye(n_train)
    
    # Cholesky decomposition
    L = cholesky_decomposition(K)
    
    # Solve for alpha: K alpha = y
    alpha = cholesky_solve(L, y_train)
    
    # Predictions
    n_test = X_test.shape[0]
    predictions = np.zeros(n_test)
    variances = np.zeros(n_test)
    
    for i in range(n_test):
        # Compute kernel vector
        k_star = np.array([kernel_func(X_test[i], X_train[j]) for j in range(n_train)])
        
        # Mean prediction
        predictions[i] = k_star @ alpha
        
        # Variance prediction
        v = la.solve_triangular(L, k_star, lower=True)
        k_star_star = kernel_func(X_test[i], X_test[i])
        variances[i] = k_star_star - v @ v
    
    return predictions, variances

# Example kernel function
def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> float:
    """RBF kernel function"""
    return np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)

# Example usage
np.random.seed(42)
X_train = np.linspace(0, 2*np.pi, 10).reshape(-1, 1)
y_train = np.sin(X_train.flatten()) + 0.1 * np.random.randn(10)

X_test = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)

predictions, variances = gaussian_process_regression(X_train, y_train, X_test, rbf_kernel)

print(f"GP predictions shape: {predictions.shape}")
print(f"GP variances shape: {variances.shape}")
```

## Jordan Canonical Form

### Definition and Theory

**Jordan canonical form** represents a matrix A as:
```
A = PJP⁻¹
```

Where J is a block diagonal matrix with Jordan blocks on the diagonal.

### Jordan Blocks

A Jordan block of size k with eigenvalue λ is:
```
J_k(λ) = [λ  1  0  ...  0]
         [0  λ  1  ...  0]
         [0  0  λ  ...  0]
         [⋮  ⋮  ⋮  ⋱  ⋮]
         [0  0  0  ...  λ]
```

### Implementation

```python
def jordan_form(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jordan canonical form (simplified implementation)"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # For simplicity, assume distinct eigenvalues
    # In practice, this is much more complex
    n = A.shape[0]
    J = np.zeros_like(A, dtype=complex)
    P = np.zeros_like(A, dtype=complex)
    
    # Sort eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Create Jordan form (diagonal for distinct eigenvalues)
    for i in range(n):
        J[i, i] = eigenvalues[i]
        P[:, i] = eigenvectors[:, i]
    
    return J, P

def matrix_exponential_jordan(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Compute matrix exponential using Jordan form"""
    J, P = jordan_form(A)
    
    # For diagonal Jordan form: exp(Jt) = diag(exp(λᵢt))
    exp_J = np.diag(np.exp(eigenvalues * t))
    
    # exp(At) = P * exp(Jt) * P⁻¹
    exp_A = P @ exp_J @ np.linalg.inv(P)
    
    return np.real(exp_A)

# Example
A = np.array([[1, 1], [0, 2]])
J, P = jordan_form(A)
exp_A = matrix_exponential_jordan(A, t=1.0)

print(f"Jordan form J:")
print(J)
print(f"Matrix exponential exp(A):")
print(exp_A)
```

## Performance Comparison

```python
def benchmark_decompositions(n: int = 100) -> dict:
    """Benchmark different decomposition methods"""
    import time
    
    # Generate test matrix
    A = np.random.randn(n, n)
    A = A @ A.T  # Make positive definite for Cholesky
    
    results = {}
    
    # QR decomposition
    start_time = time.time()
    Q, R = np.linalg.qr(A)
    results['QR'] = time.time() - start_time
    
    # LU decomposition
    start_time = time.time()
    L, U, P = lu_with_pivoting(A)
    results['LU'] = time.time() - start_time
    
    # Cholesky decomposition
    start_time = time.time()
    L_chol = cholesky_decomposition(A)
    results['Cholesky'] = time.time() - start_time
    
    # SVD (for comparison)
    start_time = time.time()
    U_svd, S, Vt = np.linalg.svd(A)
    results['SVD'] = time.time() - start_time
    
    return results

# Benchmark
n = 200
benchmark_results = benchmark_decompositions(n)
print(f"Decomposition Benchmark (Matrix size: {n}x{n}):")
for method, time_taken in benchmark_results.items():
    print(f"{method:10}: {time_taken:.4f}s")
```

## Advanced Applications

### 1. Kalman Filter Implementation

```python
class KalmanFilter:
    """Kalman filter using Cholesky decomposition for numerical stability"""
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        
        # Initialize state and covariance
        self.n_states = F.shape[0]
        self.n_obs = H.shape[0]
        
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states)
    
    def predict(self):
        """Prediction step"""
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray):
        """Update step using Cholesky decomposition"""
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Cholesky decomposition for numerical stability
        L_S = cholesky_decomposition(S)
        
        # Kalman gain
        K = self.P @ self.H.T @ la.solve_triangular(L_S, 
                                                   la.solve_triangular(L_S, 
                                                                      np.eye(self.n_obs), 
                                                                      lower=True), 
                                                   lower=True)
        
        # Update state
        y = z - self.H @ self.x  # Innovation
        self.x = self.x + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
    
    def filter(self, measurements: np.ndarray) -> np.ndarray:
        """Run Kalman filter on sequence of measurements"""
        n_steps = measurements.shape[0]
        filtered_states = np.zeros((n_steps, self.n_states))
        
        for t in range(n_steps):
            self.predict()
            self.update(measurements[t])
            filtered_states[t] = self.x.copy()
        
        return filtered_states

# Example: 1D position tracking
dt = 0.1
F = np.array([[1, dt], [0, 1]])  # Constant velocity model
H = np.array([[1, 0]])  # Observe position only
Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * 0.1  # Process noise
R = np.array([[1.0]])  # Measurement noise

kf = KalmanFilter(F, H, Q, R)

# Simulate measurements
true_positions = np.linspace(0, 10, 100)
measurements = true_positions + np.random.randn(100) * 0.5

filtered_states = kf.filter(measurements.reshape(-1, 1))
filtered_positions = filtered_states[:, 0]

print(f"Kalman filter applied to {len(measurements)} measurements")
```

### 2. Robust Linear Regression

```python
def robust_regression_qr(X: np.ndarray, y: np.ndarray, 
                        max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Robust regression using iteratively reweighted least squares"""
    n, p = X.shape
    weights = np.ones(n)
    
    for iteration in range(max_iter):
        # Weighted least squares using QR
        W_sqrt = np.diag(np.sqrt(weights))
        X_weighted = W_sqrt @ X
        y_weighted = W_sqrt @ y
        
        # Solve using QR decomposition
        Q, R = np.linalg.qr(X_weighted)
        b = Q.T @ y_weighted
        coefficients = la.solve_triangular(R, b)
        
        # Compute residuals
        residuals = y - X @ coefficients
        
        # Update weights (Huber's robust function)
        sigma = np.median(np.abs(residuals)) / 0.6745  # Robust scale estimate
        threshold = 1.345 * sigma
        
        weights = np.where(np.abs(residuals) <= threshold, 
                          1.0, 
                          threshold / np.abs(residuals))
        
        # Check convergence
        if iteration > 0 and np.linalg.norm(coefficients - coefficients_old) < tol:
            break
        
        coefficients_old = coefficients.copy()
    
    return coefficients

# Example with outliers
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
true_coeffs = np.array([2.0, -1.5])
y = X @ true_coeffs + 0.1 * np.random.randn(n)

# Add outliers
outlier_indices = np.random.choice(n, 10, replace=False)
y[outlier_indices] += 5 * np.random.randn(10)

# Compare ordinary and robust regression
coeffs_ols = qr_least_squares(X, y)
coeffs_robust = robust_regression_qr(X, y)

print(f"True coefficients: {true_coeffs}")
print(f"OLS coefficients: {coeffs_ols}")
print(f"Robust coefficients: {coeffs_robust}")
```

## Practice Problems

### Theoretical Challenges

1. **Prove the uniqueness** of Cholesky decomposition for positive definite matrices
2. **Show that QR decomposition** is numerically stable for least squares problems
3. **Prove that LU decomposition** exists if and only if all leading principal minors are non-zero
4. **Analyze the computational complexity** of each decomposition method

### Implementation Challenges

1. **Implement a parallel QR decomposition** using distributed computing
2. **Create a sparse matrix version** of LU decomposition
3. **Build a matrix factorization library** with automatic method selection
4. **Implement the QR algorithm** for computing eigenvalues of large matrices

### Research Applications

1. **Apply Cholesky decomposition** to Gaussian process regression with millions of data points
2. **Use QR decomposition** for online learning algorithms
3. **Implement robust matrix decompositions** for corrupted data
4. **Create a distributed matrix factorization system** for recommendation engines

## Next Steps

In the next tutorial, we'll explore **Linear Transformations and Change of Basis**, covering:
- **Coordinate systems** and basis transformations
- **Linear maps** and their matrix representations
- **Similarity transformations** and canonical forms
- **Affine transformations** and homogeneous coordinates
- **Applications** in computer graphics and robotics

This foundation in matrix decompositions will enable you to understand and implement the most sophisticated numerical algorithms used in modern machine learning and scientific computing.
