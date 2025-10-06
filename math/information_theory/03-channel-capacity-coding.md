# Information Theory Tutorial 03: Channel Capacity and Coding Theory

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand channel capacity and Shannon's theorem
- Apply error-correcting codes for reliable communication
- Use source coding for data compression
- Understand rate-distortion theory
- Apply coding theory to machine learning problems

## Channel Capacity

### Definition
Channel capacity C is the maximum rate at which information can be reliably transmitted over a communication channel:

C = max I(X;Y)

Where the maximum is over all possible input distributions p(x).

### Shannon's Channel Coding Theorem
For a channel with capacity C and any rate R < C, there exists a code that achieves arbitrarily small error probability.

**Converse**: For any rate R > C, reliable communication is impossible.

### Examples

**Example 1**: Binary Symmetric Channel (BSC)
- Input: X ∈ {0,1}
- Output: Y ∈ {0,1}
- Error probability: p
- Capacity: C = 1 - H(p) where H(p) = -p log p - (1-p) log(1-p)

**Example 2**: Binary Erasure Channel (BEC)
- Input: X ∈ {0,1}
- Output: Y ∈ {0, ?, 1}
- Erasure probability: ε
- Capacity: C = 1 - ε

**Example 3**: Additive White Gaussian Noise (AWGN)
- Input: X with power constraint E[X²] ≤ P
- Output: Y = X + N where N ~ N(0, σ²)
- Capacity: C = (1/2) log(1 + P/σ²)

## Error-Correcting Codes

### Linear Codes
A linear code C is a subspace of F_q^n where F_q is a finite field.

**Parameters**:
- Length: n
- Dimension: k
- Minimum distance: d
- Notation: [n, k, d]_q

### Hamming Codes
Hamming codes are perfect codes that can correct single errors.

**Example**: [7, 4, 3]_2 Hamming code
- Can correct 1 error
- Rate: R = 4/7 ≈ 0.571
- Generator matrix G and parity-check matrix H

### Reed-Solomon Codes
Reed-Solomon codes are maximum distance separable (MDS) codes.

**Properties**:
- Can correct up to (n-k)/2 errors
- Used in CDs, DVDs, QR codes
- Efficient decoding algorithms

### Low-Density Parity-Check (LDPC) Codes
LDPC codes have sparse parity-check matrices and can achieve rates close to capacity.

**Properties**:
- Iterative decoding (belief propagation)
- Near-capacity performance
- Used in modern communication systems

## Source Coding

### Source Coding Theorem
For a source with entropy H(X), we can compress the source to an average of H(X) + ε bits per symbol, for any ε > 0.

### Huffman Coding
Huffman coding is an optimal prefix-free code for given symbol probabilities.

**Algorithm**:
1. Create leaf nodes for each symbol with their probabilities
2. Repeatedly merge two nodes with smallest probabilities
3. Assign 0/1 to edges
4. Read codewords from root to leaves

**Example**: Symbols {A, B, C, D} with probabilities {0.4, 0.3, 0.2, 0.1}
- Average codeword length: 1.9 bits
- Entropy: H(X) = 1.846 bits
- Efficiency: 97.2%

### Arithmetic Coding
Arithmetic coding achieves compression close to entropy limit.

**Advantages**:
- Better compression than Huffman
- Handles adaptive probabilities
- Used in modern compression algorithms

### Lempel-Ziv (LZ) Algorithms
LZ algorithms are universal compression algorithms that don't require knowledge of source statistics.

**LZ77**: Uses sliding window and look-ahead buffer
**LZ78**: Builds dictionary of phrases
**LZW**: Used in GIF, TIFF formats

## Rate-Distortion Theory

### Definition
Rate-distortion theory studies the trade-off between compression rate and reconstruction quality.

**Rate-Distortion Function**:
R(D) = min I(X;X̂) subject to E[d(X,X̂)] ≤ D

Where d(x,x̂) is a distortion measure.

### Distortion Measures

**Hamming Distortion**:
d(x,x̂) = 0 if x = x̂, 1 otherwise

**Squared Error Distortion**:
d(x,x̂) = (x - x̂)²

**Absolute Error Distortion**:
d(x,x̂) = |x - x̂|

### Examples

**Example 1**: Binary source with Hamming distortion
- Source: X ~ Bernoulli(p)
- Distortion: D = 0 → R(D) = H(p)
- Distortion: D = p → R(D) = 0

**Example 2**: Gaussian source with squared error
- Source: X ~ N(0, σ²)
- Rate-distortion function: R(D) = (1/2) log(σ²/D) for D ≤ σ²

## Applications to Machine Learning

### Neural Network Compression
Information theory provides principled approaches to neural network compression:

**Weight Quantization**:
- Reduce precision of weights
- Minimize information loss
- Rate-distortion trade-off

**Pruning**:
- Remove redundant connections
- Information-theoretic criteria for pruning
- Maintain model performance

### Federated Learning
Channel capacity concepts apply to federated learning:

**Communication Efficiency**:
- Compress model updates
- Quantize gradients
- Reduce communication rounds

**Privacy-Utility Trade-off**:
- Differential privacy
- Information-theoretic privacy measures
- Rate-distortion analysis

### Generative Models
Coding theory principles in generative models:

**Variational Autoencoders**:
- Encoder: source coding
- Decoder: channel coding
- Rate-distortion optimization

**Generative Adversarial Networks**:
- Generator: learns to compress data
- Discriminator: learns to detect compression artifacts
- Information-theoretic analysis

## Advanced Coding Techniques

### Polar Codes
Polar codes achieve capacity for symmetric channels.

**Properties**:
- Channel polarization
- Successive cancellation decoding
- Used in 5G communication

### Turbo Codes
Turbo codes use parallel concatenation of convolutional codes.

**Properties**:
- Iterative decoding
- Near-Shannon limit performance
- Used in 3G/4G systems

### Fountain Codes
Fountain codes are rateless codes that can generate unlimited codewords.

**Properties**:
- No fixed rate
- Robust to erasures
- Used in content distribution

## Practical Implementation

### Python Implementation Example

```python
import numpy as np
from scipy.stats import entropy

def channel_capacity_bsc(error_prob):
    """Calculate capacity of Binary Symmetric Channel"""
    if error_prob == 0 or error_prob == 1:
        return 1.0
    h = entropy([error_prob, 1-error_prob], base=2)
    return 1 - h

def huffman_coding(symbols, probabilities):
    """Simple Huffman coding implementation"""
    # Create nodes
    nodes = [(prob, symbol) for symbol, prob in zip(symbols, probabilities)]
    
    # Build tree
    while len(nodes) > 1:
        nodes.sort()
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = (left[0] + right[0], left, right)
        nodes.append(merged)
    
    # Extract codes
    codes = {}
    def extract_codes(node, code=""):
        if len(node) == 2:  # Leaf node
            codes[node[1]] = code
        else:  # Internal node
            extract_codes(node[1], code + "0")
            extract_codes(node[2], code + "1")
    
    extract_codes(nodes[0])
    return codes

def calculate_compression_ratio(symbols, probabilities, codes):
    """Calculate compression ratio"""
    avg_length = sum(prob * len(codes[symbol]) 
                    for symbol, prob in zip(symbols, probabilities))
    entropy_limit = entropy(probabilities, base=2)
    return entropy_limit / avg_length
```

## Practice Problems

### Problem 1
Calculate the capacity of a Binary Symmetric Channel with error probability p = 0.1.

**Solution**:
C = 1 - H(0.1) = 1 - (-0.1 log 0.1 - 0.9 log 0.9)
= 1 - (-0.1(-3.32) - 0.9(-0.152))
= 1 - (0.332 + 0.137) = 0.531 bits

### Problem 2
Design a Huffman code for symbols {A, B, C, D, E} with probabilities {0.3, 0.25, 0.2, 0.15, 0.1}.

**Solution**:
Step 1: Merge E(0.1) and D(0.15) → ED(0.25)
Step 2: Merge ED(0.25) and B(0.25) → BED(0.5)
Step 3: Merge C(0.2) and A(0.3) → AC(0.5)
Step 4: Merge AC(0.5) and BED(0.5) → ACBED(1.0)

Codes:
- A: 00, B: 10, C: 01, D: 111, E: 110

Average length: 0.3×2 + 0.25×2 + 0.2×2 + 0.15×3 + 0.1×3 = 2.25 bits

### Problem 3
For a Gaussian source X ~ N(0, 1) with squared error distortion, find the rate-distortion function for D = 0.5.

**Solution**:
R(D) = (1/2) log(σ²/D) = (1/2) log(1/0.5) = (1/2) log(2) = 0.5 bits

## Key Takeaways
- Channel capacity determines maximum reliable communication rate
- Error-correcting codes enable reliable communication over noisy channels
- Source coding achieves compression close to entropy limit
- Rate-distortion theory quantifies compression-quality trade-offs
- Coding theory principles apply to machine learning problems

## Next Steps
In the next tutorial, we'll explore applications of information theory in machine learning, including feature selection, variational inference, and the information bottleneck principle.
