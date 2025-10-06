# NLP Tutorial Module 2: Mathematical Foundations for NLP

## Learning Objectives
By the end of this module, you will be able to:
- Understand vector spaces and their properties in NLP
- Apply distance and similarity metrics to text
- Work with probability distributions for language modeling
- Understand information theory concepts in NLP
- Apply linear algebra operations to text data
- Implement statistical methods for text analysis

## Introduction to Mathematical Foundations

Natural Language Processing relies heavily on mathematical concepts to represent, analyze, and process text data. This module covers the essential mathematical foundations that underpin modern NLP systems.

### Key Mathematical Areas in NLP

1. **Linear Algebra**: Vector representations, matrix operations
2. **Probability Theory**: Language modeling, uncertainty quantification
3. **Information Theory**: Entropy, mutual information, perplexity
4. **Statistics**: Hypothesis testing, confidence intervals
5. **Optimization**: Gradient descent, convex optimization

## Vector Spaces and Text Representation

### Vector Space Model

Text documents can be represented as vectors in high-dimensional spaces, where each dimension corresponds to a feature (word, n-gram, etc.).

#### Basic Vector Operations

```python
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Create sample document vectors
doc1 = np.array([1, 2, 0, 3, 0])  # "cat dog mouse cat cat"
doc2 = np.array([0, 1, 1, 2, 1])  # "dog mouse cat cat bird"
doc3 = np.array([2, 0, 3, 0, 0])  # "cat cat cat mouse mouse mouse"

print("Document vectors:")
print(f"Doc1: {doc1}")
print(f"Doc2: {doc2}")
print(f"Doc3: {doc3}")

# Vector operations
print(f"\nVector addition (doc1 + doc2): {doc1 + doc2}")
print(f"Scalar multiplication (2 * doc1): {2 * doc1}")
print(f"Dot product (doc1 · doc2): {np.dot(doc1, doc2)}")

# Vector norms
print(f"\nL2 norm of doc1: {np.linalg.norm(doc1):.3f}")
print(f"L1 norm of doc1: {np.sum(np.abs(doc1))}")
```

### Distance and Similarity Metrics

#### Euclidean Distance
Measures straight-line distance between vectors.

```python
def euclidean_distance(v1, v2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((v1 - v2) ** 2))

# Example usage
dist_12 = euclidean_distance(doc1, doc2)
dist_13 = euclidean_distance(doc1, doc3)
print(f"Euclidean distance (doc1, doc2): {dist_12:.3f}")
print(f"Euclidean distance (doc1, doc3): {dist_13:.3f}")
```

#### Cosine Similarity
Measures the cosine of the angle between vectors, ignoring magnitude.

```python
def cosine_similarity_custom(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

# Example usage
cos_sim_12 = cosine_similarity_custom(doc1, doc2)
cos_sim_13 = cosine_similarity_custom(doc1, doc3)
print(f"Cosine similarity (doc1, doc2): {cos_sim_12:.3f}")
print(f"Cosine similarity (doc1, doc3): {cos_sim_13:.3f}")

# Using scikit-learn
cos_sim_matrix = cosine_similarity([doc1, doc2, doc3])
print(f"\nCosine similarity matrix:\n{cos_sim_matrix}")
```

#### Manhattan Distance
Measures distance by summing absolute differences.

```python
def manhattan_distance(v1, v2):
    """Calculate Manhattan distance between two vectors"""
    return np.sum(np.abs(v1 - v2))

# Example usage
manhattan_12 = manhattan_distance(doc1, doc2)
print(f"Manhattan distance (doc1, doc2): {manhattan_12}")
```

### Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection.

#### Mathematical Definition

- **Term Frequency (TF)**: `tf(t,d) = count(t,d) / |d|`
- **Inverse Document Frequency (IDF)**: `idf(t,D) = log(|D| / |{d ∈ D : t ∈ d}|)`
- **TF-IDF**: `tfidf(t,d,D) = tf(t,d) × idf(t,D)`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and dog are friends",
    "the mat is on the floor"
]

# TF-IDF implementation
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Convert to DataFrame for better visualization
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=[f"doc_{i+1}" for i in range(len(documents))]
)

print("TF-IDF Matrix:")
print(tfidf_df.round(3))

# Calculate TF-IDF manually
def calculate_tfidf_manual(documents):
    """Manual TF-IDF calculation"""
    # Step 1: Calculate term frequencies
    tf_matrix = []
    all_words = set()
    
    for doc in documents:
        words = doc.split()
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
            all_words.add(word)
        tf_matrix.append(word_count)
    
    # Step 2: Calculate IDF
    idf_scores = {}
    total_docs = len(documents)
    
    for word in all_words:
        docs_with_word = sum(1 for doc_words in tf_matrix if word in doc_words)
        idf_scores[word] = np.log(total_docs / docs_with_word)
    
    # Step 3: Calculate TF-IDF
    tfidf_matrix = []
    for doc_words in tf_matrix:
        doc_length = sum(doc_words.values())
        tfidf_scores = {}
        for word, count in doc_words.items():
            tf = count / doc_length
            idf = idf_scores[word]
            tfidf_scores[word] = tf * idf
        tfidf_matrix.append(tfidf_scores)
    
    return tfidf_matrix, idf_scores

# Manual calculation
manual_tfidf, idf_scores = calculate_tfidf_manual(documents)
print(f"\nManual TF-IDF calculation:")
for i, scores in enumerate(manual_tfidf):
    print(f"Doc {i+1}: {scores}")
```

## Probability Theory in NLP

### Language Models

Language models assign probabilities to sequences of words.

#### N-gram Language Model

```python
from collections import defaultdict, Counter
import math

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.vocabulary = set()
        
    def train(self, texts):
        """Train the n-gram model"""
        for text in texts:
            tokens = text.split()
            self.vocabulary.update(tokens)
            
            # Add padding
            padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            # Count n-grams
            for i in range(len(padded_tokens) - self.n + 1):
                ngram = tuple(padded_tokens[i:i + self.n])
                prefix = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[prefix][word] += 1
    
    def get_probability(self, prefix, word):
        """Get probability of word given prefix"""
        prefix = tuple(prefix)
        if prefix not in self.ngram_counts:
            return 0
        
        total_count = sum(self.ngram_counts[prefix].values())
        if total_count == 0:
            return 0
        
        return self.ngram_counts[prefix][word] / total_count
    
    def get_sentence_probability(self, sentence):
        """Calculate probability of entire sentence"""
        tokens = sentence.split()
        padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
        
        probability = 1.0
        for i in range(len(padded_tokens) - self.n + 1):
            prefix = tuple(padded_tokens[i:i + self.n - 1])
            word = padded_tokens[i + self.n - 1]
            prob = self.get_probability(prefix, word)
            probability *= prob
        
        return probability
    
    def get_perplexity(self, test_sentences):
        """Calculate perplexity on test data"""
        total_log_prob = 0
        total_words = 0
        
        for sentence in test_sentences:
            tokens = sentence.split()
            total_words += len(tokens)
            prob = self.get_sentence_probability(sentence)
            if prob > 0:
                total_log_prob += math.log(prob)
            else:
                # Handle zero probability
                total_log_prob += -float('inf')
        
        if total_log_prob == -float('inf'):
            return float('inf')
        
        avg_log_prob = total_log_prob / len(test_sentences)
        perplexity = math.exp(-avg_log_prob)
        return perplexity

# Example usage
training_texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and dog are friends",
    "the mat is on the floor"
]

# Train bigram model
bigram_model = NGramLanguageModel(2)
bigram_model.train(training_texts)

# Test sentences
test_sentence = "the cat sat on the mat"
prob = bigram_model.get_sentence_probability(test_sentence)
print(f"Probability of '{test_sentence}': {prob:.6f}")

# Calculate perplexity
test_sentences = ["the cat is happy", "the dog runs fast"]
perplexity = bigram_model.get_perplexity(test_sentences)
print(f"Perplexity: {perplexity:.3f}")
```

### Smoothing Techniques

#### Laplace Smoothing (Add-One Smoothing)

```python
class SmoothedNGramModel(NGramLanguageModel):
    def __init__(self, n, alpha=1.0):
        super().__init__(n)
        self.alpha = alpha
    
    def get_probability(self, prefix, word):
        """Get smoothed probability of word given prefix"""
        prefix = tuple(prefix)
        
        # Count for this specific n-gram
        count = self.ngram_counts[prefix][word]
        
        # Total count for this prefix
        total_count = sum(self.ngram_counts[prefix].values())
        
        # Vocabulary size
        vocab_size = len(self.vocabulary)
        
        # Laplace smoothing
        smoothed_prob = (count + self.alpha) / (total_count + self.alpha * vocab_size)
        
        return smoothed_prob

# Example usage
smoothed_model = SmoothedNGramModel(2, alpha=1.0)
smoothed_model.train(training_texts)

prob = smoothed_model.get_probability(['the'], 'cat')
print(f"Smoothed probability P(cat|the): {prob:.4f}")
```

## Information Theory in NLP

### Entropy

Entropy measures the uncertainty or information content of a probability distribution.

```python
import math

def entropy(probabilities):
    """Calculate Shannon entropy"""
    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

def conditional_entropy(joint_probs, marginal_probs):
    """Calculate conditional entropy H(Y|X)"""
    cond_entropy = 0
    for x, x_prob in marginal_probs.items():
        if x_prob > 0:
            y_probs = [joint_probs[(x, y)] / x_prob 
                      for y in joint_probs.keys() if y[0] == x]
            cond_entropy += x_prob * entropy(y_probs)
    return cond_entropy

# Example: Calculate entropy of word distribution
word_counts = {'the': 50, 'cat': 20, 'dog': 15, 'sat': 10, 'mat': 5}
total_count = sum(word_counts.values())
word_probs = [count / total_count for count in word_counts.values()]

word_entropy = entropy(word_probs)
print(f"Word entropy: {word_entropy:.3f} bits")
```

### Mutual Information

Mutual information measures the amount of information shared between two variables.

```python
def mutual_information(joint_probs, x_marginal, y_marginal):
    """Calculate mutual information I(X;Y)"""
    mi = 0
    for (x, y), joint_prob in joint_probs.items():
        if joint_prob > 0 and x_marginal[x] > 0 and y_marginal[y] > 0:
            mi += joint_prob * math.log2(joint_prob / (x_marginal[x] * y_marginal[y]))
    return mi

# Example: Calculate mutual information between adjacent words
def calculate_word_mi(texts, window_size=2):
    """Calculate mutual information between words in a window"""
    word_counts = Counter()
    pair_counts = Counter()
    
    for text in texts:
        words = text.split()
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            for word in window:
                word_counts[word] += 1
            
            for j in range(len(window)):
                for k in range(j + 1, len(window)):
                    pair = tuple(sorted([window[j], window[k]]))
                    pair_counts[pair] += 1
    
    total_words = sum(word_counts.values())
    total_pairs = sum(pair_counts.values())
    
    # Calculate probabilities
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    pair_probs = {pair: count / total_pairs for pair, count in pair_counts.items()}
    
    # Calculate mutual information
    mi_scores = {}
    for (w1, w2), pair_prob in pair_probs.items():
        if w1 in word_probs and w2 in word_probs:
            mi = pair_prob * math.log2(pair_prob / (word_probs[w1] * word_probs[w2]))
            mi_scores[(w1, w2)] = mi
    
    return mi_scores

# Example usage
texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and dog are friends"
]

mi_scores = calculate_word_mi(texts)
print("Mutual information scores:")
for pair, score in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{pair}: {score:.3f}")
```

## Linear Algebra Operations for NLP

### Matrix Operations for Text Processing

```python
import numpy as np
from scipy.sparse import csr_matrix

# Document-term matrix
def create_document_term_matrix(documents, vocabulary):
    """Create document-term matrix"""
    matrix = np.zeros((len(documents), len(vocabulary)))
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
    
    for doc_idx, doc in enumerate(documents):
        words = doc.split()
        for word in words:
            if word in vocab_dict:
                matrix[doc_idx, vocab_dict[word]] += 1
    
    return matrix, vocab_dict

# Example usage
documents = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and dog are friends"
]

vocabulary = list(set(" ".join(documents).split()))
dt_matrix, vocab_dict = create_document_term_matrix(documents, vocabulary)

print("Document-Term Matrix:")
print(f"Vocabulary: {vocabulary}")
print(f"Matrix shape: {dt_matrix.shape}")
print(dt_matrix)

# Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(dt_matrix, full_matrices=False)

print(f"\nSVD Results:")
print(f"U shape: {U.shape}")
print(f"s shape: {s.shape}")
print(f"Vt shape: {Vt.shape}")
print(f"Singular values: {s}")

# Dimensionality reduction using SVD
k = 2  # Number of components to keep
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

print(f"\nReduced representation (k={k}):")
print(f"U_k: {U_k}")
print(f"s_k: {s_k}")
```

### Eigenvalue Decomposition

```python
# Covariance matrix
def calculate_covariance_matrix(vectors):
    """Calculate covariance matrix of vectors"""
    mean_vector = np.mean(vectors, axis=0)
    centered_vectors = vectors - mean_vector
    covariance_matrix = np.dot(centered_vectors.T, centered_vectors) / (len(vectors) - 1)
    return covariance_matrix, mean_vector

# Example usage
vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
cov_matrix, mean_vec = calculate_covariance_matrix(vectors)

print("Covariance Matrix:")
print(cov_matrix)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Principal Component Analysis (PCA)
def pca(vectors, n_components=2):
    """Perform PCA on vectors"""
    # Center the data
    mean_vector = np.mean(vectors, axis=0)
    centered_vectors = vectors - mean_vector
    
    # Calculate covariance matrix
    cov_matrix = np.dot(centered_vectors.T, centered_vectors) / (len(vectors) - 1)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    transformed_data = np.dot(centered_vectors, components)
    
    return transformed_data, components, eigenvalues

# Example usage
pca_result, components, explained_variance = pca(vectors, n_components=2)
print(f"\nPCA Results:")
print(f"Transformed data:\n{pca_result}")
print(f"Components:\n{components}")
print(f"Explained variance: {explained_variance}")
```

## Statistical Methods for Text Analysis

### Chi-Square Test for Feature Selection

```python
from scipy.stats import chi2_contingency
import pandas as pd

def chi_square_feature_selection(documents, labels, vocabulary):
    """Perform chi-square test for feature selection"""
    chi_scores = {}
    
    for word in vocabulary:
        # Create contingency table
        word_present_class1 = sum(1 for i, doc in enumerate(documents) 
                                 if word in doc and labels[i] == 0)
        word_absent_class1 = sum(1 for i, doc in enumerate(documents) 
                                if word not in doc and labels[i] == 0)
        word_present_class2 = sum(1 for i, doc in enumerate(documents) 
                                 if word in doc and labels[i] == 1)
        word_absent_class2 = sum(1 for i, doc in enumerate(documents) 
                                if word not in doc and labels[i] == 1)
        
        contingency_table = [[word_present_class1, word_absent_class1],
                           [word_present_class2, word_absent_class2]]
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        chi_scores[word] = (chi2, p_value)
    
    return chi_scores

# Example usage
documents = [
    "good movie great film",
    "bad movie terrible film",
    "excellent movie wonderful film",
    "awful movie horrible film"
]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative
vocabulary = ["good", "bad", "movie", "film", "great", "terrible", "excellent", "awful", "wonderful", "horrible"]

chi_scores = chi_square_feature_selection(documents, labels, vocabulary)
print("Chi-square scores:")
for word, (chi2, p_value) in sorted(chi_scores.items(), key=lambda x: x[1][0], reverse=True):
    print(f"{word}: χ²={chi2:.3f}, p={p_value:.3f}")
```

### Hypothesis Testing for Text Analysis

```python
from scipy.stats import ttest_ind, mannwhitneyu

def compare_word_frequencies(group1_docs, group2_docs, word):
    """Compare word frequencies between two groups"""
    # Calculate word frequencies for each document
    group1_freqs = []
    for doc in group1_docs:
        words = doc.split()
        freq = words.count(word) / len(words) if len(words) > 0 else 0
        group1_freqs.append(freq)
    
    group2_freqs = []
    for doc in group2_docs:
        words = doc.split()
        freq = words.count(word) / len(words) if len(words) > 0 else 0
        group2_freqs.append(freq)
    
    # Perform t-test
    t_stat, p_value = ttest_ind(group1_freqs, group2_freqs)
    
    # Perform Mann-Whitney U test (non-parametric)
    u_stat, u_p_value = mannwhitneyu(group1_freqs, group2_freqs, alternative='two-sided')
    
    return {
        'word': word,
        'group1_mean': np.mean(group1_freqs),
        'group2_mean': np.mean(group2_freqs),
        't_statistic': t_stat,
        't_p_value': p_value,
        'u_statistic': u_stat,
        'u_p_value': u_p_value
    }

# Example usage
positive_docs = [
    "great movie amazing film",
    "excellent performance wonderful acting",
    "fantastic story brilliant plot"
]

negative_docs = [
    "terrible movie awful film",
    "bad performance poor acting",
    "boring story dull plot"
]

results = compare_word_frequencies(positive_docs, negative_docs, "movie")
print("Statistical comparison:")
for key, value in results.items():
    print(f"{key}: {value}")
```

## Practical Exercises

### Exercise 1: Vector Space Operations
Implement a complete text similarity system that:
- Creates document vectors using TF-IDF
- Calculates multiple similarity metrics
- Ranks documents by similarity to a query

### Exercise 2: Language Model Implementation
Build a trigram language model with:
- Laplace smoothing
- Good-Turing smoothing
- Perplexity calculation
- Text generation capabilities

### Exercise 3: Information Theoretic Analysis
Analyze a text corpus to:
- Calculate word and bigram entropies
- Find the most informative word pairs (mutual information)
- Compare entropy across different text genres

## Assessment Questions

1. **What is the difference between cosine similarity and Euclidean distance?**
   - Cosine similarity: measures angle, ignores magnitude
   - Euclidean distance: measures straight-line distance
   - Cosine similarity is better for text similarity
   - Euclidean distance is sensitive to document length

2. **How does TF-IDF address the limitations of simple word frequency?**
   - Reduces importance of common words
   - Increases importance of rare but meaningful words
   - Balances local and global word importance
   - Provides better document representations

3. **What is perplexity and why is it useful for language models?**
   - Perplexity: measure of model uncertainty
   - Lower perplexity = better model
   - Allows comparison between different models
   - Related to entropy and information content

## Key Takeaways

- Vector spaces provide a mathematical framework for text representation
- Probability theory enables language modeling and uncertainty quantification
- Information theory helps measure and optimize information content
- Linear algebra operations enable dimensionality reduction and feature extraction
- Statistical methods provide rigorous approaches to text analysis
- Mathematical foundations are essential for understanding modern NLP systems

## Next Steps

In the next module, we'll explore traditional NLP models including n-gram models, Hidden Markov Models, and probabilistic context-free grammars. These models form the foundation for understanding more advanced neural approaches.
