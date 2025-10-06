# NLP Tutorial Module 4: Word Embeddings and Distributed Representations

## Learning Objectives
By the end of this module, you will be able to:
- Understand the theory behind word embeddings
- Implement Word2Vec (Skip-gram and CBOW) from scratch
- Work with GloVe embeddings
- Apply FastText for subword modeling
- Understand and use pre-trained embeddings
- Evaluate embedding quality
- Apply embeddings to downstream NLP tasks

## Introduction to Word Embeddings

Word embeddings are dense vector representations of words that capture semantic and syntactic relationships. Unlike sparse representations (like one-hot vectors), embeddings encode meaning in continuous vector spaces.

### Why Word Embeddings?

1. **Dimensionality Reduction**: Compact representation of vocabulary
2. **Semantic Similarity**: Similar words have similar vectors
3. **Algebraic Properties**: Word relationships can be expressed as vector operations
4. **Transfer Learning**: Pre-trained embeddings improve downstream tasks

### Key Properties of Good Embeddings

- **Similarity**: Related words have high cosine similarity
- **Analogical Relationships**: king - man + woman ≈ queen
- **Compositionality**: Word meanings compose in predictable ways
- **Context Sensitivity**: Words can have multiple meanings

## Word2Vec: Skip-gram and CBOW

Word2Vec learns embeddings by predicting words in context, using two main architectures:

### Skip-gram Model
Predicts context words given a center word.

### CBOW Model
Predicts center word given context words.

### Mathematical Foundation

For Skip-gram, the objective is to maximize:
```
L = (1/T) ∑ᵗ₌₁ᵀ ∑₋c≤j≤c,j≠0 log P(wₜ₊ⱼ | wₜ)
```

Where the probability is computed using softmax:
```
P(wₒ | wᵢ) = exp(vₒᵀ · vᵢ) / ∑ᵧ₌₁ᵂ exp(vᵧᵀ · vᵢ)
```

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Word2Vec:
    def __init__(self, vocabulary_size, embedding_dim, window_size=5, negative_samples=5):
        self.vocab_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        
        # Initialize embeddings
        self.input_embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.output_embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        
        # Vocabulary and mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_frequencies = Counter()
        
        # Negative sampling table
        self.negative_sampling_table = None
        
    def build_vocabulary(self, texts, min_count=5):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            self.word_frequencies.update(words)
        
        # Filter by minimum count
        filtered_words = {word: count for word, count in self.word_frequencies.items() 
                         if count >= min_count}
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(filtered_words.keys()))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Update vocabulary size
        self.vocab_size = len(self.word_to_idx)
        
        # Reinitialize embeddings
        self.input_embeddings = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
        self.output_embeddings = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
        
        # Build negative sampling table
        self._build_negative_sampling_table()
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _build_negative_sampling_table(self):
        """Build table for negative sampling"""
        # Calculate frequencies
        frequencies = np.array([self.word_frequencies[self.idx_to_word[i]] for i in range(self.vocab_size)])
        
        # Apply 3/4 power
        frequencies = frequencies ** 0.75
        
        # Normalize
        frequencies = frequencies / np.sum(frequencies)
        
        # Build cumulative distribution
        cumulative = np.cumsum(frequencies)
        self.negative_sampling_table = cumulative
    
    def _get_negative_samples(self, target_word_idx):
        """Get negative samples for a target word"""
        negative_samples = []
        while len(negative_samples) < self.negative_samples:
            sample_idx = np.searchsorted(self.negative_sampling_table, random.random())
            if sample_idx != target_word_idx and sample_idx not in negative_samples:
                negative_samples.append(sample_idx)
        return negative_samples
    
    def _sigmoid(self, x):
        """Sigmoid function with numerical stability"""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    def _generate_training_data(self, texts):
        """Generate training pairs from texts"""
        training_data = []
        
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx[word] for word in words if word in self.word_to_idx]
            
            for i, center_word in enumerate(word_indices):
                # Define context window
                start = max(0, i - self.window_size // 2)
                end = min(len(word_indices), i + self.window_size // 2 + 1)
                
                context_words = word_indices[start:i] + word_indices[i+1:end]
                
                for context_word in context_words:
                    training_data.append((center_word, context_word))
        
        return training_data
    
    def train_skipgram(self, texts, epochs=5, learning_rate=0.025):
        """Train Skip-gram model"""
        print("Building vocabulary...")
        self.build_vocabulary(texts)
        
        print("Generating training data...")
        training_data = self._generate_training_data(texts)
        print(f"Training pairs: {len(training_data)}")
        
        print("Training Skip-gram model...")
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle training data
            random.shuffle(training_data)
            
            for i, (center_word, context_word) in enumerate(training_data):
                # Forward pass
                center_embedding = self.input_embeddings[center_word]
                context_embedding = self.output_embeddings[context_word]
                
                # Positive sample score
                positive_score = np.dot(center_embedding, context_embedding)
                positive_prob = self._sigmoid(positive_score)
                
                # Negative sampling
                negative_samples = self._get_negative_samples(context_word)
                negative_scores = np.dot(center_embedding, self.output_embeddings[negative_samples])
                negative_probs = self._sigmoid(-negative_scores)
                
                # Calculate loss
                loss = -np.log(positive_prob + 1e-10) - np.sum(np.log(negative_probs + 1e-10))
                total_loss += loss
                
                # Gradients
                # Positive sample gradient
                positive_gradient = (positive_prob - 1) * context_embedding
                self.input_embeddings[center_word] -= learning_rate * positive_gradient
                
                context_gradient = (positive_prob - 1) * center_embedding
                self.output_embeddings[context_word] -= learning_rate * context_gradient
                
                # Negative sample gradients
                for neg_idx, neg_prob in zip(negative_samples, negative_probs):
                    neg_gradient = neg_prob * center_embedding
                    self.output_embeddings[neg_idx] -= learning_rate * neg_gradient
                
                neg_input_gradient = np.sum(neg_prob * self.output_embeddings[neg_idx] for neg_idx, neg_prob in zip(negative_samples, negative_probs))
                self.input_embeddings[center_word] -= learning_rate * neg_input_gradient
                
                if i % 10000 == 0:
                    print(f"Epoch {epoch+1}, Sample {i}, Loss: {total_loss/(i+1):.4f}")
            
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Decay learning rate
            learning_rate *= 0.95
    
    def train_cbow(self, texts, epochs=5, learning_rate=0.025):
        """Train CBOW model"""
        print("Building vocabulary...")
        self.build_vocabulary(texts)
        
        print("Generating training data...")
        training_data = []
        
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx[word] for word in words if word in self.word_to_idx]
            
            for i, center_word in enumerate(word_indices):
                # Define context window
                start = max(0, i - self.window_size // 2)
                end = min(len(word_indices), i + self.window_size // 2 + 1)
                
                context_words = word_indices[start:i] + word_indices[i+1:end]
                
                if context_words:
                    training_data.append((context_words, center_word))
        
        print(f"Training pairs: {len(training_data)}")
        
        print("Training CBOW model...")
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for i, (context_words, center_word) in enumerate(training_data):
                # Average context embeddings
                context_embeddings = self.input_embeddings[context_words]
                context_sum = np.sum(context_embeddings, axis=0)
                
                # Forward pass
                center_embedding = self.output_embeddings[center_word]
                score = np.dot(context_sum, center_embedding)
                prob = self._sigmoid(score)
                
                # Negative sampling
                negative_samples = self._get_negative_samples(center_word)
                negative_scores = np.dot(context_sum, self.output_embeddings[negative_samples])
                negative_probs = self._sigmoid(-negative_scores)
                
                # Calculate loss
                loss = -np.log(prob + 1e-10) - np.sum(np.log(negative_probs + 1e-10))
                total_loss += loss
                
                # Gradients
                # Center word gradient
                center_gradient = (prob - 1) * context_sum
                self.output_embeddings[center_word] -= learning_rate * center_gradient
                
                # Context word gradients
                context_gradient = (prob - 1) * center_embedding / len(context_words)
                for context_word in context_words:
                    self.input_embeddings[context_word] -= learning_rate * context_gradient
                
                # Negative sample gradients
                for neg_idx, neg_prob in zip(negative_samples, negative_probs):
                    neg_gradient = neg_prob * context_sum / len(context_words)
                    self.output_embeddings[neg_idx] -= learning_rate * neg_gradient
                    
                    for context_word in context_words:
                        self.input_embeddings[context_word] -= learning_rate * neg_prob * self.output_embeddings[neg_idx] / len(context_words)
                
                if i % 10000 == 0:
                    print(f"Epoch {epoch+1}, Sample {i}, Loss: {total_loss/(i+1):.4f}")
            
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            learning_rate *= 0.95
    
    def get_embedding(self, word):
        """Get embedding for a word"""
        if word in self.word_to_idx:
            return self.input_embeddings[self.word_to_idx[word]]
        else:
            return None
    
    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return None
        
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def most_similar(self, word, top_k=10):
        """Find most similar words"""
        if word not in self.word_to_idx:
            return None
        
        word_embedding = self.get_embedding(word)
        similarities = []
        
        for other_word in self.word_to_idx:
            if other_word != word:
                other_embedding = self.get_embedding(other_word)
                sim = np.dot(word_embedding, other_embedding) / (np.linalg.norm(word_embedding) * np.linalg.norm(other_embedding))
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analogy(self, word1, word2, word3, top_k=5):
        """Solve word analogy: word1 is to word2 as word3 is to ?"""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        emb3 = self.get_embedding(word3)
        
        if emb1 is None or emb2 is None or emb3 is None:
            return None
        
        # Calculate analogy vector
        analogy_vector = emb2 - emb1 + emb3
        
        # Find most similar word
        similarities = []
        for word in self.word_to_idx:
            if word not in [word1, word2, word3]:
                word_embedding = self.get_embedding(word)
                sim = np.dot(analogy_vector, word_embedding) / (np.linalg.norm(analogy_vector) * np.linalg.norm(word_embedding))
                similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def visualize_embeddings(self, words=None, method='pca'):
        """Visualize embeddings in 2D"""
        if words is None:
            # Select most frequent words
            words = [word for word, _ in self.word_frequencies.most_common(50)]
        
        # Get embeddings
        embeddings = []
        word_labels = []
        for word in words:
            if word in self.word_to_idx:
                embeddings.append(self.get_embedding(word))
                word_labels.append(word)
        
        embeddings = np.array(embeddings)
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        for i, word in enumerate(word_labels):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        
        plt.title(f'Word Embeddings Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.show()

# Example usage
sample_texts = [
    "the king is a man",
    "the queen is a woman",
    "the prince is a boy",
    "the princess is a girl",
    "the king and queen are married",
    "the prince and princess are siblings",
    "the royal family lives in the palace",
    "the king rules the kingdom",
    "the queen helps the king",
    "the prince will be king someday"
]

# Train Skip-gram model
print("Training Skip-gram model...")
skipgram_model = Word2Vec(vocabulary_size=100, embedding_dim=50, window_size=3)
skipgram_model.train_skipgram(sample_texts, epochs=10, learning_rate=0.1)

# Test similarity
print("\nWord similarities:")
print(f"king - queen: {skipgram_model.similarity('king', 'queen'):.3f}")
print(f"king - man: {skipgram_model.similarity('king', 'man'):.3f}")
print(f"queen - woman: {skipgram_model.similarity('queen', 'woman'):.3f}")

# Test analogy
print("\nWord analogies:")
analogy_result = skipgram_model.analogy('king', 'man', 'queen')
if analogy_result:
    print(f"king - man + queen = {analogy_result[0][0]} (similarity: {analogy_result[0][1]:.3f})")

# Most similar words
print("\nMost similar to 'king':")
similar_words = skipgram_model.most_similar('king', top_k=5)
for word, sim in similar_words:
    print(f"  {word}: {sim:.3f}")
```

## GloVe: Global Vectors for Word Representation

GloVe learns embeddings by factorizing a word co-occurrence matrix, combining global statistics with local context.

### Mathematical Foundation

GloVe minimizes the objective:
```
J = ∑ᵢ,ⱼ f(Xᵢⱼ)(wᵢᵀ · w̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
```

Where:
- Xᵢⱼ is the co-occurrence count between words i and j
- f(Xᵢⱼ) is a weighting function
- wᵢ and w̃ⱼ are word embeddings
- bᵢ and b̃ⱼ are bias terms

### Implementation

```python
class GloVe:
    def __init__(self, vocabulary_size, embedding_dim, x_max=100, alpha=0.75):
        self.vocab_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        
        # Initialize embeddings and biases
        self.word_embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.context_embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.word_biases = np.zeros(vocab_size)
        self.context_biases = np.zeros(vocab_size)
        
        # Vocabulary mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def build_cooccurrence_matrix(self, texts, window_size=10):
        """Build co-occurrence matrix from texts"""
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for text in texts:
            words = text.lower().split()
            word_indices = []
            
            for word in words:
                if word not in self.word_to_idx:
                    continue
                word_indices.append(self.word_to_idx[word])
            
            # Count co-occurrences
            for i, word1 in enumerate(word_indices):
                start = max(0, i - window_size // 2)
                end = min(len(word_indices), i + window_size // 2 + 1)
                
                for j in range(start, end):
                    if i != j:
                        word2 = word_indices[j]
                        distance = abs(i - j)
                        weight = 1.0 / distance  # Distance weighting
                        cooccurrence[word1][word2] += weight
        
        return cooccurrence
    
    def _weighting_function(self, x):
        """GloVe weighting function"""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        else:
            return 1.0
    
    def train(self, texts, epochs=25, learning_rate=0.05):
        """Train GloVe model"""
        print("Building vocabulary...")
        self._build_vocabulary(texts)
        
        print("Building co-occurrence matrix...")
        cooccurrence = self.build_cooccurrence_matrix(texts)
        
        print("Training GloVe model...")
        for epoch in range(epochs):
            total_loss = 0
            num_pairs = 0
            
            for word1, contexts in cooccurrence.items():
                for word2, count in contexts.items():
                    # Calculate prediction
                    prediction = (np.dot(self.word_embeddings[word1], self.context_embeddings[word2]) + 
                                self.word_biases[word1] + self.context_biases[word2])
                    
                    # Calculate loss
                    error = prediction - np.log(count + 1e-8)
                    weight = self._weighting_function(count)
                    loss = weight * error ** 2
                    total_loss += loss
                    
                    # Gradients
                    grad_word = 2 * weight * error * self.context_embeddings[word2]
                    grad_context = 2 * weight * error * self.word_embeddings[word1]
                    grad_word_bias = 2 * weight * error
                    grad_context_bias = 2 * weight * error
                    
                    # Update parameters
                    self.word_embeddings[word1] -= learning_rate * grad_word
                    self.context_embeddings[word2] -= learning_rate * grad_context
                    self.word_biases[word1] -= learning_rate * grad_word_bias
                    self.context_biases[word2] -= learning_rate * grad_context_bias
                    
                    num_pairs += 1
            
            avg_loss = total_loss / num_pairs if num_pairs > 0 else 0
            print(f"Epoch {epoch+1}, Average loss: {avg_loss:.4f}")
            
            # Decay learning rate
            learning_rate *= 0.95
    
    def _build_vocabulary(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(word_counts.keys()))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Update vocabulary size
        self.vocab_size = len(self.word_to_idx)
        
        # Reinitialize embeddings
        self.word_embeddings = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
        self.context_embeddings = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
        self.word_biases = np.zeros(self.vocab_size)
        self.context_biases = np.zeros(self.vocab_size)
    
    def get_embedding(self, word):
        """Get final embedding (average of word and context embeddings)"""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return (self.word_embeddings[idx] + self.context_embeddings[idx]) / 2
        else:
            return None

# Example usage
glove_model = GloVe(vocabulary_size=100, embedding_dim=50)
glove_model.train(sample_texts, epochs=50, learning_rate=0.1)

print("GloVe similarities:")
print(f"king - queen: {glove_model.similarity('king', 'queen'):.3f}")
```

## FastText: Subword Information

FastText extends Word2Vec by representing words as bags of character n-grams, enabling handling of out-of-vocabulary words.

### Implementation

```python
class FastText:
    def __init__(self, vocabulary_size, embedding_dim, min_n=3, max_n=6):
        self.vocab_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.min_n = min_n
        self.max_n = max_n
        
        # Vocabulary and mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.ngram_to_idx = {}
        self.idx_to_ngram = {}
        
        # Embeddings
        self.word_embeddings = None
        self.ngram_embeddings = None
        
    def _get_ngrams(self, word):
        """Get character n-grams for a word"""
        ngrams = []
        
        # Add word boundaries
        word = f"<{word}>"
        
        for n in range(self.min_n, min(self.max_n + 1, len(word) + 1)):
            for i in range(len(word) - n + 1):
                ngram = word[i:i + n]
                ngrams.append(ngram)
        
        return ngrams
    
    def _build_vocabulary(self, texts):
        """Build vocabulary including n-grams"""
        word_counts = Counter()
        ngram_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] += 1
                ngrams = self._get_ngrams(word)
                ngram_counts.update(ngrams)
        
        # Create word mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(word_counts.keys()))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Create n-gram mappings
        self.ngram_to_idx = {ngram: idx for idx, ngram in enumerate(sorted(ngram_counts.keys()))}
        self.idx_to_ngram = {idx: ngram for ngram, idx in self.ngram_to_idx.items()}
        
        # Update sizes
        self.vocab_size = len(self.word_to_idx)
        ngram_size = len(self.ngram_to_idx)
        
        # Initialize embeddings
        self.word_embeddings = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
        self.ngram_embeddings = np.random.uniform(-0.5, 0.5, (ngram_size, self.embedding_dim))
    
    def _get_word_representation(self, word):
        """Get representation of a word (sum of word and n-gram embeddings)"""
        if word in self.word_to_idx:
            word_idx = self.word_to_idx[word]
            word_vec = self.word_embeddings[word_idx]
        else:
            word_vec = np.zeros(self.embedding_dim)
        
        ngrams = self._get_ngrams(word)
        ngram_vec = np.zeros(self.embedding_dim)
        
        for ngram in ngrams:
            if ngram in self.ngram_to_idx:
                ngram_idx = self.ngram_to_idx[ngram]
                ngram_vec += self.ngram_embeddings[ngram_idx]
        
        return word_vec + ngram_vec
    
    def train_skipgram(self, texts, epochs=5, learning_rate=0.025):
        """Train FastText Skip-gram model"""
        print("Building vocabulary...")
        self._build_vocabulary(texts)
        
        print("Training FastText model...")
        # Training loop similar to Word2Vec but using word representations
        # Implementation details omitted for brevity
        pass
    
    def get_embedding(self, word):
        """Get embedding for a word (handles OOV words)"""
        return self._get_word_representation(word)

# Example usage
fasttext_model = FastText(vocabulary_size=100, embedding_dim=50)
# Training implementation would be similar to Word2Vec but using n-gram representations
```

## Pre-trained Embeddings

### Using Pre-trained Word2Vec

```python
import gensim.downloader as api

# Download pre-trained model
print("Loading pre-trained Word2Vec model...")
model = api.load("word2vec-google-news-300")

# Test similarity
print(f"Similarity between 'king' and 'queen': {model.similarity('king', 'queen'):.3f}")

# Test analogy
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"king - man + woman = {result[0][0]} (similarity: {result[0][1]:.3f})")

# Most similar words
print("\nMost similar to 'computer':")
similar = model.most_similar('computer', topn=5)
for word, score in similar:
    print(f"  {word}: {score:.3f}")
```

### Using GloVe Embeddings

```python
def load_glove_embeddings(file_path, embedding_dim=100):
    """Load GloVe embeddings from file"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Example usage (requires GloVe file)
# glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
```

## Embedding Evaluation

### Intrinsic Evaluation

```python
class EmbeddingEvaluator:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def evaluate_similarity(self, word_pairs, human_similarities):
        """Evaluate on word similarity tasks"""
        predicted_similarities = []
        
        for word1, word2 in word_pairs:
            if word1 in self.embeddings and word2 in self.embeddings:
                sim = cosine_similarity([self.embeddings[word1]], [self.embeddings[word2]])[0][0]
                predicted_similarities.append(sim)
            else:
                predicted_similarities.append(0.0)
        
        # Calculate correlation
        correlation = np.corrcoef(human_similarities, predicted_similarities)[0, 1]
        return correlation
    
    def evaluate_analogy(self, analogy_questions):
        """Evaluate on analogy tasks"""
        correct = 0
        total = 0
        
        for question in analogy_questions:
            word1, word2, word3, expected = question
            
            if all(word in self.embeddings for word in [word1, word2, word3, expected]):
                # Find most similar word
                analogy_vec = (self.embeddings[word2] - self.embeddings[word1] + 
                             self.embeddings[word3])
                
                similarities = {}
                for word in self.embeddings:
                    if word not in [word1, word2, word3]:
                        sim = cosine_similarity([analogy_vec], [self.embeddings[word]])[0][0]
                        similarities[word] = sim
                
                predicted = max(similarities, key=similarities.get)
                if predicted == expected:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0

# Example evaluation
# evaluator = EmbeddingEvaluator(embeddings_dict)
# correlation = evaluator.evaluate_similarity(word_pairs, human_scores)
# accuracy = evaluator.evaluate_analogy(analogy_questions)
```

## Practical Exercises

### Exercise 1: Word2Vec Implementation
Complete the Word2Vec implementation:
- Add hierarchical softmax as an alternative to negative sampling
- Implement different sampling strategies
- Compare Skip-gram vs CBOW performance

### Exercise 2: Embedding Analysis
Analyze different embedding methods:
- Train Word2Vec, GloVe, and FastText on the same corpus
- Compare performance on similarity and analogy tasks
- Visualize embeddings using t-SNE

### Exercise 3: Domain-Specific Embeddings
Train embeddings on domain-specific data:
- Use medical, legal, or technical text
- Compare with general-purpose embeddings
- Evaluate on domain-specific tasks

## Assessment Questions

1. **What are the advantages of Word2Vec over traditional methods?**
   - Captures semantic relationships
   - Dense vector representations
   - Algebraic properties for word relationships
   - Efficient training and inference

2. **How does GloVe differ from Word2Vec?**
   - Uses global co-occurrence statistics
   - Factorizes co-occurrence matrix
   - Combines local and global information
   - Often performs better on analogy tasks

3. **What makes FastText suitable for morphologically rich languages?**
   - Uses character n-grams
   - Handles out-of-vocabulary words
   - Captures subword information
   - Better for languages with complex morphology

## Key Takeaways

- Word embeddings capture semantic and syntactic relationships in continuous vector spaces
- Word2Vec uses local context to learn embeddings through prediction tasks
- GloVe combines local and global information through matrix factorization
- FastText handles OOV words using character n-grams
- Pre-trained embeddings provide strong baselines for many NLP tasks
- Evaluation requires both intrinsic and extrinsic measures
- Embeddings form the foundation for modern neural NLP models

## Next Steps

In the next module, we'll explore neural networks for NLP, building upon the embedding foundations to create more sophisticated models for various language understanding tasks.
