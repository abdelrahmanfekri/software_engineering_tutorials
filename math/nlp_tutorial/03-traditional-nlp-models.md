# NLP Tutorial Module 3: Traditional NLP Models

## Learning Objectives
By the end of this module, you will be able to:
- Understand and implement n-gram language models
- Work with Hidden Markov Models (HMMs)
- Apply probabilistic context-free grammars
- Implement Naive Bayes classifiers
- Use Maximum Entropy models
- Apply traditional models to real-world NLP tasks

## Introduction to Traditional NLP Models

Before the deep learning revolution, NLP relied heavily on statistical and probabilistic models. These traditional approaches, while less powerful than modern neural methods, provide important foundations and are still useful in specific contexts.

### Types of Traditional Models

1. **Language Models**: N-gram models, Hidden Markov Models
2. **Sequence Labeling**: HMMs, Maximum Entropy Markov Models
3. **Parsing**: Probabilistic Context-Free Grammars
4. **Classification**: Naive Bayes, Maximum Entropy
5. **Topic Modeling**: Latent Dirichlet Allocation

## N-gram Language Models

### Theory and Implementation

N-gram models predict the next word based on the previous n-1 words, using the Markov assumption.

#### Mathematical Foundation

For an n-gram model, the probability of a sequence is:
```
P(w₁, w₂, ..., wₙ) = ∏ᵢ₌₁ⁿ P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁)
```

```python
import numpy as np
from collections import defaultdict, Counter
import math
import random

class AdvancedNGramModel:
    def __init__(self, n, smoothing='kneser_ney'):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.vocabulary = set()
        self.total_words = 0
        
    def train(self, texts):
        """Train the n-gram model with various smoothing options"""
        for text in texts:
            tokens = self._tokenize(text)
            self.vocabulary.update(tokens)
            self.total_words += len(tokens)
            
            # Add padding
            padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            # Count n-grams
            for i in range(len(padded_tokens) - self.n + 1):
                ngram = tuple(padded_tokens[i:i + self.n])
                prefix = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[prefix][word] += 1
    
    def _tokenize(self, text):
        """Simple tokenization"""
        return text.lower().split()
    
    def get_probability(self, prefix, word, smoothing=None):
        """Get probability with specified smoothing method"""
        if smoothing is None:
            smoothing = self.smoothing
            
        prefix = tuple(prefix)
        
        if smoothing == 'mle':
            return self._mle_probability(prefix, word)
        elif smoothing == 'laplace':
            return self._laplace_probability(prefix, word)
        elif smoothing == 'good_turing':
            return self._good_turing_probability(prefix, word)
        elif smoothing == 'kneser_ney':
            return self._kneser_ney_probability(prefix, word)
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing}")
    
    def _mle_probability(self, prefix, word):
        """Maximum Likelihood Estimation"""
        if prefix not in self.ngram_counts:
            return 0
        
        total_count = sum(self.ngram_counts[prefix].values())
        if total_count == 0:
            return 0
        
        return self.ngram_counts[prefix][word] / total_count
    
    def _laplace_probability(self, prefix, word, alpha=1.0):
        """Laplace (Add-One) Smoothing"""
        if prefix not in self.ngram_counts:
            count = 0
            total_count = 0
        else:
            count = self.ngram_counts[prefix][word]
            total_count = sum(self.ngram_counts[prefix].values())
        
        vocab_size = len(self.vocabulary)
        return (count + alpha) / (total_count + alpha * vocab_size)
    
    def _good_turing_probability(self, prefix, word):
        """Good-Turing Smoothing"""
        if prefix not in self.ngram_counts:
            return 1 / len(self.vocabulary)  # Uniform distribution for unseen n-grams
        
        counts = self.ngram_counts[prefix]
        word_count = counts[word]
        total_count = sum(counts.values())
        
        # Count of counts
        count_of_counts = Counter(counts.values())
        
        if word_count == 0:
            # Unseen word
            return count_of_counts[1] / (total_count * len(self.vocabulary))
        else:
            # Seen word
            if word_count + 1 in count_of_counts:
                smoothed_count = (word_count + 1) * count_of_counts[word_count + 1] / count_of_counts[word_count]
            else:
                smoothed_count = word_count
            
            return smoothed_count / total_count
    
    def _kneser_ney_probability(self, prefix, word, discount=0.75):
        """Kneser-Ney Smoothing (simplified version)"""
        if len(prefix) == 0:
            # Unigram case
            word_count = sum(counts[word] for counts in self.ngram_counts.values())
            total_count = sum(sum(counts.values()) for counts in self.ngram_counts.values())
            return word_count / total_count
        
        # Higher-order n-gram
        if prefix not in self.ngram_counts:
            # Back off to lower-order model
            return self._kneser_ney_probability(prefix[1:], word, discount)
        
        counts = self.ngram_counts[prefix]
        word_count = counts[word]
        total_count = sum(counts.values())
        
        if total_count == 0:
            return 0
        
        # Discount counts
        discounted_count = max(word_count - discount, 0)
        
        # Calculate continuation probability (simplified)
        continuation_prob = len(counts) / len(self.vocabulary)
        
        # Interpolation weight
        lambda_weight = discount * len(counts) / total_count
        
        # Final probability
        prob = discounted_count / total_count + lambda_weight * continuation_prob
        return prob
    
    def generate_text(self, max_length=20, start_words=None):
        """Generate text using the trained model"""
        if start_words is None:
            start_words = ['<START>'] * (self.n - 1)
        
        generated = start_words.copy()
        
        for _ in range(max_length):
            prefix = tuple(generated[-(self.n-1):])
            
            # Get possible next words and their probabilities
            if prefix in self.ngram_counts:
                word_probs = {}
                for word in self.ngram_counts[prefix]:
                    prob = self.get_probability(list(prefix), word)
                    word_probs[word] = prob
            else:
                # Fallback to uniform distribution over vocabulary
                word_probs = {word: 1/len(self.vocabulary) for word in self.vocabulary}
            
            # Sample next word
            words = list(word_probs.keys())
            probabilities = list(word_probs.values())
            
            if not words:
                break
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                probabilities = [1/len(words)] * len(words)
            
            next_word = np.random.choice(words, p=probabilities)
            
            if next_word == '<END>':
                break
            
            generated.append(next_word)
        
        return ' '.join(generated[len(start_words):])
    
    def perplexity(self, test_texts):
        """Calculate perplexity on test data"""
        total_log_prob = 0
        total_words = 0
        
        for text in test_texts:
            tokens = self._tokenize(text)
            padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            total_words += len(tokens)
            
            for i in range(len(padded_tokens) - self.n + 1):
                prefix = padded_tokens[i:i + self.n - 1]
                word = padded_tokens[i + self.n - 1]
                prob = self.get_probability(prefix, word)
                
                if prob > 0:
                    total_log_prob += math.log(prob)
                else:
                    total_log_prob += -float('inf')
        
        if total_log_prob == -float('inf'):
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        return math.exp(-avg_log_prob)

# Example usage
training_texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and dog are friends",
    "the mat is on the floor",
    "the park has many trees"
]

test_texts = [
    "the cat is happy",
    "the dog runs fast"
]

# Train different models
models = {}
for n in [2, 3]:
    for smoothing in ['laplace', 'good_turing']:
        model = AdvancedNGramModel(n, smoothing)
        model.train(training_texts)
        models[f"{n}gram_{smoothing}"] = model

# Compare models
print("Model Comparison:")
for name, model in models.items():
    perplexity = model.perplexity(test_texts)
    print(f"{name}: Perplexity = {perplexity:.3f}")

# Generate text
print("\nText Generation:")
for name, model in models.items():
    generated = model.generate_text(max_length=10)
    print(f"{name}: {generated}")
```

## Hidden Markov Models (HMMs)

HMMs are probabilistic models that assume the system being modeled is a Markov process with unobserved (hidden) states.

### Theory

An HMM consists of:
- **States**: Hidden variables
- **Observations**: Visible outputs
- **Transition Probabilities**: P(stateᵢ₊₁ | stateᵢ)
- **Emission Probabilities**: P(observation | state)

### Implementation

```python
class HiddenMarkovModel:
    def __init__(self, states, observations):
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_observations = len(observations)
        
        # Transition probabilities
        self.transitions = np.zeros((self.n_states, self.n_states))
        
        # Emission probabilities
        self.emissions = np.zeros((self.n_states, self.n_observations))
        
        # Initial state probabilities
        self.initial = np.zeros(self.n_states)
        
        # Create mappings
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.obs_to_idx = {obs: i for i, obs in enumerate(observations)}
    
    def train_from_sequences(self, state_sequences, observation_sequences):
        """Train HMM from aligned state-observation sequences"""
        # Count transitions
        for state_seq, obs_seq in zip(state_sequences, observation_sequences):
            # Initial state
            if state_seq:
                self.initial[self.state_to_idx[state_seq[0]]] += 1
            
            # Transitions and emissions
            for i in range(len(state_seq)):
                state_idx = self.state_to_idx[state_seq[i]]
                obs_idx = self.obs_to_idx[obs_seq[i]]
                self.emissions[state_idx, obs_idx] += 1
                
                if i < len(state_seq) - 1:
                    next_state_idx = self.state_to_idx[state_seq[i + 1]]
                    self.transitions[state_idx, next_state_idx] += 1
        
        # Normalize probabilities
        self.initial = self.initial / np.sum(self.initial) if np.sum(self.initial) > 0 else np.ones(self.n_states) / self.n_states
        
        for i in range(self.n_states):
            # Transition probabilities
            total_transitions = np.sum(self.transitions[i])
            if total_transitions > 0:
                self.transitions[i] = self.transitions[i] / total_transitions
            else:
                self.transitions[i] = np.ones(self.n_states) / self.n_states
            
            # Emission probabilities
            total_emissions = np.sum(self.emissions[i])
            if total_emissions > 0:
                self.emissions[i] = self.emissions[i] / total_emissions
            else:
                self.emissions[i] = np.ones(self.n_observations) / self.n_observations
    
    def forward_algorithm(self, observations):
        """Forward algorithm for computing P(observations | model)"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        obs_idx = self.obs_to_idx[observations[0]]
        alpha[0] = self.initial * self.emissions[:, obs_idx]
        
        # Induction
        for t in range(1, T):
            obs_idx = self.obs_to_idx[observations[t]]
            for j in range(self.n_states):
                alpha[t, j] = self.emissions[j, obs_idx] * np.sum(alpha[t-1] * self.transitions[:, j])
        
        return alpha, np.sum(alpha[T-1])
    
    def backward_algorithm(self, observations):
        """Backward algorithm"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[T-1] = 1.0
        
        # Induction
        for t in range(T-2, -1, -1):
            obs_idx = self.obs_to_idx[observations[t+1]]
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transitions[i] * self.emissions[:, obs_idx] * beta[t+1])
        
        return beta
    
    def viterbi_algorithm(self, observations):
        """Viterbi algorithm for finding most likely state sequence"""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        obs_idx = self.obs_to_idx[observations[0]]
        delta[0] = self.initial * self.emissions[:, obs_idx]
        
        # Recursion
        for t in range(1, T):
            obs_idx = self.obs_to_idx[observations[t]]
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1] * self.transitions[:, j]) * self.emissions[j, obs_idx]
                psi[t, j] = np.argmax(delta[t-1] * self.transitions[:, j])
        
        # Termination
        best_path_prob = np.max(delta[T-1])
        best_path_pointer = np.argmax(delta[T-1])
        
        # Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = best_path_pointer
        
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
        
        return [self.states[i] for i in best_path], best_path_prob

# Example: POS Tagging with HMM
def create_pos_hmm():
    """Create HMM for POS tagging"""
    states = ['DT', 'NN', 'VB', 'IN', 'JJ']
    observations = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'park', 'big', 'red']
    
    hmm = HiddenMarkovModel(states, observations)
    
    # Training data (state sequences, observation sequences)
    state_sequences = [
        ['DT', 'NN', 'VB', 'IN', 'DT', 'NN'],
        ['DT', 'JJ', 'NN', 'VB', 'IN', 'DT', 'NN']
    ]
    
    observation_sequences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'big', 'dog', 'ran', 'in', 'the', 'park']
    ]
    
    hmm.train_from_sequences(state_sequences, observation_sequences)
    
    return hmm

# Example usage
pos_hmm = create_pos_hmm()

# Test sequence
test_obs = ['the', 'big', 'cat', 'sat']
best_states, prob = pos_hmm.viterbi_algorithm(test_obs)

print("POS Tagging with HMM:")
print(f"Observations: {test_obs}")
print(f"Best state sequence: {best_states}")
print(f"Probability: {prob:.6f}")

# Forward algorithm
alpha, likelihood = pos_hmm.forward_algorithm(test_obs)
print(f"Likelihood: {likelihood:.6f}")
```

## Probabilistic Context-Free Grammars (PCFGs)

PCFGs extend context-free grammars with probability distributions over productions.

### Implementation

```python
from collections import defaultdict
import random

class ProbabilisticCFG:
    def __init__(self):
        self.productions = defaultdict(list)  # Nonterminal -> [(probability, rhs)]
        self.lexicon = defaultdict(list)      # Terminal -> [(probability, pos)]
        self.start_symbol = 'S'
    
    def add_production(self, lhs, rhs, probability):
        """Add a production rule"""
        if isinstance(rhs, str):
            # Lexical rule (terminal)
            self.lexicon[rhs].append((probability, lhs))
        else:
            # Phrase structure rule
            self.productions[lhs].append((probability, rhs))
    
    def parse_probability(self, sentence):
        """Calculate probability of sentence using CKY algorithm"""
        words = sentence.split()
        n = len(words)
        
        # Initialize chart
        chart = defaultdict(lambda: defaultdict(float))
        backpointers = defaultdict(dict)
        
        # Fill lexical entries
        for i, word in enumerate(words):
            if word in self.lexicon:
                for prob, pos in self.lexicon[word]:
                    chart[i][i+1][pos] = prob
        
        # Fill phrase structure entries
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for A in chart[i][k]:
                        for B in chart[k][j]:
                            for lhs in self.productions:
                                for prob, rhs in self.productions[lhs]:
                                    if len(rhs) == 2 and rhs[0] == A and rhs[1] == B:
                                        new_prob = chart[i][k][A] * chart[k][j][B] * prob
                                        if new_prob > chart[i][j][lhs]:
                                            chart[i][j][lhs] = new_prob
                                            backpointers[i][j][lhs] = (k, A, B)
        
        return chart[0][n][self.start_symbol]
    
    def generate_sentence(self, max_length=10):
        """Generate sentence from the grammar"""
        return self._generate_from_symbol(self.start_symbol, 0, max_length)
    
    def _generate_from_symbol(self, symbol, depth, max_depth):
        """Recursively generate from a symbol"""
        if depth > max_depth:
            return ""
        
        if symbol in self.lexicon:
            # Terminal symbol
            terminals = self.lexicon[symbol]
            if terminals:
                prob, _ = random.choices(terminals, weights=[p for p, _ in terminals])[0]
                return symbol
            else:
                return ""
        
        if symbol in self.productions:
            # Non-terminal symbol
            productions = self.productions[symbol]
            if productions:
                prob, rhs = random.choices(productions, weights=[p for p, _ in productions])[0]
                result = []
                for symbol_rhs in rhs:
                    generated = self._generate_from_symbol(symbol_rhs, depth + 1, max_depth)
                    if generated:
                        result.append(generated)
                return " ".join(result)
        
        return ""

# Example usage
def create_simple_pcfg():
    """Create a simple PCFG"""
    grammar = ProbabilisticCFG()
    
    # Phrase structure rules
    grammar.add_production('S', ['NP', 'VP'], 0.8)
    grammar.add_production('S', ['NP', 'VP', 'PP'], 0.2)
    grammar.add_production('NP', ['DT', 'NN'], 0.6)
    grammar.add_production('NP', ['DT', 'JJ', 'NN'], 0.4)
    grammar.add_production('VP', ['VB'], 0.5)
    grammar.add_production('VP', ['VB', 'NP'], 0.3)
    grammar.add_production('VP', ['VB', 'PP'], 0.2)
    grammar.add_production('PP', ['IN', 'NP'], 1.0)
    
    # Lexical rules
    grammar.add_production('DT', 'the', 0.8)
    grammar.add_production('DT', 'a', 0.2)
    grammar.add_production('NN', 'cat', 0.3)
    grammar.add_production('NN', 'dog', 0.3)
    grammar.add_production('NN', 'mat', 0.2)
    grammar.add_production('NN', 'park', 0.2)
    grammar.add_production('VB', 'sat', 0.4)
    grammar.add_production('VB', 'ran', 0.3)
    grammar.add_production('VB', 'jumped', 0.3)
    grammar.add_production('JJ', 'big', 0.5)
    grammar.add_production('JJ', 'small', 0.5)
    grammar.add_production('IN', 'on', 0.6)
    grammar.add_production('IN', 'in', 0.4)
    
    return grammar

# Example usage
pcfg = create_simple_pcfg()

# Test parsing
test_sentence = "the cat sat on the mat"
prob = pcfg.parse_probability(test_sentence)
print(f"Probability of '{test_sentence}': {prob:.6f}")

# Generate sentences
print("\nGenerated sentences:")
for i in range(5):
    sentence = pcfg.generate_sentence()
    prob = pcfg.parse_probability(sentence)
    print(f"{i+1}. {sentence} (prob: {prob:.6f})")
```

## Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem with strong independence assumptions.

### Implementation

```python
from collections import defaultdict, Counter
import math

class NaiveBayesClassifier:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.total_documents = 0
        self.vocabulary = set()
        
    def train(self, documents, labels):
        """Train the Naive Bayes classifier"""
        self.total_documents = len(documents)
        
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            
            # Tokenize document
            words = self._tokenize(doc)
            self.vocabulary.update(words)
            
            # Count features for this class
            for word in words:
                self.feature_counts[label][word] += 1
    
    def _tokenize(self, text):
        """Simple tokenization"""
        return text.lower().split()
    
    def predict(self, document):
        """Predict class for a document"""
        words = self._tokenize(document)
        
        best_class = None
        best_score = float('-inf')
        
        for class_label in self.class_counts:
            score = self._calculate_log_probability(words, class_label)
            if score > best_score:
                best_score = score
                best_class = class_label
        
        return best_class, best_score
    
    def _calculate_log_probability(self, words, class_label):
        """Calculate log probability of document given class"""
        # Prior probability
        prior = math.log(self.class_counts[class_label] / self.total_documents)
        
        # Likelihood
        likelihood = 0
        class_word_count = sum(self.feature_counts[class_label].values())
        
        for word in words:
            word_count = self.feature_counts[class_label][word]
            # Laplace smoothing
            prob = (word_count + self.smoothing) / (class_word_count + self.smoothing * len(self.vocabulary))
            likelihood += math.log(prob)
        
        return prior + likelihood
    
    def predict_proba(self, document):
        """Predict class probabilities"""
        words = self._tokenize(document)
        
        log_probs = {}
        for class_label in self.class_counts:
            log_probs[class_label] = self._calculate_log_probability(words, class_label)
        
        # Convert log probabilities to probabilities
        max_log_prob = max(log_probs.values())
        probs = {}
        total_prob = 0
        
        for class_label, log_prob in log_probs.items():
            prob = math.exp(log_prob - max_log_prob)
            probs[class_label] = prob
            total_prob += prob
        
        # Normalize
        for class_label in probs:
            probs[class_label] /= total_prob
        
        return probs

# Example usage
training_docs = [
    "great movie amazing film",
    "terrible movie awful film",
    "excellent performance wonderful acting",
    "bad performance poor acting",
    "fantastic story brilliant plot",
    "boring story dull plot"
]

training_labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']

# Train classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(training_docs, training_labels)

# Test predictions
test_docs = [
    "good movie with great acting",
    "bad film with poor plot",
    "amazing story and wonderful performance"
]

print("Naive Bayes Predictions:")
for doc in test_docs:
    prediction, score = nb_classifier.predict(doc)
    probabilities = nb_classifier.predict_proba(doc)
    print(f"Document: {doc}")
    print(f"Prediction: {prediction} (score: {score:.3f})")
    print(f"Probabilities: {probabilities}")
    print()
```

## Maximum Entropy Models

Maximum Entropy models (also known as Logistic Regression) find the probability distribution that maximizes entropy while satisfying feature constraints.

### Implementation

```python
import numpy as np
from scipy.optimize import minimize

class MaximumEntropyModel:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.feature_names = None
        self.class_labels = None
        
    def extract_features(self, documents):
        """Extract binary features from documents"""
        vocabulary = set()
        for doc in documents:
            vocabulary.update(doc.lower().split())
        
        self.feature_names = sorted(vocabulary)
        feature_matrix = np.zeros((len(documents), len(self.feature_names)))
        
        for i, doc in enumerate(documents):
            words = doc.lower().split()
            for j, word in enumerate(self.feature_names):
                if word in words:
                    feature_matrix[i, j] = 1
        
        return feature_matrix
    
    def train(self, documents, labels):
        """Train the Maximum Entropy model"""
        self.class_labels = sorted(set(labels))
        n_classes = len(self.class_labels)
        
        # Extract features
        X = self.extract_features(documents)
        n_features = X.shape[1]
        
        # Convert labels to integers
        label_to_idx = {label: i for i, label in enumerate(self.class_labels)}
        y = np.array([label_to_idx[label] for label in labels])
        
        # Initialize weights
        self.weights = np.random.normal(0, 0.01, (n_classes, n_features))
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass
            scores = X @ self.weights.T  # (n_docs, n_classes)
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Calculate loss (negative log likelihood)
            loss = -np.mean(np.log(probabilities[np.arange(len(y)), y] + 1e-15))
            
            # Calculate gradients
            gradients = np.zeros_like(self.weights)
            for i in range(len(y)):
                true_class = y[i]
                for c in range(n_classes):
                    if c == true_class:
                        gradients[c] += X[i] * (probabilities[i, c] - 1)
                    else:
                        gradients[c] += X[i] * probabilities[i, c]
            
            gradients /= len(y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")
    
    def predict(self, document):
        """Predict class for a document"""
        X = self.extract_features([document])
        scores = X @ self.weights.T
        probabilities = self._softmax(scores[0])
        
        best_class_idx = np.argmax(probabilities)
        best_class = self.class_labels[best_class_idx]
        
        return best_class, probabilities
    
    def _softmax(self, scores):
        """Apply softmax function"""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)
    
    def predict_proba(self, document):
        """Get class probabilities"""
        _, probabilities = self.predict(document)
        return {label: prob for label, prob in zip(self.class_labels, probabilities)}

# Example usage
maxent_classifier = MaximumEntropyModel(learning_rate=0.1, max_iterations=500)
maxent_classifier.train(training_docs, training_labels)

print("Maximum Entropy Predictions:")
for doc in test_docs:
    prediction, probabilities = maxent_classifier.predict(doc)
    print(f"Document: {doc}")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    print()
```

## Practical Exercises

### Exercise 1: N-gram Model Comparison
Compare different n-gram models (bigram, trigram) with different smoothing techniques on a larger corpus:
- Calculate perplexity on held-out test data
- Generate text samples from each model
- Analyze the trade-offs between different approaches

### Exercise 2: HMM for Named Entity Recognition
Implement an HMM for NER task:
- Define states (PERSON, ORGANIZATION, LOCATION, O)
- Train on annotated data
- Evaluate on test data using Viterbi algorithm

### Exercise 3: PCFG for Syntactic Parsing
Extend the PCFG implementation:
- Add more complex grammar rules
- Implement probabilistic CKY parsing
- Generate parse trees for test sentences

## Assessment Questions

1. **What are the advantages and disadvantages of n-gram models?**
   - Advantages: Simple, fast, good baseline
   - Disadvantages: Data sparsity, context limitation, Markov assumption

2. **How does smoothing address the zero probability problem?**
   - Redistributes probability mass from seen to unseen events
   - Prevents infinite perplexity
   - Improves generalization to unseen data

3. **What is the difference between generative and discriminative models?**
   - Generative: Model P(x,y), can generate data
   - Discriminative: Model P(y|x), focus on classification boundary

## Key Takeaways

- Traditional NLP models provide important foundations for modern approaches
- N-gram models are simple but effective for many language modeling tasks
- HMMs are powerful for sequence labeling tasks like POS tagging and NER
- PCFGs enable probabilistic syntactic parsing
- Naive Bayes is a fast and effective classifier for text
- Maximum Entropy models balance simplicity with expressiveness
- These models are still useful in specific contexts and as baselines

## Next Steps

In the next module, we'll explore word embeddings and distributed representations, which revolutionized how we represent words and documents in vector spaces. This will bridge the gap between traditional statistical methods and modern neural approaches.
