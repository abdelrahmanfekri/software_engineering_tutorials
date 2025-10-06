# NLP Tutorial Module 5: Neural Networks for NLP

## Learning Objectives
By the end of this module, you will be able to:
- Understand the fundamentals of neural networks for NLP
- Implement feedforward networks for text classification
- Build and train neural language models
- Apply neural networks to sequence labeling tasks
- Understand backpropagation and gradient-based optimization
- Implement regularization techniques for neural NLP models

## Introduction to Neural Networks in NLP

Neural networks have revolutionized NLP by enabling end-to-end learning from raw text data. Unlike traditional methods that rely on hand-crafted features, neural networks automatically learn representations that are optimized for specific tasks.

### Key Advantages of Neural Networks in NLP

1. **Automatic Feature Learning**: No need for manual feature engineering
2. **End-to-End Training**: Direct optimization of task-specific objectives
3. **Distributed Representations**: Rich, dense vector representations
4. **Scalability**: Can handle large datasets and complex models
5. **Transfer Learning**: Pre-trained models can be fine-tuned for specific tasks

### Types of Neural Networks in NLP

1. **Feedforward Networks**: For classification and regression tasks
2. **Recurrent Networks**: For sequential data processing
3. **Convolutional Networks**: For local pattern detection
4. **Attention Mechanisms**: For focusing on relevant parts of input
5. **Transformer Networks**: For parallel processing of sequences

## Feedforward Neural Networks for Text Classification

### Architecture Overview

A basic feedforward network for text classification consists of:
1. **Input Layer**: Word embeddings or document representations
2. **Hidden Layers**: Non-linear transformations
3. **Output Layer**: Task-specific predictions

### Mathematical Foundation

For a feedforward network with one hidden layer:

```
h = σ(W₁x + b₁)
y = W₂h + b₂
```

Where:
- x is the input vector
- W₁, W₂ are weight matrices
- b₁, b₂ are bias vectors
- σ is the activation function

### Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_length=100):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        words = text.lower().split()
        indices = [self.word_to_idx.get(word, 0) for word in words[:self.max_length]]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class FeedforwardTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(FeedforwardTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Feedforward layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Embedding lookup
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        
        # Feedforward layers
        hidden1 = self.relu(self.fc1(pooled))
        hidden1 = self.dropout(hidden1)
        
        hidden2 = self.relu(self.fc2(hidden1))
        hidden2 = self.dropout(hidden2)
        
        output = self.fc3(hidden2)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the neural network model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

# Example usage
def create_sample_data():
    """Create sample text classification data"""
    texts = [
        "great movie amazing film excellent",
        "terrible movie awful film bad",
        "good performance wonderful acting",
        "poor performance bad acting",
        "fantastic story brilliant plot",
        "boring story dull plot",
        "exciting adventure thrilling journey",
        "disappointing ending terrible conclusion"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    return texts, labels

# Create vocabulary
texts, labels = create_sample_data()
vocab = set()
for text in texts:
    vocab.update(text.lower().split())

word_to_idx = {word: idx + 1 for idx, word in enumerate(sorted(vocab))}  # 0 reserved for padding
word_to_idx['<PAD>'] = 0
vocab_size = len(word_to_idx)

print(f"Vocabulary size: {vocab_size}")
print(f"Word to index mapping: {word_to_idx}")

# Create datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

train_dataset = TextClassificationDataset(train_texts, train_labels, word_to_idx)
val_dataset = TextClassificationDataset(val_texts, val_labels, word_to_idx)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Create and train model
model = FeedforwardTextClassifier(
    vocab_size=vocab_size,
    embedding_dim=50,
    hidden_dim=100,
    num_classes=2,
    dropout=0.3
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

train_losses, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, num_epochs=20, learning_rate=0.001
)

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 3, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.show()
```

## Neural Language Models

Neural language models predict the next word in a sequence using neural networks, providing better generalization than n-gram models.

### Architecture

A neural language model typically consists of:
1. **Embedding Layer**: Maps words to dense vectors
2. **Hidden Layers**: Process the sequence
3. **Output Layer**: Predicts probability distribution over vocabulary

### Implementation

```python
class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(NeuralLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Feedforward layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.size()
        
        # Embedding lookup
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Reshape for feedforward processing
        embedded = embedded.view(-1, self.embedding_dim)  # (batch_size * seq_len, embedding_dim)
        
        # Feedforward layers
        hidden = embedded
        for layer in self.layers:
            hidden = self.relu(layer(hidden))
            hidden = self.dropout(hidden)
        
        # Output layer
        output = self.output_layer(hidden)
        
        # Reshape back to sequence format
        output = output.view(batch_size, seq_len, self.vocab_size)
        
        return output

class LanguageModelDataset(Dataset):
    def __init__(self, texts, word_to_idx, sequence_length=10):
        self.texts = texts
        self.word_to_idx = word_to_idx
        self.sequence_length = sequence_length
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        for text in self.texts:
            words = text.lower().split()
            indices = [self.word_to_idx.get(word, 0) for word in words]
            
            # Create sequences of length sequence_length + 1 (input + target)
            for i in range(len(indices) - self.sequence_length):
                sequence = indices[i:i + self.sequence_length + 1]
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def train_language_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the neural language model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Reshape for loss calculation
            output = output.view(-1, model.vocab_size)
            target = target.view(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                output = output.view(-1, model.vocab_size)
                target = target.view(-1)
                
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def generate_text(model, word_to_idx, idx_to_word, seed_text, max_length=20):
    """Generate text using the trained model"""
    model.eval()
    
    # Convert seed text to indices
    seed_words = seed_text.lower().split()
    seed_indices = [word_to_idx.get(word, 0) for word in seed_words]
    
    generated_indices = seed_indices.copy()
    
    with torch.no_grad():
        for _ in range(max_length - len(seed_indices)):
            # Prepare input
            input_seq = torch.tensor([generated_indices[-10:]], dtype=torch.long)  # Use last 10 words
            
            # Get prediction
            output = model(input_seq)
            next_word_logits = output[0, -1, :]
            
            # Sample next word
            probabilities = torch.softmax(next_word_logits, dim=0)
            next_word_idx = torch.multinomial(probabilities, 1).item()
            
            generated_indices.append(next_word_idx)
    
    # Convert back to text
    generated_words = [idx_to_word.get(idx, '<UNK>') for idx in generated_indices]
    return ' '.join(generated_words)

# Example usage for language modeling
language_model = NeuralLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=50,
    hidden_dim=100,
    num_layers=2,
    dropout=0.3
)

# Create language model dataset
lm_dataset = LanguageModelDataset(texts, word_to_idx, sequence_length=10)
train_size = int(0.8 * len(lm_dataset))
val_size = len(lm_dataset) - train_size

train_lm_dataset, val_lm_dataset = torch.utils.data.random_split(
    lm_dataset, [train_size, val_size]
)

train_lm_loader = DataLoader(train_lm_dataset, batch_size=4, shuffle=True)
val_lm_loader = DataLoader(val_lm_dataset, batch_size=4, shuffle=False)

# Train language model
lm_train_losses, lm_val_losses = train_language_model(
    language_model, train_lm_loader, val_lm_loader, num_epochs=15, learning_rate=0.001
)

# Create reverse mapping for generation
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Generate text
generated_text = generate_text(language_model, word_to_idx, idx_to_word, "great movie", max_length=10)
print(f"Generated text: {generated_text}")
```

## Sequence Labeling with Neural Networks

Neural networks can be applied to sequence labeling tasks like POS tagging and named entity recognition.

### Implementation

```python
class NeuralSequenceTagger(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(NeuralSequenceTagger, self).__init__()
        
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim * 2, tag_size)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.size()
        
        # Embedding lookup
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)  # (batch_size, sequence_length, hidden_dim * 2)
        
        # Output layer
        output = self.output_layer(lstm_out)  # (batch_size, sequence_length, tag_size)
        
        return output

class SequenceTaggingDataset(Dataset):
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx, max_length=50):
        self.sentences = sentences
        self.tags = tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_sequence = self.tags[idx]
        
        # Convert to indices
        word_indices = [self.word_to_idx.get(word, 0) for word in sentence]
        tag_indices = [self.tag_to_idx[tag] for tag in tag_sequence]
        
        # Pad or truncate
        if len(word_indices) < self.max_length:
            word_indices.extend([0] * (self.max_length - len(word_indices)))
            tag_indices.extend([0] * (self.max_length - len(tag_indices)))
        else:
            word_indices = word_indices[:self.max_length]
            tag_indices = tag_indices[:self.max_length]
        
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

def train_sequence_tagger(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the sequence tagging model"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Reshape for loss calculation
            output = output.view(-1, model.tag_size)
            target = target.view(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                output = output.view(-1, model.tag_size)
                target = target.view(-1)
                
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# Example usage for POS tagging
def create_pos_data():
    """Create sample POS tagging data"""
    sentences = [
        ['The', 'cat', 'sat', 'on', 'the', 'mat'],
        ['The', 'dog', 'ran', 'in', 'the', 'park'],
        ['A', 'big', 'car', 'drove', 'fast'],
        ['The', 'red', 'ball', 'bounced', 'high']
    ]
    
    tags = [
        ['DT', 'NN', 'VBD', 'IN', 'DT', 'NN'],
        ['DT', 'NN', 'VBD', 'IN', 'DT', 'NN'],
        ['DT', 'JJ', 'NN', 'VBD', 'RB'],
        ['DT', 'JJ', 'NN', 'VBD', 'RB']
    ]
    
    return sentences, tags

sentences, tags = create_pos_data()

# Create vocabularies
word_vocab = set()
tag_vocab = set()

for sentence in sentences:
    word_vocab.update(sentence)
for tag_sequence in tags:
    tag_vocab.update(tag_sequence)

word_to_idx = {word: idx + 1 for idx, word in enumerate(sorted(word_vocab))}
word_to_idx['<PAD>'] = 0

tag_to_idx = {tag: idx + 1 for idx, tag in enumerate(sorted(tag_vocab))}
tag_to_idx['<PAD>'] = 0

print(f"Word vocabulary: {word_to_idx}")
print(f"Tag vocabulary: {tag_to_idx}")

# Create dataset
tagging_dataset = SequenceTaggingDataset(sentences, tags, word_to_idx, tag_to_idx)
tagging_loader = DataLoader(tagging_dataset, batch_size=2, shuffle=True)

# Create and train model
tagger = NeuralSequenceTagger(
    vocab_size=len(word_to_idx),
    tag_size=len(tag_to_idx),
    embedding_dim=50,
    hidden_dim=100,
    num_layers=2,
    dropout=0.3
)

# Train the tagger
tagger_train_losses, tagger_val_losses = train_sequence_tagger(
    tagger, tagging_loader, tagging_loader, num_epochs=20, learning_rate=0.001
)
```

## Regularization Techniques

### Dropout

Dropout randomly sets a fraction of input units to 0 during training, preventing overfitting.

```python
class DropoutRegularization(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(DropoutRegularization, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Different dropout rates for different layers
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate * 0.5)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        pooled = torch.mean(embedded, dim=1)
        
        hidden1 = self.relu(self.fc1(pooled))
        hidden1 = self.fc_dropout(hidden1)
        
        hidden2 = self.relu(self.fc2(hidden1))
        hidden2 = self.fc_dropout(hidden2)
        
        output = self.fc3(hidden2)
        output = self.output_dropout(output)
        
        return output
```

### Batch Normalization

Batch normalization normalizes inputs to each layer, improving training stability.

```python
class BatchNormalizedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BatchNormalizedModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        
        hidden1 = self.fc1(pooled)
        hidden1 = self.bn1(hidden1)
        hidden1 = self.relu(hidden1)
        hidden1 = self.dropout(hidden1)
        
        hidden2 = self.fc2(hidden1)
        hidden2 = self.bn2(hidden2)
        hidden2 = self.relu(hidden2)
        hidden2 = self.dropout(hidden2)
        
        output = self.fc3(hidden2)
        
        return output
```

### Weight Decay and Early Stopping

```python
def train_with_regularization(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, weight_decay=1e-4):
    """Train model with regularization techniques"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses
```

## Practical Exercises

### Exercise 1: Text Classification with Different Architectures
Implement and compare different neural architectures for text classification:
- Feedforward networks with different hidden layer sizes
- Networks with different activation functions
- Models with and without regularization

### Exercise 2: Neural Language Model Training
Train a neural language model on a larger corpus:
- Implement different sampling strategies for training
- Compare with traditional n-gram models
- Evaluate using perplexity and generated text quality

### Exercise 3: Sequence Labeling Tasks
Apply neural networks to different sequence labeling tasks:
- POS tagging with different architectures
- Named Entity Recognition
- Sentiment analysis at the word level

## Assessment Questions

1. **What are the advantages of neural networks over traditional NLP methods?**
   - Automatic feature learning
   - End-to-end training
   - Better generalization
   - Scalability to large datasets

2. **How does dropout help prevent overfitting?**
   - Randomly sets units to zero during training
   - Prevents co-adaptation of features
   - Acts as a form of ensemble learning
   - Reduces model complexity

3. **What is the purpose of batch normalization?**
   - Normalizes inputs to each layer
   - Improves training stability
   - Allows higher learning rates
   - Reduces internal covariate shift

## Key Takeaways

- Neural networks enable automatic feature learning for NLP tasks
- Feedforward networks are effective for text classification
- Neural language models provide better generalization than n-gram models
- Sequence labeling tasks benefit from recurrent architectures
- Regularization techniques are crucial for preventing overfitting
- Proper hyperparameter tuning significantly affects model performance
- Neural networks form the foundation for more advanced architectures

## Next Steps

In the next module, we'll explore Recurrent Neural Networks (RNNs), which are specifically designed to handle sequential data and have been fundamental to many NLP breakthroughs before the transformer era.
