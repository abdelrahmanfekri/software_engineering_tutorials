# NLP Tutorial Module 7: Attention Mechanisms and Sequence-to-Sequence Models

## Learning Objectives
By the end of this module, you will be able to:
- Understand the concept and importance of attention mechanisms
- Implement different types of attention (self-attention, cross-attention)
- Build sequence-to-sequence models with attention
- Apply attention to machine translation tasks
- Understand the mathematical foundations of attention
- Implement attention visualization techniques

## Introduction to Attention Mechanisms

Attention mechanisms allow models to focus on relevant parts of the input when making predictions, addressing the bottleneck problem in encoder-decoder architectures and enabling better handling of long sequences.

### Why Attention?

1. **Information Bottleneck**: Fixed-size context vector limits information transfer
2. **Long Sequences**: RNNs struggle with very long sequences
3. **Selective Focus**: Not all input is equally important for each output
4. **Parallelization**: Attention can be computed in parallel

### Types of Attention

1. **Self-Attention**: Attention within a single sequence
2. **Cross-Attention**: Attention between two different sequences
3. **Global vs Local**: Attention over entire sequence vs local window
4. **Hard vs Soft**: Discrete vs continuous attention weights

## Mathematical Foundation of Attention

### Basic Attention Mechanism

The attention mechanism computes a weighted sum of values based on query-key similarity:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q: Query matrix
- K: Key matrix  
- V: Value matrix
- d_k: Dimension of key vectors

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: (batch_size, seq_len_q, seq_len_k) or None
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# Example usage
def demonstrate_attention():
    batch_size, seq_len, d_model = 2, 5, 8
    
    # Create sample inputs
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention mechanism
    attention = ScaledDotProductAttention(d_model)
    
    # Forward pass
    output, attention_weights = attention(query, key, value)
    
    print(f"Input shapes - Q: {query.shape}, K: {key.shape}, V: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights
    plt.figure(figsize=(10, 4))
    plt.imshow(attention_weights[0].detach().numpy(), cmap='Blues')
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    plt.show()
    
    return attention_weights

attention_weights = demonstrate_attention()
```

## Multi-Head Attention

Multi-head attention allows the model to attend to different representation subspaces simultaneously.

### Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights

# Example usage
def demonstrate_multi_head_attention():
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8
    
    # Create sample inputs
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attention_weights = mha(query, key, value)
    
    print(f"Multi-head attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return mha, attention_weights

mha, mha_weights = demonstrate_multi_head_attention()
```

## Sequence-to-Sequence Models with Attention

### Encoder-Decoder Architecture

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        return outputs, (hidden, cell)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2, dropout=0.1):
        super(AttentionDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            hidden_size * 2, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        
        # Output layer
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_outputs, hidden, cell):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Calculate attention weights
        attention_weights = self._calculate_attention(hidden[-1], encoder_outputs)
        
        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        # Concatenate embedded input and context
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=1)
        lstm_input = lstm_input.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Output layer
        output = self.out(lstm_out.squeeze(1))
        
        return output, hidden, cell, attention_weights
    
    def _calculate_attention(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        
        # Repeat hidden state for each encoder output
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate hidden state with encoder outputs
        concat = torch.cat([hidden_repeated, encoder_outputs], dim=2)
        
        # Calculate attention scores
        attention_scores = self.attention(concat).squeeze(2)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout=0.1):
        super(Seq2SeqWithAttention, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = AttentionDecoder(output_size, hidden_size, num_layers, dropout)
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.output_size
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(source)
        
        # Initialize decoder hidden state
        hidden = hidden[:self.decoder.num_layers]  # Take only forward direction
        cell = cell[:self.decoder.num_layers]
        
        # Initialize outputs and attention weights
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        attention_weights = torch.zeros(batch_size, target_len, source.size(1))
        
        # First input to decoder (start token)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
        
        for t in range(target_len):
            # Decoder forward pass
            output, hidden, cell, attn_weights = self.decoder(
                decoder_input, encoder_outputs, hidden, cell
            )
            
            outputs[:, t, :] = output
            attention_weights[:, t, :] = attn_weights
            
            # Teacher forcing or use previous prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=1).unsqueeze(1)
        
        return outputs, attention_weights

# Example usage
def create_sample_data():
    """Create sample parallel text data"""
    source_sentences = [
        "hello world",
        "how are you",
        "good morning",
        "nice to meet you"
    ]
    
    target_sentences = [
        "bonjour monde",
        "comment allez vous",
        "bonjour",
        "ravi de vous rencontrer"
    ]
    
    return source_sentences, target_sentences

def tokenize_data(sentences):
    """Simple tokenization"""
    tokens = []
    for sentence in sentences:
        tokens.append(sentence.lower().split())
    return tokens

def create_vocabularies(source_tokens, target_tokens):
    """Create vocabulary mappings"""
    source_vocab = set()
    target_vocab = set()
    
    for tokens in source_tokens:
        source_vocab.update(tokens)
    for tokens in target_tokens:
        target_vocab.update(tokens)
    
    source_word_to_idx = {word: idx + 1 for idx, word in enumerate(sorted(source_vocab))}
    source_word_to_idx['<PAD>'] = 0
    source_word_to_idx['<UNK>'] = len(source_word_to_idx)
    
    target_word_to_idx = {word: idx + 1 for idx, word in enumerate(sorted(target_vocab))}
    target_word_to_idx['<PAD>'] = 0
    target_word_to_idx['<SOS>'] = len(target_word_to_idx)
    target_word_to_idx['<EOS>'] = len(target_word_to_idx)
    
    return source_word_to_idx, target_word_to_idx

# Create sample data
source_sentences, target_sentences = create_sample_data()
source_tokens = tokenize_data(source_sentences)
target_tokens = tokenize_data(target_sentences)

source_word_to_idx, target_word_to_idx = create_vocabularies(source_tokens, target_tokens)

print(f"Source vocabulary: {source_word_to_idx}")
print(f"Target vocabulary: {target_word_to_idx}")

# Convert to indices
def tokens_to_indices(tokens, word_to_idx, max_length=10):
    indices = []
    for token_seq in tokens:
        seq_indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in token_seq]
        if len(seq_indices) < max_length:
            seq_indices.extend([word_to_idx['<PAD>']] * (max_length - len(seq_indices)))
        else:
            seq_indices = seq_indices[:max_length]
        indices.append(seq_indices)
    return indices

source_indices = tokens_to_indices(source_tokens, source_word_to_idx)
target_indices = tokens_to_indices(target_tokens, target_word_to_idx)

# Add SOS and EOS tokens to target
for i, target_seq in enumerate(target_indices):
    # Find first PAD token
    pad_idx = target_seq.index(target_word_to_idx['<PAD>'])
    if pad_idx > 0:
        target_seq[pad_idx] = target_word_to_idx['<EOS>']

print(f"Source indices: {source_indices}")
print(f"Target indices: {target_indices}")

# Create model
model = Seq2SeqWithAttention(
    input_size=len(source_word_to_idx),
    output_size=len(target_word_to_idx),
    hidden_size=64,
    num_layers=2,
    dropout=0.1
)

# Convert to tensors
source_tensor = torch.tensor(source_indices, dtype=torch.long)
target_tensor = torch.tensor(target_indices, dtype=torch.long)

print(f"Source tensor shape: {source_tensor.shape}")
print(f"Target tensor shape: {target_tensor.shape}")
```

## Training Sequence-to-Sequence Models

```python
def train_seq2seq_model(model, source_data, target_data, epochs=100, learning_rate=0.001):
    """Train sequence-to-sequence model with attention"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Forward pass
        outputs, attention_weights = model(source_data, target_data, teacher_forcing_ratio=0.5)
        
        # Calculate loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_data.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss:.4f}')
    
    return attention_weights

# Train the model
attention_weights = train_seq2seq_model(model, source_tensor, target_tensor, epochs=50)

# Visualize attention weights
def visualize_attention(attention_weights, source_tokens, target_tokens, sample_idx=0):
    """Visualize attention weights for a sample"""
    plt.figure(figsize=(12, 6))
    
    # Get attention weights for the sample
    attn = attention_weights[sample_idx].detach().numpy()
    
    # Create heatmap
    plt.imshow(attn, cmap='Blues')
    plt.title('Attention Weights')
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    
    # Set labels
    source_labels = source_tokens[sample_idx]
    target_labels = target_tokens[sample_idx]
    
    plt.xticks(range(len(source_labels)), source_labels)
    plt.yticks(range(len(target_labels)), target_labels)
    
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Visualize attention
visualize_attention(attention_weights, source_tokens, target_tokens, sample_idx=0)
```

## Self-Attention Implementation

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        
        # Residual connection and layer norm
        output = self.norm(x + self.dropout(attn_output))
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.self_attn(x)
        
        # Feed forward
        ff_output = self.feed_forward(attn_output)
        ff_output = self.norm2(attn_output + self.dropout(ff_output))
        
        return ff_output, attn_weights

# Example usage of transformer block
def demonstrate_transformer_block():
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8
    d_ff = 256
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create transformer block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    
    # Forward pass
    output, attention_weights = transformer_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return transformer_block, attention_weights

transformer_block, transformer_attention = demonstrate_transformer_block()
```

## Attention Visualization Techniques

```python
def create_attention_visualization(attention_weights, source_words, target_words, head=0):
    """Create detailed attention visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Attention heatmap
    attn_matrix = attention_weights[0, head].detach().numpy()
    im = ax1.imshow(attn_matrix, cmap='Blues')
    ax1.set_title(f'Attention Head {head}')
    ax1.set_xlabel('Source Position')
    ax1.set_ylabel('Target Position')
    
    # Set labels
    ax1.set_xticks(range(len(source_words)))
    ax1.set_yticks(range(len(target_words)))
    ax1.set_xticklabels(source_words)
    ax1.set_yticklabels(target_words)
    
    # Add colorbar
    plt.colorbar(im, ax=ax1)
    
    # Plot 2: Attention distribution for each target word
    for i, target_word in enumerate(target_words):
        attention_dist = attn_matrix[i, :]
        ax2.plot(attention_dist, label=target_word, marker='o')
    
    ax2.set_title('Attention Distribution by Target Word')
    ax2.set_xlabel('Source Position')
    ax2.set_ylabel('Attention Weight')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example visualization
source_words = ["hello", "world"]
target_words = ["bonjour", "monde"]

# Create dummy attention weights
dummy_attention = torch.randn(1, 8, len(target_words), len(source_words))  # (batch, heads, target_len, source_len)
create_attention_visualization(dummy_attention, source_words, target_words, head=0)
```

## Practical Exercises

### Exercise 1: Attention Mechanism Analysis
Implement and analyze different attention mechanisms:
- Compare scaled dot-product vs additive attention
- Analyze the effect of different scaling factors
- Visualize attention patterns for different tasks

### Exercise 2: Machine Translation with Attention
Build a complete machine translation system:
- Train on parallel corpora
- Implement beam search for decoding
- Evaluate using BLEU score
- Analyze attention patterns for different language pairs

### Exercise 3: Attention Ablation Studies
Conduct ablation studies on attention mechanisms:
- Compare models with and without attention
- Analyze the effect of different numbers of attention heads
- Study the impact of attention on different sequence lengths

## Assessment Questions

1. **What problem does attention solve in sequence-to-sequence models?**
   - Information bottleneck in encoder-decoder architecture
   - Difficulty handling long sequences
   - Need for selective focus on relevant input parts
   - All of the above

2. **What is the difference between self-attention and cross-attention?**
   - Self-attention: within same sequence
   - Cross-attention: between different sequences
   - Self-attention: query, key, value from same source
   - Cross-attention: query from one sequence, key/value from another

3. **How does multi-head attention improve model performance?**
   - Allows attending to different representation subspaces
   - Captures different types of relationships
   - Provides more modeling capacity
   - All of the above

## Key Takeaways

- Attention mechanisms solve the information bottleneck in encoder-decoder models
- Self-attention allows models to attend to different positions within the same sequence
- Multi-head attention captures different types of relationships simultaneously
- Attention weights provide interpretability into model decisions
- Attention enables better handling of long sequences
- Attention mechanisms are fundamental to modern transformer architectures
- Visualization of attention weights helps understand model behavior

## Next Steps

In the next module, we'll explore the Transformer architecture, which revolutionized NLP by using attention mechanisms as the primary building block, eliminating the need for recurrent connections entirely.
