# NLP Tutorial Module 8: Transformer Architecture

## Learning Objectives
By the end of this module, you will be able to:
- Understand the complete Transformer architecture
- Implement Transformer components from scratch
- Apply Transformers to various NLP tasks
- Understand self-attention and multi-head attention in detail
- Implement positional encoding and layer normalization
- Build encoder-decoder and encoder-only Transformer models
- Apply Transformers to language modeling and machine translation

## Introduction to Transformers

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized NLP by showing that attention mechanisms alone are sufficient for building powerful sequence models, eliminating the need for recurrent or convolutional layers.

### Key Innovations

1. **Self-Attention**: Allows each position to attend to all positions in the same sequence
2. **Parallel Processing**: Unlike RNNs, Transformers process all positions simultaneously
3. **Multi-Head Attention**: Captures different types of relationships simultaneously
4. **Positional Encoding**: Injects position information without recurrence
5. **Layer Normalization**: Stabilizes training and improves convergence

### Transformer Architecture Overview

```
Input Embeddings + Positional Encoding
           ↓
    [Encoder Block] × N
           ↓
    [Decoder Block] × N
           ↓
    Linear + Softmax
```

## Core Components Implementation

### Multi-Head Attention (Detailed)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Example usage
def demonstrate_multi_head_attention():
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create sample inputs
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Self-attention (query, key, value are the same)
    output, attention_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return mha, attention_weights

mha, attention_weights = demonstrate_multi_head_attention()
```

### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

# Alternative: Learnable positional encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.embedding(positions)

# Visualize positional encoding
def visualize_positional_encoding():
    d_model = 128
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # Create dummy input
    x = torch.randn(max_len, 1, d_model)
    pos_encoded = pe(x)
    
    # Plot positional encoding
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(pos_encoded[:50, 0, :50].detach().numpy(), cmap='RdYlBu')
    plt.title('Positional Encoding (First 50 positions, First 50 dimensions)')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    
    plt.subplot(1, 3, 2)
    # Plot encoding for different positions
    positions = [0, 10, 25, 49]
    for pos in positions:
        plt.plot(pos_encoded[pos, 0, :50].detach().numpy(), label=f'Position {pos}')
    plt.title('Positional Encoding Values')
    plt.xlabel('Dimension')
    plt.ylabel('Encoding Value')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Plot encoding for different dimensions at same position
    dims = [0, 1, 2, 3, 4, 5]
    for dim in dims:
        plt.plot(pos_encoded[:50, 0, dim].detach().numpy(), label=f'Dimension {dim}')
    plt.title('Positional Encoding Across Positions')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

visualize_positional_encoding()
```

### Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### Transformer Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

# Example usage
def demonstrate_encoder_block():
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    d_ff = 2048
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create encoder block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    
    # Forward pass
    output, attention_weights = encoder_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return encoder_block, attention_weights

encoder_block, encoder_attention = demonstrate_encoder_block()
```

### Transformer Decoder Block

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        # Masked self-attention
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, self_attention_weights = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        cross_attn_output, cross_attention_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attention_weights, cross_attention_weights

def create_padding_mask(seq, pad_token=0):
    """Create padding mask for attention"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """Create look-ahead mask for decoder"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

# Example usage
def demonstrate_decoder_block():
    batch_size, tgt_len, src_len, d_model = 2, 8, 10, 512
    num_heads = 8
    d_ff = 2048
    
    # Create sample inputs
    decoder_input = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    # Create masks
    src_mask = create_padding_mask(torch.ones(batch_size, src_len))
    tgt_mask = create_look_ahead_mask(tgt_len)
    
    # Create decoder block
    decoder_block = TransformerDecoderBlock(d_model, num_heads, d_ff)
    
    # Forward pass
    output, self_attn, cross_attn = decoder_block(decoder_input, encoder_output, src_mask, tgt_mask)
    
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {output.shape}")
    
    return decoder_block, self_attn, cross_attn

decoder_block, self_attn, cross_attn = demonstrate_decoder_block()
```

## Complete Transformer Model

### Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder blocks
        attention_weights = []
        for encoder_block in self.encoder_blocks:
            x, attn_weights = encoder_block(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights

# Example usage
def demonstrate_transformer_encoder():
    batch_size, seq_len = 2, 10
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create sample input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create encoder
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Forward pass
    output, attention_weights = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    
    return encoder, attention_weights

encoder, encoder_attention_weights = demonstrate_transformer_encoder()
```

### Transformer Decoder

```python
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding and positional encoding
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder blocks
        self_attention_weights = []
        cross_attention_weights = []
        
        for decoder_block in self.decoder_blocks:
            x, self_attn, cross_attn = decoder_block(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        return x, self_attention_weights, cross_attention_weights

# Example usage
def demonstrate_transformer_decoder():
    batch_size, tgt_len = 2, 8
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create sample inputs
    decoder_input = torch.randint(0, vocab_size, (batch_size, tgt_len))
    encoder_output = torch.randn(batch_size, 10, d_model)  # Dummy encoder output
    
    # Create decoder
    decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Create masks
    src_mask = create_padding_mask(torch.ones(batch_size, 10))
    tgt_mask = create_look_ahead_mask(tgt_len)
    
    # Forward pass
    output, self_attn_weights, cross_attn_weights = decoder(
        decoder_input, encoder_output, src_mask, tgt_mask
    )
    
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of self-attention weight tensors: {len(self_attn_weights)}")
    print(f"Number of cross-attention weight tensors: {len(cross_attn_weights)}")
    
    return decoder, self_attn_weights, cross_attn_weights

decoder, self_attn_weights, cross_attn_weights = demonstrate_transformer_decoder()
```

### Complete Transformer Model

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        encoder_output, encoder_attention = self.encoder(src, src_mask)
        
        # Decode target
        decoder_output, self_attention, cross_attention = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output, encoder_attention, self_attention, cross_attention

# Example usage
def demonstrate_complete_transformer():
    batch_size, src_len, tgt_len = 2, 10, 8
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create sample inputs
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # Create masks
    src_mask = create_padding_mask(torch.ones(batch_size, src_len))
    tgt_mask = create_look_ahead_mask(tgt_len)
    
    # Create transformer
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Forward pass
    output, enc_attn, self_attn, cross_attn = transformer(src, tgt, src_mask, tgt_mask)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of encoder attention layers: {len(enc_attn)}")
    print(f"Number of self-attention layers: {len(self_attn)}")
    print(f"Number of cross-attention layers: {len(cross_attn)}")
    
    return transformer, enc_attn, self_attn, cross_attn

transformer, enc_attn, self_attn, cross_attn = demonstrate_complete_transformer()
```

## GPT-style Decoder-Only Transformer

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(GPTBlock, self).__init__()
        
        # Masked multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Masked self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(GPT, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embedding
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = create_look_ahead_mask(seq_len).to(x.device)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        # Final layer norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits, attention_weights
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=None, top_p=None):
        """Generate text using the model"""
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated

# Example usage
def demonstrate_gpt():
    batch_size, seq_len = 2, 10
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create sample input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create GPT model
    gpt = GPT(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Forward pass
    logits, attention_weights = gpt(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    
    # Generate text
    generated = gpt.generate(x[:, :5], max_length=10, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    return gpt, attention_weights

gpt_model, gpt_attention_weights = demonstrate_gpt()
```

## Training and Optimization

```python
class TransformerTrainer:
    def __init__(self, model, learning_rate=0.0001, warmup_steps=4000):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def get_lr(self):
        """Learning rate scheduler with warmup"""
        return min(self.step ** (-0.5), self.step * self.warmup_steps ** (-1.5))
    
    def train_step(self, src, tgt, criterion):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Create masks
        src_mask = create_padding_mask(src)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = create_look_ahead_mask(tgt_input.size(1))
        
        # Forward pass
        output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        
        self.optimizer.step()
        self.step += 1
        
        return loss.item()

# Example training loop
def train_transformer_example():
    # Create model
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Create trainer
    trainer = TransformerTrainer(model)
    
    # Create dummy data
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Training loop
    for epoch in range(5):
        total_loss = 0
        
        for batch in range(10):  # 10 batches per epoch
            src = torch.randint(0, src_vocab_size, (batch_size, src_len))
            tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
            
            loss = trainer.train_step(src, tgt, criterion)
            total_loss += loss
            
            if batch % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}, LR: {trainer.get_lr():.6f}')
        
        avg_loss = total_loss / 10
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')

# Run training example
train_transformer_example()
```

## Practical Exercises

### Exercise 1: Transformer Architecture Analysis
Implement and analyze different Transformer configurations:
- Compare different numbers of layers and heads
- Analyze the effect of different model dimensions
- Study attention patterns across layers

### Exercise 2: Positional Encoding Comparison
Compare different positional encoding methods:
- Sinusoidal vs learnable positional encoding
- Absolute vs relative positional encoding
- Analyze their effect on model performance

### Exercise 3: Transformer for Language Modeling
Build a complete language modeling system:
- Train GPT-style model on text data
- Implement different generation strategies
- Evaluate using perplexity and generated text quality

## Assessment Questions

1. **What are the key advantages of Transformers over RNNs?**
   - Parallel processing of sequences
   - Better handling of long-range dependencies
   - More efficient training
   - All of the above

2. **What is the purpose of positional encoding in Transformers?**
   - Provides position information to the model
   - Enables the model to understand sequence order
   - Replaces recurrent connections
   - All of the above

3. **How does multi-head attention improve model performance?**
   - Allows attending to different representation subspaces
   - Captures different types of relationships
   - Provides more modeling capacity
   - All of the above

## Key Takeaways

- Transformers use attention mechanisms as the primary building block
- Self-attention allows each position to attend to all other positions
- Multi-head attention captures different types of relationships simultaneously
- Positional encoding is crucial for sequence understanding
- Layer normalization and residual connections stabilize training
- Transformers can be used for both encoder-decoder and decoder-only tasks
- The architecture scales well with model size and data

## Next Steps

In the next module, we'll explore pre-trained language models like BERT and GPT, which revolutionized NLP by showing how large-scale pre-training can dramatically improve performance on downstream tasks.
