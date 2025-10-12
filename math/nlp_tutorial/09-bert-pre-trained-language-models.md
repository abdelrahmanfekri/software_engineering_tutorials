# NLP Tutorial Module 9: BERT and Pre-trained Language Models (PhD Level)

## Learning Objectives
By the end of this module, you will be able to:
- Understand the theoretical foundations of pre-training and transfer learning
- Implement BERT and its variants from scratch with mathematical rigor
- Master advanced pre-training objectives and architectures
- Apply domain adaptation and continued pre-training techniques
- Understand scaling laws and emergence in language models
- Implement efficient fine-tuning strategies (adapters, LoRA, prefix tuning)
- Analyze model behavior through probing and interpretability methods
- Design custom pre-training objectives for specialized domains

## Theoretical Foundations of Pre-training

### Transfer Learning in NLP: A Formal Framework

Pre-training leverages the hypothesis that language models learn universal representations that transfer across tasks. Formally:

Let $\mathcal{D}_{\text{pre}} = \{x_i\}_{i=1}^N$ be a large unsupervised corpus and $\mathcal{D}_{\text{task}} = \{(x_i, y_i)\}_{i=1}^M$ be a smaller labeled dataset.

The pre-training objective minimizes:
$$\mathcal{L}_{\text{pre}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}_{\text{pre}}}[\log p_\theta(x)]$$

The fine-tuning objective minimizes:
$$\mathcal{L}_{\text{task}}(\theta | \theta^*) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{task}}}[\log p_{\theta}(y|x)]$$

where $\theta^*$ are the pre-trained parameters.

### Information-Theoretic View

From an information-theoretic perspective, pre-training maximizes the mutual information between the model's learned representations and the downstream task:

$$I(Z; Y) = H(Y) - H(Y|Z)$$

where $Z = f_\theta(X)$ are the learned representations.

## BERT: Bidirectional Encoder Representations from Transformers

### Architecture Deep Dive

BERT uses a multi-layer bidirectional Transformer encoder with the following mathematical formulation:

For input sequence $X = [x_1, ..., x_n]$:

1. **Input Embeddings**:
   $$E = \text{Emb}(X) + \text{Pos}(X) + \text{Seg}(X)$$

2. **Multi-Layer Transformer**:
   $$H^{(l)} = \text{TransformerBlock}(H^{(l-1)})$$
   
   where each block consists of:
   - Multi-head self-attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
   - Feed-forward network: $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$

3. **Output Representations**: $H^{(L)} \in \mathbb{R}^{n \times d}$

### Implementation from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math

class BERTEmbeddings(nn.Module):
    """BERT Embeddings with theoretical justification"""
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int, 
                 type_vocab_size: int, dropout: float = 0.1):
        super().__init__()
        # Token embeddings - learnable discrete representation
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Position embeddings - encode absolute position information
        # Alternative to sinusoidal encoding, allows learning position-dependent patterns
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Token type embeddings - distinguish between segments (e.g., question vs context)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # Layer normalization for stable training
        # Normalizes across feature dimension: x' = (x - μ) / σ * γ + β
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # Position IDs buffer (1, max_position_embeddings)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_length) - token indices
            token_type_ids: (batch_size, seq_length) - segment indices
            position_ids: (batch_size, seq_length) - position indices
        Returns:
            embeddings: (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length = input_ids.size()
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Compute embeddings
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Sum all embeddings (following original BERT)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERTSelfAttention(nn.Module):
    """Self-attention with theoretical foundations"""
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}")
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        # Each head learns different attention patterns
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_length) - mask for padding
        Returns:
            context_layer: (batch_size, seq_length, hidden_size)
            attention_probs: (batch_size, num_heads, seq_length, seq_length)
        """
        # Linear projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        # Q @ K^T / sqrt(d_k) - scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask (if provided)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to (batch_size, seq_length, all_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

class BERTSelfOutput(nn.Module):
    """Output projection and residual connection"""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply output projection, dropout, and residual connection"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Residual connection + layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BERTAttention(nn.Module):
    """Complete attention module with residual connections"""
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self = BERTSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = BERTSelfOutput(hidden_size, dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output, attention_probs

class BERTIntermediate(nn.Module):
    """Intermediate feed-forward layer with GELU activation"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # Smoother than ReLU, better gradient flow
        self.intermediate_act_fn = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BERTOutput(nn.Module):
    """Output projection of feed-forward network"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BERTLayer(nn.Module):
    """Single BERT transformer layer"""
    def __init__(self, hidden_size: int, num_attention_heads: int, 
                 intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BERTAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(hidden_size, intermediate_size, dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output, attention_probs

class BERTEncoder(nn.Module):
    """Stack of BERT layers"""
    def __init__(self, num_hidden_layers: int, hidden_size: int, 
                 num_attention_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        all_attentions = [] if output_attentions else None
        
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            
            if output_attentions:
                all_attentions.append(attention_probs)
        
        return hidden_states, all_attentions

class BERTPooler(nn.Module):
    """Pooler to get sentence representation from [CLS] token"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERTModel(nn.Module):
    """Complete BERT model architecture"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = BERTEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            dropout=config.hidden_dropout_prob
        )
        
        self.encoder = BERTEncoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob
        )
        
        self.pooler = BERTPooler(config.hidden_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through BERT
        
        Args:
            input_ids: (batch_size, seq_length) - input token IDs
            attention_mask: (batch_size, seq_length) - mask for padding (1 for real tokens, 0 for padding)
            token_type_ids: (batch_size, seq_length) - segment IDs
            position_ids: (batch_size, seq_length) - position IDs
            output_attentions: whether to output attention weights
            
        Returns:
            sequence_output: (batch_size, seq_length, hidden_size) - final hidden states
            pooled_output: (batch_size, hidden_size) - pooled [CLS] representation
            attentions: List of attention weights (if output_attentions=True)
        """
        # Create attention mask in correct shape
        if attention_mask is not None:
            # Expand mask to (batch_size, 1, 1, seq_length) for broadcasting
            extended_attention_mask = attention_mask[:, None, None, :]
            # Convert from 0/1 to additive mask (0 for real tokens, -10000 for padding)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        
        # Pass through encoder
        encoder_outputs, attentions = self.encoder(embedding_output, extended_attention_mask, output_attentions)
        
        # Get pooled output
        pooled_output = self.pooler(encoder_outputs)
        
        return encoder_outputs, pooled_output, attentions

# Configuration class
class BERTConfig:
    """BERT configuration"""
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, max_position_embeddings=512,
                 type_vocab_size=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size

# Instantiate BERT
config = BERTConfig()
bert_model = BERTModel(config)

# Example forward pass
batch_size, seq_length = 4, 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)
attention_mask[:, -10:] = 0  # Mask last 10 tokens

sequence_output, pooled_output, attentions = bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)

print(f"Sequence output shape: {sequence_output.shape}")  # (4, 128, 768)
print(f"Pooled output shape: {pooled_output.shape}")      # (4, 768)
print(f"Number of attention layers: {len(attentions)}")   # 12
```

### Pre-training Objectives

#### 1. Masked Language Modeling (MLM)

Mathematical formulation:
$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{i \in \mathcal{M}} \log p_\theta(x_i | x_{\setminus \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

```python
class BERTForMaskedLM(nn.Module):
    """BERT with Masked Language Modeling head"""
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        
        # MLM head
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=1e-12),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None):
        sequence_output, pooled_output, _ = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        
        prediction_scores = self.cls(sequence_output)
        
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            return masked_lm_loss, prediction_scores
        
        return prediction_scores

def create_masked_lm_predictions(tokens, mask_prob=0.15, vocab_size=30522):
    """Create masked LM predictions
    
    Masking strategy (BERT paper):
    - 80% of the time: replace with [MASK] token
    - 10% of the time: replace with random token
    - 10% of the time: keep original token
    """
    masked_tokens = tokens.clone()
    labels = tokens.clone()
    
    # Create mask (15% of tokens)
    probability_matrix = torch.full(tokens.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # We only compute loss on masked tokens
    labels[~masked_indices] = -100
    
    # 80% of the time: replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked_indices
    masked_tokens[indices_replaced] = 103  # [MASK] token ID
    
    # 10% of the time: replace with random token
    indices_random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(vocab_size, tokens.shape, dtype=torch.long)
    masked_tokens[indices_random] = random_tokens[indices_random]
    
    # 10% of the time: keep original token (do nothing)
    
    return masked_tokens, labels
```

#### 2. Next Sentence Prediction (NSP)

Binary classification task to predict if sentence B follows sentence A:

$$\mathcal{L}_{\text{NSP}} = -\mathbb{E}_{(s_A, s_B) \sim \mathcal{D}} [\log p_\theta(\text{IsNext} | s_A, s_B)]$$

```python
class BERTForNextSentencePrediction(nn.Module):
    """BERT with Next Sentence Prediction head"""
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.cls = nn.Linear(config.hidden_size, 2)  # Binary classification
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                next_sentence_label=None):
        _, pooled_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        seq_relationship_score = self.cls(pooled_output)
        
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )
            return next_sentence_loss, seq_relationship_score
        
        return seq_relationship_score

class BERTPreTraining(nn.Module):
    """BERT with both MLM and NSP objectives"""
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=1e-12),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
        # NSP head
        self.nsp_head = nn.Linear(config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output, _ = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        
        # MLM predictions
        prediction_scores = self.mlm_head(sequence_output)
        
        # NSP predictions
        seq_relationship_score = self.nsp_head(pooled_output)
        
        total_loss = 0
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
        
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )
            total_loss += next_sentence_loss
        
        return total_loss, prediction_scores, seq_relationship_score
```

## Advanced Pre-training Techniques

### 1. RoBERTa: Robustly Optimized BERT

Key improvements:
- Remove NSP objective (shown to be unnecessary)
- Dynamic masking (different mask each epoch)
- Larger batch sizes and learning rates
- Train on more data for longer

```python
class RoBERTaTrainer:
    """RoBERTa training with dynamic masking"""
    def __init__(self, model, optimizer, mask_prob=0.15):
        self.model = model
        self.optimizer = optimizer
        self.mask_prob = mask_prob
        
    def dynamic_masking(self, input_ids, vocab_size):
        """Apply dynamic masking - different mask each time"""
        return create_masked_lm_predictions(input_ids, self.mask_prob, vocab_size)
    
    def train_step(self, batch):
        """Single training step with dynamic masking"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Apply dynamic masking
        masked_input_ids, labels = self.dynamic_masking(input_ids, self.model.bert.config.vocab_size)
        
        # Forward pass
        loss, _ = self.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            masked_lm_labels=labels
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 2. ALBERT: A Lite BERT

Key innovations:
- Factorized embedding parameterization: $V \times H = V \times E + E \times H$
- Cross-layer parameter sharing
- Sentence-Order Prediction (SOP) instead of NSP

```python
class ALBERTEmbeddings(nn.Module):
    """ALBERT embeddings with factorization"""
    def __init__(self, vocab_size, embedding_size, hidden_size, max_position_embeddings,
                 type_vocab_size, dropout=0.1):
        super().__init__()
        # Factorized embedding: vocab -> embedding_size -> hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.word_embeddings_projection = nn.Linear(embedding_size, hidden_size, bias=False)
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        batch_size, seq_length = input_ids.size()
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Factorized embedding
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_projection(words_embeddings)
        
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class ALBERTLayerGroup(nn.Module):
    """Group of layers sharing parameters"""
    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.ModuleList([
            BERTLayer(config.hidden_size, config.num_attention_heads,
                     config.intermediate_size, config.hidden_dropout_prob)
        ])
        
    def forward(self, hidden_states, attention_mask=None):
        # All layers in the group share the same parameters
        layer_output = hidden_states
        for layer_module in self.albert_layers:
            layer_output, _ = layer_module(layer_output, attention_mask)
        return layer_output
```

### 3. ELECTRA: Efficiently Learning an Encoder that Classifies Token Replacements Accurately

Instead of MLM, use discriminative objective:

$$\mathcal{L}_{\text{ELECTRA}} = \mathbb{E}_{x \sim \mathcal{D}} \sum_{i=1}^n \mathbb{1}(x_i = \tilde{x}_i) \log D(x, i) + \mathbb{1}(x_i \neq \tilde{x}_i) \log(1 - D(x, i))$$

```python
class ELECTRAGenerator(nn.Module):
    """Small MLM model to generate replacements"""
    def __init__(self, config):
        super().__init__()
        # Smaller BERT model
        self.embeddings = BERTEmbeddings(config.vocab_size, config.hidden_size, 
                                        config.max_position_embeddings, config.type_vocab_size)
        self.encoder = BERTEncoder(config.num_hidden_layers // 3,  # Smaller
                                   config.hidden_size // 2,
                                   config.num_attention_heads // 2,
                                   config.intermediate_size // 2)
        self.mlm_head = nn.Linear(config.hidden_size // 2, config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        sequence_output, _ = self.encoder(embeddings, attention_mask)
        prediction_scores = self.mlm_head(sequence_output)
        return prediction_scores

class ELECTRADiscriminator(nn.Module):
    """Discriminator to detect replaced tokens"""
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.discriminator_head = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        sequence_output, _, _ = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.discriminator_head(sequence_output)
        return logits

class ELECTRAModel(nn.Module):
    """Complete ELECTRA model"""
    def __init__(self, config):
        super().__init__()
        self.generator = ELECTRAGenerator(config)
        self.discriminator = ELECTRADiscriminator(config)
        
    def forward(self, input_ids, attention_mask=None, masked_positions=None):
        # Generator: Predict masked tokens
        generator_logits = self.generator(input_ids, attention_mask)
        
        # Sample from generator predictions
        if masked_positions is not None:
            sampled_tokens = torch.multinomial(
                F.softmax(generator_logits[masked_positions], dim=-1), 1
            ).squeeze(-1)
            
            # Create corrupted input
            corrupted_input_ids = input_ids.clone()
            corrupted_input_ids[masked_positions] = sampled_tokens
            
            # Discriminator: Detect replaced tokens
            discriminator_logits = self.discriminator(
                corrupted_input_ids, attention_mask
            )
            
            # Create labels (1 for replaced, 0 for original)
            labels = torch.zeros_like(input_ids, dtype=torch.float)
            labels[masked_positions] = 1.0
            
            # Compute losses
            generator_loss = F.cross_entropy(
                generator_logits[masked_positions],
                input_ids[masked_positions]
            )
            
            discriminator_loss = F.binary_cross_entropy_with_logits(
                discriminator_logits.squeeze(-1), labels
            )
            
            return generator_loss + discriminator_loss
        
        return generator_logits
```

## Efficient Fine-tuning Methods

### 1. Adapter Modules

Add small bottleneck layers between transformer blocks:

```python
class Adapter(nn.Module):
    """Adapter module for parameter-efficient fine-tuning"""
    def __init__(self, hidden_size, adapter_size, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize near-identity
        nn.init.zeros_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
        
    def forward(self, hidden_states):
        # Down-project
        down_projected = self.down_project(hidden_states)
        down_projected = self.activation(down_projected)
        down_projected = self.dropout(down_projected)
        
        # Up-project
        up_projected = self.up_project(down_projected)
        
        # Residual connection
        return hidden_states + up_projected

class BERTLayerWithAdapter(nn.Module):
    """BERT layer with adapter modules"""
    def __init__(self, config, adapter_size=64):
        super().__init__()
        self.bert_layer = BERTLayer(config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, config.hidden_dropout_prob)
        self.adapter_after_attention = Adapter(config.hidden_size, adapter_size)
        self.adapter_after_ffn = Adapter(config.hidden_size, adapter_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # Original BERT layer
        layer_output, attention_probs = self.bert_layer(hidden_states, attention_mask)
        
        # Adapter after attention
        layer_output = self.adapter_after_attention(layer_output)
        
        # Adapter after FFN (already included in layer_output)
        layer_output = self.adapter_after_ffn(layer_output)
        
        return layer_output, attention_probs
```

### 2. LoRA (Low-Rank Adaptation)

Adapt pre-trained weights with low-rank matrices:

$$W' = W + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$

```python
class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Frozen pre-trained weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        # Original transformation
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA adaptation
        lora_result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return result + lora_result

def apply_lora_to_bert(bert_model, rank=8, alpha=16, target_modules=['query', 'value']):
    """Apply LoRA to BERT model"""
    for name, module in bert_model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create LoRA version
                lora_linear = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha
                )
                
                # Copy pre-trained weights
                lora_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.bias.data = module.bias.data.clone()
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = bert_model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, lora_linear)
    
    return bert_model

# Example usage
config = BERTConfig()
bert_model = BERTModel(config)
bert_model_with_lora = apply_lora_to_bert(bert_model, rank=8, alpha=16)

# Only train LoRA parameters
trainable_params = []
for name, param in bert_model_with_lora.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        trainable_params.append(param)
    else:
        param.requires_grad = False

print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
print(f"Total parameters: {sum(p.numel() for p in bert_model_with_lora.parameters())}")
```

### 3. Prefix Tuning

Learn continuous task-specific vectors prepended to input:

```python
class PrefixTuning(nn.Module):
    """Prefix tuning for BERT"""
    def __init__(self, config, num_prefix_tokens=10, prefix_hidden_size=512):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = config.num_hidden_layers
        
        # Prefix parameters for each layer
        self.prefix_parameters = nn.Parameter(
            torch.randn(self.num_layers, num_prefix_tokens, config.hidden_size)
        )
        
        # Optional: use MLP to generate prefix
        self.prefix_mlp = nn.Sequential(
            nn.Linear(prefix_hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def forward(self, layer_idx, hidden_states):
        """Add prefix to hidden states at specific layer"""
        batch_size = hidden_states.size(0)
        
        # Get prefix for this layer
        prefix = self.prefix_parameters[layer_idx]  # (num_prefix_tokens, hidden_size)
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_prefix_tokens, hidden_size)
        
        # Concatenate prefix with hidden states
        hidden_states_with_prefix = torch.cat([prefix, hidden_states], dim=1)
        
        return hidden_states_with_prefix

class BERTWithPrefixTuning(nn.Module):
    """BERT model with prefix tuning"""
    def __init__(self, config, num_prefix_tokens=10):
        super().__init__()
        self.bert = BERTModel(config)
        self.prefix_tuning = PrefixTuning(config, num_prefix_tokens)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        embedding_output = self.bert.embeddings(input_ids)
        
        # Add prefix at each layer
        hidden_states = embedding_output
        for layer_idx, layer in enumerate(self.bert.encoder.layer):
            # Add prefix
            hidden_states = self.prefix_tuning(layer_idx, hidden_states)
            
            # Update attention mask to account for prefix
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    (attention_mask.size(0), self.prefix_tuning.num_prefix_tokens),
                    device=attention_mask.device
                )
                extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                extended_attention_mask = None
            
            # Apply layer
            hidden_states, _ = layer(hidden_states, extended_attention_mask)
            
            # Remove prefix for next layer
            hidden_states = hidden_states[:, self.prefix_tuning.num_prefix_tokens:, :]
        
        # Pool output
        pooled_output = self.bert.pooler(hidden_states)
        
        return hidden_states, pooled_output
```

## Domain Adaptation and Continued Pre-training

### Domain-Adaptive Pre-training (DAPT)

Continue pre-training on domain-specific data:

```python
class DomainAdaptiveTrainer:
    """Trainer for domain-adaptive pre-training"""
    def __init__(self, pretrained_model, domain_corpus, learning_rate=1e-5):
        self.model = pretrained_model
        self.domain_corpus = domain_corpus
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
    def adaptive_masking(self, input_ids, domain_keywords):
        """Adaptive masking that prioritizes domain-specific terms"""
        # Standard masking
        masked_input, labels = create_masked_lm_predictions(input_ids)
        
        # Increase masking probability for domain keywords
        for keyword_id in domain_keywords:
            keyword_mask = (input_ids == keyword_id)
            # Force mask domain keywords with higher probability
            forced_mask = torch.bernoulli(torch.full(keyword_mask.shape, 0.3)).bool()
            combined_mask = keyword_mask & forced_mask
            
            masked_input[combined_mask] = 103  # [MASK] token
            labels[~combined_mask] = -100
        
        return masked_input, labels
    
    def train(self, epochs=3, domain_keywords=None):
        """Train with domain adaptation"""
        for epoch in range(epochs):
            for batch in self.domain_corpus:
                # Apply adaptive masking
                if domain_keywords:
                    masked_input, labels = self.adaptive_masking(
                        batch['input_ids'], domain_keywords
                    )
                else:
                    masked_input, labels = create_masked_lm_predictions(
                        batch['input_ids']
                    )
                
                # Forward pass
                loss, _ = self.model(
                    input_ids=masked_input,
                    attention_mask=batch['attention_mask'],
                    masked_lm_labels=labels
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

### Task-Adaptive Pre-training (TAPT)

Pre-train on unlabeled data from the target task:

```python
class TaskAdaptivePretrainer:
    """Pre-training on task-specific unlabeled data"""
    def __init__(self, model, task_corpus, warmup_steps=100):
        self.model = model
        self.task_corpus = task_corpus
        self.warmup_steps = warmup_steps
        self.step = 0
        
        # Use learning rate warmup
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps)
        )
        
    def train(self, epochs=5):
        """Task-adaptive pre-training"""
        for epoch in range(epochs):
            for batch in self.task_corpus:
                # MLM objective
                masked_input, labels = create_masked_lm_predictions(batch['input_ids'])
                
                loss, _ = self.model(
                    input_ids=masked_input,
                    attention_mask=batch['attention_mask'],
                    masked_lm_labels=labels
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                self.step += 1
```

## Model Analysis and Interpretability

### Attention Visualization

```python
class AttentionVisualizer:
    """Visualize and analyze attention patterns"""
    def __init__(self, model):
        self.model = model
        
    def get_attention_weights(self, input_ids, attention_mask=None):
        """Extract attention weights from all layers"""
        _, _, attentions = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return attentions
    
    def visualize_head_attention(self, input_ids, tokens, layer_idx=0, head_idx=0):
        """Visualize attention for specific head"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        attentions = self.get_attention_weights(input_ids)
        
        # Get specific layer and head
        attention = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
        
        # Plot heatmap
        plt.figure(figsize=(10, 10))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                   cmap='viridis', cbar=True)
        plt.title(f'Attention Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.tight_layout()
        plt.show()
    
    def compute_attention_flow(self, input_ids):
        """Compute attention flow through layers"""
        attentions = self.get_attention_weights(input_ids)
        
        # Average attention across heads and batch
        layer_attentions = []
        for layer_attn in attentions:
            # Average across batch and heads
            avg_attn = layer_attn.mean(dim=0).mean(dim=0)
            layer_attentions.append(avg_attn.detach().cpu())
        
        return layer_attentions

class ProbeClassifier(nn.Module):
    """Probing classifier to analyze learned representations"""
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_states):
        return self.classifier(hidden_states)

def probe_bert_layers(bert_model, probe_dataset, linguistic_property='pos'):
    """Probe BERT layers for linguistic knowledge
    
    Tests what linguistic information is encoded in each layer
    """
    probe_results = {}
    
    for layer_idx in range(bert_model.config.num_hidden_layers):
        # Extract representations from specific layer
        def hook_fn(module, input, output):
            return output[0]
        
        # Register hook
        handle = list(bert_model.bert.encoder.layer[layer_idx].children())[0].register_forward_hook(hook_fn)
        
        # Train probe classifier
        probe = ProbeClassifier(
            bert_model.config.hidden_size,
            num_classes=probe_dataset.num_classes
        )
        
        optimizer = torch.optim.Adam(probe.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(10):
            for batch in probe_dataset:
                # Get BERT representations
                with torch.no_grad():
                    hidden_states, _, _ = bert_model(batch['input_ids'])
                
                # Probe prediction
                logits = probe(hidden_states)
                loss = criterion(logits.view(-1, probe_dataset.num_classes), 
                               batch['labels'].view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate probe
        accuracy = evaluate_probe(probe, probe_dataset.test_loader)
        probe_results[f'layer_{layer_idx}'] = accuracy
        
        # Remove hook
        handle.remove()
    
    return probe_results
```

## Scaling Laws and Emergent Properties

### Scaling Laws (Kaplan et al., 2020)

Performance scales as:
$$L(N) = (N_c / N)^{\alpha_N}$$

where $N$ is model size, $N_c$ is a constant, and $\alpha_N \approx 0.076$.

```python
def predict_performance_from_scaling_law(model_size, data_size, compute_budget):
    """Predict model performance based on scaling laws
    
    Args:
        model_size: Number of parameters
        data_size: Training tokens
        compute_budget: FLOPs
    
    Returns:
        predicted_loss: Predicted validation loss
    """
    # Scaling law parameters (from Kaplan et al.)
    N_c = 8.8e13  # Critical model size
    D_c = 5.4e13  # Critical data size
    C_c = 3.1e8   # Critical compute
    
    alpha_N = 0.076  # Model size exponent
    alpha_D = 0.095  # Data size exponent
    alpha_C = 0.050  # Compute exponent
    
    # Predict loss from each factor
    loss_N = (N_c / model_size) ** alpha_N
    loss_D = (D_c / data_size) ** alpha_D
    loss_C = (C_c / compute_budget) ** alpha_C
    
    # Combined prediction
    predicted_loss = min(loss_N, loss_D, loss_C)
    
    return predicted_loss

def compute_optimal_allocation(compute_budget):
    """Compute optimal model size and data size for given compute
    
    Based on Chinchilla scaling laws (Hoffman et al., 2022)
    """
    # Chinchilla optimal: Model and data should scale equally
    # N_opt ∝ C^α, D_opt ∝ C^α where α ≈ 0.5
    
    # Approximate FLOPs: 6 * N * D (forward and backward pass)
    # C = 6 * N * D
    # For optimal: N ∝ D
    # Therefore: C ∝ N^2, so N ∝ sqrt(C)
    
    optimal_model_size = (compute_budget / 6) ** 0.5
    optimal_data_size = optimal_model_size
    
    return {
        'model_size': optimal_model_size,
        'training_tokens': optimal_data_size,
        'compute_budget': compute_budget
    }

# Example
compute = 1e18  # 1 exaFLOP
allocation = compute_optimal_allocation(compute)
print(f"Optimal model size: {allocation['model_size']:.2e} parameters")
print(f"Optimal training data: {allocation['training_tokens']:.2e} tokens")
```

## Advanced Topics and Research Directions

### 1. Sparse Models

```python
class SparseAttention(nn.Module):
    """Sparse attention for efficient long-range dependencies"""
    def __init__(self, hidden_size, num_heads, block_size=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Block-sparse attention pattern
        # Only attend to tokens in same block and previous blocks
        num_blocks = N // self.block_size
        attn_output = torch.zeros_like(v)
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = (block_idx + 1) * self.block_size
            
            # Local attention within block
            q_block = q[:, :, start_idx:end_idx, :]
            k_block = k[:, :, start_idx:end_idx, :]
            v_block = v[:, :, start_idx:end_idx, :]
            
            attn_scores = (q_block @ k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output[:, :, start_idx:end_idx, :] = attn_probs @ v_block
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        
        return output
```

### 2. Mixture of Experts

```python
class MixtureOfExperts(nn.Module):
    """Sparse mixture of experts layer"""
    def __init__(self, hidden_size, num_experts=8, expert_capacity=64, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        """Route tokens to top-k experts"""
        B, N, C = x.shape
        
        # Compute routing probabilities
        router_logits = self.router(x)  # (B, N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_probs[:, :, i].unsqueeze(-1)
            
            # Route to experts
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        return output
```

## Practical Exercises

### Exercise 1: Implement Custom Pre-training Objective
Design and implement a novel pre-training objective:
- Combine multiple objectives (MLM + your custom objective)
- Measure impact on downstream task performance
- Analyze what linguistic knowledge is captured

### Exercise 2: Efficient Fine-tuning Comparison
Compare different efficient fine-tuning methods:
- Implement Adapter, LoRA, and Prefix Tuning
- Evaluate on multiple tasks
- Analyze trade-offs between efficiency and performance

### Exercise 3: Domain Adaptation Pipeline
Build a complete domain adaptation pipeline:
- Start with pre-trained BERT
- Apply domain-adaptive pre-training
- Fine-tune on target task
- Measure improvement over standard fine-tuning

### Exercise 4: Scaling Analysis
Conduct scaling analysis on your models:
- Train models of different sizes (small, base, large)
- Plot performance vs. model size and data size
- Validate scaling law predictions

## Research Paper Implementations

### Key Papers to Implement:
1. **BERT** (Devlin et al., 2019)
2. **RoBERTa** (Liu et al., 2019)
3. **ALBERT** (Lan et al., 2020)
4. **ELECTRA** (Clark et al., 2020)
5. **DeBERTa** (He et al., 2021)
6. **Scaling Laws** (Kaplan et al., 2020)
7. **Chinchilla** (Hoffmann et al., 2022)

## Assessment Questions (PhD Level)

1. **Theoretical Foundations**:
   - Derive the gradient of the masked language modeling objective
   - Prove that transformer self-attention is permutation-equivariant
   - Analyze the sample complexity of pre-training vs. training from scratch

2. **Architecture Design**:
   - Why does BERT use bidirectional attention while GPT uses causal attention?
   - Analyze the trade-offs between model depth and width
   - Design a custom architecture for a specific domain

3. **Optimization and Scaling**:
   - Derive the optimal learning rate schedule for transformer training
   - Explain the relationship between batch size, learning rate, and training dynamics
   - Analyze the computational complexity of different attention mechanisms

4. **Transfer Learning Theory**:
   - Formalize the conditions under which transfer learning is beneficial
   - Analyze the relationship between pre-training data and downstream task performance
   - Design experiments to measure catastrophic forgetting

## Key Takeaways

- Pre-trained models leverage transfer learning to improve performance on downstream tasks
- BERT's bidirectional context modeling captures richer representations than unidirectional models
- Efficient fine-tuning methods (Adapter, LoRA, Prefix Tuning) enable adaptation with minimal parameters
- Scaling laws provide theoretical guidance for optimal resource allocation
- Domain and task adaptive pre-training bridge the gap between general pre-training and specific applications
- Modern research focuses on efficiency, interpretability, and emergent capabilities

## Next Steps

In the next module, we'll explore advanced transformer architectures including GPT variants, T5, and modern innovations like sparse transformers, mixture of experts, and efficient attention mechanisms.

## References

1. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Liu et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
3. Lan et al. (2020). "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
4. Clark et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"
5. Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
6. Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models"
7. Houlsby et al. (2019). "Parameter-Efficient Transfer Learning for NLP"
8. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
9. Li & Liang (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
10. He et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"

