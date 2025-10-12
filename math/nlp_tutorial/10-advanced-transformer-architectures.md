# NLP Tutorial Module 10: Advanced Transformer Architectures (PhD Level)

## Learning Objectives
By the end of this module, you will be able to:
- Understand and implement GPT-2, GPT-3, and GPT-4 architectures
- Master T5 and encoder-decoder architectures at scale
- Implement efficient attention mechanisms (Linear, Performer, Flash Attention)
- Design and build mixture-of-experts (MoE) architectures
- Understand and implement sparse and long-context transformers
- Apply architectural innovations (RoPE, ALiBi, grouped-query attention)
- Build retrieval-augmented and memory-augmented architectures
- Analyze computational complexity and optimize transformer efficiency

## GPT Architecture Evolution

### GPT-2: Language Models are Unsupervised Multitask Learners

GPT-2 scales GPT with:
- 1.5B parameters (GPT-2 XL)
- Larger context window (1024 tokens)
- Byte-Pair Encoding (BPE) tokenization
- Zero-shot task transfer

#### Mathematical Foundation

Autoregressive objective:
$$\mathcal{L} = -\sum_{i=1}^n \log p_\theta(x_i | x_{<i})$$

Factorized as:
$$p(x) = \prod_{i=1}^n p(x_i | x_1, ..., x_{i-1})$$

#### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class GPT2Attention(nn.Module):
    """GPT-2 Multi-head Causal Self-Attention"""
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.scale_attn_weights = config.scale_attn_weights
        
        # Combined QKV projection
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        
        # Output projection
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.n_positions, config.n_positions), dtype=torch.uint8))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split hidden dimension into num_heads"""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge heads back"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def _attn(self, query, key, value, attention_mask=None):
        """Compute scaled dot-product attention"""
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        """Forward pass with optional KV caching"""
        # Compute Q, K, V
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        # Use cached KV if available
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        if use_cache:
            present = (key, value)
        else:
            present = None
        
        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present

class GPT2MLP(nn.Module):
    """GPT-2 Feed-Forward Network"""
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    """GPT-2 Transformer Block"""
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)
        
    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        # Residual connection
        hidden_states = attn_output + residual
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # Residual connection
        hidden_states = residual + feed_forward_hidden_states
        
        return hidden_states, present

class GPT2Model(nn.Module):
    """Complete GPT-2 Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_dim = config.n_embd
        
        # Token + Position embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        
    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        batch_size, seq_length = input_ids.shape
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        
        # Position IDs
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Transformer blocks
        presents = [] if use_cache else None
        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            
            if use_cache:
                presents.append(present)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states, presents

class GPT2LMHeadModel(nn.Module):
    """GPT-2 with Language Modeling Head"""
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.transformer.wte.weight
        
    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, past_key_values=None):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        hidden_states = transformer_outputs[0]
        
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, lm_logits, transformer_outputs[1]
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
        """Generate text autoregressively"""
        self.eval()
        
        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            _, logits, _ = self(input_ids, use_cache=False)
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

# Configuration
class GPT2Config:
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.n_inner = None
        self.activation_function = "gelu_new"
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.02
        self.scale_attn_weights = True

# Example usage
config = GPT2Config()
gpt2 = GPT2LMHeadModel(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 50))
loss, logits, _ = gpt2(input_ids, labels=input_ids)

print(f"Loss: {loss}")
print(f"Logits shape: {logits.shape}")

# Generate text
generated = gpt2.generate(input_ids[:, :10], max_length=50, temperature=0.7, top_k=50)
print(f"Generated shape: {generated.shape}")
```

### GPT-3: Scaling to 175B Parameters

Key innovations:
- Massive scale (175B parameters)
- In-context learning without gradient updates
- Few-shot, one-shot, and zero-shot capabilities
- Emergent abilities at scale

#### Few-Shot Learning Implementation

```python
class GPT3FewShotLearner:
    """GPT-3 style few-shot learning"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def create_few_shot_prompt(self, examples, query, task_description=""):
        """Create prompt with examples for few-shot learning"""
        prompt = task_description + "\n\n"
        
        # Add examples
        for example in examples:
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        # Add query
        prompt += f"Input: {query}\n"
        prompt += "Output:"
        
        return prompt
    
    def few_shot_predict(self, examples, query, task_description="", max_length=50):
        """Make prediction using few-shot learning"""
        prompt = self.create_few_shot_prompt(examples, query, task_description)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        output_ids = self.model.generate(input_ids, max_length=max_length, temperature=0.1)
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract prediction (text after "Output:")
        prediction = output_text.split("Output:")[-1].strip()
        
        return prediction

# Example usage
few_shot_learner = GPT3FewShotLearner(gpt2, tokenizer)

examples = [
    {"input": "The movie was amazing", "output": "positive"},
    {"input": "I hated every minute of it", "output": "negative"},
    {"input": "It was okay, nothing special", "output": "neutral"}
]

query = "Best film I've seen this year!"
prediction = few_shot_learner.few_shot_predict(
    examples, query, task_description="Classify the sentiment of movie reviews"
)
print(f"Prediction: {prediction}")
```

## T5: Text-to-Text Transfer Transformer

### Unified Text-to-Text Framework

T5 frames all NLP tasks as text-to-text:
- Translation: "translate English to German: That is good" → "Das ist gut"
- Classification: "sst2 sentence: The movie is great" → "positive"
- Summarization: "summarize: [long text]" → "[summary]"

#### Mathematical Foundation

Encoder-decoder with relative position bias:
$$\text{bias}_{i,j} = \text{bias}[|i-j|]$$

where $|i-j|$ is clipped to maximum distance.

```python
class T5Attention(nn.Module):
    """T5 Attention with Relative Position Bias"""
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        
        # Q, K, V projections
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Compute relative position buckets"""
        ret = 0
        n = -relative_position
        
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        
        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        # The other half is for logarithmically bigger bins
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def compute_bias(self, query_length, key_length):
        """Compute relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values
    
    def forward(self, hidden_states, attention_mask=None, key_value_states=None,
                position_bias=None, past_key_value=None, use_cache=False):
        """
        Self-attention (if key_value_states is None) or cross-attention (if key_value_states is not None)
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q
        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        # Project K, V
        if key_value_states is None:
            # Self-attention
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
        else:
            # Cross-attention
            key_states = self.k(key_value_states)
            value_states = self.v(key_value_states)
        
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        # Use cached KV if available
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None
        
        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        
        # Add relative position bias
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(seq_length, key_states.shape[2])
            else:
                position_bias = torch.zeros((1, self.n_heads, seq_length, key_states.shape[2]),
                                           device=scores.device, dtype=scores.dtype)
        
        scores += position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Normalize
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        
        # Output projection
        attn_output = self.o(attn_output)
        
        return attn_output, present_key_value_state, position_bias

class T5LayerFF(nn.Module):
    """T5 Feed-Forward Layer"""
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states

class T5Block(nn.Module):
    """T5 Transformer Block"""
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        
        # Self-attention
        self.layer = nn.ModuleList()
        self.layer.append(
            nn.ModuleDict({
                "SelfAttention": T5Attention(config, has_relative_attention_bias=has_relative_attention_bias),
                "layer_norm": nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            })
        )
        
        # Cross-attention (decoder only)
        if self.is_decoder:
            self.layer.append(
                nn.ModuleDict({
                    "EncDecAttention": T5Attention(config),
                    "layer_norm": nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
                })
            )
        
        # Feed-forward
        self.layer.append(T5LayerFF(config))
        
    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, use_cache=False):
        
        self_attention_outputs = self.layer[0]["SelfAttention"](
            self.layer[0]["layer_norm"](hidden_states),
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value[0] if past_key_value is not None else None,
            use_cache=use_cache
        )
        
        hidden_states = hidden_states + self_attention_outputs[0]
        present_key_value_state = self_attention_outputs[1]
        
        # Cross-attention for decoder
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1]["EncDecAttention"](
                self.layer[1]["layer_norm"](hidden_states),
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=past_key_value[1] if past_key_value is not None else None,
                use_cache=use_cache
            )
            hidden_states = hidden_states + cross_attention_outputs[0]
            
            if use_cache:
                present_key_value_state = present_key_value_state + (cross_attention_outputs[1],)
        
        # Feed-forward
        hidden_states = self.layer[-1](hidden_states)
        
        return hidden_states, present_key_value_state
```

## Efficient Attention Mechanisms

### 1. Linear Attention (Performers)

Approximates softmax attention using kernel methods:

$$\text{Attention}(Q, K, V) = \phi(Q)(\phi(K)^T V)$$

where $\phi$ is a feature map.

```python
class LinearAttention(nn.Module):
    """Linear attention with kernel approximation"""
    def __init__(self, dim, heads=8, dim_head=64, eps=1e-6):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.eps = eps
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        # Apply feature map (using ELU + 1 as kernel)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: φ(Q)(φ(K)^T V)
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        
        # Normalize
        denominator = torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2))
        out = out / (denominator.unsqueeze(-1) + self.eps)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
```

### 2. Flash Attention

Memory-efficient attention using tiling:

```python
class FlashAttention(nn.Module):
    """Flash Attention: Fast and Memory-Efficient Exact Attention
    
    Uses tiling to reduce memory from O(N²) to O(N)
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        Implements Flash Attention algorithm:
        1. Tile Q, K, V
        2. Compute attention blockwise
        3. Combine results with online softmax normalization
        """
        b, n, _, h = *x.shape, self.heads
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        # Block size for tiling
        block_size = 64
        num_blocks = (n + block_size - 1) // block_size
        
        # Initialize output and normalization terms
        out = torch.zeros_like(v)
        l = torch.zeros(b, h, n, 1, device=x.device)
        m = torch.full((b, h, n, 1), float('-inf'), device=x.device)
        
        # Process blocks of K, V
        for i in range(num_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, n)
            
            # Get Q block
            q_block = q[:, :, start_i:end_i, :]
            
            for j in range(num_blocks):
                start_j = j * block_size
                end_j = min((j + 1) * block_size, n)
                
                # Get K, V blocks
                k_block = k[:, :, start_j:end_j, :]
                v_block = v[:, :, start_j:end_j, :]
                
                # Compute attention scores for block
                scores = torch.einsum('bhid,bhjd->bhij', q_block, k_block) * self.scale
                
                if mask is not None:
                    mask_block = mask[:, start_i:end_i, start_j:end_j]
                    scores = scores.masked_fill(~mask_block.unsqueeze(1), float('-inf'))
                
                # Online softmax update
                m_new = torch.maximum(m[:, :, start_i:end_i, :], scores.max(dim=-1, keepdim=True)[0])
                
                # Update normalization term
                l_new = l[:, :, start_i:end_i, :] * torch.exp(m[:, :, start_i:end_i, :] - m_new) + \
                        torch.exp(scores - m_new).sum(dim=-1, keepdim=True)
                
                # Update output
                attn_weights = torch.exp(scores - m_new)
                out[:, :, start_i:end_i, :] = out[:, :, start_i:end_i, :] * \
                    (l[:, :, start_i:end_i, :] / l_new) + \
                    torch.einsum('bhij,bhjd->bhid', attn_weights, v_block) / l_new
                
                # Update normalization state
                l[:, :, start_i:end_i, :] = l_new
                m[:, :, start_i:end_i, :] = m_new
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
```

### 3. Rotary Position Embeddings (RoPE)

Encodes position through rotation:

$$f_q(x_m, m) = (W_q x_m) e^{im\theta}$$
$$f_k(x_n, n) = (W_k x_n) e^{in\theta}$$

```python
class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        
    def forward(self, x, seq_len=None):
        """Apply rotary embeddings"""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Extend cache for longer sequences"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype))
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype))

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to Q and K"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class AttentionWithRoPE(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.rotary_emb = RotaryEmbedding(dim_head)
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len=n)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
```

## Mixture of Experts (MoE)

### Sparse MoE Implementation

```python
class TopKRouter(nn.Module):
    """Top-K router for MoE"""
    def __init__(self, dim, num_experts, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Routing network
        self.router = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        """Route tokens to top-k experts
        
        Returns:
            expert_weights: (batch, seq_len, top_k)
            expert_indices: (batch, seq_len, top_k)
            load_balancing_loss: auxiliary loss for load balancing
        """
        # Compute routing logits
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        
        # Get top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Load balancing loss (encourage uniform distribution across experts)
        # f_i = fraction of tokens assigned to expert i
        # P_i = average routing probability to expert i
        # Loss = num_experts * sum(f_i * P_i)
        gates_per_expert = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            gates_per_expert[i] = (expert_indices == i).float().sum()
        
        fraction_per_expert = gates_per_expert / gates_per_expert.sum()
        prob_per_expert = router_probs.mean(dim=[0, 1])
        
        load_balancing_loss = self.num_experts * (fraction_per_expert * prob_per_expert).sum()
        
        return expert_weights, expert_indices, load_balancing_loss

class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    def __init__(self, dim, num_experts=8, expert_capacity=64, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = TopKRouter(dim, num_experts, top_k)
        
        # Experts (each is a feed-forward network)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        """Forward pass with expert routing"""
        batch_size, seq_len, dim = x.shape
        
        # Get routing decisions
        expert_weights, expert_indices, load_balancing_loss = self.router(x)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each token with its top-k experts
        x_flat = x.view(-1, dim)
        output_flat = output.view(-1, dim)
        
        for i in range(self.top_k):
            # Get expert index and weight for this position
            expert_idx = expert_indices[:, :, i].reshape(-1)
            expert_weight = expert_weights[:, :, i].reshape(-1, 1)
            
            # Route tokens to experts
            for expert_id in range(self.num_experts):
                expert_mask = (expert_idx == expert_id)
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output_flat[expert_mask] += expert_weight[expert_mask] * expert_output
        
        output = output_flat.view(batch_size, seq_len, dim)
        
        return output, load_balancing_loss

class SwitchTransformerBlock(nn.Module):
    """Transformer block with Switch (1-expert routing) MoE"""
    def __init__(self, config):
        super().__init__()
        self.attention = GPT2Attention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config.n_embd, num_experts=config.num_experts, top_k=1)
        
    def forward(self, x, attention_mask=None):
        # Attention with residual
        attn_output, _ = self.attention(self.ln_1(x), attention_mask)
        x = x + attn_output
        
        # MoE with residual
        moe_output, load_balancing_loss = self.moe(self.ln_2(x))
        x = x + moe_output
        
        return x, load_balancing_loss
```

## Long-Context Transformers

### 1. Longformer: Sparse Attention Patterns

```python
class LongformerAttention(nn.Module):
    """Longformer attention with sliding window + global attention"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.window_size = config.attention_window
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
    def _sliding_chunks_attention(self, q, k, v):
        """Compute attention using sliding window chunks"""
        batch_size, seq_len, num_heads, head_dim = q.size()
        
        # Reshape for chunked computation
        chunk_size = self.window_size
        num_chunks = seq_len // chunk_size
        
        # Pad sequence if needed
        padding = chunk_size - (seq_len % chunk_size)
        if padding > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, padding))
            k = F.pad(k, (0, 0, 0, 0, 0, padding))
            v = F.pad(v, (0, 0, 0, 0, 0, padding))
        
        # Reshape into chunks
        q = q.view(batch_size, num_chunks, chunk_size, num_heads, head_dim)
        k = k.view(batch_size, num_chunks, chunk_size, num_heads, head_dim)
        v = v.view(batch_size, num_chunks, chunk_size, num_heads, head_dim)
        
        # Compute local attention within windows
        attn_scores = torch.einsum('bcthd,bcshd->bctsh', q, k) / math.sqrt(head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.einsum('bctsh,bcshd->bcthd', attn_probs, v)
        
        # Reshape back
        context = context.view(batch_size, -1, num_heads, head_dim)
        if padding > 0:
            context = context[:, :-padding, :, :]
        
        return context
    
    def forward(self, hidden_states, attention_mask=None, global_attention_mask=None):
        """Forward pass with sliding window and global attention"""
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Local sliding window attention
        local_context = self._sliding_chunks_attention(q, k, v)
        
        # Global attention for special tokens (e.g., [CLS])
        if global_attention_mask is not None:
            # Compute full attention for global tokens
            global_q = q[global_attention_mask]
            global_attn_scores = torch.matmul(global_q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            global_attn_probs = F.softmax(global_attn_scores, dim=-1)
            global_context = torch.matmul(global_attn_probs, v)
            
            # Merge local and global context
            local_context[global_attention_mask] = global_context
        
        # Reshape and project
        context = local_context.view(batch_size, seq_len, -1)
        output = self.output(context)
        
        return output
```

### 2. ALiBi (Attention with Linear Biases)

Instead of positional embeddings, add linear biases to attention:

$$\text{softmax}(q_i K^T + m \cdot [-(i-1), ..., -2, -1, 0])$$

```python
class ALiBiAttention(nn.Module):
    """Attention with Linear Biases (ALiBi)"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # ALiBi slopes
        self.slopes = self._get_alibi_slopes()
        
    def _get_alibi_slopes(self):
        """Get ALiBi slopes for each head"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(self.num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(self.num_heads))
        else:
            # Closest power of 2
            closest_power_of_2 = 2**math.floor(math.log2(self.num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes += get_slopes_power_of_2(2*closest_power_of_2)[:self.num_heads-closest_power_of_2]
            return torch.tensor(slopes)
    
    def _get_alibi_bias(self, seq_len, device):
        """Compute ALiBi bias matrix"""
        # Create position matrix
        context_position = torch.arange(seq_len, device=device).unsqueeze(1)
        memory_position = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Compute relative positions
        relative_position = memory_position - context_position
        
        # Apply slopes
        alibi = relative_position.unsqueeze(0) * self.slopes.view(-1, 1, 1).to(device)
        
        return alibi
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(seq_len, hidden_states.device)
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Compute attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output(context)
        
        return output
```

## Retrieval-Augmented Generation (RAG)

```python
class RetrievalAugmentedGenerator(nn.Module):
    """RAG: Retrieval-Augmented Generation"""
    def __init__(self, question_encoder, generator, retriever, num_docs=5):
        super().__init__()
        self.question_encoder = question_encoder
        self.generator = generator
        self.retriever = retriever
        self.num_docs = num_docs
        
    def retrieve(self, question):
        """Retrieve relevant documents"""
        # Encode question
        question_embedding = self.question_encoder(question)
        
        # Retrieve documents
        docs, doc_scores = self.retriever.retrieve(question_embedding, k=self.num_docs)
        
        return docs, doc_scores
    
    def forward(self, question, return_docs=False):
        """Generate answer with retrieval"""
        # Retrieve documents
        docs, doc_scores = self.retrieve(question)
        
        # Create prompts with retrieved context
        prompts = []
        for doc in docs:
            prompt = f"Context: {doc}\n\nQuestion: {question}\n\nAnswer:"
            prompts.append(prompt)
        
        # Generate answers for each context
        outputs = []
        for prompt in prompts:
            output = self.generator.generate(prompt)
            outputs.append(output)
        
        # Marginalize over documents (weighted by retrieval scores)
        doc_scores = F.softmax(torch.tensor(doc_scores), dim=0)
        
        if return_docs:
            return outputs, docs, doc_scores
        
        # Return most likely answer
        best_idx = torch.argmax(doc_scores)
        return outputs[best_idx]

class DenseRetriever(nn.Module):
    """Dense passage retrieval"""
    def __init__(self, encoder, document_index):
        super().__init__()
        self.encoder = encoder
        self.document_index = document_index
        
    def encode(self, text):
        """Encode text to dense vector"""
        return self.encoder(text)
    
    def retrieve(self, query_embedding, k=5):
        """Retrieve top-k documents"""
        # Compute similarity with all documents
        doc_scores = torch.matmul(query_embedding, self.document_index.T)
        
        # Get top-k
        top_scores, top_indices = torch.topk(doc_scores, k)
        
        # Retrieve documents
        retrieved_docs = [self.get_document(idx) for idx in top_indices]
        
        return retrieved_docs, top_scores.tolist()
    
    def get_document(self, idx):
        """Get document by index"""
        # Implementation depends on document storage
        pass
```

## Practical Exercises

### Exercise 1: Implement Efficient Attention
Implement and compare different efficient attention mechanisms:
- Linear attention
- Flash attention
- Sparse attention patterns
Measure memory usage and speed

### Exercise 2: Build MoE Model
Create a complete MoE model:
- Implement routing with load balancing
- Train on large-scale data
- Analyze expert specialization

### Exercise 3: Long-Context Architecture
Design architecture for 32K+ context:
- Implement Longformer or ALiBi
- Test on long-document tasks
- Compare with standard transformer

### Exercise 4: RAG System
Build end-to-end RAG system:
- Implement dense retrieval
- Integrate with generator
- Evaluate on open-domain QA

## Research Directions

### Current Frontiers:
1. **Efficient Attention**: Sub-quadratic attention mechanisms
2. **Sparse Models**: Conditional computation and MoE
3. **Long Context**: Architectures for 100K+ tokens
4. **Multimodal**: Integrating vision, audio, and text
5. **Architecture Search**: Automated design of optimal architectures

## Key Papers

1. Radford et al. (2019) - GPT-2
2. Brown et al. (2020) - GPT-3
3. Raffel et al. (2020) - T5
4. Choromanski et al. (2021) - Performers (Linear Attention)
5. Dao et al. (2022) - Flash Attention
6. Su et al. (2021) - RoFormer (RoPE)
7. Press et al. (2022) - ALiBi
8. Fedus et al. (2021) - Switch Transformers
9. Beltagy et al. (2020) - Longformer
10. Lewis et al. (2020) - RAG

## Key Takeaways

- GPT scales autoregressive modeling to achieve emergent capabilities
- T5 unifies all NLP tasks in text-to-text framework
- Efficient attention mechanisms reduce O(n²) complexity
- MoE enables scaling to trillions of parameters with sparse activation
- Long-context architectures are crucial for document understanding
- Retrieval augmentation extends model knowledge beyond parameters

## Next Steps

In the next module, we'll explore prompt engineering and in-context learning, which enable powerful zero-shot and few-shot capabilities without fine-tuning.

