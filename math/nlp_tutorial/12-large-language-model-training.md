# NLP Tutorial Module 12: Large Language Model Training

## Learning Objectives
By the end of this module, you will be able to:
- Understand the principles of training large language models
- Implement distributed training strategies
- Apply efficient optimization techniques for LLMs
- Handle memory optimization and model parallelism
- Implement gradient accumulation and mixed precision training
- Understand scaling laws and model size considerations
- Apply advanced techniques like gradient checkpointing and parameter-efficient training

## Introduction to Large Language Model Training

Training large language models (LLMs) presents unique challenges due to their massive scale, requiring specialized techniques for efficient training, memory management, and optimization.

### Key Challenges in LLM Training

1. **Computational Requirements**: Massive compute resources needed
2. **Memory Constraints**: Models too large for single GPU
3. **Training Stability**: Ensuring stable training across scales
4. **Data Requirements**: Need for large, high-quality datasets
5. **Infrastructure**: Distributed training across multiple nodes

### Training Scale Considerations

- **Model Size**: 7B to 70B+ parameters
- **Data Size**: Terabytes to petabytes of text
- **Compute**: Thousands of GPUs for weeks/months
- **Memory**: 100GB+ per GPU for large models

## Distributed Training Strategies

### Data Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from torch.utils.data import DataLoader, Dataset

class DistributedTrainer:
    def __init__(self, model, rank, world_size, backend='nccl'):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.backend = backend
        
        # Initialize distributed process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        
        # Move model to device
        self.device = torch.device(f'cuda:{rank}')
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[rank])
        
    def train_step(self, batch, optimizer, criterion):
        """Single training step with distributed training"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def cleanup(self):
        """Cleanup distributed process group"""
        dist.destroy_process_group()

def setup_distributed_training():
    """Setup distributed training environment"""
    world_size = torch.cuda.device_count()
    
    def train_worker(rank, world_size):
        # Create model
        model = YourLargeLanguageModel()  # Replace with actual model
        
        # Create trainer
        trainer = DistributedTrainer(model, rank, world_size)
        
        # Create distributed sampler
        dataset = YourDataset()  # Replace with actual dataset
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(10):
            sampler.set_epoch(epoch)
            for batch in dataloader:
                loss = trainer.train_step(batch, optimizer, criterion)
                if rank == 0:  # Only print from rank 0
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')
        
        trainer.cleanup()
    
    # Launch distributed training
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

# Example usage
# setup_distributed_training()
```

### Model Parallelism

```python
class ModelParallelTransformer(nn.Module):
    def __init__(self, config):
        super(ModelParallelTransformer, self).__init__()
        self.config = config
        
        # Split layers across devices
        self.embedding = nn.Embedding(config.vocab_size, config.d_model).cuda(0)
        
        # Split transformer blocks across devices
        self.layers = nn.ModuleList()
        layers_per_device = config.num_layers // config.num_devices
        
        for i in range(config.num_layers):
            device_id = i // layers_per_device
            layer = TransformerBlock(config).cuda(device_id)
            self.layers.append(layer)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size).cuda(config.num_devices - 1)
    
    def forward(self, input_ids):
        # Embedding on device 0
        x = self.embedding(input_ids.cuda(0))
        
        # Pass through layers on different devices
        for layer in self.layers:
            x = layer(x.cuda(layer.cuda_device))
        
        # Output projection on last device
        logits = self.lm_head(x.cuda(self.config.num_devices - 1))
        
        return logits

class PipelineParallelTransformer(nn.Module):
    """Pipeline parallelism implementation"""
    def __init__(self, config):
        super(PipelineParallelTransformer, self).__init__()
        self.config = config
        
        # Split model into stages
        self.stages = nn.ModuleList()
        layers_per_stage = config.num_layers // config.num_stages
        
        for stage in range(config.num_stages):
            stage_layers = nn.ModuleList()
            start_layer = stage * layers_per_stage
            end_layer = min((stage + 1) * layers_per_stage, config.num_layers)
            
            for i in range(start_layer, end_layer):
                stage_layers.append(TransformerBlock(config))
            
            self.stages.append(stage_layers)
    
    def forward(self, x):
        # Forward pass through pipeline stages
        for stage in self.stages:
            for layer in stage:
                x = layer(x)
        return x
```

## Memory Optimization Techniques

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, config):
        super(CheckpointedTransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(config.d_model, config.num_heads)
        self.feed_forward = FeedForward(config.d_model, config.d_ff)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # Use gradient checkpointing to save memory
        def attention_block(x):
            attn_output, _ = self.attention(x, x, x)
            return self.norm1(x + self.dropout(attn_output))
        
        def feed_forward_block(x):
            ff_output = self.feed_forward(x)
            return self.norm2(x + self.dropout(ff_output))
        
        # Checkpoint the attention block
        x = checkpoint(attention_block, x, use_reentrant=False)
        
        # Checkpoint the feed-forward block
        x = checkpoint(feed_forward_block, x, use_reentrant=False)
        
        return x

def apply_gradient_checkpointing(model, checkpoint_ratio=0.5):
    """Apply gradient checkpointing to a fraction of layers"""
    total_layers = len(model.layers)
    checkpoint_layers = int(total_layers * checkpoint_ratio)
    
    for i in range(checkpoint_layers):
        layer = model.layers[i]
        # Replace with checkpointed version
        model.layers[i] = CheckpointedTransformerBlock(layer.config)
    
    return model
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, batch, criterion):
        """Training step with mixed precision"""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
        
        # Backward pass with scaler
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# Example usage
def train_with_mixed_precision():
    model = YourLargeLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = MixedPrecisionTrainer(model, optimizer)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for batch in dataloader:
        loss = trainer.train_step(batch, criterion)
        print(f'Loss: {loss:.4f}')
```

### Gradient Accumulation

```python
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()
    
    def train_step(self, batch, criterion, step):
        """Training step with gradient accumulation"""
        # Forward pass
        with autocast():
            outputs = self.model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            
            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Update parameters every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps  # Return unscaled loss

# Example usage
def train_with_gradient_accumulation():
    model = YourLargeLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = GradientAccumulationTrainer(model, optimizer, accumulation_steps=8)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for step, batch in enumerate(dataloader):
        loss = trainer.train_step(batch, criterion, step)
        if step % 100 == 0:
            print(f'Step {step}, Loss: {loss:.4f}')
```

## Advanced Optimization Techniques

### Learning Rate Scheduling

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Example usage
def train_with_scheduler():
    model = YourLargeLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create scheduler
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1000, total_steps=100000)
    
    # Training loop
    for step, batch in enumerate(dataloader):
        # Training step
        loss = train_step(model, batch, optimizer)
        
        # Update learning rate
        lr = scheduler.step()
        
        if step % 100 == 0:
            print(f'Step {step}, Loss: {loss:.4f}, LR: {lr:.6f}')
```

### Parameter-Efficient Training

```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) * 0.01)
        
    def forward(self, x):
        return x @ (self.lora_A @ self.lora_B) * (self.alpha / self.rank)

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear_layer, rank=16, alpha=16):
        super(LoRALinear, self).__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha)
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora_to_model(model, rank=16, alpha=16, target_modules=['q_proj', 'v_proj']):
    """Apply LoRA to specified modules in the model"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            # Replace with LoRA version
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = model
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, LoRALinear(module, rank, alpha))
    
    return model

# Example usage
def train_with_lora():
    model = YourLargeLanguageModel()
    
    # Apply LoRA to attention layers
    model = apply_lora_to_model(model, rank=16, alpha=16)
    
    # Only train LoRA parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            trainable_params.append(param)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    
    # Training loop
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        print(f'Loss: {loss:.4f}')
```

## Data Loading and Preprocessing for LLMs

```python
class LLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        
        # Create input and target (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

class DynamicBatching:
    """Dynamic batching based on sequence length"""
    def __init__(self, batch_size, max_tokens_per_batch=8192):
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
    
    def create_batches(self, dataset):
        """Create batches with similar sequence lengths"""
        # Sort by sequence length
        sorted_indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i]['input_ids']))
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in sorted_indices:
            sample = dataset[idx]
            sample_tokens = len(sample['input_ids'])
            
            # Check if adding this sample would exceed limits
            if (len(current_batch) >= self.batch_size or 
                current_tokens + sample_tokens > self.max_tokens_per_batch):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(sample)
            current_tokens += sample_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

# Example usage
def create_llm_dataloader():
    # Sample texts
    texts = [
        "This is a sample text for training.",
        "Another example of training data.",
        "More text data for the language model.",
        # ... more texts
    ]
    
    # Create dataset
    dataset = LLMDataset(texts, tokenizer, max_length=512)
    
    # Create dynamic batcher
    batcher = DynamicBatching(batch_size=8, max_tokens_per_batch=4096)
    batches = batcher.create_batches(dataset)
    
    return batches
```

## Monitoring and Logging

```python
import wandb
import json
from collections import defaultdict

class LLMTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metrics = defaultdict(list)
        
        # Initialize wandb
        wandb.init(
            project="llm-training",
            config=config,
            name=config.run_name
        )
    
    def log_metrics(self, metrics_dict, step):
        """Log metrics to wandb and local storage"""
        # Log to wandb
        wandb.log(metrics_dict, step=step)
        
        # Store locally
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def log_model_parameters(self):
        """Log model parameter statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        param_stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
        self.log_metrics(param_stats, 0)
    
    def log_gradient_norms(self, step):
        """Log gradient norms for monitoring training stability"""
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[f'grad_norm/{name}'] = grad_norm
        
        if grad_norms:
            self.log_metrics(grad_norms, step)
    
    def log_learning_rates(self, optimizer, step):
        """Log learning rates for each parameter group"""
        lr_dict = {}
        for i, param_group in enumerate(optimizer.param_groups):
            lr_dict[f'learning_rate/group_{i}'] = param_group['lr']
        
        self.log_metrics(lr_dict, step)
    
    def save_checkpoint(self, epoch, step, optimizer, scheduler, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save locally
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        
        # Save to wandb
        wandb.save(f'checkpoint_epoch_{epoch}_step_{step}.pt')
    
    def cleanup(self):
        """Cleanup resources"""
        wandb.finish()

# Example usage
def train_with_monitoring():
    model = YourLargeLanguageModel()
    config = YourConfig()
    
    trainer = LLMTrainer(model, config)
    trainer.log_model_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1000, total_steps=100000)
    
    # Training loop
    for epoch in range(10):
        for step, batch in enumerate(dataloader):
            loss = train_step(model, batch, optimizer)
            
            # Log metrics
            trainer.log_metrics({'loss': loss}, step)
            trainer.log_gradient_norms(step)
            trainer.log_learning_rates(optimizer, step)
            
            # Save checkpoint
            if step % 1000 == 0:
                trainer.save_checkpoint(epoch, step, optimizer, scheduler, loss)
    
    trainer.cleanup()
```

## Practical Exercises

### Exercise 1: Distributed Training Setup
Set up distributed training for a medium-scale language model:
- Implement data parallelism across multiple GPUs
- Add model parallelism for larger models
- Monitor training efficiency and communication overhead

### Exercise 2: Memory Optimization
Implement memory optimization techniques:
- Apply gradient checkpointing to reduce memory usage
- Use mixed precision training
- Implement gradient accumulation for larger effective batch sizes

### Exercise 3: Parameter-Efficient Training
Experiment with parameter-efficient training methods:
- Implement LoRA for fine-tuning
- Compare with full fine-tuning
- Analyze the trade-offs between efficiency and performance

## Assessment Questions

1. **What are the main challenges in training large language models?**
   - Computational requirements
   - Memory constraints
   - Training stability
   - All of the above

2. **How does gradient accumulation help in LLM training?**
   - Allows larger effective batch sizes
   - Reduces memory requirements
   - Enables training on smaller GPUs
   - All of the above

3. **What is the purpose of mixed precision training?**
   - Reduces memory usage
   - Speeds up training
   - Maintains numerical stability
   - All of the above

## Key Takeaways

- LLM training requires specialized techniques for efficiency and stability
- Distributed training is essential for large-scale models
- Memory optimization techniques enable training larger models
- Parameter-efficient training methods reduce computational requirements
- Proper monitoring and logging are crucial for debugging and optimization
- Scaling laws help predict performance improvements with model size
- Advanced optimization techniques improve training efficiency

## Next Steps

In the next module, we'll explore prompt engineering and in-context learning, which are crucial for effectively using pre-trained language models for various tasks.
