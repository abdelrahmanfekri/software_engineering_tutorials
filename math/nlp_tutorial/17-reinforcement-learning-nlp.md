# NLP Tutorial Module 17: Reinforcement Learning in NLP

## Learning Objectives
By the end of this module, you will be able to:
- Understand reinforcement learning fundamentals for NLP
- Implement policy gradient methods for text generation
- Build reward models for language tasks
- Apply Proximal Policy Optimization (PPO) to NLP
- Implement self-play and adversarial training
- Use RL for dialogue systems and conversational AI
- Apply RLHF (Reinforcement Learning from Human Feedback)
- Build multi-agent RL systems for NLP
- Evaluate RL-based NLP models
- Design reward functions for language tasks

## Introduction to Reinforcement Learning in NLP

Reinforcement Learning (RL) has revolutionized NLP by enabling models to learn optimal behaviors through interaction with their environment. In 2025, RL is widely used for training language models, dialogue systems, and text generation models through techniques like RLHF (Reinforcement Learning from Human Feedback).

### Key RL Concepts for NLP

1. **Agent**: The language model making decisions
2. **Environment**: The text generation or dialogue context
3. **State**: Current text or conversation state
4. **Action**: Next token or response to generate
5. **Reward**: Quality score or human preference
6. **Policy**: Strategy for generating text

## Policy Gradient Methods for Text Generation

### REINFORCE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt

@dataclass
class RLConfig:
    """Configuration for RL training"""
    learning_rate: float = 1e-5
    gamma: float = 0.99
    batch_size: int = 32
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

class REINFORCETrainer:
    """REINFORCE algorithm for text generation"""
    
    def __init__(self, model, tokenizer, config: RLConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'policy_entropies': []
        }
    
    def generate_sequence(self, prompt: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Generate a sequence using current policy"""
        if max_length is None:
            max_length = self.config.max_length
            
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated_ids = input_ids.clone()
        log_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length - len(input_ids[0])):
                # Get logits for next token
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :] / self.config.temperature
                
                # Apply top-k and top-p filtering
                if self.config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                if self.config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from policy
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Calculate log probability
                log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)
                log_probs.append(log_prob)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return {
            'generated_ids': generated_ids,
            'log_probs': torch.cat(log_probs, dim=1),
            'input_length': len(input_ids[0])
        }
    
    def compute_reward(self, generated_text: str, reference_text: str = None) -> float:
        """Compute reward for generated text"""
        # Multiple reward components
        rewards = {}
        
        # 1. Length penalty (encourage reasonable length)
        length = len(generated_text.split())
        rewards['length'] = 1.0 if 5 <= length <= 50 else 0.5
        
        # 2. Repetition penalty
        words = generated_text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0
        rewards['diversity'] = repetition_ratio
        
        # 3. Coherence (simplified - would use more sophisticated metrics)
        rewards['coherence'] = 0.8  # Mock coherence score
        
        # 4. Reference similarity (if reference provided)
        if reference_text:
            # Simplified similarity (would use BLEU, ROUGE, etc.)
            ref_words = set(reference_text.lower().split())
            gen_words = set(words)
            overlap = len(ref_words.intersection(gen_words))
            rewards['similarity'] = overlap / len(ref_words) if ref_words else 0
        else:
            rewards['similarity'] = 0.5
        
        # Weighted combination
        weights = {
            'length': 0.2,
            'diversity': 0.3,
            'coherence': 0.3,
            'similarity': 0.2
        }
        
        total_reward = sum(weights[key] * rewards[key] for key in weights)
        return total_reward
    
    def train_episode(self, prompt: str, reference_text: str = None) -> Dict[str, float]:
        """Train on a single episode"""
        # Generate sequence
        generation_result = self.generate_sequence(prompt)
        generated_ids = generation_result['generated_ids']
        log_probs = generation_result['log_probs']
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Compute reward
        reward = self.compute_reward(generated_text, reference_text)
        
        # Compute policy loss (REINFORCE)
        policy_loss = -torch.mean(log_probs) * reward
        
        # Add entropy bonus for exploration
        entropy = -torch.sum(F.softmax(log_probs, dim=-1) * F.log_softmax(log_probs, dim=-1), dim=-1)
        entropy_bonus = 0.01 * torch.mean(entropy)
        
        total_loss = policy_loss - entropy_bonus
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record statistics
        episode_stats = {
            'reward': reward,
            'loss': total_loss.item(),
            'policy_entropy': torch.mean(entropy).item(),
            'generated_text': generated_text
        }
        
        return episode_stats
    
    def train_batch(self, prompts: List[str], reference_texts: List[str] = None) -> Dict[str, float]:
        """Train on a batch of episodes"""
        if reference_texts is None:
            reference_texts = [None] * len(prompts)
        
        batch_stats = []
        
        for prompt, reference in zip(prompts, reference_texts):
            episode_stats = self.train_episode(prompt, reference)
            batch_stats.append(episode_stats)
        
        # Average statistics
        avg_stats = {
            'avg_reward': np.mean([s['reward'] for s in batch_stats]),
            'avg_loss': np.mean([s['loss'] for s in batch_stats]),
            'avg_entropy': np.mean([s['policy_entropy'] for s in batch_stats])
        }
        
        return avg_stats
    
    def train(self, prompts: List[str], reference_texts: List[str] = None, 
              num_epochs: int = 100) -> Dict[str, List]:
        """Train the model using REINFORCE"""
        
        for epoch in range(num_epochs):
            # Sample batch
            batch_indices = random.sample(range(len(prompts)), 
                                        min(self.config.batch_size, len(prompts)))
            batch_prompts = [prompts[i] for i in batch_indices]
            batch_references = [reference_texts[i] if reference_texts else None 
                              for i in batch_indices]
            
            # Train on batch
            batch_stats = self.train_batch(batch_prompts, batch_references)
            
            # Record statistics
            self.training_stats['episodes'].append(epoch)
            self.training_stats['rewards'].append(batch_stats['avg_reward'])
            self.training_stats['losses'].append(batch_stats['avg_loss'])
            self.training_stats['policy_entropies'].append(batch_stats['avg_entropy'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Reward={batch_stats['avg_reward']:.3f}, "
                      f"Loss={batch_stats['avg_loss']:.3f}, "
                      f"Entropy={batch_stats['avg_entropy']:.3f}")
        
        return self.training_stats
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward curve
        axes[0, 0].plot(self.training_stats['episodes'], self.training_stats['rewards'])
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        
        # Loss curve
        axes[0, 1].plot(self.training_stats['episodes'], self.training_stats['losses'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Loss')
        
        # Entropy curve
        axes[1, 0].plot(self.training_stats['episodes'], self.training_stats['policy_entropies'])
        axes[1, 0].set_title('Policy Entropy')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Entropy')
        
        # Combined view
        axes[1, 1].plot(self.training_stats['episodes'], self.training_stats['rewards'], label='Reward')
        axes[1, 1].plot(self.training_stats['episodes'], 
                       np.array(self.training_stats['losses']) * 10, label='Loss (×10)')
        axes[1, 1].set_title('Training Progress')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
def demonstrate_reinforce():
    """Demonstrate REINFORCE training"""
    print("REINFORCE Training for Text Generation")
    
    # Load a small model for demonstration
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Configuration
    config = RLConfig(
        learning_rate=1e-5,
        batch_size=4,
        max_length=30
    )
    
    # Initialize trainer
    trainer = REINFORCETrainer(model, tokenizer, config)
    
    # Training prompts
    prompts = [
        "The weather today is",
        "I think that artificial intelligence",
        "The best way to learn",
        "Technology has changed",
        "In the future, we will"
    ]
    
    # Reference texts (optional)
    reference_texts = [
        "The weather today is sunny and warm.",
        "I think that artificial intelligence will revolutionize many industries.",
        "The best way to learn is through practice and repetition.",
        "Technology has changed the way we communicate.",
        "In the future, we will have flying cars."
    ]
    
    print("Starting training...")
    training_stats = trainer.train(prompts, reference_texts, num_epochs=20)
    
    # Plot results
    trainer.plot_training_progress()
    
    # Test generation
    test_prompt = "The future of AI is"
    test_generation = trainer.generate_sequence(test_prompt)
    test_text = tokenizer.decode(test_generation['generated_ids'][0], skip_special_tokens=True)
    
    print(f"\nTest generation: {test_text}")
    
    return trainer, training_stats

# Run demonstration
reinforce_trainer, reinforce_stats = demonstrate_reinforce()
```

## Proximal Policy Optimization (PPO) for NLP

### PPO Implementation

```python
class PPOTrainer:
    """PPO algorithm for text generation"""
    
    def __init__(self, model, tokenizer, config: RLConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # PPO specific parameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Value function (simplified - would use separate network)
        self.value_function = nn.Linear(model.config.n_embd, 1)
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'kl_divergences': []
        }
    
    def generate_sequence_with_value(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Generate sequence and estimate value"""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated_ids = input_ids.clone()
        log_probs = []
        values = []
        
        for _ in range(self.config.max_length - len(input_ids[0])):
            # Get model outputs
            outputs = self.model(generated_ids)
            logits = outputs.logits[:, -1, :] / self.config.temperature
            
            # Apply filtering
            if self.config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample action
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Calculate log probability
            log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)
            log_probs.append(log_prob)
            
            # Estimate value
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last hidden state
            value = self.value_function(hidden_states)
            values.append(value)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return {
            'generated_ids': generated_ids,
            'log_probs': torch.cat(log_probs, dim=1),
            'values': torch.cat(values, dim=1),
            'input_length': len(input_ids[0])
        }
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE"""
        gamma = self.config.gamma
        lam = 0.95  # GAE parameter
        
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Compute advantages using GAE
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = delta + gamma * lam * (advantages[:, t + 1] if t < seq_len - 1 else 0)
        
        # Compute returns
        for t in range(seq_len):
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return advantages, returns
    
    def train_ppo_step(self, old_log_probs: torch.Tensor, 
                      log_probs: torch.Tensor,
                      values: torch.Tensor,
                      advantages: torch.Tensor,
                      returns: torch.Tensor) -> Dict[str, float]:
        """Single PPO training step"""
        
        # Calculate probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Calculate clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        # Calculate value loss
        value_loss = F.mse_loss(values, returns)
        
        # Calculate entropy bonus
        entropy = -torch.sum(F.softmax(log_probs, dim=-1) * F.log_softmax(log_probs, dim=-1), dim=-1)
        entropy_bonus = self.entropy_coef * torch.mean(entropy)
        
        # Calculate KL divergence
        kl_div = torch.mean(old_log_probs - log_probs)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - entropy_bonus
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'kl_divergence': kl_div.item(),
            'entropy': torch.mean(entropy).item()
        }
    
    def train_episode_ppo(self, prompt: str, reference_text: str = None) -> Dict[str, float]:
        """Train on a single episode using PPO"""
        
        # Generate sequence
        generation_result = self.generate_sequence_with_value(prompt)
        generated_ids = generation_result['generated_ids']
        old_log_probs = generation_result['log_probs']
        old_values = generation_result['values']
        
        # Decode and compute rewards
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        reward = self.compute_reward(generated_text, reference_text)
        
        # Create reward tensor
        seq_len = old_log_probs.shape[1]
        rewards = torch.full((1, seq_len), reward, dtype=torch.float32)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, old_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple PPO updates
        total_stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'total_loss': 0,
            'kl_divergence': 0,
            'entropy': 0
        }
        
        num_updates = 4  # Number of PPO updates per episode
        
        for _ in range(num_updates):
            # Re-generate with current policy
            current_result = self.generate_sequence_with_value(prompt)
            current_log_probs = current_result['log_probs']
            current_values = current_result['values']
            
            # PPO update
            stats = self.train_ppo_step(
                old_log_probs, current_log_probs, current_values, 
                advantages, returns
            )
            
            # Accumulate statistics
            for key in total_stats:
                total_stats[key] += stats[key]
            
            # Early stopping if KL divergence is too large
            if stats['kl_divergence'] > 0.02:
                break
        
        # Average statistics
        for key in total_stats:
            total_stats[key] /= num_updates
        
        total_stats['reward'] = reward
        total_stats['generated_text'] = generated_text
        
        return total_stats
    
    def compute_reward(self, generated_text: str, reference_text: str = None) -> float:
        """Compute reward (same as REINFORCE)"""
        # Multiple reward components
        rewards = {}
        
        # Length penalty
        length = len(generated_text.split())
        rewards['length'] = 1.0 if 5 <= length <= 50 else 0.5
        
        # Repetition penalty
        words = generated_text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0
        rewards['diversity'] = repetition_ratio
        
        # Coherence
        rewards['coherence'] = 0.8
        
        # Reference similarity
        if reference_text:
            ref_words = set(reference_text.lower().split())
            gen_words = set(words)
            overlap = len(ref_words.intersection(gen_words))
            rewards['similarity'] = overlap / len(ref_words) if ref_words else 0
        else:
            rewards['similarity'] = 0.5
        
        # Weighted combination
        weights = {'length': 0.2, 'diversity': 0.3, 'coherence': 0.3, 'similarity': 0.2}
        total_reward = sum(weights[key] * rewards[key] for key in weights)
        
        return total_reward

# Example usage
def demonstrate_ppo():
    """Demonstrate PPO training"""
    print("PPO Training for Text Generation")
    
    # Load model (same as REINFORCE)
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    config = RLConfig(learning_rate=1e-5, batch_size=4, max_length=30)
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(model, tokenizer, config)
    
    prompts = [
        "The weather today is",
        "I think that artificial intelligence",
        "The best way to learn"
    ]
    
    reference_texts = [
        "The weather today is sunny and warm.",
        "I think that artificial intelligence will revolutionize many industries.",
        "The best way to learn is through practice and repetition."
    ]
    
    print("Starting PPO training...")
    
    # Train for a few episodes
    for epoch in range(10):
        episode_stats = ppo_trainer.train_episode_ppo(prompts[0], reference_texts[0])
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Reward={episode_stats['reward']:.3f}, "
                  f"Policy Loss={episode_stats['policy_loss']:.3f}, "
                  f"KL Div={episode_stats['kl_divergence']:.3f}")
    
    return ppo_trainer

ppo_demo = demonstrate_ppo()
```

## Reward Modeling and Human Feedback

### Reward Model Implementation

```python
class RewardModel(nn.Module):
    """Neural network for predicting human preferences"""
    
    def __init__(self, base_model, hidden_size=256):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.config.n_embd, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Pool over sequence length (mean pooling)
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Predict reward
        reward = self.reward_head(pooled_output)
        
        return reward

class RewardModelTrainer:
    """Trainer for reward models"""
    
    def __init__(self, reward_model, tokenizer, learning_rate=1e-5):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_step(self, text1: str, text2: str, preference: int):
        """Train on a single preference pair"""
        # preference: 0 if text1 is preferred, 1 if text2 is preferred
        
        # Tokenize both texts
        inputs1 = self.tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
        inputs2 = self.tokenizer(text2, return_tensors='pt', padding=True, truncation=True)
        
        # Get reward predictions
        reward1 = self.reward_model(inputs1['input_ids'], inputs1['attention_mask'])
        reward2 = self.reward_model(inputs2['input_ids'], inputs2['attention_mask'])
        
        # Compute preference loss
        if preference == 0:  # text1 preferred
            loss = -F.logsigmoid(reward1 - reward2)
        else:  # text2 preferred
            loss = -F.logsigmoid(reward2 - reward1)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_reward(self, text: str) -> float:
        """Predict reward for a single text"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            reward = self.reward_model(inputs['input_ids'], inputs['attention_mask'])
        
        return reward.item()
    
    def rank_texts(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Rank texts by predicted reward"""
        rewards = []
        for text in texts:
            reward = self.predict_reward(text)
            rewards.append((text, reward))
        
        # Sort by reward (descending)
        rewards.sort(key=lambda x: x[1], reverse=True)
        
        return rewards

# Example usage
def demonstrate_reward_modeling():
    """Demonstrate reward model training"""
    print("Reward Model Training")
    
    # Load base model
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create reward model
    reward_model = RewardModel(base_model)
    
    # Initialize trainer
    reward_trainer = RewardModelTrainer(reward_model, tokenizer)
    
    # Training data (preference pairs)
    training_data = [
        ("This movie is absolutely fantastic!", "This movie is okay.", 0),
        ("I love this product, it's amazing.", "The product is fine.", 0),
        ("Terrible service, very disappointed.", "The service could be better.", 0),
        ("Great experience overall.", "It was an okay experience.", 0)
    ]
    
    print("Training reward model...")
    
    # Train for several epochs
    for epoch in range(5):
        total_loss = 0
        for text1, text2, preference in training_data:
            loss = reward_trainer.train_step(text1, text2, preference)
            total_loss += loss
        
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.3f}")
    
    # Test ranking
    test_texts = [
        "This is amazing!",
        "This is okay.",
        "This is terrible.",
        "This is great!"
    ]
    
    ranked_texts = reward_trainer.rank_texts(test_texts)
    
    print("\nRanked texts by predicted reward:")
    for i, (text, reward) in enumerate(ranked_texts, 1):
        print(f"{i}. {text} (reward: {reward:.3f})")
    
    return reward_trainer

reward_demo = demonstrate_reward_modeling()
```

## RLHF (Reinforcement Learning from Human Feedback)

### RLHF Implementation

```python
class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer"""
    
    def __init__(self, policy_model, reward_model, tokenizer, config):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize PPO trainer for policy optimization
        self.ppo_trainer = PPOTrainer(policy_model, tokenizer, config)
        
        # Human feedback data
        self.human_feedback_data = []
        
    def collect_human_feedback(self, prompt: str, num_responses: int = 4) -> List[Dict]:
        """Collect human feedback for generated responses"""
        
        # Generate multiple responses
        responses = []
        for _ in range(num_responses):
            generation_result = self.ppo_trainer.generate_sequence_with_value(prompt)
            response_text = self.tokenizer.decode(
                generation_result['generated_ids'][0], skip_special_tokens=True
            )
            responses.append(response_text)
        
        # In practice, this would collect human ratings
        # For demonstration, we'll simulate human preferences
        human_ratings = []
        for i, response in enumerate(responses):
            # Simulate human rating (1-10 scale)
            rating = random.uniform(1, 10)
            human_ratings.append({
                'response': response,
                'rating': rating,
                'response_id': i
            })
        
        # Sort by rating
        human_ratings.sort(key=lambda x: x['rating'], reverse=True)
        
        return human_ratings
    
    def train_reward_model_from_feedback(self, feedback_data: List[Dict]):
        """Train reward model using human feedback"""
        
        # Create preference pairs from rankings
        preference_pairs = []
        
        for i in range(len(feedback_data)):
            for j in range(i + 1, len(feedback_data)):
                better_response = feedback_data[i]['response']
                worse_response = feedback_data[j]['response']
                
                # Create preference pair (0 means first response is preferred)
                preference_pairs.append((better_response, worse_response, 0))
        
        # Train reward model on preference pairs
        reward_trainer = RewardModelTrainer(self.reward_model, self.tokenizer)
        
        for better, worse, preference in preference_pairs:
            reward_trainer.train_step(better, worse, preference)
    
    def rlhf_iteration(self, prompts: List[str]) -> Dict[str, float]:
        """Single RLHF iteration"""
        
        iteration_stats = {
            'policy_rewards': [],
            'reward_model_losses': [],
            'human_satisfaction': []
        }
        
        for prompt in prompts:
            # 1. Collect human feedback
            feedback_data = self.collect_human_feedback(prompt)
            
            # 2. Train reward model
            self.train_reward_model_from_feedback(feedback_data)
            
            # 3. Train policy using reward model
            # Generate with current policy
            generation_result = self.ppo_trainer.generate_sequence_with_value(prompt)
            response_text = self.tokenizer.decode(
                generation_result['generated_ids'][0], skip_special_tokens=True
            )
            
            # Get reward from trained reward model
            reward = self.reward_model.predict_reward(response_text)
            
            # Train policy with this reward
            episode_stats = self.ppo_trainer.train_episode_ppo(prompt)
            
            # Record statistics
            iteration_stats['policy_rewards'].append(reward)
            iteration_stats['reward_model_losses'].append(episode_stats['total_loss'])
            
            # Simulate human satisfaction (would be collected from humans)
            human_satisfaction = random.uniform(0.6, 0.9)
            iteration_stats['human_satisfaction'].append(human_satisfaction)
        
        # Average statistics
        avg_stats = {
            'avg_policy_reward': np.mean(iteration_stats['policy_rewards']),
            'avg_reward_model_loss': np.mean(iteration_stats['reward_model_losses']),
            'avg_human_satisfaction': np.mean(iteration_stats['human_satisfaction'])
        }
        
        return avg_stats
    
    def train_rlhf(self, prompts: List[str], num_iterations: int = 10) -> Dict[str, List]:
        """Full RLHF training loop"""
        
        training_history = {
            'iterations': [],
            'policy_rewards': [],
            'reward_model_losses': [],
            'human_satisfaction': []
        }
        
        for iteration in range(num_iterations):
            print(f"RLHF Iteration {iteration + 1}/{num_iterations}")
            
            # Single RLHF iteration
            stats = self.rlhf_iteration(prompts)
            
            # Record history
            training_history['iterations'].append(iteration)
            training_history['policy_rewards'].append(stats['avg_policy_reward'])
            training_history['reward_model_losses'].append(stats['avg_reward_model_loss'])
            training_history['human_satisfaction'].append(stats['avg_human_satisfaction'])
            
            # Print progress
            print(f"  Policy Reward: {stats['avg_policy_reward']:.3f}")
            print(f"  Reward Model Loss: {stats['avg_reward_model_loss']:.3f}")
            print(f"  Human Satisfaction: {stats['avg_human_satisfaction']:.3f}")
        
        return training_history

# Example usage
def demonstrate_rlhf():
    """Demonstrate RLHF training"""
    print("Reinforcement Learning from Human Feedback")
    
    # Load models
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    policy_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.eos_token_id
        base_model.config.pad_token_id = tokenizer.eos_token_id
    
    # Create reward model
    reward_model = RewardModel(base_model)
    
    config = RLConfig(learning_rate=1e-5, batch_size=2, max_length=30)
    
    # Initialize RLHF trainer
    rlhf_trainer = RLHFTrainer(policy_model, reward_model, tokenizer, config)
    
    # Training prompts
    prompts = [
        "Write a helpful response to a customer complaint:",
        "Explain how to use this product:",
        "Provide a summary of this topic:"
    ]
    
    print("Starting RLHF training...")
    
    # Train for a few iterations
    history = rlhf_trainer.train_rlhf(prompts, num_iterations=3)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['iterations'], history['policy_rewards'])
    plt.title('Policy Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['iterations'], history['reward_model_losses'])
    plt.title('Reward Model Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Average Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['iterations'], history['human_satisfaction'])
    plt.title('Human Satisfaction')
    plt.xlabel('Iteration')
    plt.ylabel('Average Satisfaction')
    
    plt.tight_layout()
    plt.show()
    
    return rlhf_trainer, history

rlhf_demo = demonstrate_rlhf()
```

## Practical Exercises

### Exercise 1: REINFORCE Implementation
Implement a complete REINFORCE system:
- Build custom reward functions for different tasks
- Experiment with different exploration strategies
- Compare performance with supervised learning baselines

### Exercise 2: PPO for Dialogue Systems
Apply PPO to dialogue systems:
- Implement conversation reward modeling
- Handle multi-turn conversations
- Optimize for both helpfulness and safety

### Exercise 3: RLHF Pipeline
Build a complete RLHF system:
- Collect and process human feedback data
- Train reward models from preferences
- Optimize policies using learned rewards

## Assessment Questions

1. **What is the main advantage of RL over supervised learning for text generation?**
   - Can optimize for complex, non-differentiable objectives
   - Faster training time
   - Better generalization
   - All of the above

2. **How does PPO improve upon REINFORCE?**
   - Better sample efficiency
   - More stable training
   - Handles continuous action spaces
   - All of the above

3. **What is the key insight behind RLHF?**
   - Human preferences can be learned by neural networks
   - RL can optimize for human-aligned objectives
   - Reward models can replace human evaluators
   - All of the above

## Key Takeaways

- Reinforcement Learning enables optimization for complex objectives
- REINFORCE provides a foundation for policy gradient methods
- PPO offers more stable and sample-efficient training
- Reward modeling allows learning from human preferences
- RLHF aligns language models with human values
- Exploration-exploitation trade-offs are crucial for RL
- Evaluation requires careful consideration of multiple metrics

## Next Steps

This module provides the foundation for applying RL to NLP tasks. In the next module, we'll explore Brain-Computer Interface applications in NLP, which represents cutting-edge research in 2025.
