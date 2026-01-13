# Module 10: Reinforcement Learning for Trading

## Table of Contents
1. [RL Fundamentals for Finance](#rl-fundamentals-for-finance)
2. [Value-Based Methods](#value-based-methods)
3. [Policy Gradient Methods](#policy-gradient-methods)
4. [Model-Based RL](#model-based-rl)
5. [Multi-Agent RL](#multi-agent-rl)
6. [Advanced RL Techniques](#advanced-rl-techniques)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## RL Fundamentals for Finance

### Trading Environment Design

```python
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

class TradingEnvironment(gym.Env):
    def __init__(
        self,
        price_data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position: int = 100
    ):
        super(TradingEnvironment, self).__init__()
        
        self.price_data = price_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.total_pnl = 0
        
        n_features = price_data.shape[1]
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_pnl = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.price_data):
            self.current_step = len(self.price_data) - 1
        
        market_features = self.price_data.iloc[self.current_step].values
        
        normalized_position = self.position / self.max_position
        normalized_balance = self.balance / self.initial_balance
        normalized_pnl = self.total_pnl / self.initial_balance
        
        obs = np.concatenate([
            market_features,
            [normalized_position, normalized_balance, normalized_pnl]
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        current_price = self.price_data.iloc[self.current_step, 0]
        
        reward = 0
        old_portfolio_value = self.balance + self.position * current_price
        
        if action == 0:
            if self.position > -self.max_position:
                cost = current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.position -= 1
                reward -= self.transaction_cost * current_price
        
        elif action == 2:
            if self.position < self.max_position:
                revenue = current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.position += 1
                reward -= self.transaction_cost * current_price
        
        self.current_step += 1
        
        done = self.current_step >= len(self.price_data) - 1
        
        if not done:
            new_price = self.price_data.iloc[self.current_step, 0]
            new_portfolio_value = self.balance + self.position * new_price
            
            pnl = new_portfolio_value - old_portfolio_value
            self.total_pnl += pnl
            
            reward += pnl
            
            sharpe_penalty = -abs(self.position) * 0.01
            reward += sharpe_penalty
        else:
            final_price = self.price_data.iloc[self.current_step, 0]
            final_portfolio_value = self.balance + self.position * final_price
            reward = final_portfolio_value - self.initial_balance
        
        obs = self._get_observation()
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.balance + self.position * self.price_data.iloc[self.current_step, 0],
            'total_pnl': self.total_pnl
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        current_price = self.price_data.iloc[self.current_step, 0]
        portfolio_value = self.balance + self.position * current_price
        
        print(f"Step: {self.current_step}")
        print(f"Price: {current_price:.2f}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Position: {self.position}")
        print(f"Portfolio Value: {portfolio_value:.2f}")
        print(f"Total PnL: {self.total_pnl:.2f}")
        print("-" * 50)
```

### Continuous Action Space Environment

```python
class ContinuousActionTradingEnv(gym.Env):
    def __init__(
        self,
        price_data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001
    ):
        super(ContinuousActionTradingEnv, self).__init__()
        
        self.price_data = price_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0
        
        n_features = price_data.shape[1]
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 2,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.price_data):
            self.current_step = len(self.price_data) - 1
        
        market_features = self.price_data.iloc[self.current_step].values
        
        current_price = self.price_data.iloc[self.current_step, 0]
        portfolio_value = self.balance + self.shares * current_price
        
        normalized_balance = self.balance / portfolio_value if portfolio_value > 0 else 0
        normalized_shares = (self.shares * current_price) / portfolio_value if portfolio_value > 0 else 0
        
        obs = np.concatenate([
            market_features,
            [normalized_balance, normalized_shares]
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = float(action[0])
        
        current_price = self.price_data.iloc[self.current_step, 0]
        portfolio_value = self.balance + self.shares * current_price
        
        target_shares_value = portfolio_value * (action + 1) / 2
        target_shares = target_shares_value / current_price
        
        shares_to_trade = target_shares - self.shares
        
        if shares_to_trade > 0:
            cost = shares_to_trade * current_price * (1 + self.transaction_cost)
            if cost <= self.balance:
                self.balance -= cost
                self.shares += shares_to_trade
                transaction_cost_paid = shares_to_trade * current_price * self.transaction_cost
            else:
                transaction_cost_paid = 0
        else:
            revenue = abs(shares_to_trade) * current_price * (1 - self.transaction_cost)
            self.balance += revenue
            self.shares += shares_to_trade
            transaction_cost_paid = abs(shares_to_trade) * current_price * self.transaction_cost
        
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        new_price = self.price_data.iloc[self.current_step, 0]
        new_portfolio_value = self.balance + self.shares * new_price
        
        reward = (new_portfolio_value - portfolio_value) - transaction_cost_paid
        
        obs = self._get_observation()
        
        info = {
            'balance': self.balance,
            'shares': self.shares,
            'portfolio_value': new_portfolio_value
        }
        
        return obs, reward, done, info
```

## Value-Based Methods

### Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = deque(maxlen=buffer_size)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### Double DQN and Dueling DQN

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class DoubleDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=kwargs.get('learning_rate', 0.001))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
```

## Policy Gradient Methods

### REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99
    ):
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        action_probs = self.policy(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        self.log_probs.append(action_dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward: float):
        self.rewards.append(reward)
    
    def train(self):
        returns = []
        G = 0
        
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()
```

### Proximal Policy Optimization (PPO)

```python
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCriticNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.actor(x)
        value = self.critic(x)
        
        return action_probs, value


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_clip: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.01,
        epochs: int = 10
    ):
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.memory = []
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ):
        self.memory.append((state, action, log_prob, value, reward, done))
    
    def train(self):
        if not self.memory:
            return
        
        states, actions, old_log_probs, old_values, rewards, dones = zip(*self.memory)
        
        returns = []
        advantages = []
        G = 0
        A = 0
        
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G * (1 - dones[i])
            returns.insert(0, G)
            
            if i < len(rewards) - 1:
                td_error = rewards[i] + self.gamma * old_values[i+1] * (1 - dones[i]) - old_values[i]
                A = td_error + self.gamma * 0.95 * A * (1 - dones[i])
            else:
                A = G - old_values[i]
            
            advantages.insert(0, A)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            action_probs, values = self.policy(states)
            
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.memory = []
        
        return loss.item()
```

### Soft Actor-Critic (SAC)

```python
class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from copy import deepcopy
        
        self.actor = self._build_actor(state_dim, action_dim).to(self.device)
        
        self.critic1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2 = self._build_critic(state_dim, action_dim).to(self.device)
        
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        self.replay_buffer = deque(maxlen=100000)
        
    def _build_actor(self, state_dim: int, action_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def _build_critic(self, state_dim: int, action_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if not evaluate:
            action += np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1, 1)
        
        return action
    
    def train(self, batch_size: int = 256):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions = self.actor(next_states)
            
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2)
            
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        new_actions = self.actor(states)
        q_new = self.critic1(torch.cat([states, new_actions], dim=1))
        
        actor_loss = -q_new.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Model-Based RL

### World Model for Trading

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(WorldModel, self).__init__()
        
        self.transition_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        
        next_state = self.transition_model(x)
        reward = self.reward_model(x)
        
        return next_state, reward


class ModelBasedRLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.world_model = WorldModel(state_dim, action_dim).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.replay_buffer = deque(maxlen=10000)
        
    def train_world_model(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, _ = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=3).float()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        pred_next_states, pred_rewards = self.world_model(states, actions_one_hot)
        
        transition_loss = nn.MSELoss()(pred_next_states, next_states)
        reward_loss = nn.MSELoss()(pred_rewards, rewards)
        
        loss = transition_loss + reward_loss
        
        self.world_optimizer.zero_grad()
        loss.backward()
        self.world_optimizer.step()
        
        return loss.item()
    
    def plan_with_model(
        self,
        state: np.ndarray,
        horizon: int = 10,
        n_samples: int = 100
    ) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_tensor = state_tensor.repeat(n_samples, 1)
        
        total_rewards = torch.zeros(n_samples, 3).to(self.device)
        
        for _ in range(horizon):
            action_probs = self.policy(state_tensor)
            
            for action in range(3):
                action_tensor = torch.nn.functional.one_hot(
                    torch.full((n_samples,), action, dtype=torch.long),
                    num_classes=3
                ).float().to(self.device)
                
                next_state, reward = self.world_model(state_tensor, action_tensor)
                total_rewards[:, action] += reward.squeeze()
                
        best_action = total_rewards.mean(dim=0).argmax().item()
        
        return best_action
```

## Multi-Agent RL

### Cooperative Trading Agents

```python
class MultiAgentTradingEnv:
    def __init__(
        self,
        price_data: pd.DataFrame,
        n_agents: int = 2,
        initial_balance: float = 10000.0
    ):
        self.price_data = price_data
        self.n_agents = n_agents
        self.initial_balance = initial_balance
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balances = [self.initial_balance] * self.n_agents
        self.positions = [0] * self.n_agents
        
        return self._get_observations()
    
    def _get_observations(self) -> List[np.ndarray]:
        observations = []
        
        market_state = self.price_data.iloc[self.current_step].values
        
        for i in range(self.n_agents):
            agent_state = np.array([
                self.balances[i] / self.initial_balance,
                self.positions[i] / 100
            ])
            
            obs = np.concatenate([market_state, agent_state])
            observations.append(obs)
        
        return observations
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        current_price = self.price_data.iloc[self.current_step, 0]
        
        rewards = []
        
        for i, action in enumerate(actions):
            old_value = self.balances[i] + self.positions[i] * current_price
            
            if action == 0 and self.positions[i] > -100:
                self.balances[i] -= current_price
                self.positions[i] -= 1
            elif action == 2 and self.positions[i] < 100:
                self.balances[i] += current_price
                self.positions[i] += 1
            
            new_value = self.balances[i] + self.positions[i] * current_price
            rewards.append(new_value - old_value)
        
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        observations = self._get_observations()
        
        info = {
            'balances': self.balances,
            'positions': self.positions
        }
        
        return observations, rewards, done, info
```

## Advanced RL Techniques

### Hierarchical RL for Trading

```python
class HierarchicalTradingAgent:
    def __init__(
        self,
        state_dim: int,
        n_strategies: int = 3,
        actions_per_strategy: int = 3
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.meta_controller = PolicyNetwork(state_dim, n_strategies).to(self.device)
        
        self.sub_controllers = nn.ModuleList([
            PolicyNetwork(state_dim, actions_per_strategy).to(self.device)
            for _ in range(n_strategies)
        ])
        
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=0.001)
        self.sub_optimizers = [
            optim.Adam(controller.parameters(), lr=0.001)
            for controller in self.sub_controllers
        ]
        
    def select_strategy(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            strategy_probs = self.meta_controller(state_tensor)
        
        strategy_dist = torch.distributions.Categorical(strategy_probs)
        strategy = strategy_dist.sample().item()
        
        return strategy
    
    def select_action(self, state: np.ndarray, strategy: int) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.sub_controllers[strategy](state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        
        return action
```

## PhD-Level Research Topics

### Decision Transformer for Trading

The Decision Transformer reformulates RL as sequence modeling, enabling training on historical (offline) data without bootstrapping.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DecisionTransformerTrading(nn.Module):
    """Decision Transformer for offline reinforcement learning in trading"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_ep_len: int = 1000,
        max_return: float = 100.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_return = max_return
        
        # Embeddings for returns-to-go, states, and actions
        self.return_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        
        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_ep_len, d_model)
        
        # Layer norm
        self.embed_ln = nn.LayerNorm(d_model)
        
        # GPT-style transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh()  # Actions bounded [-1, 1]
        )
        
    def forward(
        self,
        returns_to_go: torch.Tensor,  # (batch, seq_len, 1)
        states: torch.Tensor,          # (batch, seq_len, state_dim)
        actions: torch.Tensor,         # (batch, seq_len, action_dim)
        timesteps: torch.Tensor        # (batch, seq_len)
    ) -> torch.Tensor:
        batch_size, seq_len = states.shape[:2]
        
        # Embed tokens
        return_emb = self.return_embed(returns_to_go / self.max_return)
        state_emb = self.state_embed(states)
        action_emb = self.action_embed(actions)
        time_emb = self.timestep_embed(timesteps)
        
        # Add time embeddings
        return_emb = return_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        
        # Interleave: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # Stack along sequence dimension
        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
        stacked = stacked.reshape(batch_size, seq_len * 3, self.d_model)
        stacked = self.embed_ln(stacked)
        
        # Causal mask
        total_len = seq_len * 3
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=states.device),
            diagonal=1
        ).bool()
        
        # Transform
        hidden = self.transformer(stacked, stacked, tgt_mask=causal_mask)
        
        # Extract state positions for action prediction (positions 1, 4, 7, ...)
        state_hidden = hidden[:, 1::3]
        
        # Predict actions
        action_preds = self.action_head(state_hidden)
        
        return action_preds
    
    def get_action(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get action for the current state given context"""
        action_preds = self.forward(returns_to_go, states, actions, timesteps)
        return action_preds[:, -1]


class DecisionTransformerTrainer:
    """Trainer for Decision Transformer on offline trading data"""
    def __init__(
        self,
        model: DecisionTransformerTrading,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def train_step(self, batch: dict) -> float:
        self.model.train()
        
        returns_to_go = batch['returns_to_go']
        states = batch['states']
        actions = batch['actions']
        timesteps = batch['timesteps']
        
        # Forward pass
        action_preds = self.model(returns_to_go, states, actions, timesteps)
        
        # Loss on all actions except the last
        loss = F.mse_loss(action_preds[:, :-1], actions[:, 1:])
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        
        # Learning rate warmup
        self.step += 1
        if self.step < self.warmup_steps:
            lr_mult = min(1.0, self.step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_mult * 1e-4
        
        return loss.item()
```

### Offline RL with Conservative Q-Learning

```python
class ConservativeQLearning(nn.Module):
    """Conservative Q-Learning (CQL) for offline trading"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_actions_sample: int = 10,
        cql_alpha: float = 1.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions_sample = n_actions_sample
        self.cql_alpha = cql_alpha
        
        # Twin Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        output = self.policy(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if deterministic:
            return torch.tanh(mean)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action)
    
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Tuple[torch.Tensor, dict]:
        batch_size = states.shape[0]
        
        # Standard TD loss
        with torch.no_grad():
            next_actions = self.get_action(next_states)
            next_q1 = self.q1(torch.cat([next_states, next_actions], dim=-1))
            next_q2 = self.q2(torch.cat([next_states, next_actions], dim=-1))
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + gamma * (1 - dones) * next_q
        
        current_q1 = self.q1(torch.cat([states, actions], dim=-1))
        current_q2 = self.q2(torch.cat([states, actions], dim=-1))
        
        td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # CQL regularization: penalize high Q-values for OOD actions
        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size, self.n_actions_sample, self.action_dim
        ).uniform_(-1, 1).to(states.device)
        
        # Q-values for random actions
        states_repeat = states.unsqueeze(1).repeat(1, self.n_actions_sample, 1)
        states_flat = states_repeat.reshape(-1, states.shape[-1])
        actions_flat = random_actions.reshape(-1, self.action_dim)
        
        random_q1 = self.q1(torch.cat([states_flat, actions_flat], dim=-1))
        random_q2 = self.q2(torch.cat([states_flat, actions_flat], dim=-1))
        
        random_q1 = random_q1.reshape(batch_size, self.n_actions_sample)
        random_q2 = random_q2.reshape(batch_size, self.n_actions_sample)
        
        # CQL penalty: logsumexp over random actions - Q(s, a) from data
        cql_loss1 = (
            torch.logsumexp(random_q1, dim=1).mean() - current_q1.mean()
        )
        cql_loss2 = (
            torch.logsumexp(random_q2, dim=1).mean() - current_q2.mean()
        )
        cql_loss = self.cql_alpha * (cql_loss1 + cql_loss2)
        
        total_loss = td_loss + cql_loss
        
        info = {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item()
        }
        
        return total_loss, info


class ImplicitQLearning(nn.Module):
    """Implicit Q-Learning (IQL) for offline trading - avoids OOD action evaluation"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        expectile: float = 0.7
    ):
        super().__init__()
        self.expectile = expectile
        
        # Value function V(s)
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q-function Q(s, a)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy (advantage-weighted regression)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric L2 loss for expectile regression"""
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * (diff ** 2)).mean()
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Target Q (using target value function)
        with torch.no_grad():
            next_v = self.value(next_states)
            target_q = rewards + gamma * (1 - dones) * next_v
        
        # Q-function loss
        sa = torch.cat([states, actions], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # Value function loss (expectile regression)
        with torch.no_grad():
            q_min = torch.min(q1, q2)
        
        v = self.value(states)
        v_loss = self.expectile_loss(q_min - v)
        
        # Policy loss (advantage-weighted behavior cloning)
        with torch.no_grad():
            advantage = q_min - v
            weight = torch.exp(advantage * 3.0).clamp(max=100.0)
        
        policy_actions = self.policy(states)
        policy_loss = (weight * F.mse_loss(policy_actions, actions, reduction='none').mean(dim=-1)).mean()
        
        info = {
            'q_loss': q_loss.item(),
            'v_loss': v_loss.item(),
            'policy_loss': policy_loss.item(),
            'advantage_mean': advantage.mean().item()
        }
        
        return q_loss, v_loss, policy_loss, info
```

### Distributional RL for Risk-Aware Trading

```python
class QuantileNetwork(nn.Module):
    """Quantile Regression DQN for distributional RL"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 51,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.action_dim = action_dim
        
        # Quantile embedding (cosine basis)
        self.embedding_dim = 64
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Quantile embedding network
        self.quantile_embed = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, action_dim)
        
        # Register tau values
        self.register_buffer(
            'tau',
            torch.linspace(0.5 / n_quantiles, 1 - 0.5 / n_quantiles, n_quantiles)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns quantile values for each action: (batch, n_quantiles, action_dim)"""
        batch_size = state.shape[0]
        
        # Encode state
        state_encoded = self.state_encoder(state)  # (batch, hidden)
        
        # Cosine embedding of tau
        i_pi = torch.arange(1, self.embedding_dim + 1, device=state.device).float() * 3.14159
        tau_expand = self.tau.unsqueeze(-1)  # (n_quantiles, 1)
        cos_embed = torch.cos(tau_expand * i_pi)  # (n_quantiles, embed_dim)
        
        # Embed quantiles
        quantile_embed = self.quantile_embed(cos_embed)  # (n_quantiles, hidden)
        
        # Combine state and quantile embeddings
        state_expand = state_encoded.unsqueeze(1)  # (batch, 1, hidden)
        quantile_expand = quantile_embed.unsqueeze(0)  # (1, n_quantiles, hidden)
        combined = state_expand * quantile_expand  # (batch, n_quantiles, hidden)
        
        # Output quantile values
        quantile_values = self.output(combined)  # (batch, n_quantiles, action_dim)
        
        return quantile_values
    
    def quantile_huber_loss(
        self,
        quantiles: torch.Tensor,  # (batch, n_quantiles, action_dim)
        target_quantiles: torch.Tensor,  # (batch, n_quantiles, 1)
        actions: torch.Tensor,  # (batch,)
        kappa: float = 1.0
    ) -> torch.Tensor:
        # Get quantiles for selected actions
        batch_size = quantiles.shape[0]
        action_indices = actions.unsqueeze(1).unsqueeze(2).expand(-1, self.n_quantiles, -1)
        selected_quantiles = quantiles.gather(2, action_indices.long()).squeeze(-1)  # (batch, n_quantiles)
        
        # Compute pairwise TD errors
        td_errors = target_quantiles - selected_quantiles.unsqueeze(1)  # (batch, n_quantiles, n_quantiles)
        
        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors ** 2,
            kappa * (td_errors.abs() - 0.5 * kappa)
        )
        
        # Quantile loss
        tau_expand = self.tau.view(1, 1, -1)
        quantile_loss = torch.abs(tau_expand - (td_errors < 0).float()) * huber_loss
        
        return quantile_loss.sum(dim=2).mean()


class CVaRPortfolioRL(nn.Module):
    """Risk-sensitive RL using CVaR (Conditional Value at Risk) objective"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.05,  # CVaR level
        hidden_dim: int = 256
    ):
        super().__init__()
        self.alpha = alpha
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Value network for CVaR estimation
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # VaR threshold learner
        self.var_threshold = nn.Parameter(torch.zeros(1))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.policy(state)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        return mean, torch.exp(log_std)
    
    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action)
    
    def cvar_loss(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute CVaR loss for risk-sensitive optimization"""
        var = self.var_threshold
        
        # CVaR = VaR + E[(VaR - return)^+ ] / alpha
        excess_loss = F.relu(var - returns)
        cvar = var + excess_loss.mean() / self.alpha
        
        return cvar
    
    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        lambda_risk: float = 0.5
    ) -> torch.Tensor:
        # Expected return objective
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(torch.atanh(torch.clamp(actions, -0.999, 0.999))).sum(-1)
        
        expected_return_loss = -(log_prob * returns).mean()
        
        # CVaR penalty
        cvar_penalty = self.cvar_loss(returns)
        
        # Combined loss
        total_loss = (1 - lambda_risk) * expected_return_loss + lambda_risk * cvar_penalty
        
        return total_loss
```

### Model-Based RL with World Models

```python
class DreamerV3Finance(nn.Module):
    """Dreamer-style world model for trading simulation"""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        discrete_latent_size: int = 32,
        discrete_latent_classes: int = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.discrete_size = discrete_latent_size
        self.discrete_classes = discrete_latent_classes
        
        # Encoder: observation -> posterior latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, discrete_latent_size * discrete_latent_classes)
        )
        
        # Sequence model (RSSM style)
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        
        # Prior network
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, discrete_latent_size * discrete_latent_classes)
        )
        
        # Decoder: latent -> observation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 255)  # Symlog twohot encoding
        )
        
        # Continue predictor
        self.continue_predictor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode_posterior(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([obs, hidden], dim=-1)
        logits = self.encoder(combined)
        logits = logits.view(-1, self.discrete_size, self.discrete_classes)
        
        # Straight-through estimator
        probs = F.softmax(logits, dim=-1)
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.discrete_classes).float()
        latent = one_hot + probs - probs.detach()
        latent = latent.view(-1, self.discrete_size * self.discrete_classes)
        
        return latent[:, :self.latent_dim], logits
    
    def encode_prior(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.prior(hidden)
        logits = logits.view(-1, self.discrete_size, self.discrete_classes)
        
        probs = F.softmax(logits, dim=-1)
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.discrete_classes).float()
        latent = one_hot + probs - probs.detach()
        latent = latent.view(-1, self.discrete_size * self.discrete_classes)
        
        return latent[:, :self.latent_dim], logits
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor
    ) -> dict:
        # Get posterior latent
        latent_post, post_logits = self.encode_posterior(obs, hidden)
        
        # Update hidden state
        rnn_input = torch.cat([latent_post, action], dim=-1)
        next_hidden = self.rnn(rnn_input, hidden)
        
        # Get prior latent
        latent_prior, prior_logits = self.encode_prior(next_hidden)
        
        # Decode
        decode_input = torch.cat([next_hidden, latent_post], dim=-1)
        obs_pred = self.decoder(decode_input)
        reward_logits = self.reward_predictor(decode_input)
        continue_logits = self.continue_predictor(decode_input)
        
        return {
            'hidden': next_hidden,
            'latent_post': latent_post,
            'latent_prior': latent_prior,
            'post_logits': post_logits,
            'prior_logits': prior_logits,
            'obs_pred': obs_pred,
            'reward_logits': reward_logits,
            'continue_logits': continue_logits
        }
    
    def imagine(
        self,
        initial_hidden: torch.Tensor,
        initial_latent: torch.Tensor,
        policy: nn.Module,
        horizon: int
    ) -> dict:
        """Imagine trajectories using learned world model"""
        hidden = initial_hidden
        latent = initial_latent
        
        imagined_states = []
        imagined_rewards = []
        imagined_continues = []
        
        for _ in range(horizon):
            # Get action from policy
            state = torch.cat([hidden, latent], dim=-1)
            action = policy(state)
            
            # Step world model
            rnn_input = torch.cat([latent, action], dim=-1)
            hidden = self.rnn(rnn_input, hidden)
            latent, _ = self.encode_prior(hidden)
            
            # Predict reward and continue
            decode_input = torch.cat([hidden, latent], dim=-1)
            reward = self.reward_predictor(decode_input)
            cont = torch.sigmoid(self.continue_predictor(decode_input))
            
            imagined_states.append(state)
            imagined_rewards.append(reward)
            imagined_continues.append(cont)
        
        return {
            'states': torch.stack(imagined_states, dim=1),
            'rewards': torch.stack(imagined_rewards, dim=1),
            'continues': torch.stack(imagined_continues, dim=1)
        }
```

### Safe RL for Trading

```python
class SafeRLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        risk_threshold: float = 0.05
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.safety_critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.risk_threshold = risk_threshold
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.safety_optimizer = optim.Adam(self.safety_critic.parameters(), lr=0.001)
        
    def select_safe_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            risk_score = self.safety_critic(state_tensor).item()
        
        if risk_score > self.risk_threshold:
            action = 1
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        
        return action
```

## Implementation

### Complete RL Trading System

```python
class RLTradingSystem:
    def __init__(
        self,
        price_data: pd.DataFrame,
        agent_type: str = 'ppo',
        **agent_kwargs
    ):
        self.price_data = price_data
        
        self.env = TradingEnvironment(price_data)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        if agent_type == 'dqn':
            self.agent = DQNAgent(state_dim, action_dim, **agent_kwargs)
        elif agent_type == 'ppo':
            self.agent = PPOAgent(state_dim, action_dim, **agent_kwargs)
        elif agent_type == 'sac':
            self.agent = SACAgent(state_dim, action_dim, **agent_kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
    def train(self, n_episodes: int = 1000):
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if hasattr(self.agent, 'select_action'):
                    if isinstance(self.agent, PPOAgent):
                        action, log_prob, value = self.agent.select_action(state)
                    else:
                        action = self.agent.select_action(state)
                
                next_state, reward, done, info = self.env.step(action)
                
                if isinstance(self.agent, DQNAgent):
                    self.agent.store_transition(state, action, reward, next_state, done)
                    self.agent.train()
                    
                    if episode % 10 == 0:
                        self.agent.update_target_network()
                
                elif isinstance(self.agent, PPOAgent):
                    self.agent.store_transition(state, action, log_prob, value, reward, done)
                
                episode_reward += reward
                state = next_state
            
            if isinstance(self.agent, PPOAgent):
                self.agent.train()
            
            episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return episode_rewards
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        total_returns = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_return = 0
            done = False
            
            while not done:
                if hasattr(self.agent, 'select_action'):
                    action = self.agent.select_action(state, training=False)
                
                state, reward, done, info = self.env.step(action)
                episode_return += reward
            
            total_returns.append(episode_return)
        
        return {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'min_return': np.min(total_returns),
            'max_return': np.max(total_returns)
        }
```
