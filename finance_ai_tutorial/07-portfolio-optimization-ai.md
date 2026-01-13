# Module 7: Portfolio Optimization with AI

## Table of Contents
1. [Modern Portfolio Theory Extensions](#modern-portfolio-theory-extensions)
2. [Deep Reinforcement Learning for Portfolios](#deep-reinforcement-learning-for-portfolios)
3. [Hierarchical Risk Parity](#hierarchical-risk-parity)
4. [Multi-Objective Optimization](#multi-objective-optimization)
5. [Factor Investing with AI](#factor-investing-with-ai)
6. [Advanced Techniques](#advanced-techniques)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Modern Portfolio Theory Extensions

### Robust Portfolio Optimization

```python
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from typing import Tuple, Optional

class RobustPortfolioOptimizer:
    def __init__(self, returns: np.ndarray, risk_aversion: float = 1.0):
        self.returns = returns
        self.risk_aversion = risk_aversion
        self.n_assets = returns.shape[1]
        
    def mean_variance_optimization(
        self,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        mu = np.mean(self.returns, axis=0)
        Sigma = np.cov(self.returns.T)
        
        w = cp.Variable(self.n_assets)
        
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
    
    def robust_optimization(
        self,
        uncertainty_set: str = 'box',
        epsilon: float = 0.1
    ) -> np.ndarray:
        mu = np.mean(self.returns, axis=0)
        Sigma = np.cov(self.returns.T)
        
        w = cp.Variable(self.n_assets)
        
        if uncertainty_set == 'box':
            worst_case_return = mu @ w - epsilon * cp.norm(w, 1)
        elif uncertainty_set == 'ellipsoidal':
            worst_case_return = mu @ w - epsilon * cp.norm(Sigma @ w, 2)
        else:
            worst_case_return = mu @ w
        
        portfolio_variance = cp.quad_form(w, Sigma)
        
        objective = cp.Maximize(worst_case_return - self.risk_aversion * portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
    
    def black_litterman(
        self,
        market_weights: np.ndarray,
        views_matrix: np.ndarray,
        views_returns: np.ndarray,
        tau: float = 0.025,
        omega_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Sigma = np.cov(self.returns.T)
        
        market_return = np.mean(self.returns, axis=0)
        
        Pi = self.risk_aversion * Sigma @ market_weights
        
        if omega_diag is None:
            omega_diag = np.diag(np.diag(views_matrix @ (tau * Sigma) @ views_matrix.T))
        
        M_inv = np.linalg.inv(np.linalg.inv(tau * Sigma) + views_matrix.T @ np.linalg.inv(omega_diag) @ views_matrix)
        mu_bl = M_inv @ (np.linalg.inv(tau * Sigma) @ Pi + views_matrix.T @ np.linalg.inv(omega_diag) @ views_returns)
        
        Sigma_bl = Sigma + M_inv
        
        w = cp.Variable(self.n_assets)
        
        portfolio_return = mu_bl @ w
        portfolio_variance = cp.quad_form(w, Sigma_bl)
        
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
```

### Bayesian Portfolio Optimization

```python
import pymc3 as pm
from scipy import stats

class BayesianPortfolioOptimizer:
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.n_periods = returns.shape[0]
        
    def estimate_posterior_returns(
        self,
        n_samples: int = 2000,
        n_chains: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sd=0.1, shape=self.n_assets)
            
            packed_chol = pm.LKJCholeskyCov(
                'chol_cov',
                n=self.n_assets,
                eta=2,
                sd_dist=pm.Exponential.dist(1.0)
            )
            
            chol = pm.expand_packed_triangular(self.n_assets, packed_chol)
            
            returns_dist = pm.MvNormal(
                'returns',
                mu=mu,
                chol=chol,
                observed=self.returns
            )
            
            trace = pm.sample(n_samples, chains=n_chains, return_inferencedata=False)
        
        posterior_mu = trace['mu'].mean(axis=0)
        posterior_cov = np.cov(trace['mu'].T)
        
        return posterior_mu, posterior_cov
    
    def bayesian_portfolio_weights(
        self,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        posterior_mu, posterior_cov = self.estimate_posterior_returns()
        
        inv_cov = np.linalg.inv(posterior_cov)
        
        ones = np.ones(self.n_assets)
        
        numerator = inv_cov @ posterior_mu
        denominator = ones.T @ inv_cov @ posterior_mu
        
        weights = numerator / denominator
        
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()
        
        return weights
```

## Deep Reinforcement Learning for Portfolios

### Continuous Action Space Portfolio RL

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class PortfolioActor(nn.Module):
    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 256):
        super(PortfolioActor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_assets)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        
        weights = F.softmax(x, dim=-1)
        
        return weights


class PortfolioCritic(nn.Module):
    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 256):
        super(PortfolioCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + n_assets, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        
        return q_value


class TD3PortfolioAgent:
    def __init__(
        self,
        state_dim: int,
        n_assets: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2
    ):
        self.n_assets = n_assets
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.actor = PortfolioActor(state_dim, n_assets).to(self.device)
        self.actor_target = PortfolioActor(state_dim, n_assets).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1 = PortfolioCritic(state_dim, n_assets).to(self.device)
        self.critic1_target = PortfolioCritic(state_dim, n_assets).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = PortfolioCritic(state_dim, n_assets).to(self.device)
        self.critic2_target = PortfolioCritic(state_dim, n_assets).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        self.replay_buffer = deque(maxlen=100000)
        self.total_it = 0
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, 0.1, size=self.n_assets)
            action = action + noise
            action = np.abs(action)
            action = action / action.sum()
        
        return action
    
    def train(self, batch_size: int = 256):
        if len(self.replay_buffer) < batch_size:
            return
        
        self.total_it += 1
        
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state)
            next_action = next_action + noise
            next_action = torch.abs(next_action)
            next_action = next_action / next_action.sum(dim=-1, keepdim=True)
            
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        if self.total_it % self.policy_delay == 0:
            actor_action = self.actor(state)
            actor_loss = -self.critic1(state, actor_action).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### Transaction Cost Modeling

```python
class TransactionCostModel:
    def __init__(
        self,
        fixed_cost: float = 0.0001,
        proportional_cost: float = 0.0005,
        market_impact_coef: float = 0.01
    ):
        self.fixed_cost = fixed_cost
        self.proportional_cost = proportional_cost
        self.market_impact_coef = market_impact_coef
        
    def calculate_costs(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        volumes: np.ndarray
    ) -> float:
        trade_values = np.abs(target_weights - current_weights) * portfolio_value
        
        fixed_costs = np.sum(trade_values > 0) * self.fixed_cost * portfolio_value
        
        proportional_costs = np.sum(trade_values) * self.proportional_cost
        
        market_impact = self.market_impact_coef * np.sum(
            (trade_values ** 2) / (volumes * portfolio_value + 1e-6)
        )
        
        total_cost = fixed_costs + proportional_costs + market_impact
        
        return total_cost
```

## Hierarchical Risk Parity

### HRP Implementation

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.n_assets = returns.shape[1]
        
    def compute_correlation_matrix(self) -> np.ndarray:
        return np.corrcoef(self.returns.T)
    
    def compute_distance_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        return np.sqrt((1 - corr_matrix) / 2)
    
    def quasi_diagonalization(self, linkage_matrix: np.ndarray) -> list:
        sorted_indices = []
        
        def _recursive_bisection(cluster):
            if cluster < self.n_assets:
                sorted_indices.append(cluster)
            else:
                left = int(linkage_matrix[cluster - self.n_assets, 0])
                right = int(linkage_matrix[cluster - self.n_assets, 1])
                _recursive_bisection(left)
                _recursive_bisection(right)
        
        _recursive_bisection(2 * self.n_assets - 2)
        
        return sorted_indices
    
    def compute_cluster_variance(
        self,
        cov_matrix: np.ndarray,
        cluster_items: list
    ) -> float:
        cov_slice = cov_matrix[np.ix_(cluster_items, cluster_items)]
        
        inv_diag = 1 / np.diag(cov_slice)
        w = inv_diag / inv_diag.sum()
        
        cluster_var = np.dot(w, np.dot(cov_slice, w))
        
        return cluster_var
    
    def recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sorted_indices: list
    ) -> np.ndarray:
        weights = np.ones(self.n_assets)
        
        clusters = [sorted_indices]
        
        while len(clusters) > 0:
            clusters = [
                cluster[i:j]
                for cluster in clusters
                for i, j in [(0, len(cluster) // 2), (len(cluster) // 2, len(cluster))]
                if len(cluster) > 1
            ] + [cluster for cluster in clusters if len(cluster) == 1]
            
            if all(len(cluster) == 1 for cluster in clusters):
                break
            
            for i in range(0, len(clusters), 2):
                if i + 1 < len(clusters):
                    left_cluster = clusters[i]
                    right_cluster = clusters[i + 1]
                    
                    left_var = self.compute_cluster_variance(cov_matrix, left_cluster)
                    right_var = self.compute_cluster_variance(cov_matrix, right_cluster)
                    
                    alpha = 1 - left_var / (left_var + right_var)
                    
                    weights[left_cluster] *= alpha
                    weights[right_cluster] *= (1 - alpha)
        
        return weights
    
    def optimize(self) -> np.ndarray:
        corr_matrix = self.compute_correlation_matrix()
        
        dist_matrix = self.compute_distance_matrix(corr_matrix)
        
        linkage_matrix = linkage(squareform(dist_matrix), method='single')
        
        sorted_indices = self.quasi_diagonalization(linkage_matrix)
        
        cov_matrix = np.cov(self.returns.T)
        
        weights = self.recursive_bisection(cov_matrix, sorted_indices)
        
        return weights / weights.sum()
```

### Machine Learning Enhanced HRP

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class MLEnhancedHRP:
    def __init__(self, returns: np.ndarray, features: Optional[np.ndarray] = None):
        self.returns = returns
        self.features = features
        self.hrp = HierarchicalRiskParity(returns)
        
    def predict_covariance(
        self,
        lookback: int = 252
    ) -> np.ndarray:
        if self.features is None:
            return np.cov(self.returns[-lookback:].T)
        
        models = {}
        scaler = StandardScaler()
        
        X = scaler.fit_transform(self.features[:-1])
        
        for i in range(self.returns.shape[1]):
            for j in range(i, self.returns.shape[1]):
                y = self.returns[1:, i] * self.returns[1:, j]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X[-lookback:], y[-lookback:])
                
                models[(i, j)] = model
        
        predicted_cov = np.zeros((self.returns.shape[1], self.returns.shape[1]))
        
        current_features = scaler.transform(self.features[-1:])
        
        for (i, j), model in models.items():
            pred = model.predict(current_features)[0]
            predicted_cov[i, j] = pred
            if i != j:
                predicted_cov[j, i] = pred
        
        return predicted_cov
    
    def dynamic_hrp(self) -> np.ndarray:
        predicted_cov = self.predict_covariance()
        
        corr_matrix = predicted_cov / np.outer(np.sqrt(np.diag(predicted_cov)), np.sqrt(np.diag(predicted_cov)))
        
        dist_matrix = self.hrp.compute_distance_matrix(corr_matrix)
        
        linkage_matrix = linkage(squareform(dist_matrix), method='single')
        
        sorted_indices = self.hrp.quasi_diagonalization(linkage_matrix)
        
        weights = self.hrp.recursive_bisection(predicted_cov, sorted_indices)
        
        return weights / weights.sum()
```

## Multi-Objective Optimization

### Pareto-Optimal Portfolio Frontier

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as moo_minimize

class PortfolioProblem(Problem):
    def __init__(self, returns: np.ndarray, esg_scores: Optional[np.ndarray] = None):
        self.returns = returns
        self.esg_scores = esg_scores
        self.n_assets = returns.shape[1]
        
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns.T)
        
        n_obj = 3 if esg_scores is not None else 2
        
        super().__init__(
            n_var=self.n_assets,
            n_obj=n_obj,
            n_constr=1,
            xl=0.0,
            xu=1.0
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        portfolio_returns = X @ self.mean_returns
        
        portfolio_risks = np.sqrt(np.sum((X @ self.cov_matrix) * X, axis=1))
        
        objectives = [
            -portfolio_returns,
            portfolio_risks
        ]
        
        if self.esg_scores is not None:
            esg_performance = X @ self.esg_scores
            objectives.append(-esg_performance)
        
        out["F"] = np.column_stack(objectives)
        
        out["G"] = np.abs(X.sum(axis=1) - 1.0)


class MultiObjectivePortfolioOptimizer:
    def __init__(
        self,
        returns: np.ndarray,
        esg_scores: Optional[np.ndarray] = None
    ):
        self.returns = returns
        self.esg_scores = esg_scores
        
    def optimize(self, pop_size: int = 100, n_gen: int = 200) -> np.ndarray:
        problem = PortfolioProblem(self.returns, self.esg_scores)
        
        algorithm = NSGA2(pop_size=pop_size)
        
        res = moo_minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=False
        )
        
        return res.X
    
    def select_portfolio_from_pareto(
        self,
        pareto_front: np.ndarray,
        risk_preference: float = 0.5,
        esg_preference: float = 0.0
    ) -> np.ndarray:
        returns = -pareto_front[:, 0]
        risks = pareto_front[:, 1]
        
        normalized_returns = (returns - returns.min()) / (returns.max() - returns.min())
        normalized_risks = (risks - risks.min()) / (risks.max() - risks.min())
        
        scores = (
            (1 - risk_preference) * normalized_returns -
            risk_preference * normalized_risks
        )
        
        if self.esg_scores is not None and pareto_front.shape[1] > 2:
            esg = -pareto_front[:, 2]
            normalized_esg = (esg - esg.min()) / (esg.max() - esg.min())
            scores += esg_preference * normalized_esg
        
        best_idx = np.argmax(scores)
        
        return pareto_front[best_idx]
```

## Factor Investing with AI

### ML-based Factor Discovery

```python
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans

class MLFactorDiscovery:
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.n_assets = returns.shape[1]
        
    def pca_factors(self, n_factors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components=n_factors)
        
        factors = pca.fit_transform(self.returns)
        
        loadings = pca.components_.T
        
        return factors, loadings
    
    def ica_factors(self, n_factors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        ica = FastICA(n_components=n_factors, random_state=42)
        
        factors = ica.fit_transform(self.returns)
        
        loadings = ica.mixing_
        
        return factors, loadings
    
    def autoencoder_factors(
        self,
        n_factors: int = 5,
        hidden_dim: int = 64,
        n_epochs: int = 100
    ) -> Tuple[np.ndarray, torch.nn.Module]:
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dim):
                super(Autoencoder, self).__init__()
                
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim)
                )
                
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x):
                latent = self.encoder(x)
                reconstructed = self.decoder(latent)
                return reconstructed, latent
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Autoencoder(self.n_assets, n_factors, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X = torch.FloatTensor(self.returns).to(device)
        
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            reconstructed, latent = model(X)
            loss = criterion(reconstructed, X)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            _, factors = model(X)
            factors = factors.cpu().numpy()
        
        return factors, model
    
    def dynamic_factor_portfolio(
        self,
        n_factors: int = 5,
        method: str = 'pca'
    ) -> np.ndarray:
        if method == 'pca':
            factors, loadings = self.pca_factors(n_factors)
        elif method == 'ica':
            factors, loadings = self.ica_factors(n_factors)
        else:
            factors, _ = self.autoencoder_factors(n_factors)
            loadings = np.eye(self.n_assets, n_factors)
        
        factor_returns = np.mean(factors[-20:], axis=0)
        
        factor_weights = np.abs(factor_returns) / np.sum(np.abs(factor_returns))
        
        asset_weights = loadings @ factor_weights
        
        asset_weights = np.abs(asset_weights)
        asset_weights = asset_weights / asset_weights.sum()
        
        return asset_weights
```

## Advanced Techniques

### Distributionally Robust Optimization

```python
class DistributionallyRobustOptimizer:
    def __init__(self, returns: np.ndarray, epsilon: float = 0.1):
        self.returns = returns
        self.epsilon = epsilon
        self.n_assets = returns.shape[1]
        
    def wasserstein_dro(self) -> np.ndarray:
        mu = np.mean(self.returns, axis=0)
        Sigma = np.cov(self.returns.T)
        
        w = cp.Variable(self.n_assets)
        kappa = cp.Variable()
        
        portfolio_return = mu @ w
        portfolio_std = cp.norm(Sigma @ w, 2)
        
        worst_case_return = portfolio_return - self.epsilon * portfolio_std - kappa
        
        objective = cp.Maximize(worst_case_return)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            kappa >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
    
    def cvar_optimization(
        self,
        alpha: float = 0.05,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        n_samples = self.returns.shape[0]
        
        w = cp.Variable(self.n_assets)
        zeta = cp.Variable()
        s = cp.Variable(n_samples)
        
        portfolio_returns = self.returns @ w
        
        cvar = zeta + (1 / (alpha * n_samples)) * cp.sum(s)
        
        objective = cp.Minimize(cvar)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            s >= 0,
            s >= -portfolio_returns - zeta
        ]
        
        if target_return is not None:
            mu = np.mean(self.returns, axis=0)
            constraints.append(mu @ w >= target_return)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
```

### Online Portfolio Selection

```python
class OnlinePortfolioSelection:
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
        self.cumulative_return = 1.0
        
    def update_follow_the_winner(
        self,
        returns: np.ndarray,
        eta: float = 0.5
    ) -> np.ndarray:
        wealth_factors = 1 + returns
        
        gradients = wealth_factors / (self.weights @ wealth_factors)
        
        self.weights = self.weights * np.exp(eta * gradients)
        
        self.weights = self.weights / self.weights.sum()
        
        self.cumulative_return *= (self.weights @ wealth_factors)
        
        return self.weights
    
    def update_exponential_gradient(
        self,
        returns: np.ndarray,
        eta: float = 0.05
    ) -> np.ndarray:
        wealth_factors = 1 + returns
        
        self.weights = self.weights * np.exp(eta * returns)
        
        self.weights = self.weights / self.weights.sum()
        
        self.cumulative_return *= (self.weights @ wealth_factors)
        
        return self.weights
    
    def update_online_newton_step(
        self,
        returns: np.ndarray,
        beta: float = 1.0,
        delta: float = 0.125
    ) -> np.ndarray:
        grad = returns
        
        A = np.outer(returns, returns)
        
        if not hasattr(self, 'A_sum'):
            self.A_sum = np.eye(self.n_assets) * delta
        
        self.A_sum += A
        
        A_inv = np.linalg.inv(self.A_sum)
        
        self.weights = self.weights + beta * A_inv @ grad
        
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum()
        
        return self.weights
```

## PhD-Level Research Topics

### Quantum Portfolio Optimization

```python
try:
    from qiskit import Aer, QuantumCircuit
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit_optimization.applications import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    
    class QuantumPortfolioOptimizer:
        def __init__(self, returns: np.ndarray, budget: int):
            self.returns = returns
            self.budget = budget
            self.n_assets = returns.shape[1]
            
        def optimize_qaoa(self, p: int = 1) -> np.ndarray:
            mu = np.mean(self.returns, axis=0)
            Sigma = np.cov(self.returns.T)
            
            portfolio = PortfolioOptimization(
                expected_returns=mu,
                covariances=Sigma,
                risk_factor=0.5,
                budget=self.budget
            )
            
            qp = portfolio.to_quadratic_program()
            
            qubo = QuadraticProgramToQubo().convert(qp)
            
            qaoa = QAOA(optimizer=COBYLA(), reps=p, quantum_instance=Aer.get_backend('qasm_simulator'))
            
            result = qaoa.compute_minimum_eigenvalue(qubo)
            
            return result.x
            
except ImportError:
    pass
```

### Mean-Variance-CVaR Optimization

```python
class MeanVarianceCVaROptimizer:
    def __init__(
        self,
        returns: np.ndarray,
        alpha: float = 0.05,
        lambda_var: float = 0.5,
        lambda_cvar: float = 0.5
    ):
        self.returns = returns
        self.alpha = alpha
        self.lambda_var = lambda_var
        self.lambda_cvar = lambda_cvar
        self.n_assets = returns.shape[1]
        self.n_samples = returns.shape[0]
        
    def optimize(self) -> np.ndarray:
        mu = np.mean(self.returns, axis=0)
        Sigma = np.cov(self.returns.T)
        
        w = cp.Variable(self.n_assets)
        zeta = cp.Variable()
        s = cp.Variable(self.n_samples)
        
        portfolio_return = mu @ w
        
        portfolio_variance = cp.quad_form(w, Sigma)
        
        portfolio_returns = self.returns @ w
        cvar = zeta + (1 / (self.alpha * self.n_samples)) * cp.sum(s)
        
        objective = cp.Maximize(
            portfolio_return -
            self.lambda_var * portfolio_variance -
            self.lambda_cvar * cvar
        )
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            s >= 0,
            s >= -portfolio_returns - zeta
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
```

## Practical Implementation

### Complete Portfolio Management System

```python
class PortfolioManagementSystem:
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.n_assets = returns.shape[1]
        
        self.robust_optimizer = RobustPortfolioOptimizer(returns)
        self.hrp = HierarchicalRiskParity(returns)
        self.transaction_cost_model = TransactionCostModel()
        
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1000000.0
        
    def rebalance(
        self,
        method: str = 'robust',
        target_return: Optional[float] = None
    ) -> Dict[str, any]:
        if method == 'robust':
            target_weights = self.robust_optimizer.robust_optimization()
        elif method == 'hrp':
            target_weights = self.hrp.optimize()
        elif method == 'mean_variance':
            target_weights = self.robust_optimizer.mean_variance_optimization(target_return)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        volumes = np.ones(self.n_assets) * 1e6
        transaction_costs = self.transaction_cost_model.calculate_costs(
            self.current_weights,
            target_weights,
            self.portfolio_value,
            volumes
        )
        
        net_target_weights = target_weights
        
        self.current_weights = net_target_weights
        self.portfolio_value -= transaction_costs
        
        return {
            'target_weights': target_weights,
            'actual_weights': net_target_weights,
            'transaction_costs': transaction_costs,
            'portfolio_value': self.portfolio_value
        }
    
    def evaluate_performance(
        self,
        realized_returns: np.ndarray
    ) -> Dict[str, float]:
        portfolio_returns = realized_returns @ self.current_weights
        
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-6) * np.sqrt(252)
        
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annual_return': np.mean(portfolio_returns) * 252,
            'annual_volatility': np.std(portfolio_returns) * np.sqrt(252),
            'max_drawdown': max_drawdown
        }
```
