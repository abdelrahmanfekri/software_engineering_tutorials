# Module 11: Execution Algorithms and Market Microstructure

## Table of Contents
1. [Optimal Execution Theory](#optimal-execution-theory)
2. [Standard Execution Algorithms](#standard-execution-algorithms)
3. [Market Microstructure](#market-microstructure)
4. [Advanced Execution Strategies](#advanced-execution-strategies)
5. [Machine Learning for Execution](#machine-learning-for-execution)
6. [High-Frequency Microstructure](#high-frequency-microstructure)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Optimal Execution Theory

### Almgren-Chriss Framework

```python
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List

class AlmgrenChrissModel:
    def __init__(
        self,
        volatility: float,
        eta: float,
        gamma: float,
        lambda_param: float = 0.01
    ):
        self.sigma = volatility
        self.eta = eta
        self.gamma = gamma
        self.lambda_param = lambda_param
        
    def temporary_impact(self, trade_rate: float) -> float:
        return self.eta * trade_rate
    
    def permanent_impact(self, trade_rate: float) -> float:
        return self.gamma * trade_rate
    
    def optimal_trajectory(
        self,
        total_shares: int,
        total_time: float,
        num_steps: int
    ) -> np.ndarray:
        dt = total_time / num_steps
        
        kappa = np.sqrt(self.lambda_param * self.sigma**2 / self.eta)
        
        tau = np.arange(0, num_steps + 1) * dt
        
        sinh_kappa_t = np.sinh(kappa * (total_time - tau))
        sinh_kappa_T = np.sinh(kappa * total_time)
        
        trajectory = total_shares * sinh_kappa_t / sinh_kappa_T
        
        return trajectory
    
    def calculate_expected_cost(
        self,
        trajectory: np.ndarray,
        total_time: float
    ) -> float:
        num_steps = len(trajectory) - 1
        dt = total_time / num_steps
        
        trade_list = -np.diff(trajectory)
        
        temporary_cost = np.sum(self.eta * (trade_list / dt) * trade_list)
        
        permanent_cost = self.gamma * np.sum(
            (trajectory[:-1] - np.cumsum(trade_list) / 2) * trade_list
        )
        
        risk_cost = (self.lambda_param * self.sigma**2 / 2) * np.sum(
            trajectory[:-1]**2 * dt
        )
        
        total_cost = temporary_cost + permanent_cost + risk_cost
        
        return total_cost
    
    def calculate_implementation_shortfall(
        self,
        execution_prices: np.ndarray,
        quantities: np.ndarray,
        arrival_price: float
    ) -> Dict[str, float]:
        vwap = np.sum(execution_prices * quantities) / np.sum(quantities)
        
        implementation_shortfall = (vwap - arrival_price) / arrival_price
        
        total_cost = np.sum((execution_prices - arrival_price) * quantities)
        
        return {
            'implementation_shortfall': implementation_shortfall,
            'total_cost': total_cost,
            'vwap': vwap
        }
```

### Optimal Execution with Stochastic Control

```python
class StochasticOptimalExecution:
    def __init__(
        self,
        volatility: float,
        alpha: float,
        kappa: float,
        gamma: float
    ):
        self.sigma = volatility
        self.alpha = alpha
        self.kappa = kappa
        self.gamma = gamma
        
    def value_function(
        self,
        inventory: float,
        time_remaining: float
    ) -> float:
        q = inventory
        t = time_remaining
        
        theta = self.kappa * np.sqrt(1 + (self.gamma / self.kappa)**2)
        
        sinh_theta_t = np.sinh(theta * t)
        cosh_theta_t = np.cosh(theta * t)
        
        value = -(self.alpha * q**2 * cosh_theta_t) / (2 * sinh_theta_t)
        
        return value
    
    def optimal_trading_rate(
        self,
        inventory: float,
        time_remaining: float
    ) -> float:
        q = inventory
        t = time_remaining
        
        theta = self.kappa * np.sqrt(1 + (self.gamma / self.kappa)**2)
        
        sinh_theta_t = np.sinh(theta * t)
        cosh_theta_t = np.cosh(theta * t)
        
        rate = (theta * q * cosh_theta_t) / sinh_theta_t
        
        return rate
    
    def simulate_execution(
        self,
        initial_inventory: int,
        total_time: float,
        num_steps: int,
        price_process: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = total_time / num_steps
        
        if price_process is None:
            price_process = 100 + self.sigma * np.sqrt(dt) * np.cumsum(np.random.randn(num_steps + 1))
        
        inventory = np.zeros(num_steps + 1)
        inventory[0] = initial_inventory
        
        trading_rates = np.zeros(num_steps)
        execution_costs = np.zeros(num_steps)
        
        for i in range(num_steps):
            time_remaining = total_time - i * dt
            
            rate = self.optimal_trading_rate(inventory[i], time_remaining)
            
            shares_to_trade = min(rate * dt, inventory[i])
            
            trading_rates[i] = shares_to_trade / dt
            
            price_impact = self.alpha * trading_rates[i]
            execution_price = price_process[i] + price_impact
            
            execution_costs[i] = shares_to_trade * execution_price
            
            inventory[i + 1] = inventory[i] - shares_to_trade
        
        return inventory, trading_rates, execution_costs
```

## Standard Execution Algorithms

### VWAP (Volume-Weighted Average Price)

```python
class VWAPExecutionAlgorithm:
    def __init__(self, target_shares: int, total_time: float):
        self.target_shares = target_shares
        self.total_time = total_time
        self.executed_shares = 0
        self.executed_value = 0
        
    def calculate_target_schedule(
        self,
        historical_volume_profile: np.ndarray,
        num_intervals: int
    ) -> np.ndarray:
        normalized_profile = historical_volume_profile / np.sum(historical_volume_profile)
        
        target_schedule = self.target_shares * normalized_profile
        
        return target_schedule
    
    def execute_interval(
        self,
        interval_target: int,
        current_price: float,
        current_volume: int,
        max_participation: float = 0.10
    ) -> Tuple[int, float]:
        max_shares = int(current_volume * max_participation)
        
        shares_to_execute = min(interval_target, max_shares, self.target_shares - self.executed_shares)
        
        if shares_to_execute > 0:
            self.executed_shares += shares_to_execute
            self.executed_value += shares_to_execute * current_price
        
        return shares_to_execute, current_price
    
    def calculate_vwap(self) -> float:
        if self.executed_shares == 0:
            return 0.0
        
        return self.executed_value / self.executed_shares
    
    def simulate_execution(
        self,
        price_series: np.ndarray,
        volume_series: np.ndarray,
        volume_profile: np.ndarray
    ) -> pd.DataFrame:
        num_intervals = len(price_series)
        
        target_schedule = self.calculate_target_schedule(volume_profile, num_intervals)
        
        results = pd.DataFrame({
            'price': price_series,
            'volume': volume_series,
            'target': target_schedule,
            'executed': 0,
            'remaining': self.target_shares
        })
        
        for i in range(num_intervals):
            shares, price = self.execute_interval(
                int(target_schedule[i]),
                price_series[i],
                volume_series[i]
            )
            
            results.loc[i, 'executed'] = shares
            results.loc[i, 'remaining'] = self.target_shares - self.executed_shares
        
        results['cumulative_executed'] = results['executed'].cumsum()
        results['vwap'] = self.calculate_vwap()
        
        return results
```

### TWAP (Time-Weighted Average Price)

```python
class TWAPExecutionAlgorithm:
    def __init__(self, target_shares: int, total_time: float, num_intervals: int):
        self.target_shares = target_shares
        self.total_time = total_time
        self.num_intervals = num_intervals
        self.shares_per_interval = target_shares / num_intervals
        
        self.executed_shares = 0
        self.executed_value = 0
        
    def execute_interval(
        self,
        current_price: float,
        randomize: bool = False,
        randomization_factor: float = 0.2
    ) -> Tuple[int, float]:
        shares_to_execute = self.shares_per_interval
        
        if randomize:
            variation = np.random.uniform(
                1 - randomization_factor,
                1 + randomization_factor
            )
            shares_to_execute *= variation
        
        shares_to_execute = int(min(
            shares_to_execute,
            self.target_shares - self.executed_shares
        ))
        
        if shares_to_execute > 0:
            self.executed_shares += shares_to_execute
            self.executed_value += shares_to_execute * current_price
        
        return shares_to_execute, current_price
    
    def calculate_twap(self) -> float:
        if self.executed_shares == 0:
            return 0.0
        
        return self.executed_value / self.executed_shares
```

### POV (Percentage of Volume)

```python
class POVExecutionAlgorithm:
    def __init__(
        self,
        target_shares: int,
        target_participation: float = 0.10,
        min_participation: float = 0.05,
        max_participation: float = 0.20
    ):
        self.target_shares = target_shares
        self.target_participation = target_participation
        self.min_participation = min_participation
        self.max_participation = max_participation
        
        self.executed_shares = 0
        self.executed_value = 0
        
    def calculate_adaptive_participation(
        self,
        remaining_shares: int,
        remaining_time: float,
        expected_volume: int
    ) -> float:
        required_rate = remaining_shares / remaining_time if remaining_time > 0 else float('inf')
        
        expected_rate = expected_volume / remaining_time if remaining_time > 0 else 0
        
        if expected_rate == 0:
            participation = self.max_participation
        else:
            participation = required_rate / expected_rate
        
        participation = np.clip(
            participation,
            self.min_participation,
            self.max_participation
        )
        
        return participation
    
    def execute_interval(
        self,
        current_price: float,
        current_volume: int,
        participation_rate: Optional[float] = None
    ) -> Tuple[int, float]:
        if participation_rate is None:
            participation_rate = self.target_participation
        
        shares_to_execute = int(current_volume * participation_rate)
        
        shares_to_execute = min(
            shares_to_execute,
            self.target_shares - self.executed_shares
        )
        
        if shares_to_execute > 0:
            self.executed_shares += shares_to_execute
            self.executed_value += shares_to_execute * current_price
        
        return shares_to_execute, current_price
```

## Market Microstructure

### Limit Order Book Dynamics

```python
class LimitOrderBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        
        self.bid_prices = []
        self.ask_prices = []
        
    def add_order(
        self,
        side: str,
        price: float,
        quantity: int,
        order_id: str
    ):
        if side == 'bid':
            if price not in self.bids:
                self.bids[price] = {}
            self.bids[price][order_id] = quantity
            self._update_bid_prices()
        else:
            if price not in self.asks:
                self.asks[price] = {}
            self.asks[price][order_id] = quantity
            self._update_ask_prices()
    
    def cancel_order(self, side: str, price: float, order_id: str):
        if side == 'bid' and price in self.bids:
            if order_id in self.bids[price]:
                del self.bids[price][order_id]
                if not self.bids[price]:
                    del self.bids[price]
                self._update_bid_prices()
        elif side == 'ask' and price in self.asks:
            if order_id in self.asks[price]:
                del self.asks[price][order_id]
                if not self.asks[price]:
                    del self.asks[price]
                self._update_ask_prices()
    
    def _update_bid_prices(self):
        self.bid_prices = sorted(self.bids.keys(), reverse=True)
    
    def _update_ask_prices(self):
        self.ask_prices = sorted(self.asks.keys())
    
    def get_best_bid(self) -> Optional[float]:
        return self.bid_prices[0] if self.bid_prices else None
    
    def get_best_ask(self) -> Optional[float]:
        return self.ask_prices[0] if self.ask_prices else None
    
    def get_mid_price(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        
        return None
    
    def get_spread(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        
        return None
    
    def get_depth(self, side: str, levels: int = 5) -> List[Tuple[float, int]]:
        if side == 'bid':
            prices = self.bid_prices[:levels]
            return [(price, sum(self.bids[price].values())) for price in prices]
        else:
            prices = self.ask_prices[:levels]
            return [(price, sum(self.asks[price].values())) for price in prices]
    
    def match_market_order(
        self,
        side: str,
        quantity: int
    ) -> List[Tuple[float, int]]:
        executions = []
        remaining = quantity
        
        if side == 'buy':
            prices = self.ask_prices.copy()
            book = self.asks
        else:
            prices = self.bid_prices.copy()
            book = self.bids
        
        for price in prices:
            if remaining == 0:
                break
            
            level_quantity = sum(book[price].values())
            executed = min(remaining, level_quantity)
            
            executions.append((price, executed))
            remaining -= executed
            
            if executed == level_quantity:
                del book[price]
            else:
                total = sum(book[price].values())
                for order_id in list(book[price].keys()):
                    order_qty = book[price][order_id]
                    remove_qty = int(order_qty * executed / total)
                    book[price][order_id] -= remove_qty
                    if book[price][order_id] <= 0:
                        del book[price][order_id]
        
        if side == 'buy':
            self._update_ask_prices()
        else:
            self._update_bid_prices()
        
        return executions
```

### Kyle's Lambda (Market Impact)

```python
class KyleLambdaEstimator:
    def __init__(self):
        self.lambda_estimate = None
        
    def estimate_kyle_lambda(
        self,
        order_flow: np.ndarray,
        price_changes: np.ndarray
    ) -> float:
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(order_flow.reshape(-1, 1), price_changes)
        
        self.lambda_estimate = model.coef_[0]
        
        return self.lambda_estimate
    
    def calculate_signed_volume(
        self,
        trades: pd.DataFrame
    ) -> np.ndarray:
        signed_volume = trades['volume'].values * np.sign(trades['price'].diff().values)
        
        return signed_volume
    
    def calculate_price_impact(
        self,
        order_size: int,
        side: str
    ) -> float:
        if self.lambda_estimate is None:
            raise ValueError("Lambda not estimated. Call estimate_kyle_lambda first.")
        
        signed_order = order_size if side == 'buy' else -order_size
        
        impact = self.lambda_estimate * signed_order
        
        return impact
```

## Advanced Execution Strategies

### Adaptive Execution with RL

```python
import torch
import torch.nn as nn

class AdaptiveExecutionAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
    def select_execution_rate(
        self,
        state: np.ndarray
    ) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        execution_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        return execution_rates[action.item()]
```

### Smart Order Routing

```python
class SmartOrderRouter:
    def __init__(self, venues: List[str]):
        self.venues = venues
        self.venue_characteristics = {}
        
    def update_venue_stats(
        self,
        venue: str,
        liquidity: float,
        fee: float,
        latency: float,
        fill_rate: float
    ):
        self.venue_characteristics[venue] = {
            'liquidity': liquidity,
            'fee': fee,
            'latency': latency,
            'fill_rate': fill_rate
        }
    
    def calculate_venue_score(
        self,
        venue: str,
        order_size: int,
        urgency: float
    ) -> float:
        if venue not in self.venue_characteristics:
            return 0.0
        
        stats = self.venue_characteristics[venue]
        
        liquidity_score = min(stats['liquidity'] / order_size, 1.0)
        
        fee_score = 1.0 / (1.0 + stats['fee'])
        
        latency_score = 1.0 / (1.0 + stats['latency'])
        
        fill_rate_score = stats['fill_rate']
        
        score = (
            0.4 * liquidity_score +
            0.2 * fee_score +
            0.2 * latency_score * urgency +
            0.2 * fill_rate_score
        )
        
        return score
    
    def route_order(
        self,
        order_size: int,
        urgency: float = 0.5
    ) -> Dict[str, int]:
        venue_scores = {
            venue: self.calculate_venue_score(venue, order_size, urgency)
            for venue in self.venues
        }
        
        total_score = sum(venue_scores.values())
        
        if total_score == 0:
            return {venue: order_size // len(self.venues) for venue in self.venues}
        
        allocation = {
            venue: int(order_size * score / total_score)
            for venue, score in venue_scores.items()
        }
        
        remaining = order_size - sum(allocation.values())
        if remaining > 0:
            best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
            allocation[best_venue] += remaining
        
        return allocation
```

## Machine Learning for Execution

### Fill Rate Prediction

```python
from sklearn.ensemble import GradientBoostingRegressor

class FillRatePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
    def engineer_features(
        self,
        order_data: pd.DataFrame
    ) -> pd.DataFrame:
        features = pd.DataFrame()
        
        features['order_size_ratio'] = order_data['order_size'] / order_data['market_volume']
        
        features['spread_bps'] = (
            (order_data['ask_price'] - order_data['bid_price']) /
            order_data['mid_price'] * 10000
        )
        
        features['volatility'] = order_data['price'].pct_change().rolling(20).std()
        
        features['order_imbalance'] = (
            (order_data['bid_volume'] - order_data['ask_volume']) /
            (order_data['bid_volume'] + order_data['ask_volume'])
        )
        
        features['time_of_day'] = pd.to_datetime(order_data['timestamp']).dt.hour
        
        return features.fillna(0)
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    
    def predict_fill_rate(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(X)
        
        return np.clip(predictions, 0, 1)
```

### Market Impact Prediction

```python
class MarketImpactPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
    def calculate_permanent_impact(
        self,
        pre_trade_price: float,
        post_trade_price: float,
        mid_price_before: float
    ) -> float:
        permanent_impact = (post_trade_price - mid_price_before) / mid_price_before
        
        return permanent_impact
    
    def calculate_temporary_impact(
        self,
        execution_price: float,
        mid_price: float
    ) -> float:
        temporary_impact = (execution_price - mid_price) / mid_price
        
        return temporary_impact
    
    def engineer_features(
        self,
        trade_data: pd.DataFrame
    ) -> pd.DataFrame:
        features = pd.DataFrame()
        
        features['order_size_pct'] = trade_data['order_size'] / trade_data['adv']
        
        features['volatility'] = trade_data['returns'].rolling(20).std()
        
        features['relative_spread'] = trade_data['spread'] / trade_data['mid_price']
        
        features['order_imbalance'] = (
            (trade_data['bid_depth'] - trade_data['ask_depth']) /
            (trade_data['bid_depth'] + trade_data['ask_depth'])
        )
        
        return features.fillna(0)
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    
    def predict_impact(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
```

## High-Frequency Microstructure

### Queue Position Modeling

```python
class QueuePositionModel:
    def __init__(self):
        self.queue_positions = {}
        
    def estimate_queue_position(
        self,
        order_id: str,
        price_level: float,
        submission_time: float,
        order_book: LimitOrderBook
    ) -> int:
        if price_level not in order_book.bids and price_level not in order_book.asks:
            return 0
        
        orders_at_level = order_book.bids.get(price_level, {})
        
        orders_before = sum(
            1 for oid, qty in orders_at_level.items()
            if oid < order_id
        )
        
        return orders_before
    
    def calculate_fill_probability(
        self,
        queue_position: int,
        total_depth: int,
        expected_volume: int
    ) -> float:
        if total_depth == 0:
            return 0.0
        
        position_ratio = queue_position / total_depth
        
        volume_ratio = expected_volume / total_depth
        
        fill_prob = max(0, volume_ratio - position_ratio)
        
        return min(fill_prob, 1.0)
```

### Latency Arbitrage Detection

```python
class LatencyArbitrageDetector:
    def __init__(self, threshold_ms: float = 1.0):
        self.threshold_ms = threshold_ms
        
    def detect_quote_stuffing(
        self,
        order_events: pd.DataFrame,
        window_seconds: float = 1.0
    ) -> pd.Series:
        order_events['timestamp'] = pd.to_datetime(order_events['timestamp'])
        
        order_rate = order_events.set_index('timestamp').resample(f'{window_seconds}S').size()
        
        mean_rate = order_rate.mean()
        std_rate = order_rate.std()
        
        threshold = mean_rate + 3 * std_rate
        
        stuffing_periods = order_rate > threshold
        
        return stuffing_periods
    
    def detect_layering(
        self,
        order_book_snapshots: pd.DataFrame
    ) -> List[Dict]:
        layering_events = []
        
        for i in range(1, len(order_book_snapshots)):
            prev_snapshot = order_book_snapshots.iloc[i-1]
            curr_snapshot = order_book_snapshots.iloc[i]
            
            depth_increase = curr_snapshot['bid_depth'] > prev_snapshot['bid_depth'] * 1.5
            
            price_moved = curr_snapshot['ask_price'] < prev_snapshot['ask_price']
            
            if depth_increase and price_moved:
                layering_events.append({
                    'timestamp': curr_snapshot['timestamp'],
                    'type': 'potential_layering',
                    'depth_change': curr_snapshot['bid_depth'] - prev_snapshot['bid_depth']
                })
        
        return layering_events
```

## PhD-Level Research Topics

### Optimal Execution with Ambiguity

```python
class AmbiguityAverseExecution:
    def __init__(self, risk_aversion: float, ambiguity_aversion: float):
        self.gamma = risk_aversion
        self.kappa = ambiguity_aversion
        
    def robust_optimal_trajectory(
        self,
        total_shares: int,
        total_time: float,
        num_steps: int,
        volatility_range: Tuple[float, float]
    ) -> np.ndarray:
        sigma_min, sigma_max = volatility_range
        
        worst_case_sigma = sigma_max
        
        dt = total_time / num_steps
        
        eta = 0.01
        
        kappa = np.sqrt(self.gamma * worst_case_sigma**2 / eta)
        
        tau = np.arange(0, num_steps + 1) * dt
        
        sinh_kappa_t = np.sinh(kappa * (total_time - tau))
        sinh_kappa_T = np.sinh(kappa * total_time)
        
        trajectory = total_shares * sinh_kappa_t / sinh_kappa_T
        
        return trajectory
```

## Implementation

### Complete Execution System

```python
class ExecutionManagementSystem:
    def __init__(self):
        self.almgren_chriss = AlmgrenChrissModel(
            volatility=0.02,
            eta=0.01,
            gamma=0.001
        )
        
        self.vwap_algo = None
        self.twap_algo = None
        self.pov_algo = None
        
        self.order_book = LimitOrderBook()
        self.smart_router = SmartOrderRouter(['NYSE', 'NASDAQ', 'BATS'])
        
    def execute_order(
        self,
        symbol: str,
        total_shares: int,
        algorithm: str,
        **kwargs
    ) -> pd.DataFrame:
        if algorithm == 'vwap':
            self.vwap_algo = VWAPExecutionAlgorithm(total_shares, kwargs.get('total_time', 1.0))
            return self.vwap_algo.simulate_execution(
                kwargs['price_series'],
                kwargs['volume_series'],
                kwargs['volume_profile']
            )
        
        elif algorithm == 'twap':
            self.twap_algo = TWAPExecutionAlgorithm(
                total_shares,
                kwargs.get('total_time', 1.0),
                kwargs.get('num_intervals', 10)
            )
            
        elif algorithm == 'optimal':
            trajectory = self.almgren_chriss.optimal_trajectory(
                total_shares,
                kwargs.get('total_time', 1.0),
                kwargs.get('num_steps', 10)
            )
            
            return pd.DataFrame({'inventory': trajectory})
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
```
