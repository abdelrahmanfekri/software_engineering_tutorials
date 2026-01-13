# Module 16: Agent-Based Market Simulation

## Table of Contents
1. [Agent-Based Modeling Fundamentals](#agent-based-modeling-fundamentals)
2. [Market Participant Agents](#market-participant-agents)
3. [Market Mechanisms](#market-mechanisms)
4. [Behavioral Finance Integration](#behavioral-finance-integration)
5. [Systemic Risk Analysis](#systemic-risk-analysis)
6. [Policy Impact Testing](#policy-impact-testing)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Agent-Based Modeling Fundamentals

### Base Agent Class

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional

class TradingAgent(ABC):
    def __init__(self, agent_id: str, initial_cash: float, initial_shares: int = 0):
        self.agent_id = agent_id
        self.cash = initial_cash
        self.shares = initial_shares
        self.wealth_history = []
        self.trade_history = []
        
    @abstractmethod
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        pass
    
    def update_wealth(self, price: float):
        wealth = self.cash + self.shares * price
        self.wealth_history.append(wealth)
        
    def execute_trade(self, action: str, quantity: int, price: float):
        if action == 'buy':
            cost = quantity * price
            if cost <= self.cash:
                self.cash -= cost
                self.shares += quantity
                self.trade_history.append({
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': len(self.wealth_history)
                })
        elif action == 'sell':
            if quantity <= self.shares:
                self.cash += quantity * price
                self.shares -= quantity
                self.trade_history.append({
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': len(self.wealth_history)
                })


class MarketSimulator:
    def __init__(self, agents: List[TradingAgent], initial_price: float = 100.0):
        self.agents = agents
        self.current_price = initial_price
        self.price_history = [initial_price]
        self.volume_history = []
        self.order_book = {'bids': [], 'asks': []}
        
    def step(self):
        market_state = self.get_market_state()
        
        orders = []
        for agent in self.agents:
            action, quantity, limit_price = agent.decide_action(market_state)
            if action in ['buy', 'sell'] and quantity > 0:
                orders.append({
                    'agent': agent,
                    'action': action,
                    'quantity': quantity,
                    'price': limit_price
                })
        
        self._process_orders(orders)
        
        for agent in self.agents:
            agent.update_wealth(self.current_price)
        
    def get_market_state(self) -> Dict:
        return {
            'current_price': self.current_price,
            'price_history': self.price_history[-100:],
            'volume': self.volume_history[-1] if self.volume_history else 0,
            'bid_ask_spread': self._calculate_spread()
        }
    
    def _process_orders(self, orders: List[Dict]):
        buy_orders = sorted(
            [o for o in orders if o['action'] == 'buy'],
            key=lambda x: x['price'],
            reverse=True
        )
        
        sell_orders = sorted(
            [o for o in orders if o['action'] == 'sell'],
            key=lambda x: x['price']
        )
        
        trades = []
        total_volume = 0
        
        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]
            
            if buy_order['price'] >= sell_order['price']:
                trade_price = (buy_order['price'] + sell_order['price']) / 2
                trade_quantity = min(buy_order['quantity'], sell_order['quantity'])
                
                buy_order['agent'].execute_trade('buy', trade_quantity, trade_price)
                sell_order['agent'].execute_trade('sell', trade_quantity, trade_price)
                
                trades.append({
                    'price': trade_price,
                    'quantity': trade_quantity
                })
                
                total_volume += trade_quantity
                
                buy_order['quantity'] -= trade_quantity
                sell_order['quantity'] -= trade_quantity
                
                if buy_order['quantity'] == 0:
                    buy_orders.pop(0)
                if sell_order['quantity'] == 0:
                    sell_orders.pop(0)
            else:
                break
        
        if trades:
            vwap = sum(t['price'] * t['quantity'] for t in trades) / sum(t['quantity'] for t in trades)
            self.current_price = vwap
        
        self.price_history.append(self.current_price)
        self.volume_history.append(total_volume)
    
    def _calculate_spread(self) -> float:
        if not self.order_book['bids'] or not self.order_book['asks']:
            return 0.0
        
        best_bid = max(self.order_book['bids'], key=lambda x: x['price'])['price']
        best_ask = min(self.order_book['asks'], key=lambda x: x['price'])['price']
        
        return best_ask - best_bid
    
    def run_simulation(self, num_steps: int):
        for _ in range(num_steps):
            self.step()
```

## Market Participant Agents

### Fundamental Trader Agent

```python
class FundamentalTrader(TradingAgent):
    def __init__(self, agent_id: str, initial_cash: float, fundamental_value: float, confidence: float = 0.1):
        super().__init__(agent_id, initial_cash)
        self.fundamental_value = fundamental_value
        self.confidence = confidence
        
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        current_price = market_state['current_price']
        
        deviation = (self.fundamental_value - current_price) / current_price
        
        if abs(deviation) < self.confidence:
            return 'hold', 0, current_price
        
        trade_size = int(min(self.cash / current_price, 10) * abs(deviation))
        
        if deviation > self.confidence:
            return 'buy', trade_size, current_price * 1.01
        else:
            return 'sell', min(trade_size, self.shares), current_price * 0.99


class TechnicalTrader(TradingAgent):
    def __init__(self, agent_id: str, initial_cash: float, lookback_period: int = 20):
        super().__init__(agent_id, initial_cash)
        self.lookback_period = lookback_period
        
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        price_history = market_state['price_history']
        current_price = market_state['current_price']
        
        if len(price_history) < self.lookback_period:
            return 'hold', 0, current_price
        
        sma = np.mean(price_history[-self.lookback_period:])
        
        trend_strength = (current_price - sma) / sma
        
        trade_size = min(int(self.cash / current_price / 10), 5)
        
        if trend_strength > 0.02:
            return 'buy', trade_size, current_price * 1.005
        elif trend_strength < -0.02:
            return 'sell', min(trade_size, self.shares), current_price * 0.995
        else:
            return 'hold', 0, current_price


class NoiseTrader(TradingAgent):
    def __init__(self, agent_id: str, initial_cash: float, noise_level: float = 0.1):
        super().__init__(agent_id, initial_cash)
        self.noise_level = noise_level
        
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        current_price = market_state['current_price']
        
        random_action = np.random.random()
        
        trade_size = max(1, int(np.random.exponential(5)))
        
        if random_action < 0.33:
            return 'buy', trade_size, current_price * (1 + np.random.uniform(0, self.noise_level))
        elif random_action < 0.66:
            return 'sell', min(trade_size, self.shares), current_price * (1 - np.random.uniform(0, self.noise_level))
        else:
            return 'hold', 0, current_price
```

### Market Maker Agent

```python
class MarketMaker(TradingAgent):
    def __init__(
        self,
        agent_id: str,
        initial_cash: float,
        spread: float = 0.02,
        max_inventory: int = 100
    ):
        super().__init__(agent_id, initial_cash)
        self.spread = spread
        self.max_inventory = max_inventory
        
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        current_price = market_state['current_price']
        
        mid_price = current_price
        
        inventory_skew = self.shares / self.max_inventory if self.max_inventory > 0 else 0
        
        bid_price = mid_price * (1 - self.spread / 2 - inventory_skew * 0.01)
        ask_price = mid_price * (1 + self.spread / 2 - inventory_skew * 0.01)
        
        if self.shares < self.max_inventory and self.cash > bid_price:
            return 'buy', 1, bid_price
        elif self.shares > 0:
            return 'sell', 1, ask_price
        else:
            return 'hold', 0, current_price
```

## Market Mechanisms

### Continuous Double Auction

```python
class ContinuousDoubleAuction:
    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.trades = []
        
    def submit_order(self, order_type: str, price: float, quantity: int, trader_id: str):
        order = {
            'type': order_type,
            'price': price,
            'quantity': quantity,
            'trader_id': trader_id,
            'timestamp': len(self.trades)
        }
        
        if order_type == 'buy':
            self.buy_orders.append(order)
            self.buy_orders.sort(key=lambda x: (-x['price'], x['timestamp']))
        else:
            self.sell_orders.append(order)
            self.sell_orders.sort(key=lambda x: (x['price'], x['timestamp']))
        
        self._match_orders()
        
    def _match_orders(self):
        while self.buy_orders and self.sell_orders:
            best_buy = self.buy_orders[0]
            best_sell = self.sell_orders[0]
            
            if best_buy['price'] >= best_sell['price']:
                trade_price = (best_buy['price'] + best_sell['price']) / 2
                trade_quantity = min(best_buy['quantity'], best_sell['quantity'])
                
                self.trades.append({
                    'price': trade_price,
                    'quantity': trade_quantity,
                    'buyer': best_buy['trader_id'],
                    'seller': best_sell['trader_id'],
                    'timestamp': len(self.trades)
                })
                
                best_buy['quantity'] -= trade_quantity
                best_sell['quantity'] -= trade_quantity
                
                if best_buy['quantity'] == 0:
                    self.buy_orders.pop(0)
                if best_sell['quantity'] == 0:
                    self.sell_orders.pop(0)
            else:
                break
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        best_bid = self.buy_orders[0]['price'] if self.buy_orders else None
        best_ask = self.sell_orders[0]['price'] if self.sell_orders else None
        
        return best_bid, best_ask
```

## Behavioral Finance Integration

### Herding Behavior Agent

```python
class HerdingAgent(TradingAgent):
    def __init__(
        self,
        agent_id: str,
        initial_cash: float,
        herding_strength: float = 0.7
    ):
        super().__init__(agent_id, initial_cash)
        self.herding_strength = herding_strength
        self.network_connections = []
        
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        current_price = market_state['current_price']
        
        if not self.network_connections:
            return 'hold', 0, current_price
        
        neighbor_actions = [agent.trade_history[-1]['action'] 
                          for agent in self.network_connections 
                          if agent.trade_history]
        
        if not neighbor_actions:
            return 'hold', 0, current_price
        
        buy_count = neighbor_actions.count('buy')
        sell_count = neighbor_actions.count('sell')
        
        if buy_count > sell_count and np.random.random() < self.herding_strength:
            trade_size = min(int(self.cash / current_price / 10), 5)
            return 'buy', trade_size, current_price * 1.01
        elif sell_count > buy_count and np.random.random() < self.herding_strength:
            trade_size = min(5, self.shares)
            return 'sell', trade_size, current_price * 0.99
        else:
            return 'hold', 0, current_price
    
    def add_connection(self, other_agent: 'HerdingAgent'):
        self.network_connections.append(other_agent)


class ProspectTheoryAgent(TradingAgent):
    def __init__(
        self,
        agent_id: str,
        initial_cash: float,
        reference_price: float,
        loss_aversion: float = 2.0
    ):
        super().__init__(agent_id, initial_cash)
        self.reference_price = reference_price
        self.loss_aversion = loss_aversion
        
    def decide_action(self, market_state: Dict) -> Tuple[str, int, float]:
        current_price = market_state['current_price']
        
        gain_loss = current_price - self.reference_price
        
        if gain_loss > 0:
            utility = gain_loss ** 0.88
        else:
            utility = -self.loss_aversion * (abs(gain_loss) ** 0.88)
        
        trade_size = min(int(self.cash / current_price / 10), 5)
        
        if utility > 0.1:
            return 'sell', min(trade_size, self.shares), current_price * 0.995
        elif utility < -0.1:
            return 'buy', trade_size, current_price * 1.005
        else:
            return 'hold', 0, current_price
```

## Systemic Risk Analysis

### Contagion Simulator

```python
class ContagionSimulator:
    def __init__(self, agents: List[TradingAgent], network_edges: List[Tuple[int, int]]):
        self.agents = agents
        self.network = self._build_network(network_edges)
        self.failed_agents = set()
        
    def _build_network(self, edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        network = {i: [] for i in range(len(self.agents))}
        
        for source, target in edges:
            network[source].append(target)
            network[target].append(source)
        
        return network
    
    def simulate_shock(
        self,
        initial_shock_agents: List[int],
        shock_magnitude: float
    ) -> Dict[str, any]:
        for agent_idx in initial_shock_agents:
            self.agents[agent_idx].cash *= (1 - shock_magnitude)
            if self.agents[agent_idx].cash < 0:
                self.failed_agents.add(agent_idx)
        
        contagion_rounds = 0
        max_rounds = 10
        
        while contagion_rounds < max_rounds:
            new_failures = set()
            
            for failed_agent in self.failed_agents:
                for connected_agent in self.network[failed_agent]:
                    if connected_agent not in self.failed_agents:
                        exposure_loss = self.agents[connected_agent].cash * 0.1
                        
                        self.agents[connected_agent].cash -= exposure_loss
                        
                        if self.agents[connected_agent].cash < self.agents[connected_agent].cash * 0.3:
                            new_failures.add(connected_agent)
            
            if not new_failures:
                break
            
            self.failed_agents.update(new_failures)
            contagion_rounds += 1
        
        return {
            'total_failures': len(self.failed_agents),
            'failure_rate': len(self.failed_agents) / len(self.agents),
            'contagion_rounds': contagion_rounds,
            'failed_agents': list(self.failed_agents)
        }
```

## Policy Impact Testing

### Circuit Breaker Simulator

```python
class CircuitBreakerSimulator:
    def __init__(
        self,
        threshold_pct: float = 0.07,
        halt_duration: int = 15
    ):
        self.threshold_pct = threshold_pct
        self.halt_duration = halt_duration
        self.is_halted = False
        self.halt_remaining = 0
        self.trigger_history = []
        
    def check_trigger(
        self,
        current_price: float,
        reference_price: float,
        current_time: int
    ) -> bool:
        price_change = (current_price - reference_price) / reference_price
        
        if abs(price_change) >= self.threshold_pct and not self.is_halted:
            self.is_halted = True
            self.halt_remaining = self.halt_duration
            self.trigger_history.append({
                'time': current_time,
                'price_change': price_change,
                'trigger_price': current_price
            })
            
            return True
        
        return False
    
    def update(self):
        if self.is_halted:
            self.halt_remaining -= 1
            
            if self.halt_remaining <= 0:
                self.is_halted = False
    
    def can_trade(self) -> bool:
        return not self.is_halted


class TransactionTaxSimulator:
    def __init__(self, tax_rate: float = 0.001):
        self.tax_rate = tax_rate
        self.total_tax_collected = 0
        self.trade_volume_history = []
        
    def apply_tax(self, trade_value: float) -> float:
        tax = trade_value * self.tax_rate
        self.total_tax_collected += tax
        
        return tax
    
    def simulate_impact(
        self,
        agents: List[TradingAgent],
        market_sim: MarketSimulator,
        num_steps: int
    ) -> Dict[str, any]:
        baseline_volume = []
        
        for _ in range(num_steps):
            market_sim.step()
            baseline_volume.append(market_sim.volume_history[-1])
        
        with_tax_volume = []
        
        for _ in range(num_steps):
            market_sim.step()
            
            if market_sim.volume_history:
                volume = market_sim.volume_history[-1]
                tax = self.apply_tax(volume * market_sim.current_price)
                with_tax_volume.append(volume)
        
        return {
            'baseline_avg_volume': np.mean(baseline_volume),
            'with_tax_avg_volume': np.mean(with_tax_volume),
            'volume_reduction_pct': (np.mean(baseline_volume) - np.mean(with_tax_volume)) / np.mean(baseline_volume) * 100,
            'total_tax_collected': self.total_tax_collected
        }
```

## PhD-Level Research Topics

### Complexity Economics Framework

```python
class ComplexityEconomicsModel:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agents = []
        self.emergence_metrics = []
        
    def measure_market_entropy(self, price_history: List[float]) -> float:
        returns = np.diff(np.log(price_history))
        
        hist, _ = np.histogram(returns, bins=20)
        
        probabilities = hist / hist.sum()
        
        probabilities = probabilities[probabilities > 0]
        
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        return entropy
    
    def calculate_hurst_exponent(self, time_series: np.ndarray) -> float:
        lags = range(2, 20)
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return poly[0]
    
    def detect_phase_transitions(
        self,
        order_parameter: np.ndarray,
        window_size: int = 50
    ) -> List[int]:
        transitions = []
        
        for i in range(window_size, len(order_parameter)):
            window = order_parameter[i-window_size:i]
            
            mean = np.mean(window)
            std = np.std(window)
            
            if abs(order_parameter[i] - mean) > 3 * std:
                transitions.append(i)
        
        return transitions
```

## Implementation

### Complete ABM System

```python
class ComprehensiveABMSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.agents = self._create_agents()
        self.simulator = MarketSimulator(self.agents)
        self.circuit_breaker = CircuitBreakerSimulator()
        self.contagion_sim = None
        
    def _create_agents(self) -> List[TradingAgent]:
        agents = []
        
        num_fundamental = self.config.get('num_fundamental', 20)
        num_technical = self.config.get('num_technical', 20)
        num_noise = self.config.get('num_noise', 10)
        num_market_makers = self.config.get('num_market_makers', 5)
        
        for i in range(num_fundamental):
            agents.append(FundamentalTrader(
                f'fundamental_{i}',
                10000,
                100 + np.random.randn() * 5
            ))
        
        for i in range(num_technical):
            agents.append(TechnicalTrader(
                f'technical_{i}',
                10000
            ))
        
        for i in range(num_noise):
            agents.append(NoiseTrader(
                f'noise_{i}',
                10000
            ))
        
        for i in range(num_market_makers):
            agents.append(MarketMaker(
                f'mm_{i}',
                50000
            ))
        
        return agents
    
    def run_simulation(
        self,
        num_steps: int,
        with_policies: bool = False
    ) -> Dict[str, any]:
        for step in range(num_steps):
            if with_policies:
                if self.circuit_breaker.is_halted:
                    self.circuit_breaker.update()
                    continue
                
                if len(self.simulator.price_history) >= 2:
                    self.circuit_breaker.check_trigger(
                        self.simulator.current_price,
                        self.simulator.price_history[0],
                        step
                    )
            
            self.simulator.step()
        
        results = {
            'price_history': self.simulator.price_history,
            'volume_history': self.simulator.volume_history,
            'final_price': self.simulator.current_price,
            'price_volatility': np.std(np.diff(np.log(self.simulator.price_history))),
            'total_volume': sum(self.simulator.volume_history)
        }
        
        if with_policies:
            results['circuit_breaker_triggers'] = len(self.circuit_breaker.trigger_history)
        
        return results
```
