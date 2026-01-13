# Module 12: Backtesting and Strategy Evaluation

## Table of Contents
1. [Backtesting Frameworks](#backtesting-frameworks)
2. [Data Quality and Preparation](#data-quality-and-preparation)
3. [Transaction Cost Modeling](#transaction-cost-modeling)
4. [Performance Metrics](#performance-metrics)
5. [Statistical Validation](#statistical-validation)
6. [Overfitting Prevention](#overfitting-prevention)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Backtesting Frameworks

### Event-Driven Backtesting Engine

```python
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class EventType(Enum):
    MARKET = 1
    SIGNAL = 2
    ORDER = 3
    FILL = 4

class Event(ABC):
    @abstractmethod
    def __init__(self):
        pass

class MarketEvent(Event):
    def __init__(self, timestamp: pd.Timestamp, symbol: str, data: Dict):
        self.type = EventType.MARKET
        self.timestamp = timestamp
        self.symbol = symbol
        self.data = data

class SignalEvent(Event):
    def __init__(self, timestamp: pd.Timestamp, symbol: str, signal_type: str, strength: float):
        self.type = EventType.SIGNAL
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength

class OrderEvent(Event):
    def __init__(self, timestamp: pd.Timestamp, symbol: str, order_type: str, quantity: int, direction: str):
        self.type = EventType.ORDER
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

class FillEvent(Event):
    def __init__(self, timestamp: pd.Timestamp, symbol: str, quantity: int, direction: str, fill_price: float, commission: float):
        self.type = EventType.FILL
        self.timestamp = timestamp
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        self.commission = commission


class DataHandler(ABC):
    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def update_bars(self):
        pass


class HistoricCSVDataHandler(DataHandler):
    def __init__(self, events_queue: Queue, csv_dir: str, symbol_list: List[str]):
        self.events_queue = events_queue
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        
        self._load_data()
        
    def _load_data(self):
        for symbol in self.symbol_list:
            self.symbol_data[symbol] = pd.read_csv(
                f"{self.csv_dir}/{symbol}.csv",
                parse_dates=['timestamp'],
                index_col='timestamp'
            )
            
            self.latest_symbol_data[symbol] = []
    
    def get_latest_bars(self, symbol: str, N: int = 1) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.latest_symbol_data[symbol][-N:])
        except KeyError:
            print(f"Symbol {symbol} not available")
            return pd.DataFrame()
    
    def update_bars(self):
        for symbol in self.symbol_list:
            try:
                bar = next(self.symbol_data[symbol].iterrows())
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[symbol].append(bar[1].to_dict())
                    
                    self.events_queue.put(MarketEvent(bar[0], symbol, bar[1].to_dict()))


class Strategy(ABC):
    @abstractmethod
    def calculate_signals(self, event: MarketEvent):
        pass


class Portfolio:
    def __init__(self, data_handler: DataHandler, events_queue: Queue, start_date: pd.Timestamp, initial_capital: float = 100000.0):
        self.data_handler = data_handler
        self.events_queue = events_queue
        self.start_date = start_date
        self.initial_capital = initial_capital
        
        self.all_positions = []
        self.current_positions = {}
        
        self.all_holdings = []
        self.current_holdings = {'cash': initial_capital, 'commission': 0.0, 'total': initial_capital}
        
    def update_timeindex(self, event: MarketEvent):
        bars = {}
        for symbol in self.data_handler.symbol_list:
            bars[symbol] = self.data_handler.get_latest_bars(symbol, N=1)
        
        holdings = {'datetime': event.timestamp, 'cash': self.current_holdings['cash']}
        
        for symbol in self.data_handler.symbol_list:
            if not bars[symbol].empty:
                market_value = self.current_positions.get(symbol, 0) * bars[symbol]['close'].iloc[-1]
                holdings[symbol] = market_value
                holdings['total'] = holdings.get('total', 0) + market_value
        
        self.all_holdings.append(holdings)
    
    def update_positions_from_fill(self, fill: FillEvent):
        fill_dir = 1 if fill.direction == 'BUY' else -1
        
        self.current_positions[fill.symbol] = self.current_positions.get(fill.symbol, 0) + fill_dir * fill.quantity
        
        fill_cost = fill_dir * fill.fill_price * fill.quantity
        self.current_holdings['cash'] -= (fill_cost + fill.commission)
        self.current_holdings['commission'] += fill.commission
        
    def generate_naive_order(self, signal: SignalEvent) -> OrderEvent:
        order = None
        
        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength
        
        mkt_quantity = 100
        cur_quantity = self.current_positions.get(symbol, 0)
        order_type = 'MKT'
        
        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(signal.timestamp, symbol, order_type, mkt_quantity, 'BUY')
        
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(signal.timestamp, symbol, order_type, mkt_quantity, 'SELL')
        
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(signal.timestamp, symbol, order_type, abs(cur_quantity), 'SELL')
        
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(signal.timestamp, symbol, order_type, abs(cur_quantity), 'BUY')
        
        return order


class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, event: OrderEvent):
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, events_queue: Queue, data_handler: DataHandler):
        self.events_queue = events_queue
        self.data_handler = data_handler
        
    def execute_order(self, event: OrderEvent):
        if event.type == EventType.ORDER:
            fill_event = self._execute_market_order(event)
            if fill_event:
                self.events_queue.put(fill_event)
    
    def _execute_market_order(self, event: OrderEvent) -> Optional[FillEvent]:
        bars = self.data_handler.get_latest_bars(event.symbol, N=1)
        
        if bars.empty:
            return None
        
        fill_price = bars['close'].iloc[-1]
        
        commission = self._calculate_commission(event.quantity, fill_price)
        
        fill_event = FillEvent(
            event.timestamp,
            event.symbol,
            event.quantity,
            event.direction,
            fill_price,
            commission
        )
        
        return fill_event
    
    def _calculate_commission(self, quantity: int, fill_price: float) -> float:
        commission = max(1.0, 0.005 * quantity * fill_price)
        
        return commission


class Backtest:
    def __init__(
        self,
        csv_dir: str,
        symbol_list: List[str],
        initial_capital: float,
        start_date: pd.Timestamp,
        strategy: Strategy
    ):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.strategy = strategy
        
        self.events_queue = Queue()
        
        self.data_handler = HistoricCSVDataHandler(self.events_queue, csv_dir, symbol_list)
        self.portfolio = Portfolio(self.data_handler, self.events_queue, start_date, initial_capital)
        self.execution_handler = SimulatedExecutionHandler(self.events_queue, self.data_handler)
        
        self.signals = 0
        self.orders = 0
        self.fills = 0
        
    def _run_backtest(self):
        while self.data_handler.continue_backtest:
            self.data_handler.update_bars()
            
            while not self.events_queue.empty():
                event = self.events_queue.get()
                
                if event.type == EventType.MARKET:
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)
                    
                elif event.type == EventType.SIGNAL:
                    self.signals += 1
                    order = self.portfolio.generate_naive_order(event)
                    if order:
                        self.events_queue.put(order)
                    
                elif event.type == EventType.ORDER:
                    self.orders += 1
                    self.execution_handler.execute_order(event)
                    
                elif event.type == EventType.FILL:
                    self.fills += 1
                    self.portfolio.update_positions_from_fill(event)
    
    def _calculate_performance(self) -> pd.DataFrame:
        holdings_df = pd.DataFrame(self.portfolio.all_holdings)
        holdings_df.set_index('datetime', inplace=True)
        
        holdings_df['returns'] = holdings_df['total'].pct_change()
        holdings_df['equity_curve'] = (1 + holdings_df['returns']).cumprod()
        
        return holdings_df
    
    def run(self) -> pd.DataFrame:
        self._run_backtest()
        
        performance = self._calculate_performance()
        
        return performance
```

### Walk-Forward Analysis

```python
class WalkForwardAnalysis:
    def __init__(
        self,
        data: pd.DataFrame,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 63
    ):
        self.data = data
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        
    def generate_windows(self) -> List[Tuple[int, int, int, int]]:
        windows = []
        
        total_length = len(self.data)
        
        for i in range(0, total_length - self.train_period - self.test_period, self.step_size):
            train_start = i
            train_end = i + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period
            
            if test_end > total_length:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def run_walk_forward(
        self,
        strategy_class,
        **strategy_params
    ) -> pd.DataFrame:
        windows = self.generate_windows()
        
        results = []
        
        for train_start, train_end, test_start, test_end in windows:
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            strategy = strategy_class(**strategy_params)
            strategy.fit(train_data)
            
            predictions = strategy.predict(test_data)
            
            window_result = {
                'train_start': self.data.index[train_start],
                'train_end': self.data.index[train_end],
                'test_start': self.data.index[test_start],
                'test_end': self.data.index[test_end],
                'predictions': predictions,
                'actuals': test_data['returns'].values
            }
            
            results.append(window_result)
        
        return pd.DataFrame(results)
    
    def calculate_walk_forward_metrics(
        self,
        results: pd.DataFrame
    ) -> Dict[str, float]:
        all_predictions = np.concatenate(results['predictions'].values)
        all_actuals = np.concatenate(results['actuals'].values)
        
        sharpe = np.mean(all_predictions * all_actuals) / np.std(all_predictions * all_actuals) * np.sqrt(252)
        
        cumulative_returns = np.cumprod(1 + all_predictions * all_actuals) - 1
        
        total_return = cumulative_returns[-1]
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'num_windows': len(results)
        }
```

## Data Quality and Preparation

### Point-in-Time Data Handler

```python
class PointInTimeData:
    def __init__(self):
        self.data_snapshots = {}
        
    def add_snapshot(
        self,
        timestamp: pd.Timestamp,
        data: pd.DataFrame
    ):
        self.data_snapshots[timestamp] = data.copy()
        
    def get_data_as_of(
        self,
        timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        valid_timestamps = [ts for ts in self.data_snapshots.keys() if ts <= timestamp]
        
        if not valid_timestamps:
            return pd.DataFrame()
        
        latest_valid_timestamp = max(valid_timestamps)
        
        return self.data_snapshots[latest_valid_timestamp]
    
    def adjust_for_corporate_actions(
        self,
        prices: pd.DataFrame,
        actions: pd.DataFrame
    ) -> pd.DataFrame:
        adjusted_prices = prices.copy()
        
        for _, action in actions.iterrows():
            action_date = action['date']
            ratio = action['ratio']
            
            mask = adjusted_prices.index < action_date
            adjusted_prices.loc[mask] *= ratio
        
        return adjusted_prices


class SurvivorshipBiasCleaner:
    def __init__(self):
        pass
    
    def identify_delisted_stocks(
        self,
        universe: pd.DataFrame,
        as_of_date: pd.Timestamp
    ) -> List[str]:
        current_universe = universe[universe['last_date'] > as_of_date]
        
        return current_universe['symbol'].tolist()
    
    def create_historical_universe(
        self,
        all_stocks: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[str]:
        active_stocks = all_stocks[
            (all_stocks['first_date'] <= date) &
            (all_stocks['last_date'] >= date)
        ]
        
        return active_stocks['symbol'].tolist()
```

## Transaction Cost Modeling

### Comprehensive Transaction Cost Model

```python
class TransactionCostModel:
    def __init__(
        self,
        fixed_cost_per_trade: float = 1.0,
        proportional_cost_bps: float = 5.0,
        market_impact_model: str = 'square_root'
    ):
        self.fixed_cost = fixed_cost_per_trade
        self.proportional_cost = proportional_cost_bps / 10000
        self.market_impact_model = market_impact_model
        
    def calculate_commission(
        self,
        quantity: int,
        price: float
    ) -> float:
        trade_value = quantity * price
        
        commission = self.fixed_cost + trade_value * self.proportional_cost
        
        return commission
    
    def calculate_market_impact(
        self,
        quantity: int,
        avg_daily_volume: int,
        volatility: float,
        price: float
    ) -> float:
        participation_rate = quantity / avg_daily_volume
        
        if self.market_impact_model == 'linear':
            impact_bps = 10 * participation_rate
        
        elif self.market_impact_model == 'square_root':
            impact_bps = 10 * volatility * np.sqrt(participation_rate)
        
        elif self.market_impact_model == 'concave':
            impact_bps = 10 * volatility * (participation_rate ** 0.6)
        
        else:
            impact_bps = 0
        
        impact_dollars = price * (impact_bps / 10000) * quantity
        
        return impact_dollars
    
    def calculate_slippage(
        self,
        quantity: int,
        bid_ask_spread: float
    ) -> float:
        slippage = quantity * bid_ask_spread / 2
        
        return slippage
    
    def calculate_total_cost(
        self,
        quantity: int,
        price: float,
        avg_daily_volume: int,
        volatility: float,
        bid_ask_spread: float
    ) -> Dict[str, float]:
        commission = self.calculate_commission(quantity, price)
        
        market_impact = self.calculate_market_impact(
            quantity,
            avg_daily_volume,
            volatility,
            price
        )
        
        slippage = self.calculate_slippage(quantity, bid_ask_spread)
        
        total_cost = commission + market_impact + slippage
        
        return {
            'commission': commission,
            'market_impact': market_impact,
            'slippage': slippage,
            'total_cost': total_cost,
            'cost_bps': (total_cost / (quantity * price)) * 10000
        }
```

## Performance Metrics

### Comprehensive Performance Analytics

```python
class PerformanceAnalytics:
    def __init__(self, returns: pd.Series):
        self.returns = returns
        self.cumulative_returns = (1 + returns).cumprod() - 1
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        
        return sortino
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        cumulative_returns = (1 + self.returns).cumprod()
        
        running_max = cumulative_returns.expanding().max()
        
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        
        max_dd_end = drawdown.idxmin()
        
        max_dd_start = cumulative_returns[:max_dd_end].idxmax()
        
        recovery_date = None
        if max_dd_end < cumulative_returns.index[-1]:
            recovery_mask = cumulative_returns[max_dd_end:] >= running_max[max_dd_end]
            if recovery_mask.any():
                recovery_date = cumulative_returns[max_dd_end:][recovery_mask].index[0]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_start': max_dd_start,
            'max_drawdown_end': max_dd_end,
            'recovery_date': recovery_date
        }
    
    def calculate_calmar_ratio(self) -> float:
        annual_return = self.returns.mean() * 252
        
        max_dd = abs(self.calculate_max_drawdown()['max_drawdown'])
        
        if max_dd == 0:
            return 0.0
        
        calmar = annual_return / max_dd
        
        return calmar
    
    def calculate_information_ratio(self, benchmark_returns: pd.Series) -> float:
        excess_returns = self.returns - benchmark_returns
        
        if excess_returns.std() == 0:
            return 0.0
        
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return information_ratio
    
    def calculate_win_rate(self) -> float:
        winning_days = (self.returns > 0).sum()
        total_days = len(self.returns)
        
        return winning_days / total_days
    
    def calculate_profit_factor(self) -> float:
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    def calculate_all_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        metrics = {
            'total_return': self.cumulative_returns.iloc[-1],
            'annual_return': self.returns.mean() * 252,
            'annual_volatility': self.returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'max_drawdown': self.calculate_max_drawdown()['max_drawdown'],
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor()
        }
        
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(benchmark_returns)
        
        return metrics
```

## Statistical Validation

### Monte Carlo Simulation

```python
class MonteCarloSimulation:
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        
    def bootstrap_sharpe_distribution(
        self,
        n_simulations: int = 10000,
        block_size: int = 20
    ) -> np.ndarray:
        sharpe_ratios = []
        
        n_blocks = len(self.returns) // block_size
        
        for _ in range(n_simulations):
            sampled_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)
            
            simulated_returns = np.concatenate([
                self.returns[i*block_size:(i+1)*block_size]
                for i in sampled_blocks
            ])
            
            sharpe = np.mean(simulated_returns) / np.std(simulated_returns) * np.sqrt(252)
            sharpe_ratios.append(sharpe)
        
        return np.array(sharpe_ratios)
    
    def calculate_confidence_interval(
        self,
        metric: str = 'sharpe',
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        if metric == 'sharpe':
            distribution = self.bootstrap_sharpe_distribution()
        
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bound = np.percentile(distribution, lower_percentile)
        upper_bound = np.percentile(distribution, upper_percentile)
        
        return (lower_bound, upper_bound)


class PermutationTest:
    def __init__(self):
        pass
    
    def test_strategy_significance(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        n_permutations: int = 10000
    ) -> Dict[str, float]:
        observed_diff = np.mean(strategy_returns) - np.mean(benchmark_returns)
        
        combined = np.concatenate([strategy_returns, benchmark_returns])
        n_strategy = len(strategy_returns)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            
            perm_strategy = combined[:n_strategy]
            perm_benchmark = combined[n_strategy:]
            
            perm_diff = np.mean(perm_strategy) - np.mean(perm_benchmark)
            permuted_diffs.append(perm_diff)
        
        p_value = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / n_permutations
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

## Overfitting Prevention

### Combinatorial Purged Cross-Validation

```python
from sklearn.model_selection import KFold

class PurgedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        embargo_samples = int(n_samples * self.embargo_pct)
        
        fold_size = n_samples // self.n_splits
        
        indices = np.arange(n_samples)
        
        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            train_indices = np.concatenate([
                indices[:max(0, test_start - embargo_samples)],
                indices[min(n_samples, test_end + embargo_samples):]
            ])
            
            yield train_indices, test_indices


class FeatureImportanceValidator:
    def __init__(self, model):
        self.model = model
        
    def calculate_mdi_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> pd.Series:
        from sklearn.ensemble import RandomForestClassifier
        
        if isinstance(self.model, RandomForestClassifier):
            self.model.fit(X_train, y_train)
            
            importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            )
            
            return importance.sort_values(ascending=False)
        
        return pd.Series()
    
    def calculate_mda_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_permutations: int = 10
    ) -> pd.Series:
        self.model.fit(X_train, y_train)
        
        baseline_score = self.model.score(X_test, y_test)
        
        importance = {}
        
        for col in X_test.columns:
            scores = []
            
            for _ in range(n_permutations):
                X_test_permuted = X_test.copy()
                X_test_permuted[col] = np.random.permutation(X_test_permuted[col])
                
                permuted_score = self.model.score(X_test_permuted, y_test)
                scores.append(baseline_score - permuted_score)
            
            importance[col] = np.mean(scores)
        
        return pd.Series(importance).sort_values(ascending=False)


class DeflatedSharpeRatio:
    def __init__(self):
        pass
    
    def calculate_deflated_sharpe(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> Tuple[float, float]:
        expected_max_sharpe = self._expected_maximum_sharpe(n_trials)
        
        variance_sharpe = (1 + (1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)) / n_observations
        
        deflated_sharpe = (observed_sharpe - expected_max_sharpe) / np.sqrt(variance_sharpe)
        
        p_value = 1 - stats.norm.cdf(deflated_sharpe)
        
        return deflated_sharpe, p_value
    
    def _expected_maximum_sharpe(self, n_trials: int) -> float:
        from scipy.special import erfinv
        
        euler_mascheroni = 0.5772156649
        
        expected_max = np.sqrt(2 * np.log(n_trials)) - (
            np.log(np.log(n_trials)) + euler_mascheroni
        ) / (2 * np.sqrt(2 * np.log(n_trials)))
        
        return expected_max
```

## PhD-Level Research Topics

### Minimum Backtest Length

```python
class MinimumBacktestLength:
    def __init__(self):
        pass
    
    def calculate_min_track_record_length(
        self,
        target_sharpe: float,
        observed_sharpe: float,
        confidence_level: float = 0.95,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> int:
        from scipy.stats import norm
        
        z_alpha = norm.ppf(confidence_level)
        
        numerator = (z_alpha**2) * (1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
        
        denominator = (observed_sharpe - target_sharpe)**2
        
        if denominator == 0:
            return np.inf
        
        min_length = int(np.ceil(numerator / denominator))
        
        return min_length
```

## Implementation

### Complete Backtesting System

```python
class ComprehensiveBacktestingSystem:
    def __init__(self):
        self.transaction_cost_model = TransactionCostModel()
        self.performance_analytics = None
        self.monte_carlo = None
        
    def run_backtest(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, any]:
        self.performance_analytics = PerformanceAnalytics(strategy_returns)
        
        metrics = self.performance_analytics.calculate_all_metrics(benchmark_returns)
        
        self.monte_carlo = MonteCarloSimulation(strategy_returns.values)
        sharpe_ci = self.monte_carlo.calculate_confidence_interval('sharpe')
        
        metrics['sharpe_ci_lower'] = sharpe_ci[0]
        metrics['sharpe_ci_upper'] = sharpe_ci[1]
        
        if benchmark_returns is not None:
            perm_test = PermutationTest()
            significance = perm_test.test_strategy_significance(
                strategy_returns.values,
                benchmark_returns.values
            )
            
            metrics['p_value'] = significance['p_value']
            metrics['significant'] = significance['significant']
        
        return metrics
    
    def validate_strategy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        n_splits: int = 5
    ) -> Dict[str, any]:
        purged_cv = PurgedKFold(n_splits=n_splits)
        
        fold_scores = []
        
        for train_idx, test_idx in purged_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            fold_scores.append(score)
        
        return {
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'fold_scores': fold_scores
        }
```
