# Module 9: Algorithmic Trading Strategies

## Table of Contents
1. [Statistical Arbitrage](#statistical-arbitrage)
2. [Momentum and Trend Following](#momentum-and-trend-following)
3. [Market Making Strategies](#market-making-strategies)
4. [Volatility Trading](#volatility-trading)
5. [Machine Learning Strategies](#machine-learning-strategies)
6. [High-Frequency Trading](#high-frequency-trading)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Statistical Arbitrage

### Pairs Trading with Machine Learning

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

class PairsTradingStrategy:
    def __init__(self, lookback_period: int = 60, z_entry: float = 2.0, z_exit: float = 0.5):
        self.lookback_period = lookback_period
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.hedge_ratio = None
        self.mean_spread = None
        self.std_spread = None
        
    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        significance_level: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        n = price_data.shape[1]
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = price_data.columns[i]
                stock2 = price_data.columns[j]
                
                _, pvalue, _ = coint(price_data[stock1], price_data[stock2])
                
                if pvalue < significance_level:
                    pairs.append((stock1, stock2, pvalue))
        
        return sorted(pairs, key=lambda x: x[2])
    
    def calculate_hedge_ratio(
        self,
        price1: np.ndarray,
        price2: np.ndarray
    ) -> float:
        model = sm.OLS(price1, sm.add_constant(price2))
        results = model.fit()
        
        self.hedge_ratio = results.params[1]
        
        return self.hedge_ratio
    
    def calculate_spread(
        self,
        price1: np.ndarray,
        price2: np.ndarray,
        hedge_ratio: Optional[float] = None
    ) -> np.ndarray:
        if hedge_ratio is None:
            hedge_ratio = self.hedge_ratio
        
        spread = price1 - hedge_ratio * price2
        
        return spread
    
    def calculate_z_score(self, spread: np.ndarray) -> np.ndarray:
        self.mean_spread = np.mean(spread[-self.lookback_period:])
        self.std_spread = np.std(spread[-self.lookback_period:])
        
        z_score = (spread - self.mean_spread) / self.std_spread
        
        return z_score
    
    def generate_signals(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.DataFrame:
        self.calculate_hedge_ratio(price1.values, price2.values)
        
        spread = self.calculate_spread(price1.values, price2.values)
        
        signals = pd.DataFrame(index=price1.index)
        signals['spread'] = spread
        signals['z_score'] = self.calculate_z_score(spread)
        
        signals['position'] = 0
        
        for i in range(self.lookback_period, len(signals)):
            z = signals['z_score'].iloc[i]
            prev_pos = signals['position'].iloc[i-1]
            
            if z > self.z_entry and prev_pos == 0:
                signals.loc[signals.index[i], 'position'] = -1
            elif z < -self.z_entry and prev_pos == 0:
                signals.loc[signals.index[i], 'position'] = 1
            elif abs(z) < self.z_exit and prev_pos != 0:
                signals.loc[signals.index[i], 'position'] = 0
            else:
                signals.loc[signals.index[i], 'position'] = prev_pos
        
        signals['stock1_position'] = signals['position']
        signals['stock2_position'] = -signals['position'] * self.hedge_ratio
        
        return signals
    
    def backtest(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> Dict[str, float]:
        signals = self.generate_signals(price1, price2)
        
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        strategy_returns = (
            signals['stock1_position'].shift(1) * returns1 +
            signals['stock2_position'].shift(1) * returns2
        )
        
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': (signals['position'].diff() != 0).sum()
        }
```

### Kalman Filter for Pairs Trading

```python
from pykalman import KalmanFilter

class KalmanPairsTrading:
    def __init__(self):
        self.kf = None
        
    def fit_kalman_filter(
        self,
        price1: np.ndarray,
        price2: np.ndarray
    ):
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        
        obs_mat = np.vstack([price2, np.ones(len(price2))]).T
        
        self.kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            observation_covariance=1.0,
            transition_covariance=trans_cov
        )
        
        state_means, _ = self.kf.filter(price1)
        
        return state_means
    
    def generate_dynamic_hedge_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.DataFrame:
        state_means = self.fit_kalman_filter(price1.values, price2.values)
        
        results = pd.DataFrame(index=price1.index)
        results['hedge_ratio'] = state_means[:, 0]
        results['intercept'] = state_means[:, 1]
        
        results['spread'] = price1 - results['hedge_ratio'] * price2 - results['intercept']
        
        results['z_score'] = (
            (results['spread'] - results['spread'].rolling(60).mean()) /
            results['spread'].rolling(60).std()
        )
        
        return results
```

### Mean Reversion with Ornstein-Uhlenbeck Process

```python
class OrnsteinUhlenbeckStrategy:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        
    def estimate_parameters(self, prices: np.ndarray, dt: float = 1.0) -> Tuple[float, float, float]:
        log_prices = np.log(prices)
        
        dx = np.diff(log_prices)
        x = log_prices[:-1]
        
        self.mu = np.mean(log_prices)
        
        x_centered = x - self.mu
        
        sum_xx = np.sum(x_centered ** 2)
        sum_x_dx = np.sum(x_centered * dx)
        
        self.theta = -sum_x_dx / (sum_xx * dt)
        
        residuals = dx + self.theta * x_centered * dt
        self.sigma = np.std(residuals) / np.sqrt(dt)
        
        return self.theta, self.mu, self.sigma
    
    def calculate_ou_z_score(self, current_price: float) -> float:
        log_price = np.log(current_price)
        
        expected_mean_reversion = np.exp(self.mu)
        
        std = self.sigma / np.sqrt(2 * self.theta)
        
        z_score = (current_price - expected_mean_reversion) / std
        
        return z_score
    
    def generate_signals(
        self,
        prices: pd.Series,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> pd.DataFrame:
        self.estimate_parameters(prices.values)
        
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['z_score'] = 0.0
        signals['position'] = 0
        
        for i in range(60, len(signals)):
            z = self.calculate_ou_z_score(prices.iloc[i])
            signals.loc[signals.index[i], 'z_score'] = z
            
            prev_pos = signals['position'].iloc[i-1]
            
            if z > entry_threshold and prev_pos == 0:
                signals.loc[signals.index[i], 'position'] = -1
            elif z < -entry_threshold and prev_pos == 0:
                signals.loc[signals.index[i], 'position'] = 1
            elif abs(z) < exit_threshold:
                signals.loc[signals.index[i], 'position'] = 0
            else:
                signals.loc[signals.index[i], 'position'] = prev_pos
        
        return signals
```

## Momentum and Trend Following

### Time Series Momentum

```python
class TimeSeriesMomentumStrategy:
    def __init__(
        self,
        lookback_periods: list = [20, 60, 120],
        holding_period: int = 20
    ):
        self.lookback_periods = lookback_periods
        self.holding_period = holding_period
        
    def calculate_momentum(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        momentum_signals = pd.DataFrame(index=prices.index)
        
        for period in self.lookback_periods:
            momentum_signals[f'momentum_{period}'] = prices.pct_change(period)
        
        momentum_signals['composite_momentum'] = momentum_signals.mean(axis=1)
        
        return momentum_signals
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        momentum = self.calculate_momentum(prices)
        
        signals = pd.DataFrame(index=prices.index)
        signals['momentum'] = momentum['composite_momentum']
        
        signals['position'] = 0
        signals.loc[signals['momentum'] > threshold, 'position'] = 1
        signals.loc[signals['momentum'] < -threshold, 'position'] = -1
        
        return signals
    
    def calculate_volatility_scaled_positions(
        self,
        prices: pd.DataFrame,
        target_vol: float = 0.15
    ) -> pd.DataFrame:
        signals = self.generate_signals(prices)
        
        returns = prices.pct_change()
        realized_vol = returns.rolling(window=60).std() * np.sqrt(252)
        
        vol_scalar = target_vol / realized_vol
        vol_scalar = vol_scalar.fillna(1.0).clip(0.5, 2.0)
        
        signals['scaled_position'] = signals['position'] * vol_scalar
        
        return signals
```

### Adaptive Trend Detection

```python
from scipy.signal import find_peaks

class AdaptiveTrendStrategy:
    def __init__(self, ema_short: int = 12, ema_long: int = 26):
        self.ema_short = ema_short
        self.ema_long = ema_long
        
    def detect_trend_regime(
        self,
        prices: pd.Series,
        window: int = 100
    ) -> pd.Series:
        returns = prices.pct_change()
        
        rolling_sharpe = (
            returns.rolling(window).mean() /
            returns.rolling(window).std() * np.sqrt(252)
        )
        
        autocorr = returns.rolling(window).apply(
            lambda x: x.autocorr(), raw=False
        )
        
        trending = (rolling_sharpe.abs() > 1.0) & (autocorr > 0.1)
        
        regime = pd.Series(0, index=prices.index)
        regime[trending] = 1
        regime[~trending] = 0
        
        return regime
    
    def macd_strategy(self, prices: pd.Series) -> pd.DataFrame:
        ema_short = prices.ewm(span=self.ema_short).mean()
        ema_long = prices.ewm(span=self.ema_long).mean()
        
        macd_line = ema_short - ema_long
        
        signal_line = macd_line.ewm(span=9).mean()
        
        macd_histogram = macd_line - signal_line
        
        signals = pd.DataFrame(index=prices.index)
        signals['macd'] = macd_line
        signals['signal'] = signal_line
        signals['histogram'] = macd_histogram
        
        signals['position'] = 0
        signals.loc[macd_line > signal_line, 'position'] = 1
        signals.loc[macd_line < signal_line, 'position'] = -1
        
        return signals
    
    def adaptive_trend_following(
        self,
        prices: pd.Series
    ) -> pd.DataFrame:
        regime = self.detect_trend_regime(prices)
        
        macd_signals = self.macd_strategy(prices)
        
        mean_reversion_signals = self.generate_mean_reversion_signals(prices)
        
        adaptive_signals = pd.DataFrame(index=prices.index)
        adaptive_signals['position'] = 0
        
        trending_mask = regime == 1
        adaptive_signals.loc[trending_mask, 'position'] = macd_signals.loc[trending_mask, 'position']
        adaptive_signals.loc[~trending_mask, 'position'] = mean_reversion_signals.loc[~trending_mask, 'position']
        
        return adaptive_signals
    
    def generate_mean_reversion_signals(self, prices: pd.Series) -> pd.DataFrame:
        signals = pd.DataFrame(index=prices.index)
        
        bollinger_mid = prices.rolling(20).mean()
        bollinger_std = prices.rolling(20).std()
        
        z_score = (prices - bollinger_mid) / bollinger_std
        
        signals['position'] = 0
        signals.loc[z_score < -2, 'position'] = 1
        signals.loc[z_score > 2, 'position'] = -1
        
        return signals
```

## Market Making Strategies

### Avellaneda-Stoikov Market Making

```python
class AvellanedaStoikovMM:
    def __init__(
        self,
        risk_aversion: float = 0.1,
        terminal_time: float = 1.0,
        volatility: float = 0.02
    ):
        self.gamma = risk_aversion
        self.T = terminal_time
        self.sigma = volatility
        
    def calculate_reservation_price(
        self,
        mid_price: float,
        position: int,
        time_to_end: float
    ) -> float:
        r = mid_price - position * self.gamma * self.sigma**2 * time_to_end
        
        return r
    
    def calculate_optimal_spread(
        self,
        time_to_end: float,
        order_arrival_rate: float = 1.0
    ) -> float:
        spread = self.gamma * self.sigma**2 * time_to_end + (2 / self.gamma) * np.log(1 + self.gamma / order_arrival_rate)
        
        return spread
    
    def calculate_bid_ask(
        self,
        mid_price: float,
        position: int,
        time_to_end: float
    ) -> Tuple[float, float]:
        reservation_price = self.calculate_reservation_price(mid_price, position, time_to_end)
        
        spread = self.calculate_optimal_spread(time_to_end)
        
        bid = reservation_price - spread / 2
        ask = reservation_price + spread / 2
        
        return bid, ask
    
    def simulate_market_making(
        self,
        price_series: pd.Series,
        initial_inventory: int = 0,
        max_inventory: int = 10
    ) -> pd.DataFrame:
        results = pd.DataFrame(index=price_series.index)
        results['mid_price'] = price_series
        results['inventory'] = initial_inventory
        results['pnl'] = 0.0
        results['bid'] = 0.0
        results['ask'] = 0.0
        
        T_total = len(price_series)
        
        for i in range(1, len(results)):
            time_to_end = (T_total - i) / T_total * self.T
            
            current_inventory = results['inventory'].iloc[i-1]
            mid_price = results['mid_price'].iloc[i]
            
            bid, ask = self.calculate_bid_ask(mid_price, current_inventory, time_to_end)
            
            results.loc[results.index[i], 'bid'] = bid
            results.loc[results.index[i], 'ask'] = ask
            
            if abs(current_inventory) < max_inventory:
                price_change = price_series.iloc[i] - price_series.iloc[i-1]
                
                if price_change > 0 and np.random.random() < 0.3:
                    results.loc[results.index[i], 'inventory'] = current_inventory + 1
                    results.loc[results.index[i], 'pnl'] = results['pnl'].iloc[i-1] - bid
                
                elif price_change < 0 and np.random.random() < 0.3:
                    results.loc[results.index[i], 'inventory'] = current_inventory - 1
                    results.loc[results.index[i], 'pnl'] = results['pnl'].iloc[i-1] + ask
                else:
                    results.loc[results.index[i], 'inventory'] = current_inventory
                    results.loc[results.index[i], 'pnl'] = results['pnl'].iloc[i-1]
            else:
                results.loc[results.index[i], 'inventory'] = current_inventory
                results.loc[results.index[i], 'pnl'] = results['pnl'].iloc[i-1]
            
            results.loc[results.index[i], 'pnl'] += current_inventory * (price_series.iloc[i] - price_series.iloc[i-1])
        
        return results
```

### Order Book Imbalance Strategy

```python
class OrderBookImbalanceStrategy:
    def __init__(self, depth_levels: int = 5):
        self.depth_levels = depth_levels
        
    def calculate_imbalance(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> float:
        total_bid = np.sum(bid_volumes[:self.depth_levels])
        total_ask = np.sum(ask_volumes[:self.depth_levels])
        
        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        
        return imbalance
    
    def calculate_weighted_midprice(
        self,
        best_bid: float,
        best_ask: float,
        bid_volume: float,
        ask_volume: float
    ) -> float:
        weighted_mid = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        
        return weighted_mid
    
    def generate_signals(
        self,
        order_book_data: pd.DataFrame,
        imbalance_threshold: float = 0.3
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=order_book_data.index)
        
        signals['imbalance'] = order_book_data.apply(
            lambda row: self.calculate_imbalance(
                row['bid_volumes'],
                row['ask_volumes']
            ),
            axis=1
        )
        
        signals['weighted_mid'] = order_book_data.apply(
            lambda row: self.calculate_weighted_midprice(
                row['best_bid'],
                row['best_ask'],
                row['bid_volumes'][0],
                row['ask_volumes'][0]
            ),
            axis=1
        )
        
        signals['position'] = 0
        signals.loc[signals['imbalance'] > imbalance_threshold, 'position'] = 1
        signals.loc[signals['imbalance'] < -imbalance_threshold, 'position'] = -1
        
        return signals
```

## Volatility Trading

### Volatility Arbitrage

```python
from scipy.stats import norm

class VolatilityArbitrageStrategy:
    def __init__(self):
        self.black_scholes_model = BlackScholesModel()
        
    def calculate_implied_volatility(
        self,
        option_price: float,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = 'call'
    ) -> float:
        from scipy.optimize import brentq
        
        def objective(vol):
            bs_price = self.black_scholes_model.price(
                spot_price, strike, time_to_expiry, risk_free_rate, vol, option_type
            )
            return bs_price - option_price
        
        try:
            implied_vol = brentq(objective, 0.01, 5.0)
        except:
            implied_vol = np.nan
        
        return implied_vol
    
    def calculate_historical_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        return returns.rolling(window).std() * np.sqrt(252)
    
    def generate_signals(
        self,
        historical_vol: pd.Series,
        implied_vol: pd.Series,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=historical_vol.index)
        signals['hist_vol'] = historical_vol
        signals['implied_vol'] = implied_vol
        
        signals['vol_spread'] = (implied_vol - historical_vol) / historical_vol
        
        signals['position'] = 0
        
        signals.loc[signals['vol_spread'] > threshold, 'position'] = -1
        signals.loc[signals['vol_spread'] < -threshold, 'position'] = 1
        
        return signals


class BlackScholesModel:
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        return delta
    
    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return gamma
```

### VIX Futures Trading

```python
class VIXFuturesStrategy:
    def __init__(self):
        self.term_structure = None
        
    def calculate_contango_backwardation(
        self,
        front_month: pd.Series,
        second_month: pd.Series
    ) -> pd.Series:
        term_structure = (second_month - front_month) / front_month
        
        return term_structure
    
    def generate_signals(
        self,
        vix_spot: pd.Series,
        vix_futures: pd.Series,
        contango_threshold: float = 0.05
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=vix_spot.index)
        signals['vix_spot'] = vix_spot
        signals['vix_futures'] = vix_futures
        
        signals['basis'] = (vix_futures - vix_spot) / vix_spot
        
        signals['position'] = 0
        
        signals.loc[signals['basis'] > contango_threshold, 'position'] = -1
        signals.loc[signals['basis'] < -contango_threshold, 'position'] = 1
        
        signals['vix_level_signal'] = 0
        signals.loc[vix_spot > vix_spot.quantile(0.8), 'vix_level_signal'] = -1
        signals.loc[vix_spot < vix_spot.quantile(0.2), 'vix_level_signal'] = 1
        
        signals['combined_position'] = signals['position'] + signals['vix_level_signal']
        signals['combined_position'] = signals['combined_position'].clip(-1, 1)
        
        return signals
```

## Machine Learning Strategies

### Ensemble Trading System

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class EnsembleTradingSystem:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        self.scaler = StandardScaler()
        
    def engineer_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=price_data.index)
        
        for period in [5, 10, 20, 50]:
            features[f'return_{period}'] = price_data.pct_change(period)
            features[f'vol_{period}'] = price_data.pct_change().rolling(period).std()
            features[f'rsi_{period}'] = self.calculate_rsi(price_data, period)
        
        features['macd'] = self.calculate_macd(price_data)
        
        features['bb_position'] = self.calculate_bollinger_position(price_data)
        
        return features.dropna()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26
    
    def calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        mid = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return (prices - mid) / std
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred)
        
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def generate_signals(
        self,
        price_data: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.DataFrame:
        features = self.engineer_features(price_data)
        
        predictions = self.predict(features)
        
        signals = pd.DataFrame(index=features.index)
        signals['prediction'] = predictions
        signals['position'] = 0
        
        signals.loc[predictions > threshold, 'position'] = 1
        signals.loc[predictions < (1 - threshold), 'position'] = -1
        
        return signals
```

## High-Frequency Trading

### Microstructure Signals

```python
class MicrostructureSignals:
    def __init__(self):
        pass
    
    def calculate_effective_spread(
        self,
        trade_price: float,
        mid_price: float,
        side: str
    ) -> float:
        if side == 'buy':
            effective_spread = 2 * (trade_price - mid_price)
        else:
            effective_spread = 2 * (mid_price - trade_price)
        
        return effective_spread
    
    def calculate_price_impact(
        self,
        trade_price: float,
        mid_price_before: float,
        mid_price_after: float,
        side: str
    ) -> float:
        if side == 'buy':
            impact = mid_price_after - mid_price_before
        else:
            impact = mid_price_before - mid_price_after
        
        return impact
    
    def calculate_roll_measure(
        self,
        price_changes: np.ndarray
    ) -> float:
        autocovariance = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        
        spread = 2 * np.sqrt(-autocovariance)
        
        return max(spread, 0)
    
    def tick_rule_classification(
        self,
        current_price: float,
        previous_price: float,
        last_direction: int
    ) -> int:
        if current_price > previous_price:
            return 1
        elif current_price < previous_price:
            return -1
        else:
            return last_direction
    
    def calculate_vpin(
        self,
        volumes: np.ndarray,
        buy_volumes: np.ndarray,
        window: int = 50
    ) -> np.ndarray:
        sell_volumes = volumes - buy_volumes
        
        vpin = np.zeros(len(volumes))
        
        for i in range(window, len(volumes)):
            total_volume = np.sum(volumes[i-window:i])
            volume_imbalance = np.sum(np.abs(buy_volumes[i-window:i] - sell_volumes[i-window:i]))
            
            vpin[i] = volume_imbalance / total_volume if total_volume > 0 else 0
        
        return vpin
```

## PhD-Level Research Topics

### Game Theory in Trading

```python
class GameTheoreticTrading:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        
    def calculate_nash_equilibrium(
        self,
        payoff_matrix: np.ndarray
    ) -> np.ndarray:
        from scipy.optimize import linprog
        
        n_strategies = payoff_matrix.shape[0]
        
        c = np.zeros(n_strategies + 1)
        c[-1] = -1
        
        A_ub = np.hstack([payoff_matrix.T, -np.ones((n_strategies, 1))])
        b_ub = np.zeros(n_strategies)
        
        A_eq = np.hstack([np.ones((1, n_strategies)), np.zeros((1, 1))])
        b_eq = np.array([1])
        
        bounds = [(0, None) for _ in range(n_strategies)] + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            return result.x[:-1]
        else:
            return None
```

## Implementation

### Complete Trading System

```python
class AlgorithmicTradingSystem:
    def __init__(self):
        self.strategies = {
            'pairs': PairsTradingStrategy(),
            'momentum': TimeSeriesMomentumStrategy(),
            'ensemble': EnsembleTradingSystem()
        }
        
        self.active_positions = {}
        self.pnl = 0.0
        
    def run_strategy(
        self,
        strategy_name: str,
        price_data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        strategy = self.strategies.get(strategy_name)
        
        if strategy is None:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        signals = strategy.generate_signals(price_data, **kwargs)
        
        return signals
    
    def backtest_strategy(
        self,
        strategy_name: str,
        price_data: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict[str, any]:
        signals = self.run_strategy(strategy_name, price_data)
        
        returns = price_data.pct_change()
        
        strategy_returns = signals['position'].shift(1) * returns
        
        portfolio_value = initial_capital * (1 + strategy_returns).cumprod()
        
        total_return = (portfolio_value.iloc[-1] / initial_capital - 1)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'final_value': portfolio_value.iloc[-1]
        }
```
