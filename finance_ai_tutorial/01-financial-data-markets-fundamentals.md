# Module 1: Financial Data and Markets Fundamentals

## Table of Contents
1. [Financial Markets Structure](#financial-markets-structure)
2. [Market Data Types and Sources](#market-data-types-and-sources)
3. [Time Series Characteristics](#time-series-characteristics)
4. [Trading Mechanics and Order Types](#trading-mechanics-and-order-types)
5. [Financial Instruments](#financial-instruments)
6. [Market Microstructure Theory](#market-microstructure-theory)
7. [Data Quality and Preprocessing](#data-quality-and-preprocessing)

## Financial Markets Structure

### Market Taxonomy

#### Primary vs Secondary Markets
```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class MarketType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    OTC = "over_the_counter"
    DARK_POOL = "dark_pool"

@dataclass
class Market:
    name: str
    market_type: MarketType
    trading_hours: tuple
    tick_size: float
    lot_size: int
    
    def is_trading_active(self, timestamp: pd.Timestamp) -> bool:
        hour = timestamp.hour
        return self.trading_hours[0] <= hour < self.trading_hours[1]

class MarketStructure:
    def __init__(self):
        self.markets = {
            'NYSE': Market('NYSE', MarketType.SECONDARY, (9.5, 16), 0.01, 100),
            'NASDAQ': Market('NASDAQ', MarketType.SECONDARY, (9.5, 16), 0.01, 100),
            'CME': Market('CME', MarketType.SECONDARY, (0, 24), 0.25, 1)
        }
    
    def get_active_markets(self, timestamp: pd.Timestamp) -> List[str]:
        return [name for name, market in self.markets.items() 
                if market.is_trading_active(timestamp)]
```

### Market Participants

#### Institutional vs Retail Flow
```python
from abc import ABC, abstractmethod

class MarketParticipant(ABC):
    def __init__(self, participant_id: str, capital: float):
        self.participant_id = participant_id
        self.capital = capital
        self.positions = {}
    
    @abstractmethod
    def generate_order(self, market_data: Dict) -> Optional['Order']:
        pass

class InstitutionalTrader(MarketParticipant):
    def __init__(self, participant_id: str, capital: float, 
                 strategy_type: str = "momentum"):
        super().__init__(participant_id, capital)
        self.strategy_type = strategy_type
        self.min_order_size = 10000
    
    def generate_order(self, market_data: Dict) -> Optional['Order']:
        if self.strategy_type == "momentum":
            return self._momentum_signal(market_data)
        return None
    
    def _momentum_signal(self, market_data: Dict) -> Optional['Order']:
        prices = market_data.get('prices', [])
        if len(prices) < 20:
            return None
        
        momentum = (prices[-1] - prices[-20]) / prices[-20]
        if momentum > 0.02:
            return Order('BUY', self.min_order_size, prices[-1])
        elif momentum < -0.02:
            return Order('SELL', self.min_order_size, prices[-1])
        return None

class RetailTrader(MarketParticipant):
    def __init__(self, participant_id: str, capital: float):
        super().__init__(participant_id, capital)
        self.max_order_size = 1000
    
    def generate_order(self, market_data: Dict) -> Optional['Order']:
        prices = market_data.get('prices', [])
        if len(prices) < 5:
            return None
        
        if np.random.random() < 0.1:
            side = 'BUY' if np.random.random() < 0.5 else 'SELL'
            size = np.random.randint(100, self.max_order_size)
            return Order(side, size, prices[-1])
        return None

@dataclass
class Order:
    side: str
    size: int
    price: float
    order_type: str = 'LIMIT'
    timestamp: pd.Timestamp = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
```

### Exchange Mechanisms

#### Continuous Double Auction
```python
from sortedcontainers import SortedDict
from collections import deque

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = SortedDict()  # price -> queue of orders
        self.asks = SortedDict()
        self.trades = []
        self.order_id_counter = 0
    
    def add_order(self, order: Order) -> List['Trade']:
        order.order_id = self.order_id_counter
        self.order_id_counter += 1
        
        trades = []
        if order.side == 'BUY':
            trades = self._match_buy_order(order)
            if order.size > 0:
                self._add_to_book(order, self.bids)
        else:
            trades = self._match_sell_order(order)
            if order.size > 0:
                self._add_to_book(order, self.asks)
        
        return trades
    
    def _match_buy_order(self, order: Order) -> List['Trade']:
        trades = []
        while order.size > 0 and len(self.asks) > 0:
            best_ask_price = self.asks.keys()[0]
            if order.price >= best_ask_price:
                ask_queue = self.asks[best_ask_price]
                matched_order = ask_queue[0]
                
                trade_size = min(order.size, matched_order.size)
                trade = Trade(
                    symbol=self.symbol,
                    price=best_ask_price,
                    size=trade_size,
                    buy_order_id=order.order_id,
                    sell_order_id=matched_order.order_id,
                    timestamp=pd.Timestamp.now()
                )
                trades.append(trade)
                self.trades.append(trade)
                
                order.size -= trade_size
                matched_order.size -= trade_size
                
                if matched_order.size == 0:
                    ask_queue.popleft()
                    if len(ask_queue) == 0:
                        del self.asks[best_ask_price]
            else:
                break
        
        return trades
    
    def _match_sell_order(self, order: Order) -> List['Trade']:
        trades = []
        while order.size > 0 and len(self.bids) > 0:
            best_bid_price = self.bids.keys()[-1]
            if order.price <= best_bid_price:
                bid_queue = self.bids[best_bid_price]
                matched_order = bid_queue[0]
                
                trade_size = min(order.size, matched_order.size)
                trade = Trade(
                    symbol=self.symbol,
                    price=best_bid_price,
                    size=trade_size,
                    buy_order_id=matched_order.order_id,
                    sell_order_id=order.order_id,
                    timestamp=pd.Timestamp.now()
                )
                trades.append(trade)
                self.trades.append(trade)
                
                order.size -= trade_size
                matched_order.size -= trade_size
                
                if matched_order.size == 0:
                    bid_queue.popleft()
                    if len(bid_queue) == 0:
                        del self.bids[best_bid_price]
            else:
                break
        
        return trades
    
    def _add_to_book(self, order: Order, book: SortedDict):
        if order.price not in book:
            book[order.price] = deque()
        book[order.price].append(order)
    
    def get_best_bid_ask(self) -> tuple:
        best_bid = self.bids.keys()[-1] if len(self.bids) > 0 else None
        best_ask = self.asks.keys()[0] if len(self.asks) > 0 else None
        return best_bid, best_ask
    
    def get_mid_price(self) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_depth(self, levels: int = 5) -> Dict:
        bid_depth = []
        for i, (price, orders) in enumerate(reversed(self.bids.items())):
            if i >= levels:
                break
            total_size = sum(order.size for order in orders)
            bid_depth.append({'price': price, 'size': total_size})
        
        ask_depth = []
        for i, (price, orders) in enumerate(self.asks.items()):
            if i >= levels:
                break
            total_size = sum(order.size for order in orders)
            ask_depth.append({'price': price, 'size': total_size})
        
        return {'bids': bid_depth, 'asks': ask_depth}

@dataclass
class Trade:
    symbol: str
    price: float
    size: int
    buy_order_id: int
    sell_order_id: int
    timestamp: pd.Timestamp
```

## Market Data Types and Sources

### Level 1, 2, 3 Data

#### Market Data Hierarchy
```python
from typing import List, Tuple

class MarketDataLevel(Enum):
    LEVEL_1 = 1  # Best bid/ask
    LEVEL_2 = 2  # Full order book depth
    LEVEL_3 = 3  # Individual orders with IDs

class MarketDataFeed:
    def __init__(self, level: MarketDataLevel):
        self.level = level
        self.subscribers = []
    
    def subscribe(self, callback):
        self.subscribers.append(callback)
    
    def publish_level1(self, symbol: str, bid: float, ask: float, 
                      bid_size: int, ask_size: int):
        data = {
            'type': 'level1',
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'timestamp': pd.Timestamp.now()
        }
        for callback in self.subscribers:
            callback(data)
    
    def publish_level2(self, symbol: str, bids: List[Tuple[float, int]], 
                      asks: List[Tuple[float, int]]):
        data = {
            'type': 'level2',
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': pd.Timestamp.now()
        }
        for callback in self.subscribers:
            callback(data)
    
    def publish_level3(self, symbol: str, orders: List[Dict]):
        data = {
            'type': 'level3',
            'symbol': symbol,
            'orders': orders,
            'timestamp': pd.Timestamp.now()
        }
        for callback in self.subscribers:
            callback(data)

class MarketDataProcessor:
    def __init__(self):
        self.data_buffer = []
        self.max_buffer_size = 10000
    
    def process_tick(self, data: Dict):
        self.data_buffer.append(data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
    
    def get_vwap(self, window: int = 100) -> float:
        recent_data = self.data_buffer[-window:]
        total_volume = sum(d.get('size', 0) for d in recent_data)
        if total_volume == 0:
            return 0
        
        vwap = sum(d.get('price', 0) * d.get('size', 0) 
                   for d in recent_data) / total_volume
        return vwap
    
    def get_microstructure_features(self) -> Dict:
        if len(self.data_buffer) < 100:
            return {}
        
        recent = self.data_buffer[-100:]
        prices = [d.get('price', 0) for d in recent]
        spreads = [d.get('spread', 0) for d in recent if 'spread' in d]
        
        return {
            'volatility': np.std(prices),
            'avg_spread': np.mean(spreads) if spreads else 0,
            'price_range': max(prices) - min(prices),
            'tick_frequency': len(recent) / 60  # ticks per minute
        }
```

### OHLCV Data

#### Candlestick Aggregation
```python
class OHLCVAggregator:
    def __init__(self, interval: str = '1min'):
        self.interval = interval
        self.current_candle = None
        self.candles = []
    
    def process_trade(self, trade: Trade):
        candle_timestamp = self._get_candle_timestamp(trade.timestamp)
        
        if self.current_candle is None or self.current_candle['timestamp'] != candle_timestamp:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
            
            self.current_candle = {
                'timestamp': candle_timestamp,
                'open': trade.price,
                'high': trade.price,
                'low': trade.price,
                'close': trade.price,
                'volume': trade.size
            }
        else:
            self.current_candle['high'] = max(self.current_candle['high'], trade.price)
            self.current_candle['low'] = min(self.current_candle['low'], trade.price)
            self.current_candle['close'] = trade.price
            self.current_candle['volume'] += trade.size
    
    def _get_candle_timestamp(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        if self.interval == '1min':
            return timestamp.floor('1min')
        elif self.interval == '5min':
            return timestamp.floor('5min')
        elif self.interval == '1H':
            return timestamp.floor('1H')
        elif self.interval == '1D':
            return timestamp.floor('1D')
        return timestamp
    
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.candles)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
```

### Alternative Data Sources

#### News and Sentiment Data
```python
import requests
from datetime import datetime, timedelta

class AlternativeDataCollector:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.data_cache = {}
    
    def fetch_news_sentiment(self, symbol: str, 
                            start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        cache_key = f"news_{symbol}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        news_data = []
        current_date = start_date
        while current_date <= end_date:
            articles = self._fetch_daily_news(symbol, current_date)
            for article in articles:
                sentiment = self._analyze_sentiment(article['text'])
                news_data.append({
                    'timestamp': article['published_at'],
                    'symbol': symbol,
                    'headline': article['headline'],
                    'sentiment_score': sentiment,
                    'source': article['source']
                })
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(news_data)
        self.data_cache[cache_key] = df
        return df
    
    def _fetch_daily_news(self, symbol: str, date: datetime) -> List[Dict]:
        return []
    
    def _analyze_sentiment(self, text: str) -> float:
        return 0.0
    
    def fetch_social_media_sentiment(self, symbol: str, 
                                    platform: str = 'twitter') -> pd.DataFrame:
        return pd.DataFrame()
    
    def fetch_sec_filings(self, symbol: str, filing_type: str = '10-K') -> List[Dict]:
        return []

class MarketDataIntegrator:
    def __init__(self):
        self.price_data = {}
        self.alternative_data = {}
    
    def integrate_data(self, symbol: str, 
                      price_df: pd.DataFrame,
                      news_df: pd.DataFrame) -> pd.DataFrame:
        price_df = price_df.copy()
        
        news_agg = news_df.groupby(news_df.index.date).agg({
            'sentiment_score': ['mean', 'std', 'count']
        })
        news_agg.columns = ['news_sentiment_mean', 'news_sentiment_std', 'news_count']
        
        price_df['date'] = price_df.index.date
        integrated = price_df.merge(news_agg, left_on='date', right_index=True, how='left')
        integrated.drop('date', axis=1, inplace=True)
        integrated.fillna(0, inplace=True)
        
        return integrated
```

## Time Series Characteristics

### Stylized Facts of Financial Returns

#### Fat Tails and Volatility Clustering
```python
from scipy import stats
from arch import arch_model

class FinancialTimeSeriesAnalyzer:
    def __init__(self, returns: pd.Series):
        self.returns = returns
        self.results = {}
    
    def test_normality(self) -> Dict:
        statistic, p_value = stats.jarque_bera(self.returns.dropna())
        
        return {
            'jarque_bera_statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': stats.skew(self.returns.dropna()),
            'kurtosis': stats.kurtosis(self.returns.dropna())
        }
    
    def test_stationarity(self) -> Dict:
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(self.returns.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    
    def test_autocorrelation(self, lags: int = 20) -> Dict:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        result = acorr_ljungbox(self.returns.dropna(), lags=lags)
        
        return {
            'ljung_box_statistic': result['lb_stat'].values,
            'p_values': result['lb_pvalue'].values,
            'has_autocorrelation': any(result['lb_pvalue'] < 0.05)
        }
    
    def test_volatility_clustering(self) -> Dict:
        squared_returns = self.returns ** 2
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        result = acorr_ljungbox(squared_returns.dropna(), lags=20)
        
        return {
            'has_volatility_clustering': any(result['lb_pvalue'] < 0.05),
            'arch_effect': result['lb_stat'].values[0]
        }
    
    def fit_garch_model(self, p: int = 1, q: int = 1) -> Dict:
        model = arch_model(self.returns.dropna() * 100, 
                          vol='Garch', p=p, q=q)
        fitted = model.fit(disp='off')
        
        return {
            'params': fitted.params.to_dict(),
            'aic': fitted.aic,
            'bic': fitted.bic,
            'conditional_volatility': fitted.conditional_volatility / 100
        }
    
    def calculate_stylized_facts(self) -> Dict:
        facts = {
            'normality_test': self.test_normality(),
            'stationarity_test': self.test_stationarity(),
            'autocorrelation_test': self.test_autocorrelation(),
            'volatility_clustering_test': self.test_volatility_clustering()
        }
        
        return facts

def demonstrate_stylized_facts():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    garch_sim = np.random.randn(1000) * 0.01
    for i in range(1, len(garch_sim)):
        garch_sim[i] = garch_sim[i] * (0.05 + 0.9 * garch_sim[i-1]**2)**0.5
    
    returns = pd.Series(garch_sim, index=dates)
    
    analyzer = FinancialTimeSeriesAnalyzer(returns)
    facts = analyzer.calculate_stylized_facts()
    
    print("Stylized Facts Analysis:")
    print(f"Normality: {facts['normality_test']}")
    print(f"Stationarity: {facts['stationarity_test']}")
    print(f"Volatility Clustering: {facts['volatility_clustering_test']}")
```

### Microstructure Noise

#### Bid-Ask Bounce and Price Discreteness
```python
class MicrostructureNoiseAnalyzer:
    def __init__(self, tick_data: pd.DataFrame):
        self.tick_data = tick_data
    
    def estimate_bid_ask_bounce(self) -> float:
        if 'price' not in self.tick_data.columns:
            return 0.0
        
        price_changes = self.tick_data['price'].diff()
        
        autocorr = price_changes.autocorr(lag=1)
        
        if autocorr < 0:
            spread_estimate = 2 * np.sqrt(-autocorr * price_changes.var())
            return spread_estimate
        return 0.0
    
    def calculate_realized_variance(self, sampling_freq: str = '5min') -> float:
        if 'price' not in self.tick_data.columns:
            return 0.0
        
        resampled = self.tick_data['price'].resample(sampling_freq).last()
        returns = resampled.pct_change().dropna()
        
        rv = (returns ** 2).sum()
        return rv
    
    def estimate_microstructure_noise_variance(self) -> float:
        rv_1min = self.calculate_realized_variance('1min')
        rv_5min = self.calculate_realized_variance('5min')
        
        noise_var = (rv_1min - rv_5min) / 4
        return max(0, noise_var)
    
    def optimal_sampling_frequency(self) -> str:
        frequencies = ['1min', '2min', '5min', '10min', '15min']
        rvs = [self.calculate_realized_variance(freq) for freq in frequencies]
        
        optimal_idx = np.argmin(rvs)
        return frequencies[optimal_idx]
```

## Trading Mechanics and Order Types

### Order Types

#### Market, Limit, Stop Orders
```python
from enum import Enum
from typing import Optional

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "good_till_cancel"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"

@dataclass
class AdvancedOrder:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    display_quantity: Optional[int] = None  # For iceberg orders
    order_id: Optional[str] = None
    timestamp: Optional[pd.Timestamp] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.value}"

class OrderExecutionEngine:
    def __init__(self):
        self.pending_orders = {}
        self.executed_orders = []
    
    def submit_order(self, order: AdvancedOrder, 
                    market_data: Dict) -> Dict:
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, market_data)
        elif order.order_type == OrderType.LIMIT:
            return self._handle_limit_order(order, market_data)
        elif order.order_type == OrderType.STOP:
            return self._handle_stop_order(order, market_data)
        elif order.order_type == OrderType.ICEBERG:
            return self._handle_iceberg_order(order, market_data)
        
        return {'status': 'rejected', 'reason': 'unsupported_order_type'}
    
    def _execute_market_order(self, order: AdvancedOrder, 
                             market_data: Dict) -> Dict:
        current_price = market_data.get('last_price', 0)
        
        slippage = self._calculate_slippage(order.quantity, market_data)
        execution_price = current_price * (1 + slippage if order.side == OrderSide.BUY else 1 - slippage)
        
        execution = {
            'order_id': order.order_id,
            'status': 'filled',
            'filled_quantity': order.quantity,
            'avg_price': execution_price,
            'timestamp': pd.Timestamp.now()
        }
        
        self.executed_orders.append(execution)
        return execution
    
    def _handle_limit_order(self, order: AdvancedOrder, 
                           market_data: Dict) -> Dict:
        current_price = market_data.get('last_price', 0)
        
        if order.side == OrderSide.BUY and current_price <= order.price:
            return self._execute_market_order(order, market_data)
        elif order.side == OrderSide.SELL and current_price >= order.price:
            return self._execute_market_order(order, market_data)
        else:
            self.pending_orders[order.order_id] = order
            return {'status': 'pending', 'order_id': order.order_id}
    
    def _handle_stop_order(self, order: AdvancedOrder, 
                          market_data: Dict) -> Dict:
        current_price = market_data.get('last_price', 0)
        
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            return self._execute_market_order(order, market_data)
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            return self._execute_market_order(order, market_data)
        else:
            self.pending_orders[order.order_id] = order
            return {'status': 'pending', 'order_id': order.order_id}
    
    def _handle_iceberg_order(self, order: AdvancedOrder, 
                             market_data: Dict) -> Dict:
        display_qty = order.display_quantity or order.quantity // 10
        
        total_filled = 0
        executions = []
        
        while total_filled < order.quantity:
            slice_qty = min(display_qty, order.quantity - total_filled)
            slice_order = AdvancedOrder(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=slice_qty,
                price=order.price
            )
            
            execution = self._execute_market_order(slice_order, market_data)
            executions.append(execution)
            total_filled += execution['filled_quantity']
        
        return {
            'status': 'filled',
            'order_id': order.order_id,
            'total_filled': total_filled,
            'executions': executions
        }
    
    def _calculate_slippage(self, quantity: int, market_data: Dict) -> float:
        avg_volume = market_data.get('avg_volume', 1000000)
        participation_rate = quantity / avg_volume
        
        slippage = 0.0001 * np.sqrt(participation_rate * 100)
        return min(slippage, 0.01)
    
    def update_pending_orders(self, market_data: Dict):
        filled_orders = []
        
        for order_id, order in self.pending_orders.items():
            current_price = market_data.get('last_price', 0)
            
            should_execute = False
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_execute = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_execute = True
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_execute = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_execute = True
            
            if should_execute:
                execution = self._execute_market_order(order, market_data)
                filled_orders.append(order_id)
        
        for order_id in filled_orders:
            del self.pending_orders[order_id]
```

## Financial Instruments

### Equities

#### Stock Characteristics
```python
@dataclass
class Stock:
    symbol: str
    company_name: str
    sector: str
    market_cap: float
    shares_outstanding: int
    
    def calculate_price_from_market_cap(self) -> float:
        return self.market_cap / self.shares_outstanding
    
    def calculate_market_cap_from_price(self, price: float) -> float:
        return price * self.shares_outstanding

class StockUniverse:
    def __init__(self):
        self.stocks = {}
    
    def add_stock(self, stock: Stock):
        self.stocks[stock.symbol] = stock
    
    def get_sector_stocks(self, sector: str) -> List[Stock]:
        return [stock for stock in self.stocks.values() 
                if stock.sector == sector]
    
    def get_market_cap_weighted_index(self, symbols: List[str]) -> float:
        total_market_cap = sum(self.stocks[symbol].market_cap 
                              for symbol in symbols if symbol in self.stocks)
        
        if total_market_cap == 0:
            return 0
        
        weights = {symbol: self.stocks[symbol].market_cap / total_market_cap 
                  for symbol in symbols if symbol in self.stocks}
        
        return weights
```

### Fixed Income

#### Bond Pricing and Duration
```python
class Bond:
    def __init__(self, face_value: float, coupon_rate: float, 
                 maturity_years: float, frequency: int = 2):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.frequency = frequency
    
    def price(self, ytm: float) -> float:
        n = int(self.maturity_years * self.frequency)
        coupon = self.face_value * self.coupon_rate / self.frequency
        
        pv_coupons = sum(coupon / (1 + ytm/self.frequency)**(i+1) 
                        for i in range(n))
        pv_face = self.face_value / (1 + ytm/self.frequency)**n
        
        return pv_coupons + pv_face
    
    def duration(self, ytm: float) -> float:
        n = int(self.maturity_years * self.frequency)
        coupon = self.face_value * self.coupon_rate / self.frequency
        price = self.price(ytm)
        
        weighted_cf = sum((i+1) * coupon / (1 + ytm/self.frequency)**(i+1) 
                         for i in range(n))
        weighted_cf += n * self.face_value / (1 + ytm/self.frequency)**n
        
        duration = weighted_cf / (price * self.frequency)
        return duration
    
    def modified_duration(self, ytm: float) -> float:
        mac_duration = self.duration(ytm)
        return mac_duration / (1 + ytm/self.frequency)
    
    def convexity(self, ytm: float) -> float:
        n = int(self.maturity_years * self.frequency)
        coupon = self.face_value * self.coupon_rate / self.frequency
        price = self.price(ytm)
        
        weighted_cf = sum((i+1) * (i+2) * coupon / (1 + ytm/self.frequency)**(i+1) 
                         for i in range(n))
        weighted_cf += n * (n+1) * self.face_value / (1 + ytm/self.frequency)**n
        
        convexity = weighted_cf / (price * self.frequency**2)
        return convexity
    
    def dv01(self, ytm: float) -> float:
        return self.modified_duration(ytm) * self.price(ytm) * 0.0001
```

### Derivatives

#### Options Pricing (Black-Scholes)
```python
from scipy.stats import norm

class EuropeanOption:
    def __init__(self, option_type: str, strike: float, 
                 maturity: float, underlying_price: float,
                 risk_free_rate: float, volatility: float):
        self.option_type = option_type.upper()
        self.strike = strike
        self.maturity = maturity
        self.underlying_price = underlying_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
    
    def _d1(self) -> float:
        numerator = (np.log(self.underlying_price / self.strike) + 
                    (self.risk_free_rate + 0.5 * self.volatility**2) * self.maturity)
        denominator = self.volatility * np.sqrt(self.maturity)
        return numerator / denominator
    
    def _d2(self) -> float:
        return self._d1() - self.volatility * np.sqrt(self.maturity)
    
    def price(self) -> float:
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'CALL':
            price = (self.underlying_price * norm.cdf(d1) - 
                    self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(d2))
        else:  # PUT
            price = (self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2) - 
                    self.underlying_price * norm.cdf(-d1))
        
        return price
    
    def delta(self) -> float:
        d1 = self._d1()
        if self.option_type == 'CALL':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self) -> float:
        d1 = self._d1()
        return norm.pdf(d1) / (self.underlying_price * self.volatility * np.sqrt(self.maturity))
    
    def vega(self) -> float:
        d1 = self._d1()
        return self.underlying_price * norm.pdf(d1) * np.sqrt(self.maturity) / 100
    
    def theta(self) -> float:
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'CALL':
            theta = (-(self.underlying_price * norm.pdf(d1) * self.volatility) / 
                    (2 * np.sqrt(self.maturity)) - 
                    self.risk_free_rate * self.strike * 
                    np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(d2))
        else:
            theta = (-(self.underlying_price * norm.pdf(d1) * self.volatility) / 
                    (2 * np.sqrt(self.maturity)) + 
                    self.risk_free_rate * self.strike * 
                    np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2))
        
        return theta / 365
    
    def rho(self) -> float:
        d2 = self._d2()
        
        if self.option_type == 'CALL':
            return (self.strike * self.maturity * 
                   np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(d2) / 100)
        else:
            return (-self.strike * self.maturity * 
                   np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2) / 100)
    
    def implied_volatility(self, market_price: float, 
                          tolerance: float = 0.0001, 
                          max_iterations: int = 100) -> float:
        sigma = 0.5
        
        for _ in range(max_iterations):
            self.volatility = sigma
            price = self.price()
            vega = self.vega()
            
            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma
            
            sigma = sigma + diff / (vega * 100)
            sigma = max(0.01, min(sigma, 5.0))
        
        return sigma
```

## Market Microstructure Theory

### Information Asymmetry Models

#### Kyle's Lambda
```python
class KyleModel:
    def __init__(self, informed_traders: int, noise_traders: int):
        self.informed_traders = informed_traders
        self.noise_traders = noise_traders
        self.trades = []
    
    def calculate_kyle_lambda(self, price_changes: np.ndarray, 
                             order_flow: np.ndarray) -> float:
        if len(price_changes) != len(order_flow):
            raise ValueError("Arrays must have same length")
        
        covariance = np.cov(price_changes, order_flow)[0, 1]
        variance = np.var(order_flow)
        
        if variance == 0:
            return 0
        
        kyle_lambda = covariance / variance
        return kyle_lambda
    
    def estimate_information_content(self, trades_df: pd.DataFrame) -> Dict:
        trades_df['price_change'] = trades_df['price'].diff()
        trades_df['signed_volume'] = trades_df['volume'] * np.where(
            trades_df['side'] == 'BUY', 1, -1
        )
        
        lambda_val = self.calculate_kyle_lambda(
            trades_df['price_change'].dropna().values,
            trades_df['signed_volume'].dropna().values
        )
        
        return {
            'kyle_lambda': lambda_val,
            'price_impact_per_unit': lambda_val,
            'market_depth': 1 / lambda_val if lambda_val != 0 else np.inf
        }
```

### Adverse Selection

#### Probability of Informed Trading (PIN)
```python
class PINModel:
    def __init__(self, trades: pd.DataFrame):
        self.trades = trades
    
    def classify_trades(self) -> pd.DataFrame:
        df = self.trades.copy()
        
        df['trade_classification'] = np.where(
            df['price'] > df['mid_price'], 'BUY',
            np.where(df['price'] < df['mid_price'], 'SELL', 'UNKNOWN')
        )
        
        return df
    
    def estimate_pin(self, days: int = 60) -> Dict:
        df = self.classify_trades()
        
        daily_stats = df.groupby(df.index.date).agg({
            'trade_classification': lambda x: {
                'buys': (x == 'BUY').sum(),
                'sells': (x == 'SELL').sum()
            }
        })
        
        buys = [stats['buys'] for stats in daily_stats['trade_classification']]
        sells = [stats['sells'] for stats in daily_stats['trade_classification']]
        
        alpha = 0.5
        delta = 0.5
        mu = np.mean(buys + sells)
        epsilon_b = np.mean(buys)
        epsilon_s = np.mean(sells)
        
        pin = (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s)
        
        return {
            'pin': pin,
            'alpha': alpha,
            'delta': delta,
            'mu': mu,
            'epsilon_b': epsilon_b,
            'epsilon_s': epsilon_s
        }
```

## Data Quality and Preprocessing

### Data Cleaning Pipeline

```python
class FinancialDataCleaner:
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        initial_rows = len(df)
        
        df = df[df['volume'] > 0]
        df = df[df['high'] >= df['low']]
        df = df[(df['high'] >= df['open']) & (df['high'] >= df['close'])]
        df = df[(df['low'] <= df['open']) & (df['low'] <= df['close'])]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
        
        df['returns'] = df['close'].pct_change()
        df = df[abs(df['returns']) < 0.5]
        df.drop('returns', axis=1, inplace=True)
        
        self.cleaning_stats['removed_rows'] = initial_rows - len(df)
        self.cleaning_stats['removal_rate'] = (initial_rows - len(df)) / initial_rows
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, 
                           method: str = 'forward_fill') -> pd.DataFrame:
        df = df.copy()
        
        if method == 'forward_fill':
            df.fillna(method='ffill', inplace=True)
        elif method == 'interpolate':
            df.interpolate(method='time', inplace=True)
        elif method == 'drop':
            df.dropna(inplace=True)
        
        return df
    
    def adjust_for_corporate_actions(self, df: pd.DataFrame, 
                                    splits: List[Dict],
                                    dividends: List[Dict]) -> pd.DataFrame:
        df = df.copy()
        
        for split in splits:
            split_date = pd.Timestamp(split['date'])
            split_ratio = split['ratio']
            
            mask = df.index < split_date
            df.loc[mask, ['open', 'high', 'low', 'close']] /= split_ratio
            df.loc[mask, 'volume'] *= split_ratio
        
        for dividend in dividends:
            div_date = pd.Timestamp(dividend['date'])
            div_amount = dividend['amount']
            
            mask = df.index < div_date
            df.loc[mask, ['open', 'high', 'low', 'close']] -= div_amount
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        df['z_score'] = (df['returns'] - df['returns'].rolling(20).mean()) / df['volatility']
        df['is_anomaly'] = abs(df['z_score']) > 3
        
        return df

class DataValidator:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Dict:
        issues = []
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            issues.append("Missing required columns")
        
        if (df['high'] < df['low']).any():
            issues.append("High < Low detected")
        
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            issues.append("High < Open/Close detected")
        
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            issues.append("Low > Open/Close detected")
        
        if (df['volume'] < 0).any():
            issues.append("Negative volume detected")
        
        if df.index.duplicated().any():
            issues.append("Duplicate timestamps detected")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_rows': len(df),
            'date_range': (df.index.min(), df.index.max())
        }
```

## Summary

This module covered the foundational aspects of financial markets and data:

1. **Market Structure**: Understanding different market types, participants, and exchange mechanisms
2. **Data Types**: Level 1/2/3 data, OHLCV aggregation, and alternative data sources
3. **Time Series Properties**: Stylized facts, volatility clustering, and microstructure noise
4. **Trading Mechanics**: Order types, execution algorithms, and order flow
5. **Financial Instruments**: Equities, bonds, and derivatives with pricing models
6. **Microstructure Theory**: Information asymmetry, adverse selection, and market impact
7. **Data Quality**: Cleaning pipelines, validation, and preprocessing techniques

These fundamentals provide the necessary foundation for applying AI and machine learning techniques to financial markets in subsequent modules.

