# Module 4: Machine Learning for Finance

## Table of Contents
1. [Feature Engineering for Financial Data](#feature-engineering-for-financial-data)
2. [Classification and Regression for Trading](#classification-and-regression-for-trading)
3. [Time Series Forecasting](#time-series-forecasting)
4. [Anomaly Detection](#anomaly-detection)
5. [Model Evaluation for Finance](#model-evaluation-for-finance)
6. [Advanced ML Techniques](#advanced-ml-techniques)

## Feature Engineering for Financial Data

### Technical Indicators

```python
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import talib

class TechnicalFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
    
    def add_price_features(self):
        self.features['returns'] = self.df['close'].pct_change()
        self.features['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        for window in [5, 10, 20, 50, 200]:
            self.features[f'sma_{window}'] = self.df['close'].rolling(window).mean()
            self.features[f'ema_{window}'] = self.df['close'].ewm(span=window).mean()
            self.features[f'price_to_sma_{window}'] = self.df['close'] / self.features[f'sma_{window}']
        
        for window in [10, 20, 50]:
            self.features[f'bb_upper_{window}'] = (
                self.features[f'sma_{window}'] + 2 * self.df['close'].rolling(window).std()
            )
            self.features[f'bb_lower_{window}'] = (
                self.features[f'sma_{window}'] - 2 * self.df['close'].rolling(window).std()
            )
            self.features[f'bb_width_{window}'] = (
                (self.features[f'bb_upper_{window}'] - self.features[f'bb_lower_{window}']) / 
                self.features[f'sma_{window}']
            )
            self.features[f'bb_position_{window}'] = (
                (self.df['close'] - self.features[f'bb_lower_{window}']) / 
                (self.features[f'bb_upper_{window}'] - self.features[f'bb_lower_{window}'])
            )
        
        return self
    
    def add_momentum_features(self):
        for window in [5, 10, 20, 50]:
            self.features[f'roc_{window}'] = (
                (self.df['close'] - self.df['close'].shift(window)) / 
                self.df['close'].shift(window)
            )
        
        for window in [14, 21]:
            self.features[f'rsi_{window}'] = self._calculate_rsi(self.df['close'], window)
        
        self.features['macd'], self.features['macd_signal'], self.features['macd_hist'] = \
            self._calculate_macd(self.df['close'])
        
        for window in [14, 21]:
            self.features[f'stoch_k_{window}'], self.features[f'stoch_d_{window}'] = \
                self._calculate_stochastic(window)
        
        for window in [10, 20]:
            self.features[f'williams_r_{window}'] = self._calculate_williams_r(window)
        
        return self
    
    def add_volatility_features(self):
        for window in [5, 10, 20, 50]:
            self.features[f'volatility_{window}'] = (
                self.df['close'].pct_change().rolling(window).std() * np.sqrt(252)
            )
            self.features[f'parkinson_vol_{window}'] = self._parkinson_volatility(window)
            self.features[f'garman_klass_vol_{window}'] = self._garman_klass_volatility(window)
        
        for window in [14, 21]:
            self.features[f'atr_{window}'] = self._calculate_atr(window)
            self.features[f'atr_percent_{window}'] = (
                self.features[f'atr_{window}'] / self.df['close']
            )
        
        return self
    
    def add_volume_features(self):
        for window in [5, 10, 20]:
            self.features[f'volume_sma_{window}'] = self.df['volume'].rolling(window).mean()
            self.features[f'volume_ratio_{window}'] = (
                self.df['volume'] / self.features[f'volume_sma_{window}']
            )
        
        self.features['obv'] = self._calculate_obv()
        self.features['vwap'] = self._calculate_vwap()
        
        for window in [14, 21]:
            self.features[f'mfi_{window}'] = self._calculate_mfi(window)
        
        self.features['ad_line'] = self._calculate_ad_line()
        
        return self
    
    def add_pattern_features(self):
        self.features['doji'] = self._is_doji()
        self.features['hammer'] = self._is_hammer()
        self.features['shooting_star'] = self._is_shooting_star()
        self.features['engulfing_bullish'] = self._is_engulfing_bullish()
        self.features['engulfing_bearish'] = self._is_engulfing_bearish()
        
        return self
    
    def add_microstructure_features(self):
        self.features['hl_ratio'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.features['co_ratio'] = (self.df['close'] - self.df['open']) / self.df['close']
        
        for window in [5, 10, 20]:
            self.features[f'high_low_range_{window}'] = (
                self.df['high'].rolling(window).max() - 
                self.df['low'].rolling(window).min()
            ) / self.df['close']
        
        self.features['upper_shadow'] = (
            (self.df['high'] - self.df[['open', 'close']].max(axis=1)) / 
            (self.df['high'] - self.df['low'])
        )
        self.features['lower_shadow'] = (
            (self.df[['open', 'close']].min(axis=1) - self.df['low']) / 
            (self.df['high'] - self.df['low'])
        )
        
        return self
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_stochastic(self, window: int = 14):
        low_min = self.df['low'].rolling(window).min()
        high_max = self.df['high'].rolling(window).max()
        k = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(3).mean()
        return k, d
    
    def _calculate_williams_r(self, window: int = 14):
        high_max = self.df['high'].rolling(window).max()
        low_min = self.df['low'].rolling(window).min()
        return -100 * (high_max - self.df['close']) / (high_max - low_min)
    
    def _parkinson_volatility(self, window: int):
        return np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(self.df['high'] / self.df['low']) ** 2).rolling(window).mean())
        ) * np.sqrt(252)
    
    def _garman_klass_volatility(self, window: int):
        log_hl = np.log(self.df['high'] / self.df['low']) ** 2
        log_co = np.log(self.df['close'] / self.df['open']) ** 2
        return np.sqrt(
            0.5 * log_hl.rolling(window).mean() - 
            (2 * np.log(2) - 1) * log_co.rolling(window).mean()
        ) * np.sqrt(252)
    
    def _calculate_atr(self, window: int = 14):
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def _calculate_obv(self):
        obv = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_vwap(self):
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        return (typical_price * self.df['volume']).cumsum() / self.df['volume'].cumsum()
    
    def _calculate_mfi(self, window: int = 14):
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        money_flow = typical_price * self.df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window).sum()
        negative_mf = negative_flow.rolling(window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_ad_line(self):
        clv = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / \
              (self.df['high'] - self.df['low'])
        ad = (clv * self.df['volume']).cumsum()
        return ad
    
    def _is_doji(self, threshold=0.001):
        body = abs(self.df['close'] - self.df['open'])
        range_hl = self.df['high'] - self.df['low']
        return (body / range_hl) < threshold
    
    def _is_hammer(self):
        body = abs(self.df['close'] - self.df['open'])
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        return (lower_shadow > 2 * body) & (upper_shadow < body)
    
    def _is_shooting_star(self):
        body = abs(self.df['close'] - self.df['open'])
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        return (upper_shadow > 2 * body) & (lower_shadow < body)
    
    def _is_engulfing_bullish(self):
        prev_bearish = self.df['close'].shift(1) < self.df['open'].shift(1)
        curr_bullish = self.df['close'] > self.df['open']
        engulfing = (self.df['open'] < self.df['close'].shift(1)) & \
                   (self.df['close'] > self.df['open'].shift(1))
        return prev_bearish & curr_bullish & engulfing
    
    def _is_engulfing_bearish(self):
        prev_bullish = self.df['close'].shift(1) > self.df['open'].shift(1)
        curr_bearish = self.df['close'] < self.df['open']
        engulfing = (self.df['open'] > self.df['close'].shift(1)) & \
                   (self.df['close'] < self.df['open'].shift(1))
        return prev_bullish & curr_bearish & engulfing
    
    def get_features(self) -> pd.DataFrame:
        return self.features

class FundamentalFeatureEngineer:
    @staticmethod
    def calculate_valuation_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=fundamentals.index)
        
        features['pe_ratio'] = fundamentals['price'] / fundamentals['eps']
        features['pb_ratio'] = fundamentals['price'] / fundamentals['book_value_per_share']
        features['ps_ratio'] = fundamentals['market_cap'] / fundamentals['revenue']
        features['pcf_ratio'] = fundamentals['price'] / fundamentals['cash_flow_per_share']
        features['peg_ratio'] = features['pe_ratio'] / fundamentals['earnings_growth_rate']
        
        features['ev_ebitda'] = (
            (fundamentals['market_cap'] + fundamentals['total_debt'] - 
             fundamentals['cash']) / fundamentals['ebitda']
        )
        
        return features
    
    @staticmethod
    def calculate_profitability_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=fundamentals.index)
        
        features['roe'] = fundamentals['net_income'] / fundamentals['shareholders_equity']
        features['roa'] = fundamentals['net_income'] / fundamentals['total_assets']
        features['roic'] = (
            fundamentals['net_income'] / 
            (fundamentals['total_debt'] + fundamentals['shareholders_equity'])
        )
        
        features['gross_margin'] = (
            (fundamentals['revenue'] - fundamentals['cogs']) / fundamentals['revenue']
        )
        features['operating_margin'] = fundamentals['operating_income'] / fundamentals['revenue']
        features['net_margin'] = fundamentals['net_income'] / fundamentals['revenue']
        
        return features
    
    @staticmethod
    def calculate_liquidity_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=fundamentals.index)
        
        features['current_ratio'] = (
            fundamentals['current_assets'] / fundamentals['current_liabilities']
        )
        features['quick_ratio'] = (
            (fundamentals['current_assets'] - fundamentals['inventory']) / 
            fundamentals['current_liabilities']
        )
        features['cash_ratio'] = (
            fundamentals['cash'] / fundamentals['current_liabilities']
        )
        
        return features
    
    @staticmethod
    def calculate_leverage_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=fundamentals.index)
        
        features['debt_to_equity'] = (
            fundamentals['total_debt'] / fundamentals['shareholders_equity']
        )
        features['debt_to_assets'] = fundamentals['total_debt'] / fundamentals['total_assets']
        features['interest_coverage'] = (
            fundamentals['ebit'] / fundamentals['interest_expense']
        )
        
        return features
```

### Lag Features and Rolling Statistics

```python
class TimeSeriesFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.features = pd.DataFrame(index=df.index)
    
    def add_lag_features(self, columns: List[str], lags: List[int]):
        for col in columns:
            for lag in lags:
                self.features[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        return self
    
    def add_rolling_features(self, columns: List[str], windows: List[int]):
        for col in columns:
            for window in windows:
                self.features[f'{col}_rolling_mean_{window}'] = (
                    self.df[col].rolling(window).mean()
                )
                self.features[f'{col}_rolling_std_{window}'] = (
                    self.df[col].rolling(window).std()
                )
                self.features[f'{col}_rolling_min_{window}'] = (
                    self.df[col].rolling(window).min()
                )
                self.features[f'{col}_rolling_max_{window}'] = (
                    self.df[col].rolling(window).max()
                )
                self.features[f'{col}_rolling_skew_{window}'] = (
                    self.df[col].rolling(window).skew()
                )
                self.features[f'{col}_rolling_kurt_{window}'] = (
                    self.df[col].rolling(window).kurt()
                )
        return self
    
    def add_expanding_features(self, columns: List[str]):
        for col in columns:
            self.features[f'{col}_expanding_mean'] = self.df[col].expanding().mean()
            self.features[f'{col}_expanding_std'] = self.df[col].expanding().std()
            self.features[f'{col}_expanding_min'] = self.df[col].expanding().min()
            self.features[f'{col}_expanding_max'] = self.df[col].expanding().max()
        return self
    
    def add_ewm_features(self, columns: List[str], spans: List[int]):
        for col in columns:
            for span in spans:
                self.features[f'{col}_ewm_mean_{span}'] = (
                    self.df[col].ewm(span=span).mean()
                )
                self.features[f'{col}_ewm_std_{span}'] = (
                    self.df[col].ewm(span=span).std()
                )
        return self
    
    def add_diff_features(self, columns: List[str], periods: List[int]):
        for col in columns:
            for period in periods:
                self.features[f'{col}_diff_{period}'] = self.df[col].diff(period)
                self.features[f'{col}_pct_change_{period}'] = self.df[col].pct_change(period)
        return self
    
    def add_fourier_features(self, column: str, n_components: int = 5):
        from scipy.fft import fft
        
        values = self.df[column].dropna().values
        fft_values = fft(values)
        
        for i in range(1, n_components + 1):
            self.features[f'{column}_fft_real_{i}'] = np.real(fft_values[i])
            self.features[f'{column}_fft_imag_{i}'] = np.imag(fft_values[i])
        
        return self
    
    def get_features(self) -> pd.DataFrame:
        return self.features
```

## Classification and Regression for Trading

### Direction Prediction

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DirectionPredictor:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.feature_importance = None
    
    def _initialize_model(self):
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        return models.get(self.model_type, models['random_forest'])
    
    def prepare_labels(self, returns: pd.Series, horizon: int = 1, 
                      threshold: float = 0.0) -> pd.Series:
        future_returns = returns.shift(-horizon)
        
        labels = pd.Series(0, index=returns.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1
        
        return labels
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'accuracy': cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy'),
            'precision': cross_val_score(self.model, X, y, cv=tscv, scoring='precision_weighted'),
            'recall': cross_val_score(self.model, X, y, cv=tscv, scoring='recall_weighted'),
            'f1': cross_val_score(self.model, X, y, cv=tscv, scoring='f1_weighted')
        }
        
        return {
            metric: {'mean': scores[metric].mean(), 'std': scores[metric].std()}
            for metric in scores
        }

class ReturnPredictor:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge, Lasso
        from xgboost import XGBRegressor
        
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        return models.get(self.model_type, models['random_forest'])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = self.predict(X_test)
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
```

## Time Series Forecasting

### ARIMA and GARCH Models

```python
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, acf, pacf

class ARIMAForecaster:
    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, data: pd.Series):
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model.summary()
    
    def forecast(self, steps: int) -> pd.Series:
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_residuals(self) -> pd.Series:
        return self.fitted_model.resid
    
    @staticmethod
    def auto_arima(data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                    except:
                        continue
        
        return best_model, best_order, best_aic

class GARCHForecaster:
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, returns: pd.Series):
        self.model = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q)
        self.fitted_model = self.model.fit(disp='off')
        return self.fitted_model.summary()
    
    def forecast_variance(self, horizon: int = 1) -> pd.DataFrame:
        forecast = self.fitted_model.forecast(horizon=horizon)
        return forecast.variance / 10000
    
    def conditional_volatility(self) -> pd.Series:
        return self.fitted_model.conditional_volatility / 100
```

### Prophet and Neural Prophet

```python
class ProphetForecaster:
    def __init__(self):
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.model = None
        except ImportError:
            raise ImportError("Prophet not installed. Install with: pip install prophet")
    
    def fit(self, df: pd.DataFrame, date_col: str = 'ds', value_col: str = 'y'):
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        self.model = self.Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.model.fit(prophet_df)
    
    def forecast(self, periods: int, freq: str = 'D') -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def add_regressor(self, name: str, df: pd.DataFrame):
        self.model.add_regressor(name)
```

## Anomaly Detection

### Statistical Methods

```python
class AnomalyDetector:
    @staticmethod
    def z_score_detection(data: pd.Series, threshold: float = 3.0) -> pd.Series:
        mean = data.mean()
        std = data.std()
        z_scores = (data - mean) / std
        return abs(z_scores) > threshold
    
    @staticmethod
    def iqr_detection(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def rolling_z_score_detection(data: pd.Series, window: int = 20, 
                                  threshold: float = 3.0) -> pd.Series:
        rolling_mean = data.rolling(window).mean()
        rolling_std = data.rolling(window).std()
        z_scores = (data - rolling_mean) / rolling_std
        return abs(z_scores) > threshold

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=contamination, random_state=42)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.model.fit_predict(X)
        return predictions == -1
    
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X)

class AutoencoderAnomalyDetector:
    def __init__(self, encoding_dim: int = 10):
        import torch
        import torch.nn as nn
        
        self.encoding_dim = encoding_dim
        self.model = None
        self.threshold = None
    
    def build_model(self, input_dim: int):
        import torch.nn as nn
        
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, encoding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self.model = Autoencoder(input_dim, self.encoding_dim)
    
    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        import torch
        import torch.optim as optim
        
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        X_tensor = torch.FloatTensor(X_train)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
            self.threshold = torch.quantile(mse, 0.95).item()
    
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        import torch
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
        
        return mse.numpy() > self.threshold
```

## Model Evaluation for Finance

### Walk-Forward Analysis

```python
class WalkForwardValidator:
    def __init__(self, train_size: int, test_size: int, step_size: int):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, data: pd.DataFrame):
        n = len(data)
        splits = []
        
        start = 0
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            
            train_indices = data.index[start:train_end]
            test_indices = data.index[train_end:test_end]
            
            splits.append((train_indices, test_indices))
            start += self.step_size
        
        return splits
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      metric_func) -> List[float]:
        splits = self.split(X)
        scores = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            score = metric_func(y_test, y_pred)
            scores.append(score)
        
        return scores

class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            train_indices = np.concatenate([
                indices[:max(0, test_start - embargo_size)],
                indices[min(n_samples, test_end + embargo_size):]
            ])
            
            yield train_indices, test_indices
```

### Financial-Specific Metrics

```python
class FinancialMetrics:
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.sign(y_true) == np.sign(y_pred))
    
    @staticmethod
    def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
        correct_predictions = np.sum((y_pred > threshold) & (y_true > 0)) + \
                            np.sum((y_pred < -threshold) & (y_true < 0))
        total_predictions = np.sum(np.abs(y_pred) > threshold)
        return correct_predictions / total_predictions if total_predictions > 0 else 0
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        return gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def average_win_loss_ratio(returns: np.ndarray) -> float:
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        return avg_win / avg_loss if avg_loss > 0 else np.inf
    
    @staticmethod
    def expectancy(returns: np.ndarray) -> float:
        win_rate = np.sum(returns > 0) / len(returns)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
```

## Advanced ML Techniques

### Ensemble Methods

```python
class StackingEnsemble:
    def __init__(self, base_models: List, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        for model in self.base_models:
            model.fit(X_train, y_train)
        
        base_predictions = np.column_stack([
            model.predict(X_train) for model in self.base_models
        ])
        
        self.meta_model.fit(base_predictions, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        
        return self.meta_model.predict(base_predictions)

class VotingEnsemble:
    def __init__(self, models: List, voting: str = 'soft'):
        self.models = models
        self.voting = voting
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        for model in self.models:
            model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.voting == 'hard':
            predictions = np.array([model.predict(X) for model in self.models])
            from scipy import stats
            return stats.mode(predictions, axis=0)[0].flatten()
        else:
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)
```

## Summary

This module covered machine learning fundamentals for finance:

1. **Feature Engineering**: Technical indicators, fundamental ratios, time series features
2. **Classification**: Direction prediction, model selection, evaluation
3. **Regression**: Return prediction, performance metrics
4. **Time Series**: ARIMA, GARCH, Prophet forecasting
5. **Anomaly Detection**: Statistical methods, isolation forests, autoencoders
6. **Model Evaluation**: Walk-forward analysis, purged k-fold, financial metrics
7. **Advanced Techniques**: Ensemble methods, stacking, voting

These ML techniques form the foundation for building sophisticated trading and investment systems.

