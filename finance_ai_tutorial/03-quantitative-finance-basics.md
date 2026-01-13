# Module 3: Quantitative Finance Basics

## Table of Contents
1. [Statistical Measures for Finance](#statistical-measures-for-finance)
2. [Risk Metrics](#risk-metrics)
3. [Portfolio Theory Fundamentals](#portfolio-theory-fundamentals)
4. [Option Pricing Theory](#option-pricing-theory)
5. [Stochastic Calculus for Finance](#stochastic-calculus-for-finance)
6. [Backtesting Frameworks](#backtesting-frameworks)

## Statistical Measures for Finance

### Returns Calculation

#### Different Return Types
```python
import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats

class ReturnsCalculator:
    @staticmethod
    def simple_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change()
    
    @staticmethod
    def log_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def cumulative_returns(returns: pd.Series) -> pd.Series:
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
        return annualized
    
    @staticmethod
    def rolling_returns(prices: pd.Series, window: int) -> pd.Series:
        return prices.pct_change(window)
    
    @staticmethod
    def excess_returns(returns: pd.Series, risk_free_rate: float) -> pd.Series:
        return returns - risk_free_rate / 252

class DistributionAnalyzer:
    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
    
    def calculate_moments(self) -> dict:
        return {
            'mean': self.returns.mean(),
            'variance': self.returns.var(),
            'std': self.returns.std(),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns),
            'excess_kurtosis': stats.kurtosis(self.returns) - 3
        }
    
    def test_normality(self) -> dict:
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        ks_stat, ks_pvalue = stats.kstest(self.returns, 'norm', 
                                          args=(self.returns.mean(), self.returns.std()))
        sw_stat, sw_pvalue = stats.shapiro(self.returns[:5000])
        
        return {
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_pvalue},
            'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pvalue},
            'is_normal': jb_pvalue > 0.05 and ks_pvalue > 0.05
        }
    
    def fit_distribution(self, dist_name: str = 't'):
        if dist_name == 't':
            params = stats.t.fit(self.returns)
            return {'distribution': 't', 'df': params[0], 'loc': params[1], 'scale': params[2]}
        elif dist_name == 'norm':
            params = stats.norm.fit(self.returns)
            return {'distribution': 'norm', 'loc': params[0], 'scale': params[1]}
        elif dist_name == 'gennorm':
            params = stats.gennorm.fit(self.returns)
            return {'distribution': 'gennorm', 'beta': params[0], 'loc': params[1], 'scale': params[2]}
    
    def calculate_quantiles(self, quantiles: list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) -> dict:
        return {f'q_{int(q*100)}': self.returns.quantile(q) for q in quantiles}
```

### Correlation and Covariance

#### Multivariate Analysis
```python
class CovarianceAnalyzer:
    def __init__(self, returns_df: pd.DataFrame):
        self.returns = returns_df
    
    def calculate_covariance_matrix(self, method: str = 'sample') -> pd.DataFrame:
        if method == 'sample':
            return self.returns.cov()
        elif method == 'exponential':
            return self.returns.ewm(span=60).cov()
        elif method == 'shrinkage':
            return self._ledoit_wolf_shrinkage()
    
    def _ledoit_wolf_shrinkage(self) -> pd.DataFrame:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(self.returns)
        return pd.DataFrame(lw.covariance_, 
                          index=self.returns.columns, 
                          columns=self.returns.columns)
    
    def calculate_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        if method == 'pearson':
            return self.returns.corr(method='pearson')
        elif method == 'spearman':
            return self.returns.corr(method='spearman')
        elif method == 'kendall':
            return self.returns.corr(method='kendall')
    
    def rolling_correlation(self, asset1: str, asset2: str, window: int = 60) -> pd.Series:
        return self.returns[asset1].rolling(window).corr(self.returns[asset2])
    
    def calculate_beta(self, asset: str, market: str) -> float:
        covariance = self.returns[[asset, market]].cov().iloc[0, 1]
        market_variance = self.returns[market].var()
        return covariance / market_variance
    
    def pca_analysis(self, n_components: int = 3) -> dict:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.returns)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'principal_components': pd.DataFrame(
                principal_components, 
                index=self.returns.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
        }
```

## Risk Metrics

### Value at Risk (VaR)

#### Multiple VaR Methodologies
```python
class VaRCalculator:
    def __init__(self, returns: pd.Series, confidence_level: float = 0.95):
        self.returns = returns.dropna()
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def historical_var(self) -> float:
        return -np.percentile(self.returns, self.alpha * 100)
    
    def parametric_var(self) -> float:
        mean = self.returns.mean()
        std = self.returns.std()
        z_score = stats.norm.ppf(self.alpha)
        return -(mean + z_score * std)
    
    def cornish_fisher_var(self) -> float:
        mean = self.returns.mean()
        std = self.returns.std()
        skew = stats.skew(self.returns)
        kurt = stats.kurtosis(self.returns)
        
        z = stats.norm.ppf(self.alpha)
        z_cf = (z + (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * skew**2 / 36)
        
        return -(mean + z_cf * std)
    
    def monte_carlo_var(self, n_simulations: int = 10000) -> float:
        mean = self.returns.mean()
        std = self.returns.std()
        
        simulated_returns = np.random.normal(mean, std, n_simulations)
        return -np.percentile(simulated_returns, self.alpha * 100)
    
    def conditional_var(self, method: str = 'historical') -> float:
        if method == 'historical':
            var = self.historical_var()
            return -self.returns[self.returns <= -var].mean()
        elif method == 'parametric':
            mean = self.returns.mean()
            std = self.returns.std()
            z = stats.norm.ppf(self.alpha)
            return -(mean - std * stats.norm.pdf(z) / self.alpha)
    
    def expected_shortfall(self) -> float:
        return self.conditional_var('historical')

class RiskMetrics:
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, 
                    periods_per_year: int = 252) -> float:
        excess_returns = returns - risk_free_rate / periods_per_year
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                     periods_per_year: int = 252) -> float:
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown == 0:
            return 0
        
        annualized_return = (cumulative.iloc[-1] ** (periods_per_year / len(returns))) - 1
        return annualized_return / max_drawdown
    
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> dict:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        max_dd_start = cumulative[:max_dd_end].idxmax()
        
        recovery_date = None
        if max_dd_end < cumulative.index[-1]:
            recovery = cumulative[max_dd_end:][cumulative[max_dd_end:] >= running_max[max_dd_end]]
            if len(recovery) > 0:
                recovery_date = recovery.index[0]
        
        return {
            'max_drawdown': max_dd,
            'start_date': max_dd_start,
            'end_date': max_dd_end,
            'recovery_date': recovery_date,
            'duration_days': (max_dd_end - max_dd_start).days
        }
    
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]
        
        if returns_below.sum() == 0:
            return np.inf
        
        return returns_above.sum() / returns_below.sum()
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return active_returns.mean() / tracking_error * np.sqrt(252)
```

## Portfolio Theory Fundamentals

### Modern Portfolio Theory (Markowitz)

#### Mean-Variance Optimization
```python
from scipy.optimize import minimize

class MarkowitzPortfolio:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def portfolio_performance(self, weights: np.ndarray) -> tuple:
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std
    
    def negative_sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.0) -> float:
        p_return, p_std = self.portfolio_performance(weights)
        return -(p_return - risk_free_rate) / p_std
    
    def portfolio_variance(self, weights: np.ndarray) -> float:
        return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
    
    def optimize_max_sharpe(self, risk_free_rate: float = 0.0) -> dict:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        p_return, p_std = self.portfolio_performance(optimal_weights)
        
        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'return': p_return,
            'volatility': p_std,
            'sharpe_ratio': (p_return - risk_free_rate) / p_std
        }
    
    def optimize_min_variance(self) -> dict:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        p_return, p_std = self.portfolio_performance(optimal_weights)
        
        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'return': p_return,
            'volatility': p_std
        }
    
    def efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        min_var_port = self.optimize_min_variance()
        max_sharpe_port = self.optimize_max_sharpe()
        
        target_returns = np.linspace(
            min_var_port['return'],
            max_sharpe_port['return'],
            n_points
        )
        
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) * 252 - target_return}
            )
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_weights = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(
                self.portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                _, p_std = self.portfolio_performance(result.x)
                frontier_volatilities.append(p_std)
                frontier_weights.append(result.x)
        
        return pd.DataFrame({
            'return': target_returns[:len(frontier_volatilities)],
            'volatility': frontier_volatilities,
            'weights': frontier_weights
        })

class BlackLittermanModel:
    def __init__(self, market_caps: pd.Series, cov_matrix: pd.DataFrame,
                 risk_aversion: float = 2.5):
        self.market_caps = market_caps
        self.cov_matrix = cov_matrix
        self.risk_aversion = risk_aversion
        
        self.market_weights = market_caps / market_caps.sum()
        self.implied_returns = self._calculate_implied_returns()
    
    def _calculate_implied_returns(self) -> pd.Series:
        return self.risk_aversion * self.cov_matrix.dot(self.market_weights)
    
    def calculate_posterior_returns(self, views: pd.DataFrame, 
                                   view_confidences: np.ndarray,
                                   tau: float = 0.05) -> pd.Series:
        P = views.values
        Q = views['expected_return'].values
        Omega = np.diag(view_confidences)
        
        tau_sigma = tau * self.cov_matrix.values
        
        M_inverse = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P)
        posterior_returns = M_inverse @ (
            np.linalg.inv(tau_sigma) @ self.implied_returns.values +
            P.T @ np.linalg.inv(Omega) @ Q
        )
        
        return pd.Series(posterior_returns, index=self.cov_matrix.index)
    
    def calculate_posterior_covariance(self, views: pd.DataFrame,
                                      view_confidences: np.ndarray,
                                      tau: float = 0.05) -> pd.DataFrame:
        P = views.values
        Omega = np.diag(view_confidences)
        
        tau_sigma = tau * self.cov_matrix.values
        
        M_inverse = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P)
        posterior_cov = self.cov_matrix.values + M_inverse
        
        return pd.DataFrame(posterior_cov, 
                          index=self.cov_matrix.index,
                          columns=self.cov_matrix.columns)
```

### Capital Asset Pricing Model (CAPM)

```python
class CAPMAnalyzer:
    def __init__(self, asset_returns: pd.Series, market_returns: pd.Series,
                 risk_free_rate: float = 0.02):
        self.asset_returns = asset_returns
        self.market_returns = market_returns
        self.risk_free_rate = risk_free_rate
    
    def calculate_beta(self) -> float:
        covariance = np.cov(self.asset_returns, self.market_returns)[0, 1]
        market_variance = np.var(self.market_returns)
        return covariance / market_variance
    
    def calculate_alpha(self) -> float:
        beta = self.calculate_beta()
        
        asset_return = self.asset_returns.mean() * 252
        market_return = self.market_returns.mean() * 252
        
        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        alpha = asset_return - expected_return
        
        return alpha
    
    def run_regression(self) -> dict:
        from sklearn.linear_model import LinearRegression
        
        X = (self.market_returns - self.risk_free_rate/252).values.reshape(-1, 1)
        y = (self.asset_returns - self.risk_free_rate/252).values
        
        model = LinearRegression()
        model.fit(X, y)
        
        beta = model.coef_[0]
        alpha = model.intercept_ * 252
        r_squared = model.score(X, y)
        
        residuals = y - model.predict(X)
        residual_std = np.std(residuals) * np.sqrt(252)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'residual_volatility': residual_std,
            'information_ratio': alpha / residual_std if residual_std > 0 else 0
        }

class FamaFrenchModel:
    def __init__(self, asset_returns: pd.Series, 
                 market_returns: pd.Series,
                 smb_returns: pd.Series,  # Small Minus Big
                 hml_returns: pd.Series,  # High Minus Low
                 risk_free_rate: float = 0.02):
        self.asset_returns = asset_returns
        self.market_returns = market_returns
        self.smb_returns = smb_returns
        self.hml_returns = hml_returns
        self.risk_free_rate = risk_free_rate
    
    def run_three_factor_regression(self) -> dict:
        from sklearn.linear_model import LinearRegression
        
        y = (self.asset_returns - self.risk_free_rate/252).values
        X = np.column_stack([
            (self.market_returns - self.risk_free_rate/252).values,
            self.smb_returns.values,
            self.hml_returns.values
        ])
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_ * 252
        beta_market, beta_smb, beta_hml = model.coef_
        r_squared = model.score(X, y)
        
        return {
            'alpha': alpha,
            'beta_market': beta_market,
            'beta_smb': beta_smb,
            'beta_hml': beta_hml,
            'r_squared': r_squared
        }
```

## Option Pricing Theory

### Black-Scholes-Merton Model

```python
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesModel:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.q = q  # Dividend yield
    
    def d1(self) -> float:
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
    
    def d2(self) -> float:
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        d1, d2 = self.d1(), self.d2()
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
    
    def put_price(self) -> float:
        d1, d2 = self.d1(), self.d2()
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
                self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
    
    def call_delta(self) -> float:
        return np.exp(-self.q * self.T) * norm.cdf(self.d1())
    
    def put_delta(self) -> float:
        return -np.exp(-self.q * self.T) * norm.cdf(-self.d1())
    
    def gamma(self) -> float:
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1())) / \
               (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1()) * np.sqrt(self.T) / 100
    
    def theta_call(self) -> float:
        d1, d2 = self.d1(), self.d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
        term2 = self.q * self.S * norm.cdf(d1) * np.exp(-self.q * self.T)
        term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return (term1 - term2 + term3) / 365
    
    def theta_put(self) -> float:
        d1, d2 = self.d1(), self.d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
        term2 = -self.q * self.S * norm.cdf(-d1) * np.exp(-self.q * self.T)
        term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return (term1 + term2 + term3) / 365
    
    def rho_call(self) -> float:
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2()) / 100
    
    def rho_put(self) -> float:
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100
    
    def implied_volatility(self, option_price: float, option_type: str = 'call') -> float:
        def objective(sigma):
            self.sigma = sigma
            if option_type == 'call':
                return self.call_price() - option_price
            else:
                return self.put_price() - option_price
        
        try:
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except:
            return np.nan

class BinomialTreeModel:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 N: int = 100, option_type: str = 'call', exercise_type: str = 'european'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type
        self.exercise_type = exercise_type
        
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
    
    def price(self) -> float:
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        option_tree = np.zeros((self.N + 1, self.N + 1))
        
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        
        for j in range(self.N + 1):
            if self.option_type == 'call':
                option_tree[j, self.N] = max(0, stock_tree[j, self.N] - self.K)
            else:
                option_tree[j, self.N] = max(0, self.K - stock_tree[j, self.N])
        
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = np.exp(-self.r * self.dt) * \
                    (self.p * option_tree[j, i + 1] + (1 - self.p) * option_tree[j + 1, i + 1])
                
                if self.exercise_type == 'american':
                    if self.option_type == 'call':
                        option_tree[j, i] = max(option_tree[j, i], stock_tree[j, i] - self.K)
                    else:
                        option_tree[j, i] = max(option_tree[j, i], self.K - stock_tree[j, i])
        
        return option_tree[0, 0]
```

## Stochastic Calculus for Finance

### Geometric Brownian Motion

```python
class GeometricBrownianMotion:
    def __init__(self, S0: float, mu: float, sigma: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
    
    def simulate_path(self, T: float, N: int, n_paths: int = 1) -> np.ndarray:
        dt = T / N
        t = np.linspace(0, T, N + 1)
        
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))
        W = np.cumsum(dW, axis=1)
        W = np.column_stack([np.zeros(n_paths), W])
        
        time_component = (self.mu - 0.5 * self.sigma**2) * t
        stochastic_component = self.sigma * W
        
        S = self.S0 * np.exp(time_component + stochastic_component)
        
        return S
    
    def expected_value(self, t: float) -> float:
        return self.S0 * np.exp(self.mu * t)
    
    def variance(self, t: float) -> float:
        return self.S0**2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1)

class HestonModel:
    def __init__(self, S0: float, v0: float, kappa: float, theta: float, 
                 sigma: float, rho: float, r: float):
        self.S0 = S0  # Initial stock price
        self.v0 = v0  # Initial variance
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-term variance
        self.sigma = sigma  # Volatility of volatility
        self.rho = rho  # Correlation
        self.r = r  # Risk-free rate
    
    def simulate_path(self, T: float, N: int, n_paths: int = 1) -> tuple:
        dt = T / N
        
        S = np.zeros((n_paths, N + 1))
        v = np.zeros((n_paths, N + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(N):
            Z1 = np.random.normal(0, 1, n_paths)
            Z2 = np.random.normal(0, 1, n_paths)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v[:, i + 1] = np.maximum(
                v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + 
                self.sigma * np.sqrt(v[:, i] * dt) * W2,
                0
            )
            
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * v[:, i]) * dt + 
                np.sqrt(v[:, i] * dt) * W1
            )
        
        return S, v
```

## Backtesting Frameworks

### Event-Driven Backtesting Engine

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    def unrealized_pnl_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price

@dataclass
class Trade:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0

class Portfolio:
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
    
    def execute_trade(self, symbol: str, quantity: int, price: float, 
                     timestamp: datetime, side: str):
        commission = abs(quantity * price * self.commission_rate)
        
        trade = Trade(symbol, side, quantity, price, timestamp, commission)
        self.trades.append(trade)
        
        if side == 'BUY':
            cost = quantity * price + commission
            if cost > self.cash:
                return False
            
            self.cash -= cost
            
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_quantity = pos.quantity + quantity
                avg_price = (pos.entry_price * pos.quantity + price * quantity) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
            else:
                self.positions[symbol] = Position(symbol, quantity, price, timestamp)
        
        else:  # SELL
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                return False
            
            proceeds = quantity * price - commission
            self.cash += proceeds
            
            self.positions[symbol].quantity -= quantity
            if self.positions[symbol].quantity == 0:
                del self.positions[symbol]
        
        return True
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
        
        total_value = self.total_value(prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })
    
    def total_value(self, current_prices: Dict[str, float]) -> float:
        positions_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.current_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

class Strategy(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        pass

class BacktestEngine:
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.portfolio = Portfolio(initial_capital, commission_rate)
        self.data = {}
        self.results = {}
    
    def add_data(self, symbol: str, data: pd.DataFrame):
        self.data[symbol] = data
    
    def run(self, strategy: Strategy, start_date: datetime, end_date: datetime):
        all_dates = sorted(set().union(*[set(df.index) for df in self.data.values()]))
        all_dates = [d for d in all_dates if start_date <= d <= end_date]
        
        for current_date in all_dates:
            current_data = {}
            for symbol, df in self.data.items():
                if current_date in df.index:
                    current_data[symbol] = df.loc[:current_date]
            
            signals = strategy.generate_signals(current_data, current_date)
            
            current_prices = {
                symbol: df.loc[current_date, 'close'] 
                for symbol, df in self.data.items() 
                if current_date in df.index
            }
            
            for signal in signals:
                self.portfolio.execute_trade(
                    signal['symbol'],
                    signal['quantity'],
                    current_prices[signal['symbol']],
                    current_date,
                    signal['side']
                )
            
            self.portfolio.update_prices(current_prices, current_date)
        
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self):
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['total_value'].pct_change().dropna()
        
        total_return = (equity_df['total_value'].iloc[-1] / 
                       self.portfolio.initial_capital - 1)
        
        n_years = len(equity_df) / 252
        cagr = (equity_df['total_value'].iloc[-1] / 
               self.portfolio.initial_capital) ** (1 / n_years) - 1
        
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        self.results = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'volatility': returns.std() * np.sqrt(252),
            'total_trades': len(self.portfolio.trades),
            'final_value': equity_df['total_value'].iloc[-1],
            'equity_curve': equity_df
        }
    
    def get_results(self) -> Dict:
        return self.results
```

## Summary

This module covered quantitative finance fundamentals:

1. **Statistical Measures**: Returns calculation, distribution analysis, correlation
2. **Risk Metrics**: VaR, CVaR, Sharpe ratio, maximum drawdown
3. **Portfolio Theory**: Markowitz optimization, efficient frontier, Black-Litterman
4. **Asset Pricing**: CAPM, Fama-French factor models
5. **Option Pricing**: Black-Scholes, binomial trees, Greeks
6. **Stochastic Calculus**: GBM, Heston model, path simulation
7. **Backtesting**: Event-driven framework, portfolio management, performance metrics

These quantitative foundations are essential for building sophisticated AI-powered trading and risk management systems.

