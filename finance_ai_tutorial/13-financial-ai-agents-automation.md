# Finance AI Tutorial Module 13: Financial AI Agents and Automation

## Learning Objectives
By the end of this module, you will be able to:
- Build autonomous trading agents
- Create financial research and analysis agents
- Implement portfolio management agents
- Design multi-agent trading systems
- Apply agent orchestration frameworks to finance
- Build self-improving financial agents

## Introduction to Financial AI Agents

Financial AI agents are autonomous systems that can perceive market conditions, make decisions, execute trades, and learn from outcomes with minimal human intervention. They combine market data, alternative data sources, and sophisticated reasoning to operate effectively in financial markets.

### Key Characteristics of Financial Agents

1. **Autonomy**: Independent decision-making
2. **Reactivity**: Quick response to market changes
3. **Proactivity**: Anticipate market moves
4. **Risk Management**: Built-in risk controls
5. **Learning**: Continuous improvement
6. **Explainability**: Transparent decision process

## Autonomous Trading Agent

### Complete Trading Agent Implementation

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from collections import deque
import yfinance as yf
from datetime import datetime, timedelta

class TradingAction(Enum):
    """Trading actions"""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class Trade:
    """Represents a trade execution"""
    timestamp: datetime
    symbol: str
    action: TradingAction
    quantity: int
    price: float
    reason: str
    confidence: float

@dataclass
class Position:
    """Current position in a security"""
    symbol: str
    quantity: int
    average_cost: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.average_cost) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        return ((self.current_price - self.average_cost) / self.average_cost) * 100

class MarketDataProvider:
    """Provides market data to agents"""
    
    def __init__(self):
        self.cache = {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        return data['Close'].iloc[-1] if not data.empty else 0
    
    def get_historical_data(self, symbol: str, period: str = '1mo') -> pd.DataFrame:
        """Get historical market data"""
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_current_price(symbol)
        return quotes

class RiskManager:
    """Manages trading risks"""
    
    def __init__(self, max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.02,
                 stop_loss_pct: float = 0.05):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk per trade
            stop_loss_pct: Stop loss percentage
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.daily_loss_limit = 0.03  # 3% daily loss limit
        self.daily_pnl = 0
    
    def validate_trade(self, trade_action: TradingAction, symbol: str,
                      quantity: int, price: float, portfolio_value: float,
                      positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Validate if trade meets risk criteria"""
        
        # Calculate position value
        position_value = quantity * price
        position_size = position_value / portfolio_value
        
        # Check position size limit
        if position_size > self.max_position_size:
            return False, f"Position size {position_size:.2%} exceeds limit {self.max_position_size:.2%}"
        
        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit * portfolio_value:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        # Check if adding to existing position creates concentration
        if symbol in positions:
            current_position = positions[symbol]
            total_position_value = current_position.market_value + position_value
            total_size = total_position_value / portfolio_value
            
            if total_size > self.max_position_size:
                return False, f"Total position size {total_size:.2%} exceeds limit"
        
        return True, "Trade validated"
    
    def calculate_position_size(self, portfolio_value: float, price: float,
                               confidence: float) -> int:
        """Calculate optimal position size based on confidence"""
        # Kelly criterion inspired sizing
        max_value = portfolio_value * self.max_position_size
        confidence_adjusted = max_value * confidence
        
        quantity = int(confidence_adjusted / price)
        return max(1, quantity)
    
    def check_stop_loss(self, position: Position) -> bool:
        """Check if position hit stop loss"""
        return position.unrealized_pnl_pct < -self.stop_loss_pct * 100
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
    
    def reset_daily_tracking(self):
        """Reset daily tracking (call at end of day)"""
        self.daily_pnl = 0

class TradingMemory:
    """Memory system for trading agent"""
    
    def __init__(self, capacity: int = 1000):
        self.trades = deque(maxlen=capacity)
        self.market_observations = deque(maxlen=capacity)
        self.performance_metrics = {}
    
    def add_trade(self, trade: Trade):
        """Record a trade"""
        self.trades.append(trade)
    
    def add_market_observation(self, observation: Dict):
        """Record market observation"""
        self.market_observations.append(observation)
    
    def get_recent_trades(self, n: int = 10) -> List[Trade]:
        """Get recent trades"""
        return list(self.trades)[-n:]
    
    def calculate_performance(self) -> Dict:
        """Calculate trading performance metrics"""
        if not self.trades:
            return {}
        
        trades_list = list(self.trades)
        
        # Win rate
        profitable = sum(1 for t in trades_list if hasattr(t, 'pnl') and t.pnl > 0)
        total = len([t for t in trades_list if hasattr(t, 'pnl')])
        win_rate = profitable / total if total > 0 else 0
        
        # Average confidence
        avg_confidence = np.mean([t.confidence for t in trades_list])
        
        return {
            'total_trades': len(trades_list),
            'win_rate': win_rate,
            'avg_confidence': avg_confidence
        }

class AutonomousTradingAgent:
    """
    Autonomous trading agent with decision-making capabilities
    
    Features:
    - Market analysis
    - Signal generation
    - Risk management
    - Trade execution
    - Performance tracking
    - Continuous learning
    """
    
    def __init__(self, name: str, initial_capital: float, universe: List[str]):
        """
        Initialize trading agent
        
        Args:
            name: Agent identifier
            initial_capital: Starting capital
            universe: List of symbols to trade
        """
        self.name = name
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.universe = universe
        
        # Components
        self.market_data = MarketDataProvider()
        self.risk_manager = RiskManager()
        self.memory = TradingMemory()
        
        # Portfolio
        self.positions: Dict[str, Position] = {}
        self.pending_orders = []
        
        # State
        self.is_active = True
        self.current_time = datetime.now()
    
    @property
    def portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L"""
        return self.portfolio_value - self.initial_capital
    
    @property
    def total_pnl_pct(self) -> float:
        """Calculate total P&L percentage"""
        return (self.total_pnl / self.initial_capital) * 100
    
    def analyze_market(self, symbol: str) -> Dict:
        """
        Analyze market conditions for a symbol
        
        Returns:
            Dictionary with market analysis
        """
        # Get historical data
        hist_data = self.market_data.get_historical_data(symbol, period='3mo')
        
        if hist_data.empty:
            return {'signal': TradingAction.HOLD, 'confidence': 0}
        
        # Calculate technical indicators
        current_price = hist_data['Close'].iloc[-1]
        sma_20 = hist_data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist_data['Close'].rolling(50).mean().iloc[-1]
        
        # Calculate momentum
        returns = hist_data['Close'].pct_change()
        momentum_5d = returns.tail(5).mean()
        volatility = returns.std()
        
        # Generate signal
        signal = TradingAction.HOLD
        confidence = 0.5
        
        # Simple trend following logic
        if current_price > sma_20 > sma_50 and momentum_5d > 0:
            signal = TradingAction.BUY
            confidence = min(0.9, 0.5 + abs(momentum_5d) * 10)
        elif current_price < sma_20 < sma_50 and momentum_5d < 0:
            signal = TradingAction.SELL
            confidence = min(0.9, 0.5 + abs(momentum_5d) * 10)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'momentum': momentum_5d,
            'volatility': volatility
        }
    
    def make_trading_decision(self) -> List[Trade]:
        """
        Make trading decisions for portfolio
        
        Returns:
            List of trades to execute
        """
        trades = []
        
        # Check existing positions for exits
        for symbol, position in list(self.positions.items()):
            # Update current price
            position.current_price = self.market_data.get_current_price(symbol)
            
            # Check stop loss
            if self.risk_manager.check_stop_loss(position):
                trade = Trade(
                    timestamp=self.current_time,
                    symbol=symbol,
                    action=TradingAction.SELL,
                    quantity=position.quantity,
                    price=position.current_price,
                    reason="Stop loss triggered",
                    confidence=1.0
                )
                trades.append(trade)
                continue
            
            # Analyze for potential exit
            analysis = self.analyze_market(symbol)
            if analysis['signal'] == TradingAction.SELL and analysis['confidence'] > 0.7:
                trade = Trade(
                    timestamp=self.current_time,
                    symbol=symbol,
                    action=TradingAction.SELL,
                    quantity=position.quantity,
                    price=analysis['current_price'],
                    reason="Signal: Bearish trend",
                    confidence=analysis['confidence']
                )
                trades.append(trade)
        
        # Look for new opportunities
        for symbol in self.universe:
            if symbol in self.positions:
                continue
            
            analysis = self.analyze_market(symbol)
            
            if analysis['signal'] == TradingAction.BUY and analysis['confidence'] > 0.65:
                # Calculate position size
                quantity = self.risk_manager.calculate_position_size(
                    self.portfolio_value,
                    analysis['current_price'],
                    analysis['confidence']
                )
                
                # Validate trade
                valid, message = self.risk_manager.validate_trade(
                    TradingAction.BUY,
                    symbol,
                    quantity,
                    analysis['current_price'],
                    self.portfolio_value,
                    self.positions
                )
                
                if valid:
                    trade = Trade(
                        timestamp=self.current_time,
                        symbol=symbol,
                        action=TradingAction.BUY,
                        quantity=quantity,
                        price=analysis['current_price'],
                        reason=f"Signal: Bullish trend (Confidence: {analysis['confidence']:.2f})",
                        confidence=analysis['confidence']
                    )
                    trades.append(trade)
                else:
                    print(f"Trade rejected for {symbol}: {message}")
        
        return trades
    
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade
        
        Args:
            trade: Trade to execute
            
        Returns:
            True if successful, False otherwise
        """
        cost = trade.quantity * trade.price
        
        if trade.action == TradingAction.BUY:
            # Check if we have enough cash
            if cost > self.cash:
                print(f"Insufficient cash for trade: {cost:.2f} > {self.cash:.2f}")
                return False
            
            # Execute buy
            self.cash -= cost
            
            if trade.symbol in self.positions:
                # Add to existing position
                pos = self.positions[trade.symbol]
                total_quantity = pos.quantity + trade.quantity
                total_cost = (pos.average_cost * pos.quantity) + (trade.price * trade.quantity)
                pos.average_cost = total_cost / total_quantity
                pos.quantity = total_quantity
                pos.current_price = trade.price
            else:
                # Create new position
                self.positions[trade.symbol] = Position(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    average_cost=trade.price,
                    current_price=trade.price
                )
            
            print(f"✓ BUY {trade.quantity} {trade.symbol} @ ${trade.price:.2f} - {trade.reason}")
            
        elif trade.action == TradingAction.SELL:
            # Check if we have the position
            if trade.symbol not in self.positions:
                print(f"No position to sell: {trade.symbol}")
                return False
            
            pos = self.positions[trade.symbol]
            
            if trade.quantity > pos.quantity:
                print(f"Insufficient quantity: {trade.quantity} > {pos.quantity}")
                return False
            
            # Execute sell
            proceeds = trade.quantity * trade.price
            self.cash += proceeds
            
            # Calculate realized P&L
            realized_pnl = (trade.price - pos.average_cost) * trade.quantity
            self.risk_manager.update_daily_pnl(realized_pnl)
            
            # Update position
            if trade.quantity == pos.quantity:
                # Close entire position
                del self.positions[trade.symbol]
            else:
                # Reduce position
                pos.quantity -= trade.quantity
                pos.current_price = trade.price
            
            print(f"✓ SELL {trade.quantity} {trade.symbol} @ ${trade.price:.2f} - {trade.reason} (P&L: ${realized_pnl:.2f})")
        
        # Record trade
        self.memory.add_trade(trade)
        
        return True
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        print(f"\n{'='*60}")
        print(f"Trading Cycle - {self.current_time}")
        print(f"{'='*60}")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"P&L: ${self.total_pnl:,.2f} ({self.total_pnl_pct:.2f}%)")
        print(f"Open Positions: {len(self.positions)}")
        
        # Make trading decisions
        trades = self.make_trading_decision()
        
        # Execute trades
        for trade in trades:
            self.execute_trade(trade)
        
        # Update portfolio status
        if self.positions:
            print(f"\nCurrent Positions:")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos.quantity} shares @ ${pos.average_cost:.2f} "
                      f"(Current: ${pos.current_price:.2f}, P&L: {pos.unrealized_pnl_pct:.2f}%)")
        
        # Record market observation
        self.memory.add_market_observation({
            'timestamp': self.current_time,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'num_positions': len(self.positions),
            'pnl': self.total_pnl
        })
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            'agent_name': self.name,
            'initial_capital': self.initial_capital,
            'current_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'cash': self.cash,
            'num_positions': len(self.positions),
            'positions': {k: v.__dict__ for k, v in self.positions.items()},
            'memory_stats': self.memory.calculate_performance()
        }

# Example usage
trading_agent = AutonomousTradingAgent(
    name="AlphaAgent",
    initial_capital=100000,
    universe=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
)

# Run multiple trading cycles
for i in range(5):
    trading_agent.run_trading_cycle()
    # In practice, wait for next trading interval
    trading_agent.current_time += timedelta(days=1)

# Get final performance
report = trading_agent.get_performance_report()
print(f"\n{'='*60}")
print("Final Performance Report")
print(f"{'='*60}")
print(f"Agent: {report['agent_name']}")
print(f"Initial Capital: ${report['initial_capital']:,.2f}")
print(f"Final Value: ${report['current_value']:,.2f}")
print(f"Total P&L: ${report['total_pnl']:,.2f} ({report['total_pnl_pct']:.2f}%)")
print(f"Win Rate: {report['memory_stats'].get('win_rate', 0):.2%}")
```

## Multi-Agent Trading System

### Collaborative Trading Agents

```python
class MultiAgentTradingSystem:
    """System coordinating multiple trading agents"""
    
    def __init__(self):
        self.agents: Dict[str, AutonomousTradingAgent] = {}
        self.market_data = MarketDataProvider()
        self.shared_insights = deque(maxlen=100)
    
    def register_agent(self, agent: AutonomousTradingAgent):
        """Register an agent with the system"""
        self.agents[agent.name] = agent
        print(f"Registered agent: {agent.name}")
    
    def share_insight(self, agent_name: str, insight: Dict):
        """Share insight from one agent to others"""
        insight['source'] = agent_name
        insight['timestamp'] = datetime.now()
        self.shared_insights.append(insight)
    
    def get_consensus_signal(self, symbol: str) -> Dict:
        """Get consensus signal from all agents"""
        signals = []
        
        for agent in self.agents.values():
            analysis = agent.analyze_market(symbol)
            signals.append(analysis)
        
        # Aggregate signals
        buy_votes = sum(1 for s in signals if s['signal'] == TradingAction.BUY)
        sell_votes = sum(1 for s in signals if s['signal'] == TradingAction.SELL)
        avg_confidence = np.mean([s['confidence'] for s in signals])
        
        if buy_votes > sell_votes:
            consensus = TradingAction.BUY
        elif sell_votes > buy_votes:
            consensus = TradingAction.SELL
        else:
            consensus = TradingAction.HOLD
        
        return {
            'symbol': symbol,
            'consensus': consensus,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'agreement': max(buy_votes, sell_votes) / len(self.agents),
            'avg_confidence': avg_confidence
        }
    
    def run_coordinated_cycle(self):
        """Run coordinated trading cycle"""
        print(f"\n{'='*70}")
        print("Multi-Agent Trading Cycle")
        print(f"{'='*70}")
        
        # Each agent makes its own decisions
        for agent_name, agent in self.agents.items():
            print(f"\n{agent_name} Trading:")
            agent.run_trading_cycle()
        
        # Aggregate performance
        total_pnl = sum(agent.total_pnl for agent in self.agents.values())
        avg_pnl_pct = np.mean([agent.total_pnl_pct for agent in self.agents.values()])
        
        print(f"\nSystem Performance:")
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Average P&L%: {avg_pnl_pct:.2f}%")

# Create multi-agent system
mas = MultiAgentTradingSystem()

# Create multiple agents with different strategies
momentum_agent = AutonomousTradingAgent(
    name="MomentumAgent",
    initial_capital=50000,
    universe=['AAPL', 'MSFT', 'GOOGL']
)

value_agent = AutonomousTradingAgent(
    name="ValueAgent",
    initial_capital=50000,
    universe=['AMZN', 'TSLA', 'META']
)

# Register agents
mas.register_agent(momentum_agent)
mas.register_agent(value_agent)

# Run coordinated trading
mas.run_coordinated_cycle()

# Check consensus on a symbol
consensus = mas.get_consensus_signal('AAPL')
print(f"\nConsensus on AAPL:")
print(f"  Signal: {consensus['consensus']}")
print(f"  Agreement: {consensus['agreement']:.2%}")
print(f"  Confidence: {consensus['avg_confidence']:.2f}")
```

This module demonstrates building sophisticated autonomous trading agents with risk management, memory, and multi-agent coordination. These concepts form the foundation for production trading systems.

## Key Takeaways

- Autonomous trading agents combine market analysis, risk management, and execution
- Risk management is critical for agent-based trading
- Memory systems enable agents to learn from experience
- Multi-agent systems can provide robustness through diversification
- Agents should have built-in fail-safes and monitoring
- Performance tracking is essential for continuous improvement

## Next Steps

In the next module, we'll explore LLM-powered financial analysis and how to build advanced research agents using large language models.

