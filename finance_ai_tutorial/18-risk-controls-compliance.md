# Module 18: Risk Controls and Compliance

## Table of Contents
1. [Regulatory Requirements](#regulatory-requirements)
2. [Pre-Trade Risk Controls](#pre-trade-risk-controls)
3. [Post-Trade Risk Controls](#post-trade-risk-controls)
4. [AI Model Governance](#ai-model-governance)
5. [Explainable AI for Regulation](#explainable-ai-for-regulation)
6. [Market Abuse Prevention](#market-abuse-prevention)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Regulatory Requirements

### Regulatory Framework Manager

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class Regulation(Enum):
    MIFID_II = "MiFID II"
    DODD_FRANK = "Dodd-Frank"
    REG_NMS = "Reg NMS"
    MAR = "Market Abuse Regulation"
    EMIR = "EMIR"

@dataclass
class RegulatoryRequirement:
    regulation: Regulation
    requirement_id: str
    description: str
    check_function: callable
    severity: str = "high"

class RegulatoryComplianceManager:
    def __init__(self):
        self.requirements: List[RegulatoryRequirement] = []
        self.compliance_status: Dict[str, bool] = {}
        
    def add_requirement(self, requirement: RegulatoryRequirement):
        self.requirements.append(requirement)
        
    def check_compliance(self, trade_data: Dict) -> Dict[str, any]:
        results = {}
        
        for req in self.requirements:
            try:
                is_compliant = req.check_function(trade_data)
                
                results[req.requirement_id] = {
                    'compliant': is_compliant,
                    'regulation': req.regulation.value,
                    'description': req.description,
                    'severity': req.severity
                }
                
                self.compliance_status[req.requirement_id] = is_compliant
                
            except Exception as e:
                results[req.requirement_id] = {
                    'compliant': False,
                    'error': str(e),
                    'regulation': req.regulation.value
                }
        
        return results
    
    def generate_compliance_report(self) -> str:
        total = len(self.compliance_status)
        compliant = sum(1 for v in self.compliance_status.values() if v)
        
        report = f"Compliance Report\n"
        report += f"=" * 50 + "\n"
        report += f"Total Requirements: {total}\n"
        report += f"Compliant: {compliant} ({compliant/total*100:.1f}%)\n"
        report += f"Non-Compliant: {total-compliant}\n\n"
        
        report += "Details:\n"
        for req_id, status in self.compliance_status.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            report += f"{req_id}: {status_str}\n"
        
        return report
```

## Pre-Trade Risk Controls

### Order Size and Position Limits

```python
class PreTradeRiskControls:
    def __init__(self, config: Dict):
        self.max_order_size = config.get('max_order_size', 10000)
        self.max_position_size = config.get('max_position_size', 50000)
        self.max_order_value = config.get('max_order_value', 1000000)
        self.concentration_limit = config.get('concentration_limit', 0.10)
        
        self.current_positions = {}
        self.portfolio_value = config.get('portfolio_value', 10000000)
        
    def check_order_size_limit(self, order: Dict) -> Tuple[bool, str]:
        if order['quantity'] > self.max_order_size:
            return False, f"Order size {order['quantity']} exceeds limit {self.max_order_size}"
        
        return True, "OK"
    
    def check_position_limit(self, order: Dict) -> Tuple[bool, str]:
        symbol = order['symbol']
        direction = 1 if order['side'] == 'buy' else -1
        
        current_pos = self.current_positions.get(symbol, 0)
        new_pos = current_pos + direction * order['quantity']
        
        if abs(new_pos) > self.max_position_size:
            return False, f"Position {new_pos} would exceed limit {self.max_position_size}"
        
        return True, "OK"
    
    def check_order_value_limit(self, order: Dict) -> Tuple[bool, str]:
        order_value = order['quantity'] * order.get('price', 0)
        
        if order_value > self.max_order_value:
            return False, f"Order value {order_value} exceeds limit {self.max_order_value}"
        
        return True, "OK"
    
    def check_concentration_limit(self, order: Dict) -> Tuple[bool, str]:
        symbol = order['symbol']
        direction = 1 if order['side'] == 'buy' else -1
        
        current_pos = self.current_positions.get(symbol, 0)
        new_pos = current_pos + direction * order['quantity']
        
        position_value = abs(new_pos) * order.get('price', 0)
        concentration = position_value / self.portfolio_value
        
        if concentration > self.concentration_limit:
            return False, f"Concentration {concentration:.2%} exceeds limit {self.concentration_limit:.2%}"
        
        return True, "OK"
    
    def validate_order(self, order: Dict) -> Tuple[bool, List[str]]:
        checks = [
            self.check_order_size_limit,
            self.check_position_limit,
            self.check_order_value_limit,
            self.check_concentration_limit
        ]
        
        errors = []
        
        for check in checks:
            passed, message = check(order)
            if not passed:
                errors.append(message)
        
        return len(errors) == 0, errors


class DynamicRiskLimits:
    def __init__(self):
        self.base_limits = {}
        self.current_limits = {}
        
    def adjust_limits_by_volatility(
        self,
        symbol: str,
        volatility: float,
        base_limit: int
    ) -> int:
        vol_adjustment = max(0.5, min(1.5, 1 / (1 + volatility)))
        
        adjusted_limit = int(base_limit * vol_adjustment)
        
        return adjusted_limit
    
    def adjust_limits_by_pnl(
        self,
        current_pnl: float,
        base_limit: int,
        pnl_threshold: float = -0.05
    ) -> int:
        if current_pnl < pnl_threshold:
            reduction_factor = 1 - abs(current_pnl - pnl_threshold)
            adjusted_limit = int(base_limit * max(0.3, reduction_factor))
        else:
            adjusted_limit = base_limit
        
        return adjusted_limit
```

## Post-Trade Risk Controls

### Real-Time Position Monitoring

```python
import pandas as pd
from datetime import datetime

class PositionMonitor:
    def __init__(self):
        self.positions = {}
        self.position_history = []
        self.alerts = []
        
    def update_position(self, symbol: str, quantity: int, price: float):
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0
            }
        
        pos = self.positions[symbol]
        
        if (pos['quantity'] > 0 and quantity < 0) or (pos['quantity'] < 0 and quantity > 0):
            closing_qty = min(abs(pos['quantity']), abs(quantity))
            realized_pnl = closing_qty * (price - pos['avg_price']) * np.sign(pos['quantity'])
            pos['realized_pnl'] += realized_pnl
        
        new_quantity = pos['quantity'] + quantity
        
        if new_quantity * pos['quantity'] >= 0:
            total_cost = pos['avg_price'] * abs(pos['quantity']) + price * abs(quantity)
            pos['avg_price'] = total_cost / (abs(pos['quantity']) + abs(quantity))
        else:
            if new_quantity != 0:
                pos['avg_price'] = price
        
        pos['quantity'] = new_quantity
        
        self.position_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': new_quantity,
            'price': price
        })
        
    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        unrealized = pos['quantity'] * (current_price - pos['avg_price'])
        pos['unrealized_pnl'] = unrealized
        
        return unrealized
    
    def get_portfolio_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        total_realized = sum(pos['realized_pnl'] for pos in self.positions.values())
        
        total_unrealized = sum(
            self.calculate_unrealized_pnl(symbol, current_prices.get(symbol, 0))
            for symbol in self.positions.keys()
        )
        
        return {
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized
        }
    
    def check_risk_thresholds(
        self,
        max_loss: float,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        alerts = []
        
        pnl = self.get_portfolio_pnl(current_prices)
        
        if pnl['total_pnl'] < -max_loss:
            alerts.append({
                'type': 'PORTFOLIO_LOSS_LIMIT',
                'severity': 'CRITICAL',
                'message': f"Portfolio PnL {pnl['total_pnl']:.2f} exceeds max loss {max_loss}",
                'timestamp': datetime.now()
            })
        
        return alerts
```

## AI Model Governance

### Model Registry and Versioning

```python
from dataclasses import dataclass
from typing import Any, Dict, List
import pickle
import json

@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    version: str
    created_date: datetime
    trained_by: str
    training_data_period: str
    performance_metrics: Dict[str, float]
    validation_status: str
    approved_by: Optional[str] = None
    
class ModelRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.models: Dict[str, List[ModelMetadata]] = {}
        
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata
    ) -> str:
        if metadata.model_name not in self.models:
            self.models[metadata.model_name] = []
        
        self.models[metadata.model_name].append(metadata)
        
        model_path = f"{self.storage_path}/{metadata.model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        metadata_path = f"{self.storage_path}/{metadata.model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.__dict__, f, default=str)
        
        return metadata.model_id
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        model_path = f"{self.storage_path}/{model_id}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        metadata_path = f"{self.storage_path}/{model_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata(**metadata_dict)
        
        return model, metadata
    
    def get_approved_models(self, model_name: str) -> List[ModelMetadata]:
        if model_name not in self.models:
            return []
        
        return [
            m for m in self.models[model_name]
            if m.validation_status == 'approved'
        ]


class ModelValidator:
    def __init__(self):
        self.validation_rules = []
        
    def add_validation_rule(
        self,
        name: str,
        check_function: callable,
        threshold: float
    ):
        self.validation_rules.append({
            'name': name,
            'check': check_function,
            'threshold': threshold
        })
        
    def validate_model(
        self,
        model: Any,
        test_data: pd.DataFrame,
        test_labels: pd.Series
    ) -> Dict[str, any]:
        predictions = model.predict(test_data)
        
        results = {}
        
        for rule in self.validation_rules:
            score = rule['check'](test_labels, predictions)
            passed = score >= rule['threshold']
            
            results[rule['name']] = {
                'score': score,
                'threshold': rule['threshold'],
                'passed': passed
            }
        
        all_passed = all(r['passed'] for r in results.values())
        
        return {
            'overall_status': 'PASS' if all_passed else 'FAIL',
            'individual_checks': results
        }
```

## Explainable AI for Regulation

### SHAP-based Model Explanations

```python
import shap
import numpy as np
import pandas as pd

class ModelExplainer:
    def __init__(self, model, background_data: pd.DataFrame):
        self.model = model
        self.explainer = shap.TreeExplainer(model, background_data)
        
    def explain_prediction(
        self,
        instance: pd.DataFrame
    ) -> Dict[str, any]:
        shap_values = self.explainer.shap_values(instance)
        
        feature_importance = dict(zip(
            instance.columns,
            np.abs(shap_values[0])
        ))
        
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        explanation = {
            'prediction': self.model.predict(instance)[0],
            'feature_importance': sorted_importance,
            'top_3_features': sorted_importance[:3],
            'shap_values': shap_values[0].tolist()
        }
        
        return explanation
    
    def generate_explanation_report(
        self,
        instance: pd.DataFrame
    ) -> str:
        explanation = self.explain_prediction(instance)
        
        report = "Model Prediction Explanation\n"
        report += "=" * 50 + "\n"
        report += f"Prediction: {explanation['prediction']}\n\n"
        report += "Top Contributing Features:\n"
        
        for feature, importance in explanation['top_3_features']:
            report += f"  {feature}: {importance:.4f}\n"
        
        return report


class CounterfactualExplainer:
    def __init__(self, model, feature_ranges: Dict[str, Tuple[float, float]]):
        self.model = model
        self.feature_ranges = feature_ranges
        
    def generate_counterfactual(
        self,
        instance: pd.DataFrame,
        desired_outcome: int,
        max_iterations: int = 1000
    ) -> Optional[pd.DataFrame]:
        current = instance.copy()
        
        for iteration in range(max_iterations):
            prediction = self.model.predict(current)[0]
            
            if prediction == desired_outcome:
                return current
            
            feature = np.random.choice(list(self.feature_ranges.keys()))
            min_val, max_val = self.feature_ranges[feature]
            
            current[feature] = np.random.uniform(min_val, max_val)
        
        return None
```

## Market Abuse Prevention

### Manipulation Detection System

```python
class ManipulationDetector:
    def __init__(self):
        self.patterns = []
        self.alerts = []
        
    def detect_spoofing(
        self,
        order_book_history: List[Dict]
    ) -> List[Dict]:
        spoofing_alerts = []
        
        for i in range(1, len(order_book_history)):
            prev_book = order_book_history[i-1]
            curr_book = order_book_history[i]
            
            large_orders_added = self._check_large_orders_added(prev_book, curr_book)
            
            if large_orders_added:
                orders_cancelled = self._check_orders_cancelled(curr_book, order_book_history[min(i+1, len(order_book_history)-1)])
                
                if orders_cancelled:
                    spoofing_alerts.append({
                        'type': 'SPOOFING',
                        'timestamp': curr_book['timestamp'],
                        'description': 'Large orders added then quickly cancelled',
                        'severity': 'HIGH'
                    })
        
        return spoofing_alerts
    
    def _check_large_orders_added(
        self,
        prev_book: Dict,
        curr_book: Dict
    ) -> bool:
        prev_size = sum(o['size'] for o in prev_book.get('bids', []))
        curr_size = sum(o['size'] for o in curr_book.get('bids', []))
        
        return curr_size > prev_size * 1.5
    
    def _check_orders_cancelled(
        self,
        book1: Dict,
        book2: Dict
    ) -> bool:
        size1 = sum(o['size'] for o in book1.get('bids', []))
        size2 = sum(o['size'] for o in book2.get('bids', []))
        
        return size2 < size1 * 0.7
    
    def detect_wash_trading(
        self,
        trades: pd.DataFrame,
        threshold_accounts: int = 2
    ) -> List[Dict]:
        alerts = []
        
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            
            for account in symbol_trades['account'].unique():
                account_trades = symbol_trades[symbol_trades['account'] == account]
                
                buys = account_trades[account_trades['side'] == 'buy']
                sells = account_trades[account_trades['side'] == 'sell']
                
                matching_trades = 0
                for _, buy in buys.iterrows():
                    for _, sell in sells.iterrows():
                        if abs(buy['price'] - sell['price']) < 0.01 and \
                           abs(buy['quantity'] - sell['quantity']) == 0:
                            matching_trades += 1
                
                if matching_trades > 5:
                    alerts.append({
                        'type': 'WASH_TRADING',
                        'account': account,
                        'symbol': symbol,
                        'matching_trades': matching_trades,
                        'severity': 'HIGH'
                    })
        
        return alerts


class InsiderTradingDetector:
    def __init__(self):
        self.insider_list = set()
        self.material_events = []
        
    def add_insider(self, person_id: str):
        self.insider_list.add(person_id)
        
    def register_material_event(
        self,
        company: str,
        event_type: str,
        event_date: datetime
    ):
        self.material_events.append({
            'company': company,
            'event_type': event_type,
            'event_date': event_date
        })
        
    def check_suspicious_trading(
        self,
        trades: pd.DataFrame,
        lookback_days: int = 30
    ) -> List[Dict]:
        alerts = []
        
        for event in self.material_events:
            start_date = event['event_date'] - pd.Timedelta(days=lookback_days)
            end_date = event['event_date']
            
            relevant_trades = trades[
                (trades['symbol'] == event['company']) &
                (trades['timestamp'] >= start_date) &
                (trades['timestamp'] <= end_date)
            ]
            
            for trader in relevant_trades['trader_id'].unique():
                if trader in self.insider_list:
                    trader_trades = relevant_trades[relevant_trades['trader_id'] == trader]
                    
                    if len(trader_trades) > 0:
                        alerts.append({
                            'type': 'SUSPICIOUS_INSIDER_TRADING',
                            'trader_id': trader,
                            'company': event['company'],
                            'event_type': event['event_type'],
                            'num_trades': len(trader_trades),
                            'total_value': trader_trades['value'].sum(),
                            'severity': 'CRITICAL'
                        })
        
        return alerts
```

## PhD-Level Research Topics

### Privacy-Preserving ML for Compliance

```python
import tenseal as ts

class HomomorphicRiskModel:
    def __init__(self):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
    def encrypt_data(self, data: np.ndarray) -> ts.CKKSVector:
        return ts.ckks_vector(self.context, data.tolist())
    
    def compute_risk_score_encrypted(
        self,
        encrypted_features: ts.CKKSVector,
        weights: np.ndarray
    ) -> ts.CKKSVector:
        encrypted_weights = ts.ckks_vector(self.context, weights.tolist())
        
        risk_score = encrypted_features.dot(encrypted_weights)
        
        return risk_score
    
    def decrypt_result(self, encrypted_score: ts.CKKSVector) -> float:
        return encrypted_score.decrypt()[0]
```

## Implementation

### Complete Risk and Compliance System

```python
class ComplianceSystem:
    def __init__(self, config: Dict):
        self.regulatory_manager = RegulatoryComplianceManager()
        self.pre_trade_controls = PreTradeRiskControls(config)
        self.position_monitor = PositionMonitor()
        self.model_registry = ModelRegistry(config['model_storage_path'])
        self.manipulation_detector = ManipulationDetector()
        self.insider_trading_detector = InsiderTradingDetector()
        
        self._setup_compliance_rules()
        
    def _setup_compliance_rules(self):
        def check_best_execution(trade):
            return trade.get('execution_quality', 0) > 0.95
        
        self.regulatory_manager.add_requirement(
            RegulatoryRequirement(
                regulation=Regulation.MIFID_II,
                requirement_id="BE001",
                description="Best Execution",
                check_function=check_best_execution
            )
        )
        
    def validate_trade(self, order: Dict) -> Tuple[bool, List[str]]:
        passed, errors = self.pre_trade_controls.validate_order(order)
        
        compliance_results = self.regulatory_manager.check_compliance(order)
        
        non_compliant = [
            k for k, v in compliance_results.items()
            if not v.get('compliant', False)
        ]
        
        if non_compliant:
            errors.extend([f"Compliance failure: {r}" for r in non_compliant])
            passed = False
        
        return passed, errors
    
    def monitor_portfolio(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        alerts = self.position_monitor.check_risk_thresholds(
            max_loss=100000,
            current_prices=current_prices
        )
        
        return alerts
    
    def generate_compliance_report(self) -> str:
        return self.regulatory_manager.generate_compliance_report()
```
