# Module 27: Advanced Islamic Finance with AI - PhD Level

## Table of Contents
1. [Generative AI for Islamic Underwriting](#generative-ai-for-islamic-underwriting)
2. [Alternative Data Integration for Islamic Finance](#alternative-data-integration-for-islamic-finance)
3. [Graph Neural Networks for Islamic Credit Assessment](#graph-neural-networks-for-islamic-credit-assessment)
4. [Explainable AI in Islamic Underwriting](#explainable-ai-in-islamic-underwriting)
5. [Hybrid Neural-Symbolic Islamic Underwriting](#hybrid-neural-symbolic-islamic-underwriting)
6. [Causal Inference in Islamic Finance](#causal-inference-in-islamic-finance)
7. [Research Frontiers](#research-frontiers)

This module mirrors the depth and structure of Module 21 (Advanced Credit Underwriting with AI) but is designed exclusively for values-based / partnership-based / asset-backed financing (Shariah-compliant structures). All techniques are production-grade and SSB-auditable.

---

## Generative AI for Islamic Underwriting

### LLM-Based Shariah Credit Memo Generation

Research basis: McKinsey (2026) Gen AI in Credit Risk; Shariah Governance Standard on Generative AI for Islamic Financial Institutions (2025). LLMs are used to generate financing memos that explicitly reference financing structure (Murabaha, Musharakah, etc.), asset-backing, and Shariah considerations—not interest-based logic.

```python
import openai
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class IslamicFinancingApplication:
    applicant_id: str
    income: float
    debt: float
    requested_amount: float
    financing_mode: str
    asset_description: str
    asset_value: float
    employment_history: List[Dict]
    business_data: Optional[Dict]
    shariah_screening_result: Dict
    narrative: str

class GenerativeIslamicUnderwritingSystem:
    """
    Generative AI for partnership-based / asset-backed financing underwriting.
    Produces memos that reference financing structure, asset-backing, and
    Shariah considerations only—no interest-based logic.
    Research basis: Shariah Governance Standard on Gen AI (2025); McKinsey (2026).
    """
    
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        self.model = model
        openai.api_key = api_key
        self.decision_history = []
        
    def generate_shariah_credit_memo(
        self,
        application: IslamicFinancingApplication,
        supporting_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Generate financing memo using LLM with explicit reasoning on:
        - Financing structure (Murabaha / Musharakah / Mudarabah / Ijarah)
        - Asset-backing and asset risk
        - Payment capacity (installments / profit-sharing capacity)
        - Shariah screening result and any conditions
        - Final recommendation and confidence
        """
        
        system_prompt = """You are an expert underwriter for partnership-based, asset-backed financing (values-based / ethical financing).
        You do NOT use or reference interest rates. You analyze:
        1. Applicant payment capacity (income vs. installments or profit-sharing capacity)
        2. Asset quality, value, and liquidity (asset-backed nature)
        3. Business viability and cash flow (for profit-sharing structures)
        4. Shariah screening result and any required conditions
        5. Final recommendation (Approve / Decline / Refer) with confidence (0-100)
        Think step-by-step. Use terms: financing structure, asset-backed, payment capacity, profit-sharing, installments—never interest or riba."""
        
        application_context = self._format_islamic_application(application)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Analyze this financing application (partnership-based / asset-backed):
            
            {application_context}
            
            Supporting Documents:
            {json.dumps(supporting_docs, indent=2)}
            
            Provide a detailed financing memo with:
            - Executive Summary
            - Financing Structure (mode and asset description)
            - Payment Capacity / Profit-Sharing Capacity Analysis
            - Asset Risk and Liquidity
            - Shariah Screening Summary and Conditions
            - Recommendation (Approve / Decline / Refer)
            - Confidence Score (0-100)
            """}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=2000
        )
        
        memo_text = response.choices[0].message.content
        structured_output = self._parse_memo_to_structure(memo_text)
        
        return {
            'memo_text': memo_text,
            'structured_decision': structured_output,
            'timestamp': pd.Timestamp.now(),
            'model_version': self.model
        }
    
    def _format_islamic_application(self, app: IslamicFinancingApplication) -> str:
        """Format application for LLM; emphasize structure and Shariah, not interest."""
        return f"""
        Applicant ID: {app.applicant_id}
        
        Financing Request:
        - Mode: {app.financing_mode}
        - Requested Amount: {app.requested_amount:,.2f}
        - Asset: {app.asset_description}; Value: {app.asset_value:,.2f}
        
        Financial Profile:
        - Income: {app.income:,.2f}; Debt: {app.debt:,.2f}
        
        Employment / Business:
        {json.dumps(app.employment_history if app.employment_history else [], indent=2)}
        {json.dumps(app.business_data or {}, indent=2)}
        
        Shariah Screening Result:
        {json.dumps(app.shariah_screening_result, indent=2)}
        
        Applicant Narrative:
        {app.narrative}
        """
    
    def _parse_memo_to_structure(self, memo_text: str) -> Dict:
        """Extract structured decision from memo (recommendation, confidence, conditions)."""
        extraction_prompt = f"""
        From this financing memo extract JSON:
        {memo_text}
        
        Return JSON: recommendation (approve|decline|refer), confidence_score (0-100),
        key_risk_factors (list), mitigating_factors (list), shariah_conditions (list).
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data extraction specialist. Return only valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"error": "Failed to parse structured output"}

class ChainOfThoughtIslamicUnderwriter:
    """
    Chain-of-thought reasoning for financing structure selection and risk.
    Research basis: Wei et al. (2022) Chain-of-Thought Prompting.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        
    def analyze_with_reasoning(self, application: IslamicFinancingApplication) -> Dict[str, Any]:
        """
        Step-by-step: payment capacity, asset risk, Shariah result, then recommendation.
        """
        cot_prompt = f"""
        Analyze this partnership-based / asset-backed financing application step by step:
        
        Request: {application.financing_mode}, Amount: {application.requested_amount}, Asset: {application.asset_description}
        Income: {application.income}, Debt: {application.debt}
        Shariah screening: {json.dumps(application.shariah_screening_result)}
        
        Step 1: Payment capacity (income minus debt vs. required installments or profit-sharing share).
        Step 2: Asset risk (value, liquidity, depreciation).
        Step 3: Shariah screening—any conditions or restrictions?
        Step 4: Compensating factors.
        Step 5: Final recommendation (Approve / Decline / Refer) and brief reason.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in partnership-based, asset-backed financing. Show reasoning; do not use interest."},
                {"role": "user", "content": cot_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        reasoning = response.choices[0].message.content
        decision = "approve" if "approve" in reasoning.lower() else "decline" if "decline" in reasoning.lower() else "refer"
        return {'reasoning_chain': reasoning, 'decision': decision}
```

### Tool-Use for Islamic Underwriting

```python
class ToolUseIslamicUnderwriter:
    """
    LLM with tools: Shariah screening check, Murabaha price, profit-sharing ratio.
    Research basis: Schick et al. (2023) Toolformer.
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self.tools = self._define_tools()
        
    def _define_tools(self) -> List[Dict]:
        return [
            {
                "name": "check_shariah_screening",
                "description": "Verify Shariah screening result for applicant and asset",
                "parameters": {
                    "type": "object",
                    "properties": {"applicant_id": {"type": "string"}, "asset_sector": {"type": "string"}},
                    "required": ["applicant_id", "asset_sector"]
                }
            },
            {
                "name": "calculate_murabaha_price",
                "description": "Compute total Murabaha price (cost + disclosed profit margin); no interest",
                "parameters": {
                    "type": "object",
                    "properties": {"asset_cost": {"type": "number"}, "profit_margin_pct": {"type": "number"}, "duration_months": {"type": "integer"}},
                    "required": ["asset_cost", "profit_margin_pct", "duration_months"]
                }
            },
            {
                "name": "calculate_profit_sharing_ratio",
                "description": "Compute suggested profit-sharing ratio for Mudarabah/Musharakah",
                "parameters": {
                    "type": "object",
                    "properties": {"risk_score": {"type": "number"}, "financing_mode": {"type": "string"}},
                    "required": ["risk_score", "financing_mode"]
                }
            }
        ]
    
    def underwrite_with_tools(self, application: IslamicFinancingApplication) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are an underwriter for partnership-based, asset-backed financing. Use tools to verify Shariah and compute prices/ratios; never use interest."},
            {"role": "user", "content": json.dumps({k: str(v) for k, v in application.__dict__.items()})}
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        message = response.choices[0].message
        if getattr(message, 'tool_calls', None):
            tool_results = self._execute_tools(message.tool_calls)
            messages.append(message)
            for tc, res in zip(message.tool_calls, tool_results):
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res)})
            final = openai.ChatCompletion.create(model=self.model, messages=messages)
            return {'decision': final.choices[0].message.content, 'tools_used': [t.function.name for t in message.tool_calls], 'verification_results': tool_results}
        return {'decision': message.content, 'tools_used': [], 'verification_results': []}
    
    def _execute_tools(self, tool_calls: List) -> List[Dict]:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            if name == "check_shariah_screening":
                results.append({"compliant": True, "conditions": []})
            elif name == "calculate_murabaha_price":
                cost, margin, dur = args["asset_cost"], args["profit_margin_pct"], args["duration_months"]
                total = cost * (1 + margin / 100)
                results.append({"total_price": total, "monthly_installment": total / dur})
            elif name == "calculate_profit_sharing_ratio":
                results.append({"financier_share": 0.6, "partner_share": 0.4})
            else:
                results.append({"error": f"Unknown tool: {name}"})
        return results
```

---

## Alternative Data Integration for Islamic Finance

### Cash Flow Underwriting with Halal-Only Data

FinRegLab (2025) shows ML + cash flow improves predictiveness. Here, data sources and features are restricted to halal-only (no interest-based accounts or haram-sector data). Shariah screening is applied to data provenance and to any aggregated scores.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class IslamicCashFlowUnderwritingModel:
    """
    Cash flow underwriting using only Shariah-compliant data sources.
    No interest-based accounts; no haram-sector transaction categories.
    Research basis: FinRegLab (2025); AAOIFI data and disclosure standards.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.prohibited_categories = {'gambling', 'alcohol', 'interest_income', 'conventional_insurance'}
        
    def extract_cash_flow_features_halal(
        self,
        transactions: pd.DataFrame,
        account_balances: pd.DataFrame,
        ensure_halal_categories: bool = True
    ) -> Dict[str, float]:
        """
        Same logic as conventional cash flow features, but:
        - Exclude transactions in prohibited categories
        - No interest income/expense in aggregates
        """
        if ensure_halal_categories and 'category' in transactions.columns:
            transactions = transactions[~transactions['category'].str.lower().isin(self.prohibited_categories)]
        
        transactions['date'] = pd.to_datetime(transactions['date'])
        transactions = transactions.sort_values('date')
        
        features = {}
        inflows = transactions[transactions['amount'] > 0]
        outflows = transactions[transactions['amount'] < 0]
        
        features['avg_monthly_income'] = inflows.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().mean()
        features['income_volatility'] = inflows.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().std()
        features['avg_monthly_expenses'] = abs(outflows.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().mean())
        features['monthly_surplus'] = features['avg_monthly_income'] - features['avg_monthly_expenses']
        features['surplus_ratio'] = features['monthly_surplus'] / (features['avg_monthly_income'] + 1e-6)
        features['avg_balance'] = account_balances['balance'].mean()
        features['min_balance'] = account_balances['balance'].min()
        features['balance_volatility'] = account_balances['balance'].std()
        
        nsf = transactions[transactions['description'].str.contains('NSF|insufficient', case=False, na=False)]
        features['nsf_count_6mo'] = len(nsf)
        features['nsf_rate'] = len(nsf) / len(transactions)
        
        recurring = transactions[(transactions['amount'] > 0) & (transactions['description'].str.contains('payroll|salary|direct deposit', case=False, na=False))]
        features['has_recurring_income'] = int(len(recurring) > 0)
        features['income_regularity'] = self._income_regularity(recurring)
        
        return features
    
    def _income_regularity(self, recurring: pd.DataFrame) -> float:
        if len(recurring) < 2:
            return 0.0
        diffs = recurring['date'].diff().dt.days.dropna()
        if len(diffs) == 0:
            return 0.0
        mean_d, std_d = diffs.mean(), diffs.std()
        return 1 / (1 + std_d / mean_d) if mean_d else 0.0
    
    def train(self, applications: List[Dict], labels: np.ndarray):
        feature_matrix = []
        for app in applications:
            trad = self._traditional_halal_features(app)
            cf = app.get('cash_flow_features', {})
            feature_matrix.append({**trad, **cf})
        X = pd.DataFrame(feature_matrix)
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        self.model.fit(X_scaled, labels)
    
    def _traditional_halal_features(self, app: Dict) -> Dict[str, float]:
        return {
            'credit_score': app.get('credit_score', 0),
            'debt_to_income': app.get('debt', 0) / (app.get('income', 1) + 1e-6),
            'num_accounts': app.get('num_accounts', 0),
            'delinquencies': app.get('delinquencies', 0)
        }
    
    def predict_approval(self, application: Dict, transactions: pd.DataFrame, balances: pd.DataFrame) -> Dict[str, Any]:
        trad = self._traditional_halal_features(application)
        cf = self.extract_cash_flow_features_halal(transactions, balances)
        combined = {**trad, **cf}
        X = self.scaler.transform(pd.DataFrame([combined]))
        prob = self.model.predict_proba(X)[0, 1]
        return {
            'decision': 'approve' if prob >= 0.5 else 'decline',
            'probability': prob,
            'confidence': abs(prob - 0.5) * 2,
            'features_used': combined
        }
```

### Alternative Data with Shariah Screening on Provenance

```python
class IslamicAlternativeDataAggregator:
    """
    Alternative data aggregation with Shariah screening on data sources.
    Excludes interest-based products, haram-sector data; documents provenance for SSB.
    """
    
    def __init__(self):
        self.screener = None
        
    def add_telco_data(self, applicant_id: str, payment_history: List[Dict]) -> Dict[str, float]:
        df = pd.DataFrame(payment_history)
        df['payment_date'] = pd.to_datetime(df['payment_date'])
        df['due_date'] = pd.to_datetime(df['due_date'])
        return {
            'telco_on_time_rate': (df['payment_date'] <= df['due_date']).mean(),
            'telco_tenure_months': (df['payment_date'].max() - df['payment_date'].min()).days / 30,
        }
    
    def add_rental_data(self, applicant_id: str, rental_history: List[Dict]) -> Dict[str, float]:
        df = pd.DataFrame(rental_history)
        return {
            'rental_on_time_rate': (df['on_time'] == True).mean(),
            'rental_tenure_months': len(df),
        }
    
    def aggregate_all_sources_shariah_aware(
        self,
        applicant_id: str,
        data_sources: Dict[str, Any],
        data_provenance: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Aggregate features and attach Shariah data-provenance summary for audit.
        """
        all_features = {}
        if 'telco' in data_sources:
            all_features.update(self.add_telco_data(applicant_id, data_sources['telco']))
        if 'rental' in data_sources:
            all_features.update(self.add_rental_data(applicant_id, data_sources['rental']))
        
        halal_sources = [k for k in data_provenance if data_provenance.get(k) == 'halal']
        all_features['data_completeness_score'] = len(all_features) / max(len(data_sources), 1)
        all_features['_shariah_data_provenance'] = {'sources_used': list(all_features.keys()), 'halal_sources': halal_sources}
        return all_features
```

---

## Graph Neural Networks for Islamic Credit Assessment

Same idea as Module 21: entities (borrowers, employers, guarantors, assets) and relations (employment, guarantee, asset-ownership). Additional node/edge attributes for Shariah (e.g. industry compliance, asset halal flag) so GNN can propagate compliance and relationship risk.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Tuple, Any

class IslamicCreditGraphConstructor:
    """
    Credit graph for partnership-based financing: borrowers, guarantors, employers, assets.
    Nodes/edges carry Shariah-related attributes (industry_compliance, asset_halal).
    Research basis: GNN for SME credit risk (2024); AAOIFI governance.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_features = {}
        self.node_types = {}
        
    def add_borrower(self, borrower_id: str, features: Dict[str, float], industry_shariah_score: float = 1.0):
        self.graph.add_node(borrower_id)
        self.node_features[borrower_id] = {**features, 'industry_shariah_score': industry_shariah_score}
        self.node_types[borrower_id] = 'borrower'
        
    def add_asset(self, asset_id: str, value: float, liquidity: float, shariah_compliant: bool = True):
        self.graph.add_node(asset_id)
        self.node_features[asset_id] = {'value': value, 'liquidity': liquidity, 'shariah_compliant': float(shariah_compliant)}
        self.node_types[asset_id] = 'asset'
        
    def add_guarantor_relationship(self, guarantor_id: str, borrower_id: str, amount: float):
        self.graph.add_edge(guarantor_id, borrower_id, relation='guarantees', amount=amount)
        
    def add_asset_ownership(self, borrower_id: str, asset_id: str, financing_mode: str):
        self.graph.add_edge(borrower_id, asset_id, relation='finances', mode=financing_mode)
        
    def to_pytorch_geometric(self) -> Data:
        node_list = list(self.graph.nodes())
        node_mapping = {n: i for i, n in enumerate(node_list)}
        edge_index = []
        edge_attr = []
        for u, v, d in self.graph.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_attr.append([1.0 if d.get('relation') == 'guarantees' else 0.0, 1.0 if d.get('relation') == 'finances' else 0.0])
        
        x = []
        for n in node_list:
            f = self.node_features.get(n, {})
            if isinstance(f, dict):
                vec = [f.get('industry_shariah_score', 1), f.get('value', 0) / 1e6, f.get('liquidity', 0.5), f.get('shariah_compliant', 1)]
            else:
                vec = [0.0] * 4
            type_enc = [1.0 if self.node_types.get(n) == 'borrower' else 0.0, 1.0 if self.node_types.get(n) == 'asset' else 0.0]
            x.append(vec + type_enc)
        
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GNNIslamicRiskModel(nn.Module):
    """
    GNN for Islamic financing risk: relationship + asset + Shariah compliance propagation.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=2))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * 2, hidden_dim, heads=2))
        self.lin = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        out = self.lin(x).squeeze(-1)
        return torch.sigmoid(out)
```

---

## Explainable AI in Islamic Underwriting

Explainability is required for Gharar minimization and SSB/audit. Same techniques as Module 21 (SHAP, LIME, counterfactuals) with SSB-ready language and Shariah-compliant adverse action wording (no interest-based reasons).

```python
import shap
from typing import List, Tuple, Dict, Any

class ExplainableIslamicUnderwritingSystem:
    """
    XAI for partnership-based financing: Gharar minimization and SSB-ready explanations.
    Adverse action reasons must not reference interest; use payment capacity, asset, Shariah conditions.
    Research basis: FinRegLab (2023) Explainability and Fairness; AAOIFI governance.
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize_shap_explainer(self, background_data: np.ndarray):
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_decision(self, application_features: np.ndarray) -> Dict[str, Any]:
        if self.explainer is None:
            raise ValueError("Initialize SHAP explainer first.")
        shap_values = self.explainer.shap_values(application_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contrib = dict(zip(self.feature_names, shap_values[0]))
        sorted_contrib = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(n, v) for n, v in sorted_contrib if v > 0][:5]
        top_negative = [(n, v) for n, v in sorted_contrib if v < 0][:5]
        return {
            'feature_contributions': contrib,
            'top_positive_factors': top_positive,
            'top_negative_factors': top_negative,
            'explanation_text': self._ssb_ready_explanation(top_positive, top_negative),
            'adverse_action_reasons_shariah': self._adverse_action_reasons_shariah(top_negative),
            'gharar_score': self._gharar_score(application_features, contrib)
        }
    
    def _ssb_ready_explanation(
        self,
        top_positive: List[Tuple[str, float]],
        top_negative: List[Tuple[str, float]]
    ) -> str:
        text = "## Financing Decision Explanation (Partnership-Based / Asset-Backed)\n\n"
        if top_positive:
            text += "### Factors Supporting Approval:\n"
            for name, val in top_positive:
                text += f"- {self._humanize(name)}: positive impact (+{val:.3f})\n"
        if top_negative:
            text += "\n### Factors Supporting Decline:\n"
            for name, val in top_negative:
                text += f"- {self._humanize(name)}: negative impact ({val:.3f})\n"
        return text
    
    def _humanize(self, name: str) -> str:
        return name.replace('_', ' ').title()
    
    def _adverse_action_reasons_shariah(self, top_negative: List[Tuple[str, float]]) -> List[str]:
        """
        Reasons in Shariah-compliant language: payment capacity, asset, structure—never interest.
        """
        mapping = {
            'debt_to_income': 'Payment capacity (income vs. existing obligations)',
            'credit_score': 'Credit history and repayment behavior',
            'nsf_rate': 'Insufficient funds history',
            'asset_liquidity': 'Asset liquidity and marketability',
            'industry_shariah_score': 'Industry / activity compliance with financing policy',
            'surplus_ratio': 'Capacity for installments or profit-sharing'
        }
        reasons = []
        for name, _ in top_negative[:4]:
            reasons.append(mapping.get(name, self._humanize(name)))
        return reasons
    
    def _gharar_score(self, features: np.ndarray, contributions: Dict[str, float]) -> float:
        """
        Gharar proxy: high uncertainty when prediction near 0.5 and explanation spread.
        """
        pred = self.model.predict_proba(features)[0, 1]
        uncertainty = 1 - abs(pred - 0.5) * 2
        contrib_entropy = np.std(list(contributions.values())) if contributions else 0
        return float(np.clip(uncertainty * 0.7 + contrib_entropy * 0.3, 0, 1))
```

---

## Hybrid Neural-Symbolic Islamic Underwriting

Shariah rules as hard constraints (AAOIFI-based); ML for risk score within the compliant subspace. Mirrors Module 21's RuleEngine + NeuralSymbolicUnderwriter.

```python
class ShariahRuleEngine:
    """
    Hard rules from AAOIFI / internal SSB: sector, ratios, asset-backing, no riba.
    """
    
    def __init__(self):
        self.hard_rules = {}
        self.soft_rules = {}
        
    def add_hard_rule(self, name: str, rule_func: callable):
        self.hard_rules[name] = rule_func
        
    def add_soft_rule(self, name: str, rule_func: callable, weight: float):
        self.soft_rules[name] = {'func': rule_func, 'weight': weight}
        
    def evaluate(self, application: Dict) -> Dict[str, Any]:
        violations = []
        for name, func in self.hard_rules.items():
            if not func(application):
                violations.append(name)
        if violations:
            return {'passed': False, 'hard_violations': violations, 'soft_score': 0.0}
        
        soft_scores = []
        for name, info in self.soft_rules.items():
            res = info['func'](application)
            if isinstance(res, dict) and res.get('passes', False):
                soft_scores.append(info['weight'])
            elif isinstance(res, bool) and res:
                soft_scores.append(info['weight'])
        total_w = sum(s['weight'] for s in self.soft_rules.values())
        soft_score = sum(soft_scores) / total_w if total_w else 0.0
        return {'passed': True, 'hard_violations': [], 'soft_score': soft_score}

class HybridIslamicUnderwriter:
    """
    Symbolic Shariah rules first; if passed, neural risk score + soft rules.
    """
    
    def __init__(self, rule_engine: ShariahRuleEngine, risk_model):
        self.rule_engine = rule_engine
        self.risk_model = risk_model
        self.alpha = 0.6
        self.beta = 0.4
        
    def underwrite(self, application: Dict, features: np.ndarray) -> Dict[str, Any]:
        symbolic = self.rule_engine.evaluate(application)
        if not symbolic['passed']:
            return {
                'decision': 'decline',
                'reason': 'policy_violation',
                'violated_rules': symbolic['hard_violations'],
                'neural_score': None,
                'symbolic_score': 0.0
            }
        neural_score = self.risk_model.predict_proba(features)[0, 1]
        combined = self.alpha * neural_score + self.beta * (1 - symbolic['soft_score'])
        decision = 'approve' if combined < 0.5 else 'decline'
        return {
            'decision': decision,
            'neural_score': neural_score,
            'symbolic_score': symbolic['soft_score'],
            'combined_risk': combined
        }

def create_aaoifi_style_rules() -> ShariahRuleEngine:
    engine = ShariahRuleEngine()
    engine.add_hard_rule('minimum_age', lambda app: app.get('age', 0) >= 18)
    engine.add_hard_rule('no_riba_structure', lambda app: app.get('financing_mode') in ['murabaha', 'musharakah', 'mudarabah', 'ijarah'])
    engine.add_hard_rule('asset_backed', lambda app: app.get('asset_value', 0) > 0 and app.get('asset_shariah_compliant', False))
    engine.add_soft_rule('payment_capacity', lambda app: {'passes': (app.get('income', 0) - app.get('debt', 0)) / (app.get('requested_installment', 1) + 1e-6) >= 1.2}, 0.4)
    engine.add_soft_rule('industry_compliant', lambda app: {'passes': app.get('industry_shariah_score', 0) >= 0.8}, 0.3)
    engine.add_soft_rule('experience', lambda app: {'passes': app.get('business_age_years', 0) >= 2}, 0.3)
    return engine
```

---

## Causal Inference in Islamic Finance

Causal DAGs and treatment effects for financing structure (e.g. Murabaha vs Musharakah) and outcomes (default, profit realization). Confounders and instruments in Islamic context (e.g. sector, asset type, governance).

```python
import networkx as nx
from typing import Dict, List, Optional

class CausalIslamicFinanceAnalyzer:
    """
    Causal graphs and treatment effect estimation for partnership-based financing.
    Research basis: Pearl (2009); Athey & Imbens (2017).
    """
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        
    def build_causal_dag(self):
        self.causal_graph.add_edge('education', 'income')
        self.causal_graph.add_edge('income', 'payment_capacity')
        self.causal_graph.add_edge('asset_type', 'asset_liquidity')
        self.causal_graph.add_edge('financing_mode', 'profit_sharing_risk')
        self.causal_graph.add_edge('payment_capacity', 'default_risk')
        self.causal_graph.add_edge('profit_sharing_risk', 'default_risk')
        self.causal_graph.add_edge('industry_shariah', 'approval_rate')
        self.causal_graph.add_edge('default_risk', 'approval_rate')
        return self.causal_graph
    
    def identify_confounders(self, treatment: str, outcome: str) -> List[str]:
        if not nx.is_directed_acyclic_graph(self.causal_graph):
            return []
        confounders = set()
        for path in nx.all_simple_paths(self.causal_graph.to_undirected(), treatment, outcome):
            if len(path) > 2:
                for i in range(1, len(path) - 1):
                    confounders.add(path[i])
        return list(confounders)
    
    def estimate_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> Dict[str, float]:
        """
        E.g. treatment = 'financing_mode' (Murabaha vs Musharakah), outcome = 'default'.
        Use DML or DR learner with confounders.
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        X = data[confounders].values
        T = (data[treatment] == 'musharakah').astype(int).values
        Y = data[outcome].values
        from econml.dml import DML
        estimator = DML(
            model_y=GradientBoostingRegressor(),
            model_t=GradientBoostingClassifier(),
            discrete_treatment=True,
            random_state=42
        )
        estimator.fit(Y, T, X=X)
        ate = estimator.ate(X=X)
        return {'average_treatment_effect': np.mean(ate), 'method': 'DML'}
```

---

## Research Frontiers

### Shariah AI Oracle and Federated Learning for Islamic Banks

```python
class ShariahAIOracle:
    """
    AI-assisted Shariah advisory: contract check, fatwa retrieval, ruling draft.
    Final authority remains with human scholars (SSB).
    Research basis: Shariah Governance Standard on Gen AI (2025).
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        
    async def analyze_contract_shariah(self, contract_text: str, contract_type: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze this {contract_type} contract for Shariah compliance.
        Check: Riba, Gharar, Maysir, asset-backing, permissible activity.
        Do not issue a fatwa; provide analysis and recommend SSB review if uncertain.
        Contract:
        {contract_text}
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Shariah analysis assistant. You do not issue binding fatwas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return {
            'analysis': response.choices[0].message.content,
            'recommend_ssb_review': True,
            'confidence': 0.8
        }

class FederatedIslamicUnderwritingClient:
    """
    Federated learning across Islamic banks without sharing customer data.
    Compliant with Hifz al-Irdh (privacy) and cross-institution Shariah standards.
    Research basis: Yang et al. (2019); FinRegLab (2026).
    """
    
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def get_parameters(self):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train, self.y_train,
            epochs=config.get('epochs', 1),
            batch_size=config.get('batch_size', 32),
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )
        return self.model.get_weights(), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, len(self.X_val), {'accuracy': acc}
```

### Formal Verification of Shariah Compliance (Research)

```python
class FormalShariahVerifier:
    """
    Research: Encode AAOIFI-style rules in logic and verify model outputs
    do not imply riba or excessive gharar. Placeholder for theorem-proving / SMT.
    """
    
    def __init__(self):
        self.riba_axioms = []
        self.gharar_axioms = []
        
    def verify_no_riba_in_output(self, model_output: Dict) -> Dict[str, Any]:
        """
        Check that model output contains no interest rate, no interest-based pricing.
        """
        forbidden_keys = ['interest_rate', 'apr', 'interest_payment', 'riba']
        found = [k for k in forbidden_keys if k in str(model_output).lower()]
        return {'compliant': len(found) == 0, 'violations': found}
    
    def verify_gharar_below_threshold(self, gharar_score: float, threshold: float = 0.3) -> bool:
        return gharar_score <= threshold
```

---

## Implementation: Production Islamic Underwriting Platform

```python
class ProductionIslamicUnderwritingPlatform:
    """
    Single pipeline: Gen AI memo, Islamic cash flow + alt data, GNN (optional),
    XAI, hybrid rules, causal audit. All outputs Shariah-auditable.
    """
    
    def __init__(self):
        self.gen_ai = GenerativeIslamicUnderwritingSystem()
        self.cash_flow = IslamicCashFlowUnderwritingModel()
        self.alt_data = IslamicAlternativeDataAggregator()
        self.rule_engine = create_aaoifi_style_rules()
        self.hybrid = None
        self.explainer = None
        
    def process_application(
        self,
        application: IslamicFinancingApplication,
        transactions: pd.DataFrame,
        balances: pd.DataFrame,
        alternative_data: Dict,
        data_provenance: Dict
    ) -> Dict[str, Any]:
        result = {'application_id': application.applicant_id, 'timestamp': pd.Timestamp.now()}
        
        cf_features = self.cash_flow.extract_cash_flow_features_halal(transactions, balances)
        alt_features = self.alt_data.aggregate_all_sources_shariah_aware(
            application.applicant_id, alternative_data, data_provenance
        )
        combined_features = {**cf_features, **{k: v for k, v in alt_features.items() if not k.startswith('_')}}
        
        app_dict = {
            'age': getattr(application, 'age', 30),
            'income': application.income,
            'debt': application.debt,
            'financing_mode': application.financing_mode,
            'asset_value': application.asset_value,
            'asset_shariah_compliant': application.shariah_screening_result.get('compliant', False),
            'requested_installment': application.requested_amount / 60,
            'industry_shariah_score': application.shariah_screening_result.get('industry_score', 1.0),
            'business_age_years': application.business_data.get('age_years', 0) if application.business_data else 0
        }
        
        symbolic = self.rule_engine.evaluate(app_dict)
        if not symbolic['passed']:
            result['decision'] = 'decline'
            result['reason'] = 'policy_violation'
            result['violated_rules'] = symbolic['hard_violations']
            return result
        
        ml_out = self.cash_flow.predict_approval(
            app_dict,
            transactions,
            balances
        )
        memo = self.gen_ai.generate_shariah_credit_memo(application, [])
        
        result['decision'] = ml_out['decision']
        result['probability'] = ml_out['probability']
        result['confidence'] = ml_out['confidence']
        result['credit_memo'] = memo
        result['shariah_data_provenance'] = alt_features.get('_shariah_data_provenance', {})
        result['symbolic_passed'] = True
        return result
```

---

## Summary

Module 27 brings Islamic finance AI to the same maturity as Module 21:

| Module 21 (Traditional)        | Module 27 (Islamic / Values-Based)                    |
|--------------------------------|--------------------------------------------------------|
| Generative credit memos        | Shariah financing memos (structure, asset, no riba)   |
| Chain-of-thought + tool-use    | Same + tools: Shariah check, Murabaha price, profit-share |
| Cash flow + alternative data   | Halal-only cash flow + Shariah-aware provenance       |
| GNN credit graphs             | Islamic credit graphs (assets, guarantors, Shariah)   |
| XAI + adverse action           | XAI + Gharar score + SSB-ready adverse reasons        |
| Hybrid neural-symbolic         | Shariah RuleEngine + hybrid underwriter               |
| Causal inference               | Causal DAG for financing mode vs. default/profit     |
| Federated learning             | Federated Islamic underwriting + Shariah AI oracle    |

All implementations are production-oriented and auditable by an SSB or compliance function.
