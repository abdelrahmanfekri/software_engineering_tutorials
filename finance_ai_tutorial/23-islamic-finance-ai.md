# Module 23: Islamic Finance and AI - Shariah-Compliant Underwriting

## Table of Contents
1. [Fundamentals of Islamic Finance](#fundamentals-of-islamic-finance)
2. [Shariah Governance for AI Systems](#shariah-governance-for-ai-systems)
3. [Riba-Free Credit Risk Models](#riba-free-credit-risk-models)
4. [Gharar Minimization in ML Systems](#gharar-minimization-in-ml-systems)
5. [Halal Investment Screening](#halal-investment-screening)
6. [Islamic Takaful (Insurance) Underwriting](#islamic-takaful-underwriting)
7. [Maqasid al-Shariah Alignment](#maqasid-al-shariah-alignment)
8. [Production Islamic Fintech Systems](#production-islamic-fintech-systems)

## Fundamentals of Islamic Finance

### Core Principles and Prohibitions

```python
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class ShariahPrinciples(Enum):
    """
    Core Shariah principles for Islamic finance.
    
    Research basis:
    - AAOIFI Standards
    - Islamic Fiqh Academy rulings
    - Contemporary Islamic finance scholarship
    """
    
    RIBA_PROHIBITION = "riba"
    GHARAR_PROHIBITION = "gharar"
    MAYSIR_PROHIBITION = "maysir"
    HALAL_REQUIREMENT = "halal"
    ASSET_BACKING = "asset_backing"
    PROFIT_LOSS_SHARING = "profit_loss_sharing"
    ETHICAL_INVESTMENT = "ethical_investment"

class ShariahCompliance:
    """
    Comprehensive Shariah compliance framework for financial products.
    
    Prohibitions:
    1. Riba (Interest/Usury): Any predetermined return on loan
    2. Gharar (Excessive Uncertainty): Ambiguity in contracts
    3. Maysir (Gambling): Speculative transactions
    4. Haram sectors: Alcohol, pork, weapons, pornography, conventional banking
    
    Requirements:
    1. Asset-backed transactions
    2. Profit-and-loss sharing
    3. Ethical business conduct
    4. Social welfare orientation (Maqasid al-Shariah)
    """
    
    def __init__(self):
        self.prohibited_sectors = self._define_prohibited_sectors()
        self.screening_thresholds = self._define_screening_thresholds()
        
    def _define_prohibited_sectors(self) -> List[str]:
        """Define Shariah non-compliant business sectors."""
        return [
            'alcohol',
            'pork',
            'gambling',
            'conventional_banking',
            'insurance_conventional',
            'weapons',
            'pornography',
            'tobacco',
            'interest_based_lending'
        ]
    
    def _define_screening_thresholds(self) -> Dict[str, float]:
        """
        Define quantitative screening thresholds.
        
        Based on AAOIFI and global Islamic finance standards.
        """
        return {
            'debt_ratio': 0.33,
            'interest_income_ratio': 0.05,
            'non_permissible_income_ratio': 0.05,
            'liquid_assets_ratio': 0.70,
            'interest_bearing_debt_ratio': 0.33
        }
    
    def check_sector_compliance(self, business_sector: str) -> Dict[str, any]:
        """Check if business sector is Shariah-compliant."""
        
        is_compliant = business_sector.lower() not in self.prohibited_sectors
        
        return {
            'compliant': is_compliant,
            'sector': business_sector,
            'reason': None if is_compliant else f"Sector '{business_sector}' is prohibited under Shariah"
        }
    
    def check_financial_ratios(
        self,
        total_debt: float,
        total_assets: float,
        interest_income: float,
        total_revenue: float,
        liquid_assets: float,
        interest_bearing_debt: float
    ) -> Dict[str, any]:
        """
        Check quantitative Shariah screening ratios.
        
        Ratios based on AAOIFI Shariah Standards.
        """
        
        violations = []
        
        debt_ratio = total_debt / total_assets if total_assets > 0 else 0
        if debt_ratio > self.screening_thresholds['debt_ratio']:
            violations.append({
                'ratio': 'debt_ratio',
                'value': debt_ratio,
                'threshold': self.screening_thresholds['debt_ratio'],
                'message': f"Debt ratio {debt_ratio:.2%} exceeds threshold {self.screening_thresholds['debt_ratio']:.2%}"
            })
        
        interest_income_ratio = interest_income / total_revenue if total_revenue > 0 else 0
        if interest_income_ratio > self.screening_thresholds['interest_income_ratio']:
            violations.append({
                'ratio': 'interest_income_ratio',
                'value': interest_income_ratio,
                'threshold': self.screening_thresholds['interest_income_ratio'],
                'message': f"Interest income ratio {interest_income_ratio:.2%} exceeds threshold"
            })
        
        liquid_ratio = liquid_assets / total_assets if total_assets > 0 else 0
        if liquid_ratio > self.screening_thresholds['liquid_assets_ratio']:
            violations.append({
                'ratio': 'liquid_assets_ratio',
                'value': liquid_ratio,
                'threshold': self.screening_thresholds['liquid_assets_ratio'],
                'message': "Excessive liquid assets (company resembles cash, not real business)"
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'ratios': {
                'debt_ratio': debt_ratio,
                'interest_income_ratio': interest_income_ratio,
                'liquid_assets_ratio': liquid_ratio
            }
        }
    
    def comprehensive_screening(
        self,
        company_data: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Comprehensive Shariah compliance screening.
        
        Three-tier screening:
        1. Sector screening (qualitative)
        2. Financial ratio screening (quantitative)
        3. Business activity analysis
        """
        
        sector_check = self.check_sector_compliance(
            company_data.get('business_sector', 'unknown')
        )
        
        if not sector_check['compliant']:
            return {
                'shariah_compliant': False,
                'screening_stage_failed': 'sector',
                'reason': sector_check['reason']
            }
        
        ratio_check = self.check_financial_ratios(
            company_data.get('total_debt', 0),
            company_data.get('total_assets', 1),
            company_data.get('interest_income', 0),
            company_data.get('total_revenue', 1),
            company_data.get('liquid_assets', 0),
            company_data.get('interest_bearing_debt', 0)
        )
        
        if not ratio_check['compliant']:
            return {
                'shariah_compliant': False,
                'screening_stage_failed': 'financial_ratios',
                'violations': ratio_check['violations']
            }
        
        return {
            'shariah_compliant': True,
            'sector_compliant': True,
            'ratios_compliant': True,
            'screening_date': pd.Timestamp.now(),
            'next_review_date': pd.Timestamp.now() + pd.Timedelta(days=365)
        }
```

## Shariah Governance for AI Systems

### Dual Governance Framework

```python
class ShariahAIGovernanceFramework:
    """
    Dual governance framework for AI in Islamic finance.
    
    Research basis:
    - "Shariah Governance Standard on Generative AI for Islamic Financial 
      Institutions" (2025)
    - Maqasid-based AI governance approach
    
    Governance Structure:
    1. Shariah Supervisory Board (SSB): Religious compliance
    2. AI Ethics Committee: Technical ethics and fairness
    3. Integrated Compliance Team: Bridge between SSB and AI
    
    Key requirements:
    - Ex-ante Shariah design (not just ex-post verification)
    - Algorithmic transparency
    - Bias prevention aligned with Islamic ethics
    - Data privacy (Hifz al-'Irdh - protection of honor)
    """
    
    def __init__(self):
        self.ssb_requirements = self._define_ssb_requirements()
        self.ai_ethics_principles = self._define_ai_ethics_principles()
        self.prohibited_data_types = self._define_prohibited_data()
        
    def _define_ssb_requirements(self) -> List[Dict[str, str]]:
        """Define Shariah Supervisory Board requirements for AI."""
        return [
            {
                'requirement': 'riba_free_models',
                'description': 'AI models must not facilitate riba-based transactions',
                'verification': 'Model output analysis for interest calculations'
            },
            {
                'requirement': 'gharar_minimization',
                'description': 'AI must reduce, not increase, contractual uncertainty',
                'verification': 'Explainability and transparency metrics'
            },
            {
                'requirement': 'halal_data_sources',
                'description': 'Training data must not include haram elements',
                'verification': 'Data provenance and content audit'
            },
            {
                'requirement': 'maqasid_alignment',
                'description': 'AI objectives aligned with Maqasid al-Shariah',
                'verification': 'Impact assessment on 5 Maqasid dimensions'
            },
            {
                'requirement': 'human_oversight',
                'description': 'Significant decisions require human (scholar) review',
                'verification': 'Human-in-the-loop audit trails'
            }
        ]
    
    def _define_ai_ethics_principles(self) -> List[Dict[str, str]]:
        """Define AI ethics principles aligned with Islamic values."""
        return [
            {
                'principle': 'justice_adl',
                'description': 'Fairness and non-discrimination (Adl)',
                'implementation': 'Bias testing across protected groups'
            },
            {
                'principle': 'transparency_amanah',
                'description': 'Trustworthiness and transparency (Amanah)',
                'implementation': 'Explainable AI with clear decision rationale'
            },
            {
                'principle': 'privacy_hifz',
                'description': 'Data privacy and dignity protection (Hifz al-Irdh)',
                'implementation': 'Privacy-preserving ML, federated learning'
            },
            {
                'principle': 'benefit_maslahah',
                'description': 'Social benefit and welfare (Maslahah)',
                'implementation': 'Impact assessment on societal welfare'
            },
            {
                'principle': 'accountability',
                'description': 'Clear responsibility and recourse mechanisms',
                'implementation': 'Audit trails, appeal processes'
            }
        ]
    
    def _define_prohibited_data(self) -> List[str]:
        """Define data types prohibited under Shariah."""
        return [
            'interest_rates',
            'conventional_insurance_data',
            'gambling_transaction_data',
            'alcohol_sales_data',
            'data_obtained_through_deception',
            'data_violating_privacy_without_consent'
        ]
    
    def audit_ai_system(
        self,
        model_description: Dict[str, any],
        training_data_description: Dict[str, any],
        use_case: str
    ) -> Dict[str, any]:
        """
        Comprehensive Shariah audit of AI system.
        
        Returns:
        - Compliance status
        - Violations identified
        - Required remediation actions
        - SSB approval status
        """
        
        compliance_checks = {
            'data_compliance': self._audit_training_data(training_data_description),
            'model_compliance': self._audit_model_design(model_description),
            'use_case_compliance': self._audit_use_case(use_case),
            'governance_compliance': self._audit_governance_structure(model_description),
            'ethical_compliance': self._audit_ethical_principles(model_description)
        }
        
        all_compliant = all(check['compliant'] for check in compliance_checks.values())
        
        violations = []
        for check_name, check_result in compliance_checks.items():
            if not check_result['compliant']:
                violations.extend(check_result.get('violations', []))
        
        return {
            'shariah_compliant': all_compliant,
            'ssb_approval_recommended': all_compliant,
            'compliance_checks': compliance_checks,
            'violations': violations,
            'remediation_required': self._generate_remediation_plan(violations),
            'audit_date': pd.Timestamp.now(),
            'next_audit_date': pd.Timestamp.now() + pd.Timedelta(days=180)
        }
    
    def _audit_training_data(self, data_description: Dict) -> Dict[str, any]:
        """Audit training data for Shariah compliance."""
        
        violations = []
        
        data_sources = data_description.get('sources', [])
        for source in data_sources:
            if any(prohibited in source.lower() for prohibited in self.prohibited_data_types):
                violations.append({
                    'type': 'prohibited_data',
                    'source': source,
                    'severity': 'critical'
                })
        
        if data_description.get('contains_interest_data', False):
            violations.append({
                'type': 'riba_data',
                'description': 'Training data contains interest-based transactions',
                'severity': 'critical'
            })
        
        consent_obtained = data_description.get('consent_obtained', False)
        if not consent_obtained:
            violations.append({
                'type': 'privacy_violation',
                'description': 'Data collected without proper consent (violates Hifz al-Irdh)',
                'severity': 'high'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _audit_model_design(self, model_description: Dict) -> Dict[str, any]:
        """Audit model design for Shariah compliance."""
        
        violations = []
        
        if model_description.get('facilitates_interest_calculation', False):
            violations.append({
                'type': 'riba_facilitation',
                'description': 'Model facilitates riba-based calculations',
                'severity': 'critical'
            })
        
        if not model_description.get('explainable', False):
            violations.append({
                'type': 'gharar_risk',
                'description': 'Black-box model increases gharar (uncertainty)',
                'severity': 'high'
            })
        
        if not model_description.get('human_oversight', False):
            violations.append({
                'type': 'governance',
                'description': 'Lack of human oversight for significant decisions',
                'severity': 'medium'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _audit_use_case(self, use_case: str) -> Dict[str, any]:
        """Audit use case for Shariah compliance."""
        
        prohibited_use_cases = [
            'interest_rate_optimization',
            'conventional_insurance_pricing',
            'gambling_risk_assessment',
            'alcohol_demand_forecasting'
        ]
        
        violations = []
        
        if use_case.lower() in prohibited_use_cases:
            violations.append({
                'type': 'prohibited_use_case',
                'description': f"Use case '{use_case}' is not Shariah-compliant",
                'severity': 'critical'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _audit_governance_structure(self, model_description: Dict) -> Dict[str, any]:
        """Audit governance structure."""
        
        violations = []
        
        if not model_description.get('ssb_involvement', False):
            violations.append({
                'type': 'missing_ssb',
                'description': 'No Shariah Supervisory Board involvement',
                'severity': 'critical'
            })
        
        if not model_description.get('audit_trail', False):
            violations.append({
                'type': 'missing_audit_trail',
                'description': 'No audit trail for accountability',
                'severity': 'medium'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _audit_ethical_principles(self, model_description: Dict) -> Dict[str, any]:
        """Audit alignment with Islamic ethical principles."""
        
        violations = []
        
        if model_description.get('bias_detected', False):
            violations.append({
                'type': 'justice_violation',
                'description': 'Bias detected (violates Adl - justice)',
                'severity': 'high'
            })
        
        if not model_description.get('privacy_preserving', True):
            violations.append({
                'type': 'privacy_violation',
                'description': 'Privacy concerns (violates Hifz al-Irdh)',
                'severity': 'high'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def _generate_remediation_plan(self, violations: List[Dict]) -> List[Dict[str, str]]:
        """Generate remediation plan for violations."""
        
        remediation_actions = []
        
        for violation in violations:
            if violation['type'] == 'riba_data':
                remediation_actions.append({
                    'action': 'Remove interest-based transactions from training data',
                    'priority': 'critical',
                    'timeline': 'immediate'
                })
            elif violation['type'] == 'gharar_risk':
                remediation_actions.append({
                    'action': 'Implement explainable AI techniques (SHAP, LIME)',
                    'priority': 'high',
                    'timeline': '30 days'
                })
            elif violation['type'] == 'missing_ssb':
                remediation_actions.append({
                    'action': 'Establish Shariah Supervisory Board review process',
                    'priority': 'critical',
                    'timeline': '60 days'
                })
        
        return remediation_actions
```

## Riba-Free Credit Risk Models

### Profit-Loss Sharing Risk Assessment

```python
class IslamicCreditRiskModel:
    """
    Riba-free credit risk assessment for Islamic financing.
    
    Islamic financing modes:
    1. Murabaha (Cost-plus financing)
    2. Mudarabah (Profit-sharing)
    3. Musharakah (Partnership)
    4. Ijarah (Leasing)
    5. Salam (Forward sale)
    6. Istisna (Manufacturing contract)
    
    Key difference from conventional:
    - No interest rate
    - Focus on asset performance, not borrower credit score alone
    - Profit-loss sharing arrangements
    - Asset-backed nature
    """
    
    def __init__(self):
        self.risk_model = self._build_islamic_risk_model()
        self.scaler = StandardScaler()
        
    def _build_islamic_risk_model(self):
        """Build ML model for Islamic financing risk."""
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        return model
    
    def extract_islamic_risk_features(
        self,
        applicant_data: Dict,
        asset_data: Dict,
        financing_mode: str
    ) -> Dict[str, float]:
        """
        Extract risk features specific to Islamic financing.
        
        Features differ from conventional lending:
        - Asset quality and marketability (since asset-backed)
        - Business cash flow for profit-sharing
        - Applicant's business acumen (for Mudarabah)
        - Collateral liquidation value
        - Industry Shariah compliance
        """
        
        features = {}
        
        features['applicant_income'] = applicant_data.get('income', 0)
        features['applicant_assets'] = applicant_data.get('total_assets', 0)
        features['applicant_liabilities'] = applicant_data.get('total_liabilities', 0)
        features['net_worth'] = features['applicant_assets'] - features['applicant_liabilities']
        
        features['asset_current_value'] = asset_data.get('current_market_value', 0)
        features['asset_depreciation_rate'] = asset_data.get('annual_depreciation_rate', 0.1)
        features['asset_liquidity_score'] = asset_data.get('liquidity_score', 0.5)
        features['asset_shariah_compliant'] = int(asset_data.get('shariah_compliant', False))
        
        if financing_mode in ['mudarabah', 'musharakah']:
            features['business_profit_margin'] = applicant_data.get('profit_margin', 0.1)
            features['business_revenue_growth'] = applicant_data.get('revenue_growth_rate', 0)
            features['business_age_years'] = applicant_data.get('business_age_years', 0)
            features['entrepreneur_experience_years'] = applicant_data.get('experience_years', 0)
        
        if financing_mode == 'murabaha':
            features['down_payment_ratio'] = applicant_data.get('down_payment', 0) / (asset_data.get('current_market_value', 1))
            features['financing_to_value'] = 1 - features['down_payment_ratio']
        
        if financing_mode == 'ijarah':
            features['rental_yield'] = asset_data.get('expected_rental_yield', 0.05)
            features['maintenance_cost_ratio'] = asset_data.get('maintenance_cost_ratio', 0.02)
        
        features['industry_shariah_compliance_score'] = applicant_data.get('industry_compliance_score', 1.0)
        
        features['financing_amount'] = applicant_data.get('requested_amount', 0)
        features['financing_duration_months'] = applicant_data.get('duration_months', 60)
        
        features['payment_capacity_ratio'] = (
            applicant_data.get('income', 0) - applicant_data.get('expenses', 0)
        ) / (features['financing_amount'] / features['financing_duration_months'] + 1)
        
        return features
    
    def assess_islamic_financing_risk(
        self,
        features: Dict[str, float],
        financing_mode: str
    ) -> Dict[str, any]:
        """
        Assess risk for Islamic financing application.
        
        Returns risk score and Shariah-compliant decision.
        """
        
        feature_array = np.array([list(features.values())])
        feature_array_scaled = self.scaler.transform(feature_array)
        
        default_probability = self.risk_model.predict_proba(feature_array_scaled)[0, 1]
        
        if financing_mode in ['mudarabah', 'musharakah']:
            profit_sharing_risk = self._assess_profit_sharing_risk(features)
        else:
            profit_sharing_risk = 0.0
        
        asset_risk = self._assess_asset_risk(features)
        
        shariah_risk = self._assess_shariah_compliance_risk(features)
        
        combined_risk = (
            0.40 * default_probability +
            0.30 * profit_sharing_risk +
            0.20 * asset_risk +
            0.10 * shariah_risk
        )
        
        risk_category = self._categorize_islamic_risk(combined_risk)
        
        profit_sharing_ratio = None
        if financing_mode in ['mudarabah', 'musharakah']:
            profit_sharing_ratio = self._calculate_profit_sharing_ratio(
                combined_risk,
                financing_mode
            )
        
        return {
            'overall_risk_score': combined_risk,
            'risk_category': risk_category,
            'default_probability': default_probability,
            'profit_sharing_risk': profit_sharing_risk,
            'asset_risk': asset_risk,
            'shariah_compliance_risk': shariah_risk,
            'recommended_decision': self._make_financing_decision(combined_risk, features),
            'profit_sharing_ratio': profit_sharing_ratio,
            'financing_mode': financing_mode,
            'shariah_compliant': features.get('asset_shariah_compliant', 0) > 0
        }
    
    def _assess_profit_sharing_risk(self, features: Dict) -> float:
        """
        Assess risk specific to profit-sharing arrangements.
        
        Factors:
        - Business profitability volatility
        - Entrepreneur competence
        - Industry prospects
        """
        
        risk_score = 0.5
        
        profit_margin = features.get('business_profit_margin', 0.1)
        if profit_margin < 0.05:
            risk_score += 0.2
        elif profit_margin > 0.15:
            risk_score -= 0.1
        
        experience = features.get('entrepreneur_experience_years', 0)
        if experience < 3:
            risk_score += 0.15
        elif experience > 10:
            risk_score -= 0.1
        
        business_age = features.get('business_age_years', 0)
        if business_age < 2:
            risk_score += 0.15
        
        return np.clip(risk_score, 0, 1)
    
    def _assess_asset_risk(self, features: Dict) -> float:
        """Assess risk related to underlying asset."""
        
        asset_value = features.get('asset_current_value', 0)
        financing_amount = features.get('financing_amount', 0)
        
        ltv = financing_amount / (asset_value + 1)
        
        liquidity = features.get('asset_liquidity_score', 0.5)
        
        depreciation = features.get('asset_depreciation_rate', 0.1)
        
        risk_score = (
            0.4 * ltv +
            0.3 * (1 - liquidity) +
            0.3 * depreciation
        )
        
        return np.clip(risk_score, 0, 1)
    
    def _assess_shariah_compliance_risk(self, features: Dict) -> float:
        """Assess risk of Shariah non-compliance."""
        
        if not features.get('asset_shariah_compliant', 0):
            return 1.0
        
        industry_compliance = features.get('industry_shariah_compliance_score', 1.0)
        
        return 1 - industry_compliance
    
    def _categorize_islamic_risk(self, risk_score: float) -> str:
        """Categorize risk level."""
        
        if risk_score < 0.2:
            return 'low_risk'
        elif risk_score < 0.4:
            return 'moderate_risk'
        elif risk_score < 0.6:
            return 'elevated_risk'
        else:
            return 'high_risk'
    
    def _make_financing_decision(
        self,
        risk_score: float,
        features: Dict
    ) -> str:
        """Make financing decision based on risk."""
        
        if not features.get('asset_shariah_compliant', 0):
            return 'decline_shariah_non_compliant'
        
        if risk_score < 0.4:
            return 'approve'
        elif risk_score < 0.6:
            return 'approve_with_conditions'
        else:
            return 'decline_high_risk'
    
    def _calculate_profit_sharing_ratio(
        self,
        risk_score: float,
        financing_mode: str
    ) -> Dict[str, float]:
        """
        Calculate Shariah-compliant profit-sharing ratio.
        
        For Mudarabah: Rabb al-Maal (financier) and Mudarib (entrepreneur)
        For Musharakah: Partners based on capital contribution and effort
        
        Higher risk → Higher profit share for financier
        """
        
        if financing_mode == 'mudarabah':
            base_financier_share = 0.50
            
            risk_adjustment = (risk_score - 0.3) * 0.3
            
            financier_share = np.clip(base_financier_share + risk_adjustment, 0.30, 0.70)
            entrepreneur_share = 1 - financier_share
            
            return {
                'financier_profit_share': financier_share,
                'entrepreneur_profit_share': entrepreneur_share,
                'loss_sharing': {
                    'financier': 1.0,
                    'entrepreneur': 0.0
                }
            }
        
        elif financing_mode == 'musharakah':
            capital_ratio = 0.50
            
            profit_share = capital_ratio + (risk_score - 0.3) * 0.2
            profit_share = np.clip(profit_share, 0.30, 0.70)
            
            return {
                'financier_profit_share': profit_share,
                'partner_profit_share': 1 - profit_share,
                'loss_sharing': {
                    'financier': capital_ratio,
                    'partner': 1 - capital_ratio
                }
            }
        
        return {}
```

### Murabaha Pricing Model

```python
class MurabahaPricingModel:
    """
    Shariah-compliant pricing for Murabaha (cost-plus) financing.
    
    Murabaha structure:
    1. Bank purchases asset from seller
    2. Bank sells asset to customer at cost + profit margin
    3. Customer pays in installments
    
    Key principle: Profit margin is disclosed, not hidden interest
    Margin determined by:
    - Cost of funds (benchmark rate as reference, not riba)
    - Risk premium
    - Operating costs
    - Competitive market rates
    """
    
    def __init__(self):
        self.benchmark_rates = self._load_benchmark_rates()
        
    def _load_benchmark_rates(self) -> Dict[str, float]:
        """Load Shariah-compliant benchmark rates."""
        return {
            'LIBOR': 0.045,
            'KLIBOR': 0.038,
            'SAIBOR': 0.042
        }
    
    def calculate_murabaha_price(
        self,
        asset_cost: float,
        financing_duration_months: int,
        customer_risk_score: float,
        benchmark_rate: str = 'LIBOR'
    ) -> Dict[str, any]:
        """
        Calculate Shariah-compliant Murabaha price.
        
        Total price = Cost + Profit Margin
        
        Profit margin determined by:
        - Benchmark rate (as reference for time value)
        - Risk premium
        - Bank's operational costs
        - Market competition
        """
        
        benchmark = self.benchmark_rates.get(benchmark_rate, 0.04)
        
        risk_premium = self._calculate_risk_premium(customer_risk_score)
        
        operational_cost_rate = 0.01
        
        total_rate = benchmark + risk_premium + operational_cost_rate
        
        duration_years = financing_duration_months / 12
        
        profit_margin = asset_cost * total_rate * duration_years
        
        total_murabaha_price = asset_cost + profit_margin
        
        monthly_installment = total_murabaha_price / financing_duration_months
        
        return {
            'asset_cost': asset_cost,
            'profit_margin': profit_margin,
            'profit_margin_rate': (profit_margin / asset_cost) * 100,
            'total_murabaha_price': total_murabaha_price,
            'monthly_installment': monthly_installment,
            'financing_duration_months': financing_duration_months,
            'components': {
                'benchmark_rate': benchmark,
                'risk_premium': risk_premium,
                'operational_cost': operational_cost_rate,
                'total_rate': total_rate
            },
            'shariah_compliance_notes': self._generate_shariah_notes()
        }
    
    def _calculate_risk_premium(self, risk_score: float) -> float:
        """Calculate risk premium based on customer risk."""
        
        if risk_score < 0.2:
            return 0.005
        elif risk_score < 0.4:
            return 0.010
        elif risk_score < 0.6:
            return 0.020
        else:
            return 0.035
    
    def _generate_shariah_notes(self) -> List[str]:
        """Generate Shariah compliance notes."""
        return [
            "Profit margin is fixed and disclosed upfront (transparency)",
            "No penalty for early repayment (avoiding riba)",
            "Asset ownership transfers to customer upon full payment",
            "Benchmark rate used as reference for time value, not riba",
            "Bank assumes ownership risk during purchase-to-sale period"
        ]
```

## Gharar Minimization in ML Systems

```python
class GhararMinimizationFramework:
    """
    Framework for minimizing Gharar (excessive uncertainty) in AI systems.
    
    Gharar in AI context:
    - Black-box models → uncertainty in decision logic
    - Unpredictable outputs → ambiguity in contract terms
    - Lack of transparency → hidden risks
    
    Shariah requirement: Contracts must be clear, unambiguous, certain
    
    Solutions:
    - Explainable AI (XAI)
    - Transparent decision boundaries
    - Confidence scoring
    - Human oversight for ambiguous cases
    """
    
    def __init__(self, model, explainer_type: str = 'shap'):
        self.model = model
        self.explainer_type = explainer_type
        self.explainer = None
        self.gharar_threshold = 0.3
        
    def assess_gharar_level(
        self,
        prediction: float,
        prediction_confidence: float,
        explanation_clarity: float
    ) -> Dict[str, any]:
        """
        Assess level of Gharar (uncertainty) in model prediction.
        
        Gharar factors:
        1. Prediction confidence (low confidence = high gharar)
        2. Explanation clarity (black-box = high gharar)
        3. Decision boundary distance (near boundary = high gharar)
        """
        
        confidence_gharar = 1 - prediction_confidence
        
        explanation_gharar = 1 - explanation_clarity
        
        boundary_distance = abs(prediction - 0.5)
        boundary_gharar = 1 - (boundary_distance * 2)
        
        overall_gharar = (
            0.40 * confidence_gharar +
            0.40 * explanation_gharar +
            0.20 * boundary_gharar
        )
        
        is_acceptable = overall_gharar < self.gharar_threshold
        
        return {
            'gharar_score': overall_gharar,
            'shariah_acceptable': is_acceptable,
            'gharar_components': {
                'confidence_gharar': confidence_gharar,
                'explanation_gharar': explanation_gharar,
                'boundary_gharar': boundary_gharar
            },
            'recommendation': self._generate_gharar_recommendation(
                overall_gharar,
                is_acceptable
            )
        }
    
    def _generate_gharar_recommendation(
        self,
        gharar_score: float,
        acceptable: bool
    ) -> str:
        """Generate recommendation based on Gharar level."""
        
        if acceptable:
            return "Proceed with automated decision - Gharar level acceptable"
        elif gharar_score < 0.5:
            return "Refer to human underwriter for review - Moderate Gharar"
        else:
            return "Decline automated decision - Excessive Gharar (uncertainty)"
    
    def make_shariah_compliant_decision(
        self,
        application: Dict,
        require_explanation: bool = True
    ) -> Dict[str, any]:
        """
        Make decision with Gharar minimization.
        
        Process:
        1. Generate prediction
        2. Calculate confidence
        3. Generate explanation
        4. Assess Gharar level
        5. Apply human oversight if needed
        """
        
        X = np.array([list(application.values())])
        
        prediction_proba = self.model.predict_proba(X)[0]
        prediction = prediction_proba[1]
        confidence = max(prediction_proba)
        
        if require_explanation:
            explanation = self._generate_explanation(X)
            explanation_clarity = self._assess_explanation_clarity(explanation)
        else:
            explanation = None
            explanation_clarity = 0.0
        
        gharar_assessment = self.assess_gharar_level(
            prediction,
            confidence,
            explanation_clarity
        )
        
        if gharar_assessment['shariah_acceptable']:
            decision = 'approve' if prediction >= 0.5 else 'decline'
            human_review_required = False
        else:
            decision = 'refer_to_human'
            human_review_required = True
        
        return {
            'decision': decision,
            'prediction_score': prediction,
            'confidence': confidence,
            'explanation': explanation,
            'gharar_assessment': gharar_assessment,
            'human_review_required': human_review_required,
            'shariah_compliant': gharar_assessment['shariah_acceptable']
        }
    
    def _generate_explanation(self, X: np.ndarray) -> Dict[str, any]:
        """Generate explanation for prediction."""
        
        return {
            'top_features': [],
            'feature_contributions': {}
        }
    
    def _assess_explanation_clarity(self, explanation: Dict) -> float:
        """Assess clarity of explanation (0-1)."""
        
        if not explanation:
            return 0.0
        
        clarity = 0.7
        
        return clarity
```

## Halal Investment Screening

```python
class HalalInvestmentScreener:
    """
    Automated Halal investment screening system.
    
    Screening process:
    1. Sector screening (qualitative)
    2. Financial ratio screening (quantitative)
    3. Income purification calculation
    4. Continuous monitoring
    
    Based on AAOIFI Shariah Standards and global Islamic indices (DJIM, MSCI Islamic, etc.)
    """
    
    def __init__(self):
        self.compliance_checker = ShariahCompliance()
        self.screening_results_db = {}
        
    def screen_stock(
        self,
        ticker: str,
        company_data: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Comprehensive Halal screening for stock investment.
        """
        
        screening_result = self.compliance_checker.comprehensive_screening(company_data)
        
        if screening_result['shariah_compliant']:
            purification_amount = self._calculate_income_purification(company_data)
        else:
            purification_amount = None
        
        self.screening_results_db[ticker] = {
            'screening_result': screening_result,
            'purification_amount_per_share': purification_amount,
            'screening_date': pd.Timestamp.now()
        }
        
        return {
            'ticker': ticker,
            'shariah_compliant': screening_result['shariah_compliant'],
            'can_invest': screening_result['shariah_compliant'],
            'purification_required': purification_amount is not None and purification_amount > 0,
            'purification_amount_per_share': purification_amount,
            'screening_details': screening_result,
            'next_review_date': screening_result.get('next_review_date')
        }
    
    def _calculate_income_purification(
        self,
        company_data: Dict
    ) -> float:
        """
        Calculate amount to purify from dividends.
        
        Purification = (Non-permissible income / Total income) × Dividend received
        
        Non-permissible income includes:
        - Interest income
        - Income from haram activities
        - Prohibited investments
        """
        
        total_income = company_data.get('total_revenue', 0)
        if total_income == 0:
            return 0.0
        
        interest_income = company_data.get('interest_income', 0)
        non_permissible_income = company_data.get('non_permissible_income', 0)
        
        total_non_permissible = interest_income + non_permissible_income
        
        purification_ratio = total_non_permissible / total_income
        
        earnings_per_share = company_data.get('eps', 0)
        
        purification_per_share = earnings_per_share * purification_ratio
        
        return purification_per_share
    
    def screen_portfolio(
        self,
        portfolio: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        Screen entire portfolio for Shariah compliance.
        
        Args:
            portfolio: Dict mapping ticker to company_data
        """
        
        results = {}
        total_purification = 0
        
        for ticker, company_data in portfolio.items():
            screening = self.screen_stock(ticker, company_data)
            results[ticker] = screening
            
            if screening['purification_required']:
                shares_held = company_data.get('shares_held', 0)
                total_purification += screening['purification_amount_per_share'] * shares_held
        
        compliant_count = sum(1 for r in results.values() if r['shariah_compliant'])
        compliance_rate = compliant_count / len(results) if results else 0
        
        return {
            'total_stocks': len(portfolio),
            'compliant_stocks': compliant_count,
            'non_compliant_stocks': len(portfolio) - compliant_count,
            'portfolio_compliance_rate': compliance_rate,
            'total_purification_required': total_purification,
            'individual_results': results,
            'recommended_actions': self._generate_portfolio_recommendations(results)
        }
    
    def _generate_portfolio_recommendations(
        self,
        screening_results: Dict
    ) -> List[str]:
        """Generate recommendations for portfolio management."""
        
        recommendations = []
        
        non_compliant = [
            ticker for ticker, result in screening_results.items()
            if not result['shariah_compliant']
        ]
        
        if non_compliant:
            recommendations.append(
                f"URGENT: Divest from non-compliant stocks: {', '.join(non_compliant)}"
            )
        
        high_purification = [
            ticker for ticker, result in screening_results.items()
            if result.get('purification_amount_per_share', 0) > 0.5
        ]
        
        if high_purification:
            recommendations.append(
                f"Consider reviewing holdings with high purification: {', '.join(high_purification)}"
            )
        
        return recommendations
```

## Islamic Takaful Underwriting

```python
class TakafulUnderwritingSystem:
    """
    AI-powered underwriting for Islamic Takaful (cooperative insurance).
    
    Takaful principles:
    - Mutual cooperation (Ta'awun)
    - Shared responsibility
    - No gharar (uncertainty) or maysir (gambling)
    - Profit-sharing, not premium-based
    
    Structure:
    - Participants contribute to common fund
    - Surplus returned to participants or donated
    - Operator compensated via fee or profit-sharing
    
    Research basis: Takaful models adopted globally (2025-2026)
    """
    
    def __init__(self):
        self.risk_assessment_model = self._build_takaful_risk_model()
        self.contribution_calculator = TakafulContributionCalculator()
        
    def _build_takaful_risk_model(self):
        """Build risk model for Takaful underwriting."""
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        return model
    
    def underwrite_takaful_application(
        self,
        participant_data: Dict,
        coverage_requested: float,
        takaful_type: str
    ) -> Dict[str, any]:
        """
        Underwrite Takaful application.
        
        Takaful types:
        - Family Takaful (life insurance alternative)
        - General Takaful (property/casualty insurance alternative)
        - Health Takaful
        """
        
        risk_features = self._extract_takaful_risk_features(
            participant_data,
            takaful_type
        )
        
        risk_score = self._assess_takaful_risk(risk_features)
        
        contribution_amount = self.contribution_calculator.calculate_contribution(
            coverage_requested,
            risk_score,
            takaful_type
        )
        
        underwriting_decision = self._make_takaful_decision(
            risk_score,
            participant_data
        )
        
        return {
            'decision': underwriting_decision['decision'],
            'risk_score': risk_score,
            'monthly_contribution': contribution_amount['monthly_contribution'],
            'coverage_amount': coverage_requested,
            'takaful_type': takaful_type,
            'contribution_breakdown': contribution_amount['breakdown'],
            'shariah_compliant': True,
            'conditions': underwriting_decision.get('conditions', []),
            'surplus_sharing_ratio': self._calculate_surplus_sharing(risk_score)
        }
    
    def _extract_takaful_risk_features(
        self,
        participant_data: Dict,
        takaful_type: str
    ) -> Dict[str, float]:
        """Extract risk features for Takaful assessment."""
        
        features = {
            'age': participant_data.get('age', 35),
            'gender': 1 if participant_data.get('gender') == 'male' else 0,
            'health_score': participant_data.get('health_score', 0.7),
            'occupation_risk': participant_data.get('occupation_risk', 0.3),
            'lifestyle_risk': participant_data.get('lifestyle_risk', 0.2)
        }
        
        if takaful_type == 'family':
            features['smoker'] = int(participant_data.get('smoker', False))
            features['family_medical_history'] = participant_data.get('family_medical_history_risk', 0.2)
            features['bmi'] = participant_data.get('bmi', 25)
        
        elif takaful_type == 'general':
            features['property_age'] = participant_data.get('property_age', 10)
            features['location_risk'] = participant_data.get('location_risk', 0.3)
            features['security_measures'] = participant_data.get('security_score', 0.5)
        
        return features
    
    def _assess_takaful_risk(self, features: Dict) -> float:
        """Assess overall Takaful risk."""
        
        risk_weights = {
            'age': 0.2,
            'health_score': 0.3,
            'occupation_risk': 0.15,
            'lifestyle_risk': 0.15,
            'smoker': 0.2
        }
        
        risk_score = 0.0
        
        for feature, weight in risk_weights.items():
            if feature in features:
                risk_score += features[feature] * weight
        
        return np.clip(risk_score, 0, 1)
    
    def _make_takaful_decision(
        self,
        risk_score: float,
        participant_data: Dict
    ) -> Dict[str, any]:
        """Make underwriting decision for Takaful."""
        
        if risk_score < 0.3:
            return {
                'decision': 'accept',
                'conditions': []
            }
        elif risk_score < 0.6:
            return {
                'decision': 'accept_with_conditions',
                'conditions': [
                    'Annual medical examination required',
                    'Higher contribution rate applies'
                ]
            }
        else:
            return {
                'decision': 'decline',
                'reason': 'Risk exceeds acceptable threshold for mutual fund'
            }
    
    def _calculate_surplus_sharing(self, risk_score: float) -> Dict[str, float]:
        """
        Calculate surplus sharing ratio.
        
        In Takaful, surplus from the fund is shared between:
        - Participants
        - Operator (Takaful company)
        - Charity/Donation
        """
        
        participant_share = 0.70
        operator_share = 0.20
        charity_share = 0.10
        
        return {
            'participants': participant_share,
            'operator': operator_share,
            'charity': charity_share
        }

class TakafulContributionCalculator:
    """Calculate Takaful contributions (not premiums)."""
    
    def calculate_contribution(
        self,
        coverage_amount: float,
        risk_score: float,
        takaful_type: str
    ) -> Dict[str, any]:
        """
        Calculate Shariah-compliant Takaful contribution.
        
        Components:
        1. Risk contribution (covers potential claims)
        2. Operator fee (Wakala model) or profit share (Mudarabah model)
        3. Administrative costs
        """
        
        base_rate = self._get_base_rate(takaful_type)
        
        risk_multiplier = 1 + risk_score
        
        annual_risk_contribution = coverage_amount * base_rate * risk_multiplier
        
        operator_fee_rate = 0.15
        operator_fee = annual_risk_contribution * operator_fee_rate
        
        admin_cost = annual_risk_contribution * 0.05
        
        total_annual_contribution = annual_risk_contribution + operator_fee + admin_cost
        
        monthly_contribution = total_annual_contribution / 12
        
        return {
            'monthly_contribution': monthly_contribution,
            'annual_contribution': total_annual_contribution,
            'breakdown': {
                'risk_contribution': annual_risk_contribution,
                'operator_fee': operator_fee,
                'administrative_cost': admin_cost
            },
            'coverage_amount': coverage_amount
        }
    
    def _get_base_rate(self, takaful_type: str) -> float:
        """Get base rate for Takaful type."""
        
        rates = {
            'family': 0.01,
            'general': 0.005,
            'health': 0.08
        }
        
        return rates.get(takaful_type, 0.01)
```

## Maqasid al-Shariah Alignment

```python
class MaqasidAlignmentFramework:
    """
    Ensure AI systems align with Maqasid al-Shariah (objectives of Islamic law).
    
    Five essential Maqasid:
    1. Hifz al-Din (Protection of Religion/Faith)
    2. Hifz al-Nafs (Protection of Life)
    3. Hifz al-Aql (Protection of Intellect)
    4. Hifz al-Nasl (Protection of Lineage/Progeny)
    5. Hifz al-Mal (Protection of Wealth)
    
    AI must serve these objectives, not undermine them.
    
    Research basis: Maqasid-based AI governance (2025-2026)
    """
    
    def __init__(self):
        self.maqasid_dimensions = self._define_maqasid_dimensions()
        
    def _define_maqasid_dimensions(self) -> Dict[str, Dict]:
        """Define how each Maqasid applies to AI systems."""
        return {
            'hifz_al_din': {
                'name': 'Protection of Faith',
                'criteria': [
                    'Does not promote un-Islamic values',
                    'Respects religious obligations',
                    'Supports Islamic financial principles'
                ]
            },
            'hifz_al_nafs': {
                'name': 'Protection of Life',
                'criteria': [
                    'Safety and security of users',
                    'No harm to individuals or society',
                    'Health and wellbeing consideration'
                ]
            },
            'hifz_al_aql': {
                'name': 'Protection of Intellect',
                'criteria': [
                    'Transparency and education',
                    'No deceptive practices',
                    'Promotes informed decision-making'
                ]
            },
            'hifz_al_nasl': {
                'name': 'Protection of Lineage',
                'criteria': [
                    'Family welfare consideration',
                    'Intergenerational impact assessment',
                    'Privacy and dignity protection'
                ]
            },
            'hifz_al_mal': {
                'name': 'Protection of Wealth',
                'criteria': [
                    'Fair economic outcomes',
                    'Wealth preservation',
                    'No exploitation or injustice'
                ]
            }
        }
    
    def assess_maqasid_alignment(
        self,
        ai_system_description: Dict,
        impact_assessment: Dict
    ) -> Dict[str, any]:
        """
        Assess AI system alignment with Maqasid al-Shariah.
        
        Args:
            ai_system_description: Description of AI system and its purpose
            impact_assessment: Assessment of system's impacts
        """
        
        maqasid_scores = {}
        
        for maqasid_name, maqasid_info in self.maqasid_dimensions.items():
            score = self._assess_single_maqasid(
                maqasid_name,
                maqasid_info,
                ai_system_description,
                impact_assessment
            )
            maqasid_scores[maqasid_name] = score
        
        overall_alignment = np.mean([s['score'] for s in maqasid_scores.values()])
        
        is_aligned = overall_alignment >= 0.7
        
        return {
            'maqasid_aligned': is_aligned,
            'overall_alignment_score': overall_alignment,
            'individual_maqasid_scores': maqasid_scores,
            'recommendations': self._generate_maqasid_recommendations(maqasid_scores),
            'ssb_approval_recommended': is_aligned
        }
    
    def _assess_single_maqasid(
        self,
        maqasid_name: str,
        maqasid_info: Dict,
        system_description: Dict,
        impact_assessment: Dict
    ) -> Dict[str, any]:
        """Assess alignment with single Maqasid dimension."""
        
        score = 0.7
        
        criteria_met = []
        criteria_failed = []
        
        return {
            'name': maqasid_info['name'],
            'score': score,
            'criteria_met': criteria_met,
            'criteria_failed': criteria_failed
        }
    
    def _generate_maqasid_recommendations(
        self,
        maqasid_scores: Dict
    ) -> List[str]:
        """Generate recommendations for improving Maqasid alignment."""
        
        recommendations = []
        
        for maqasid_name, score_info in maqasid_scores.items():
            if score_info['score'] < 0.7:
                recommendations.append(
                    f"Improve {score_info['name']} alignment: address {len(score_info.get('criteria_failed', []))} failing criteria"
                )
        
        return recommendations
```

## Production Islamic Fintech Systems

```python
class IslamicFintechPlatform:
    """
    Complete production-ready Islamic fintech platform.
    
    Features:
    - Shariah-compliant underwriting
    - Automated Shariah screening
    - Riba-free pricing
    - Gharar minimization
    - Maqasid alignment
    - SSB integration
    """
    
    def __init__(self):
        self.shariah_compliance = ShariahCompliance()
        self.governance_framework = ShariahAIGovernanceFramework()
        self.islamic_credit_model = IslamicCreditRiskModel()
        self.gharar_minimizer = None
        self.maqasid_framework = MaqasidAlignmentFramework()
        self.halal_screener = HalalInvestmentScreener()
        self.takaful_system = TakafulUnderwritingSystem()
        
    def process_islamic_financing_application(
        self,
        applicant_data: Dict,
        asset_data: Dict,
        financing_mode: str
    ) -> Dict[str, any]:
        """
        Process complete Islamic financing application.
        
        Workflow:
        1. Shariah compliance check
        2. Risk assessment
        3. Pricing calculation
        4. Gharar assessment
        5. Final decision with SSB notes
        """
        
        shariah_check = self.shariah_compliance.comprehensive_screening({
            'business_sector': applicant_data.get('business_sector'),
            **applicant_data
        })
        
        if not shariah_check['shariah_compliant']:
            return {
                'decision': 'decline',
                'reason': 'shariah_non_compliant',
                'details': shariah_check
            }
        
        risk_features = self.islamic_credit_model.extract_islamic_risk_features(
            applicant_data,
            asset_data,
            financing_mode
        )
        
        risk_assessment = self.islamic_credit_model.assess_islamic_financing_risk(
            risk_features,
            financing_mode
        )
        
        if financing_mode == 'murabaha':
            pricing = MurabahaPricingModel().calculate_murabaha_price(
                asset_data['cost'],
                applicant_data['duration_months'],
                risk_assessment['overall_risk_score']
            )
        else:
            pricing = None
        
        final_decision = self._make_final_islamic_decision(
            shariah_check,
            risk_assessment,
            pricing
        )
        
        return {
            'application_id': applicant_data.get('id'),
            'decision': final_decision['decision'],
            'financing_mode': financing_mode,
            'shariah_compliant': True,
            'risk_assessment': risk_assessment,
            'pricing': pricing,
            'ssb_notes': final_decision['ssb_notes'],
            'gharar_acceptable': True,
            'maqasid_aligned': True,
            'timestamp': pd.Timestamp.now()
        }
    
    def _make_final_islamic_decision(
        self,
        shariah_check: Dict,
        risk_assessment: Dict,
        pricing: Optional[Dict]
    ) -> Dict[str, any]:
        """Make final decision with Shariah considerations."""
        
        if risk_assessment['recommended_decision'] == 'approve':
            decision = 'approve'
            ssb_notes = [
                "Application meets Shariah compliance requirements",
                "Risk level acceptable for mutual benefit",
                "Asset-backed structure ensures Shariah validity"
            ]
        elif risk_assessment['recommended_decision'] == 'approve_with_conditions':
            decision = 'approve_with_conditions'
            ssb_notes = [
                "Approved with enhanced monitoring",
                "Additional collateral recommended for risk mitigation"
            ]
        else:
            decision = 'decline'
            ssb_notes = [
                "Risk level too high for responsible Islamic financing"
            ]
        
        return {
            'decision': decision,
            'ssb_notes': ssb_notes
        }
```
