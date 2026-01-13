# Module 15: Agentic Investment Research Systems

## Table of Contents
1. [Automated Company Analysis](#automated-company-analysis)
2. [Multi-Source Information Synthesis](#multi-source-information-synthesis)
3. [Investment Thesis Generation](#investment-thesis-generation)
4. [Competitive Intelligence](#competitive-intelligence)
5. [Due Diligence Automation](#due-diligence-automation)
6. [Research Report Generation](#research-report-generation)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## Automated Company Analysis

### Financial Statement Analysis Agent

```python
from typing import Dict, List
import pandas as pd
import numpy as np
from openai import OpenAI

class FinancialStatementAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def analyze_income_statement(
        self,
        income_statement: pd.DataFrame
    ) -> Dict[str, any]:
        revenue_growth = income_statement['revenue'].pct_change().mean()
        
        margin_analysis = {
            'gross_margin': (income_statement['gross_profit'] / income_statement['revenue']).mean(),
            'operating_margin': (income_statement['operating_income'] / income_statement['revenue']).mean(),
            'net_margin': (income_statement['net_income'] / income_statement['revenue']).mean()
        }
        
        margin_trends = {
            'gross_margin_trend': (income_statement['gross_profit'] / income_statement['revenue']).pct_change().mean(),
            'operating_margin_trend': (income_statement['operating_income'] / income_statement['revenue']).pct_change().mean()
        }
        
        return {
            'revenue_growth': revenue_growth,
            'margin_analysis': margin_analysis,
            'margin_trends': margin_trends
        }
    
    def analyze_balance_sheet(
        self,
        balance_sheet: pd.DataFrame
    ) -> Dict[str, any]:
        liquidity_ratios = {
            'current_ratio': (balance_sheet['current_assets'] / balance_sheet['current_liabilities']).mean(),
            'quick_ratio': ((balance_sheet['current_assets'] - balance_sheet['inventory']) /
                           balance_sheet['current_liabilities']).mean()
        }
        
        leverage_ratios = {
            'debt_to_equity': (balance_sheet['total_debt'] / balance_sheet['shareholders_equity']).mean(),
            'debt_to_assets': (balance_sheet['total_debt'] / balance_sheet['total_assets']).mean()
        }
        
        efficiency_ratios = {
            'asset_turnover': balance_sheet['revenue'] / balance_sheet['total_assets']
        }
        
        return {
            'liquidity': liquidity_ratios,
            'leverage': leverage_ratios,
            'efficiency': efficiency_ratios
        }
    
    def generate_narrative_analysis(
        self,
        financial_metrics: Dict[str, any]
    ) -> str:
        prompt = f"""
        Based on the following financial metrics, provide a comprehensive analysis:
        
        {financial_metrics}
        
        Please analyze:
        1. Financial health and stability
        2. Growth trends
        3. Profitability analysis
        4. Areas of concern
        5. Strengths and opportunities
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content


class ValuationAgent:
    def __init__(self):
        pass
    
    def dcf_valuation(
        self,
        free_cash_flows: List[float],
        terminal_growth_rate: float,
        discount_rate: float
    ) -> float:
        present_values = []
        
        for i, fcf in enumerate(free_cash_flows):
            pv = fcf / ((1 + discount_rate) ** (i + 1))
            present_values.append(pv)
        
        terminal_value = (
            free_cash_flows[-1] * (1 + terminal_growth_rate) /
            (discount_rate - terminal_growth_rate)
        )
        
        terminal_pv = terminal_value / ((1 + discount_rate) ** len(free_cash_flows))
        
        enterprise_value = sum(present_values) + terminal_pv
        
        return enterprise_value
    
    def comparable_company_analysis(
        self,
        target_metrics: Dict[str, float],
        peer_metrics: pd.DataFrame
    ) -> Dict[str, float]:
        multiples = {
            'ev_to_sales': peer_metrics['ev'] / peer_metrics['sales'],
            'ev_to_ebitda': peer_metrics['ev'] / peer_metrics['ebitda'],
            'price_to_earnings': peer_metrics['market_cap'] / peer_metrics['net_income']
        }
        
        median_multiples = {k: v.median() for k, v in multiples.items()}
        
        implied_valuations = {
            'by_sales': target_metrics['sales'] * median_multiples['ev_to_sales'],
            'by_ebitda': target_metrics['ebitda'] * median_multiples['ev_to_ebitda'],
            'by_earnings': target_metrics['net_income'] * median_multiples['price_to_earnings']
        }
        
        return implied_valuations
```

## Multi-Source Information Synthesis

### Knowledge Graph Builder

```python
import networkx as nx
from collections import defaultdict

class FinancialKnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_attributes = defaultdict(dict)
        
    def add_company(
        self,
        ticker: str,
        name: str,
        sector: str,
        industry: str
    ):
        self.graph.add_node(
            ticker,
            type='company',
            name=name,
            sector=sector,
            industry=industry
        )
        
    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        attributes: Dict = None
    ):
        self.graph.add_edge(
            source,
            target,
            type=relationship_type,
            **(attributes or {})
        )
        
    def add_financial_metric(
        self,
        ticker: str,
        metric_name: str,
        value: float,
        date: str
    ):
        if ticker not in self.entity_attributes:
            self.entity_attributes[ticker] = {}
        
        if metric_name not in self.entity_attributes[ticker]:
            self.entity_attributes[ticker][metric_name] = []
        
        self.entity_attributes[ticker][metric_name].append({
            'value': value,
            'date': date
        })
        
    def find_related_companies(
        self,
        ticker: str,
        relationship_types: List[str] = None,
        max_depth: int = 2
    ) -> List[str]:
        if ticker not in self.graph:
            return []
        
        related = set()
        
        for node in nx.single_source_shortest_path_length(
            self.graph,
            ticker,
            cutoff=max_depth
        ).keys():
            if node != ticker:
                related.add(node)
        
        return list(related)
    
    def get_company_context(
        self,
        ticker: str
    ) -> Dict[str, any]:
        if ticker not in self.graph:
            return {}
        
        node_data = self.graph.nodes[ticker]
        
        relationships = {
            'suppliers': [],
            'customers': [],
            'competitors': [],
            'partners': []
        }
        
        for _, target, edge_data in self.graph.out_edges(ticker, data=True):
            rel_type = edge_data.get('type')
            if rel_type in relationships:
                relationships[rel_type].append(target)
        
        metrics = self.entity_attributes.get(ticker, {})
        
        return {
            'company_info': node_data,
            'relationships': relationships,
            'financial_metrics': metrics
        }


class MultiSourceSynthesizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def synthesize_information(
        self,
        news_articles: List[str],
        sec_filings: List[str],
        earnings_transcripts: List[str],
        analyst_reports: List[str]
    ) -> Dict[str, str]:
        all_sources = {
            'news': news_articles,
            'sec_filings': sec_filings,
            'earnings': earnings_transcripts,
            'analyst_reports': analyst_reports
        }
        
        synthesis = {}
        
        for source_type, documents in all_sources.items():
            if documents:
                combined_text = "\n\n".join(documents[:5])
                
                prompt = f"""
                Synthesize the following {source_type} into key insights:
                
                {combined_text}
                
                Provide:
                1. Main themes
                2. Key facts and figures
                3. Important developments
                4. Sentiment and tone
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a research analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                synthesis[source_type] = response.choices[0].message.content
        
        return synthesis
```

## Investment Thesis Generation

### Thesis Generation Agent

```python
class InvestmentThesisGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_bull_case(
        self,
        company_data: Dict[str, any],
        market_analysis: Dict[str, any]
    ) -> str:
        prompt = f"""
        Generate a bullish investment thesis for the following company:
        
        Company Data: {company_data}
        Market Analysis: {market_analysis}
        
        Focus on:
        1. Growth catalysts
        2. Competitive advantages
        3. Market opportunities
        4. Financial strengths
        5. Valuation upside
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an investment analyst focused on identifying opportunities."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def generate_bear_case(
        self,
        company_data: Dict[str, any],
        risk_factors: List[str]
    ) -> str:
        prompt = f"""
        Generate a bearish investment thesis for the following company:
        
        Company Data: {company_data}
        Risk Factors: {risk_factors}
        
        Focus on:
        1. Key risks and challenges
        2. Competitive threats
        3. Market headwinds
        4. Financial weaknesses
        5. Valuation concerns
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a critical investment analyst focused on risk assessment."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def identify_catalysts(
        self,
        company_info: Dict[str, any],
        industry_trends: List[str]
    ) -> List[Dict[str, str]]:
        prompt = f"""
        Identify potential catalysts for the following company:
        
        Company Info: {company_info}
        Industry Trends: {industry_trends}
        
        List catalysts with:
        - Description
        - Timeline
        - Probability
        - Potential impact
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an investment analyst identifying market catalysts."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return self._parse_catalysts(response.choices[0].message.content)
    
    def _parse_catalysts(self, text: str) -> List[Dict[str, str]]:
        catalysts = []
        
        lines = text.split('\n')
        current_catalyst = {}
        
        for line in lines:
            if line.strip() and ':' in line:
                key, value = line.split(':', 1)
                current_catalyst[key.strip().lower()] = value.strip()
                
                if len(current_catalyst) == 4:
                    catalysts.append(current_catalyst)
                    current_catalyst = {}
        
        return catalysts
```

## Competitive Intelligence

### Competitive Analysis Agent

```python
class CompetitiveIntelligenceAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def analyze_market_share(
        self,
        company_revenue: float,
        competitor_revenues: Dict[str, float]
    ) -> Dict[str, float]:
        total_market = company_revenue + sum(competitor_revenues.values())
        
        market_shares = {
            'company': company_revenue / total_market * 100
        }
        
        for competitor, revenue in competitor_revenues.items():
            market_shares[competitor] = revenue / total_market * 100
        
        return market_shares
    
    def compare_financials(
        self,
        company_metrics: Dict[str, float],
        peer_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        comparison = pd.DataFrame({
            'Company': company_metrics,
            'Peer_Median': peer_metrics.median(),
            'Peer_75th_Percentile': peer_metrics.quantile(0.75),
            'Peer_25th_Percentile': peer_metrics.quantile(0.25)
        })
        
        comparison['Vs_Median'] = (
            (comparison['Company'] - comparison['Peer_Median']) /
            comparison['Peer_Median'] * 100
        )
        
        return comparison
    
    def analyze_competitive_positioning(
        self,
        company_name: str,
        strengths: List[str],
        weaknesses: List[str],
        competitor_info: Dict[str, Dict]
    ) -> str:
        prompt = f"""
        Analyze the competitive positioning of {company_name}:
        
        Strengths: {strengths}
        Weaknesses: {weaknesses}
        Competitors: {competitor_info}
        
        Provide:
        1. Competitive advantages
        2. Competitive disadvantages
        3. Market positioning
        4. Strategic recommendations
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a competitive strategy analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
```

## Due Diligence Automation

### Red Flag Detection System

```python
class RedFlagDetector:
    def __init__(self):
        self.red_flags = []
        
    def check_accounting_quality(
        self,
        financial_statements: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        flags = []
        
        income_stmt = financial_statements['income_statement']
        balance_sheet = financial_statements['balance_sheet']
        cash_flow = financial_statements['cash_flow']
        
        net_income = income_stmt['net_income'].iloc[-1]
        operating_cash_flow = cash_flow['operating_cash_flow'].iloc[-1]
        
        if operating_cash_flow < net_income * 0.7:
            flags.append({
                'type': 'Cash Flow Quality',
                'severity': 'High',
                'description': 'Operating cash flow significantly lower than net income',
                'details': f'OCF: {operating_cash_flow}, NI: {net_income}'
            })
        
        revenue_growth = income_stmt['revenue'].pct_change().iloc[-1]
        ar_growth = balance_sheet['accounts_receivable'].pct_change().iloc[-1]
        
        if ar_growth > revenue_growth * 1.5:
            flags.append({
                'type': 'Revenue Quality',
                'severity': 'Medium',
                'description': 'Accounts receivable growing faster than revenue',
                'details': f'AR Growth: {ar_growth:.2%}, Revenue Growth: {revenue_growth:.2%}'
            })
        
        current_assets = balance_sheet['current_assets'].iloc[-1]
        current_liabilities = balance_sheet['current_liabilities'].iloc[-1]
        
        if current_assets / current_liabilities < 1.0:
            flags.append({
                'type': 'Liquidity Risk',
                'severity': 'High',
                'description': 'Current ratio below 1.0',
                'details': f'Current Ratio: {current_assets/current_liabilities:.2f}'
            })
        
        return flags
    
    def check_governance_issues(
        self,
        proxy_statement: Dict[str, any],
        board_info: Dict[str, any]
    ) -> List[Dict[str, str]]:
        flags = []
        
        if proxy_statement.get('ceo_compensation_ratio', 0) > 300:
            flags.append({
                'type': 'Governance',
                'severity': 'Medium',
                'description': 'High CEO to median employee pay ratio',
                'details': f"Ratio: {proxy_statement['ceo_compensation_ratio']}"
            })
        
        independent_directors = board_info.get('independent_directors', 0)
        total_directors = board_info.get('total_directors', 1)
        
        if independent_directors / total_directors < 0.5:
            flags.append({
                'type': 'Governance',
                'severity': 'High',
                'description': 'Insufficient board independence',
                'details': f'{independent_directors}/{total_directors} independent directors'
            })
        
        return flags
```

## Research Report Generation

### Automated Report Writer

```python
class InvestmentReportGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_executive_summary(
        self,
        company_analysis: Dict[str, any],
        valuation: Dict[str, float],
        recommendation: str
    ) -> str:
        prompt = f"""
        Generate an executive summary for an investment report:
        
        Company Analysis: {company_analysis}
        Valuation: {valuation}
        Recommendation: {recommendation}
        
        Include:
        - Investment recommendation
        - Target price
        - Key investment highlights
        - Main risks
        - Conclusion
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior equity research analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def generate_full_report(
        self,
        ticker: str,
        all_analysis: Dict[str, any]
    ) -> str:
        sections = []
        
        sections.append(f"# Investment Research Report: {ticker}\n\n")
        
        sections.append("## Executive Summary\n")
        sections.append(self.generate_executive_summary(
            all_analysis['company_analysis'],
            all_analysis['valuation'],
            all_analysis['recommendation']
        ))
        sections.append("\n\n")
        
        sections.append("## Company Overview\n")
        sections.append(str(all_analysis.get('company_overview', '')))
        sections.append("\n\n")
        
        sections.append("## Financial Analysis\n")
        sections.append(str(all_analysis.get('financial_analysis', '')))
        sections.append("\n\n")
        
        sections.append("## Valuation\n")
        sections.append(str(all_analysis.get('valuation_details', '')))
        sections.append("\n\n")
        
        sections.append("## Investment Thesis\n")
        sections.append("### Bull Case\n")
        sections.append(all_analysis.get('bull_case', ''))
        sections.append("\n\n### Bear Case\n")
        sections.append(all_analysis.get('bear_case', ''))
        sections.append("\n\n")
        
        sections.append("## Risks\n")
        sections.append(str(all_analysis.get('risk_analysis', '')))
        sections.append("\n\n")
        
        sections.append("## Recommendation\n")
        sections.append(all_analysis.get('final_recommendation', ''))
        
        return "".join(sections)
```

## PhD-Level Research Topics

### Multi-Agent Research System

```python
from typing import List, Dict, Any

class ResearchAgent:
    def __init__(self, name: str, specialty: str, api_key: str):
        self.name = name
        self.specialty = specialty
        self.client = OpenAI(api_key=api_key)
        self.memory = []
        
    def analyze(self, data: Dict[str, Any]) -> str:
        prompt = f"""
        As a {self.specialty} specialist, analyze the following:
        
        {data}
        
        Provide your expert analysis focusing on your area of expertise.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a {self.specialty} expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis = response.choices[0].message.content
        self.memory.append(analysis)
        
        return analysis


class MultiAgentResearchSystem:
    def __init__(self, api_key: str):
        self.agents = {
            'fundamental': ResearchAgent('Fundamental Analyst', 'fundamental analysis', api_key),
            'technical': ResearchAgent('Technical Analyst', 'technical analysis', api_key),
            'quant': ResearchAgent('Quantitative Analyst', 'quantitative modeling', api_key),
            'macro': ResearchAgent('Macro Analyst', 'macroeconomic analysis', api_key)
        }
        
        self.coordinator = OpenAI(api_key=api_key)
        
    def conduct_research(
        self,
        ticker: str,
        data: Dict[str, Any]
    ) -> Dict[str, str]:
        analyses = {}
        
        for agent_type, agent in self.agents.items():
            relevant_data = self._filter_data_for_agent(data, agent_type)
            analysis = agent.analyze(relevant_data)
            analyses[agent_type] = analysis
        
        consensus = self._synthesize_analyses(analyses)
        
        return {
            'individual_analyses': analyses,
            'consensus': consensus
        }
    
    def _filter_data_for_agent(
        self,
        data: Dict[str, Any],
        agent_type: str
    ) -> Dict[str, Any]:
        if agent_type == 'fundamental':
            return {k: v for k, v in data.items() if k in ['financials', 'company_info', 'industry']}
        elif agent_type == 'technical':
            return {k: v for k, v in data.items() if k in ['price_history', 'volume', 'indicators']}
        elif agent_type == 'quant':
            return {k: v for k, v in data.items() if k in ['returns', 'factors', 'correlations']}
        elif agent_type == 'macro':
            return {k: v for k, v in data.items() if k in ['economic_data', 'sector_trends', 'rates']}
        
        return data
    
    def _synthesize_analyses(
        self,
        analyses: Dict[str, str]
    ) -> str:
        combined_analyses = "\n\n".join([
            f"{agent_type.upper()} ANALYSIS:\n{analysis}"
            for agent_type, analysis in analyses.items()
        ])
        
        prompt = f"""
        Synthesize the following analyses from different specialists into a coherent investment recommendation:
        
        {combined_analyses}
        
        Provide:
        1. Overall assessment
        2. Key agreements across analysts
        3. Key disagreements
        4. Final recommendation
        5. Confidence level
        """
        
        response = self.coordinator.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research director synthesizing team analyses."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
```

## Implementation

### Complete Agentic Research System

```python
class ComprehensiveResearchSystem:
    def __init__(self, api_key: str):
        self.financial_analyzer = FinancialStatementAnalyzer(api_key)
        self.valuation_agent = ValuationAgent()
        self.knowledge_graph = FinancialKnowledgeGraphBuilder()
        self.thesis_generator = InvestmentThesisGenerator(api_key)
        self.red_flag_detector = RedFlagDetector()
        self.report_generator = InvestmentReportGenerator(api_key)
        self.multi_agent_system = MultiAgentResearchSystem(api_key)
        
    def research_company(
        self,
        ticker: str,
        data_sources: Dict[str, Any]
    ) -> Dict[str, Any]:
        financial_analysis = self.financial_analyzer.analyze_income_statement(
            data_sources['income_statement']
        )
        
        valuation = self.valuation_agent.dcf_valuation(
            data_sources['free_cash_flows'],
            data_sources['terminal_growth'],
            data_sources['discount_rate']
        )
        
        bull_case = self.thesis_generator.generate_bull_case(
            data_sources['company_data'],
            data_sources['market_analysis']
        )
        
        bear_case = self.thesis_generator.generate_bear_case(
            data_sources['company_data'],
            data_sources['risk_factors']
        )
        
        red_flags = self.red_flag_detector.check_accounting_quality(
            data_sources['financial_statements']
        )
        
        multi_agent_analysis = self.multi_agent_system.conduct_research(
            ticker,
            data_sources
        )
        
        full_analysis = {
            'ticker': ticker,
            'financial_analysis': financial_analysis,
            'valuation': valuation,
            'bull_case': bull_case,
            'bear_case': bear_case,
            'red_flags': red_flags,
            'multi_agent_analysis': multi_agent_analysis
        }
        
        report = self.report_generator.generate_full_report(ticker, full_analysis)
        
        full_analysis['report'] = report
        
        return full_analysis
```
