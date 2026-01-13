# Finance AI Tutorial Module 14: LLM-Powered Financial Analysis

## Learning Objectives
By the end of this module, you will be able to:
- Use GPT-4 and other LLMs for financial research
- Implement prompt engineering for financial analysis
- Build retrieval-augmented generation (RAG) for finance
- Fine-tune LLMs on financial data
- Create financial AI copilots and assistants
- Deploy LLM-based financial analysis systems

## Introduction to LLMs in Finance

Large Language Models have revolutionized financial analysis by enabling natural language understanding of complex financial documents, automated research, and intelligent decision support systems.

### Applications of LLMs in Finance

1. **Research Automation**: Analyzing reports and extracting insights
2. **Investment Analysis**: Evaluating companies and opportunities
3. **Risk Assessment**: Identifying risks in documents
4. **Market Commentary**: Generating reports and summaries
5. **Customer Service**: Financial chatbots and advisors
6. **Regulatory Compliance**: Document review and analysis

## Financial LLM System Architecture

```python
import openai
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import json
import pandas as pd
from datetime import datetime

@dataclass
class FinancialQuery:
    """Represents a financial analysis query"""
    question: str
    context: Optional[str] = None
    symbols: Optional[List[str]] = None
    time_period: Optional[str] = None
    analysis_type: str = "general"  # general, technical, fundamental, sentiment

@dataclass
class FinancialInsight:
    """Represents a financial insight from LLM"""
    query: str
    response: str
    confidence: float
    sources: List[str]
    timestamp: datetime
    metadata: Dict

class FinancialLLMSystem:
    """
    Comprehensive LLM system for financial analysis
    
    Features:
    - Multi-source data integration
    - Prompt engineering for finance
    - Context management
    - Citation and sourcing
    - Confidence scoring
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize financial LLM system
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        openai.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.knowledge_base = {}
        
        # Financial analysis templates
        self.prompt_templates = {
            'fundamental_analysis': """
            You are an expert financial analyst. Analyze the following company information:
            
            {context}
            
            Provide a comprehensive fundamental analysis covering:
            1. Business model and competitive advantages
            2. Financial health (revenue, profitability, cash flow)
            3. Growth prospects
            4. Key risks
            5. Valuation assessment
            6. Investment recommendation (Buy/Hold/Sell)
            
            Be specific, quantitative, and cite sources.
            """,
            
            'technical_analysis': """
            You are an expert technical analyst. Given the following market data:
            
            {context}
            
            Provide technical analysis including:
            1. Trend identification
            2. Support and resistance levels
            3. Key indicators (RSI, MACD, moving averages)
            4. Chart patterns
            5. Short-term price outlook
            6. Trading recommendations
            """,
            
            'earnings_analysis': """
            You are a financial analyst specializing in earnings analysis. Review:
            
            {context}
            
            Analyze:
            1. Revenue and EPS vs. expectations
            2. Key metrics and trends
            3. Management commentary and outlook
            4. Notable items (one-time charges, etc.)
            5. Impact on stock price
            6. Revised estimates
            """,
            
            'risk_assessment': """
            You are a risk management specialist. Evaluate:
            
            {context}
            
            Assess risks in the following categories:
            1. Market risk
            2. Credit risk
            3. Operational risk
            4. Regulatory risk
            5. Reputational risk
            6. Overall risk rating (Low/Medium/High)
            7. Risk mitigation strategies
            """,
            
            'portfolio_advice': """
            You are a portfolio manager. Given:
            
            Current Portfolio: {portfolio}
            Market Conditions: {market_context}
            Client Profile: {client_profile}
            
            Provide:
            1. Portfolio assessment
            2. Asset allocation recommendations
            3. Specific position adjustments
            4. Risk management suggestions
            5. Rebalancing strategy
            """
        }
    
    def analyze_company(self, symbol: str, analysis_type: str = 'fundamental') -> FinancialInsight:
        """
        Analyze a company using LLM
        
        Args:
            symbol: Stock ticker symbol
            analysis_type: Type of analysis to perform
            
        Returns:
            FinancialInsight with analysis results
        """
        # Gather context (in practice, fetch real data)
        context = self._gather_company_context(symbol)
        
        # Select appropriate prompt template
        template = self.prompt_templates.get(
            f'{analysis_type}_analysis',
            self.prompt_templates['fundamental_analysis']
        )
        
        # Format prompt
        prompt = template.format(context=context)
        
        # Query LLM
        response = self._query_llm(prompt)
        
        # Parse and structure response
        insight = FinancialInsight(
            query=f"Analyze {symbol} ({analysis_type})",
            response=response,
            confidence=self._calculate_confidence(response),
            sources=self._extract_sources(context),
            timestamp=datetime.now(),
            metadata={
                'symbol': symbol,
                'analysis_type': analysis_type,
                'model': self.model
            }
        )
        
        return insight
    
    def _gather_company_context(self, symbol: str) -> str:
        """Gather comprehensive context about a company"""
        # In practice, fetch from multiple sources:
        # - Financial statements
        # - News articles
        # - Analyst reports
        # - Market data
        
        context = f"""
        Company: {symbol}
        
        Recent Financial Data:
        - Revenue (TTM): $50.2B (+15% YoY)
        - Net Income (TTM): $8.5B (+22% YoY)
        - EPS (TTM): $5.25
        - P/E Ratio: 25.3
        - Market Cap: $850B
        
        Recent News:
        - Q4 earnings beat expectations
        - Announced new product launch
        - Expanding into emerging markets
        
        Analyst Consensus:
        - Average Price Target: $185
        - Recommendation: Buy (12 Buy, 5 Hold, 1 Sell)
        
        Key Metrics:
        - Gross Margin: 42%
        - Operating Margin: 28%
        - ROE: 35%
        - Debt/Equity: 0.85
        """
        
        return context
    
    def _query_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Query the LLM with a prompt"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst with deep knowledge of markets, companies, and investment strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for response"""
        # Simplified confidence scoring based on response characteristics
        # In practice, use more sophisticated methods
        
        confidence_indicators = [
            'high confidence',
            'strong evidence',
            'clear trend',
            'definitive',
            'certain'
        ]
        
        uncertainty_indicators = [
            'uncertain',
            'unclear',
            'limited data',
            'may',
            'might',
            'possibly'
        ]
        
        response_lower = response.lower()
        
        confidence_score = 0.5  # Base confidence
        
        # Adjust based on indicators
        for indicator in confidence_indicators:
            if indicator in response_lower:
                confidence_score += 0.1
        
        for indicator in uncertainty_indicators:
            if indicator in response_lower:
                confidence_score -= 0.1
        
        return max(0.0, min(1.0, confidence_score))
    
    def _extract_sources(self, context: str) -> List[str]:
        """Extract sources from context"""
        # In practice, track actual sources
        return ["Financial Statements", "Market Data", "News Articles"]
    
    def multi_company_comparison(self, symbols: List[str], 
                                 criteria: List[str]) -> Dict:
        """
        Compare multiple companies using LLM
        
        Args:
            symbols: List of company symbols to compare
            criteria: Comparison criteria
            
        Returns:
            Comparative analysis
        """
        # Gather context for all companies
        contexts = [self._gather_company_context(symbol) for symbol in symbols]
        
        # Create comparison prompt
        comparison_prompt = f"""
        Compare the following companies on these criteria: {', '.join(criteria)}
        
        Companies:
        {chr(10).join(contexts)}
        
        Provide:
        1. Side-by-side comparison table
        2. Strengths and weaknesses of each
        3. Best choice for different investor profiles
        4. Risk-return assessment
        5. Final ranking with justification
        """
        
        response = self._query_llm(comparison_prompt)
        
        return {
            'symbols': symbols,
            'criteria': criteria,
            'analysis': response,
            'timestamp': datetime.now()
        }
    
    def generate_investment_thesis(self, symbol: str) -> Dict:
        """
        Generate comprehensive investment thesis
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Complete investment thesis
        """
        # Gather comprehensive data
        context = self._gather_company_context(symbol)
        
        thesis_prompt = f"""
        Generate a comprehensive investment thesis for {symbol}.
        
        Context:
        {context}
        
        Your thesis should include:
        
        1. EXECUTIVE SUMMARY
           - Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
           - Target price with 12-month horizon
           - Key investment highlights (3-5 bullets)
        
        2. COMPANY OVERVIEW
           - Business model and market position
           - Competitive advantages and moat
           - Industry trends and dynamics
        
        3. FINANCIAL ANALYSIS
           - Historical performance trends
           - Profitability metrics
           - Cash flow analysis
           - Balance sheet health
        
        4. GROWTH DRIVERS
           - Near-term catalysts
           - Long-term growth opportunities
           - Addressable market size
        
        5. VALUATION
           - Multiple approaches (P/E, DCF, comparable companies)
           - Fair value estimate
           - Upside/downside scenarios
        
        6. RISK FACTORS
           - Company-specific risks
           - Industry and market risks
           - Risk mitigation factors
        
        7. CONCLUSION
           - Investment decision rationale
           - Position sizing recommendation
           - Key metrics to monitor
        
        Format professionally as an analyst report.
        """
        
        response = self._query_llm(thesis_prompt, temperature=0.2)
        
        return {
            'symbol': symbol,
            'thesis': response,
            'generated_at': datetime.now(),
            'analyst': 'AI Financial Analyst'
        }

# Example usage (requires OpenAI API key)
# llm_system = FinancialLLMSystem(api_key="your-api-key", model="gpt-4")

# Analyze a company
# insight = llm_system.analyze_company("AAPL", analysis_type='fundamental')
# print("Analysis:", insight.response)
# print(f"Confidence: {insight.confidence:.2f}")

# Compare companies
# comparison = llm_system.multi_company_comparison(
#     symbols=['AAPL', 'MSFT', 'GOOGL'],
#     criteria=['growth potential', 'valuation', 'risk profile']
# )
# print(comparison['analysis'])

# Generate investment thesis
# thesis = llm_system.generate_investment_thesis("TSLA")
# print(thesis['thesis'])
```

## Retrieval-Augmented Generation (RAG) for Finance

### Financial RAG System

```python
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

class FinancialRAGSystem:
    """
    Retrieval-Augmented Generation system for financial analysis
    
    Combines vector database with LLM for context-aware analysis
    """
    
    def __init__(self, api_key: str, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize RAG system"""
        self.llm = FinancialLLMSystem(api_key)
        self.encoder = SentenceTransformer(embedding_model)
        
        # Document storage
        self.documents = []
        self.document_embeddings = None
        self.index = None
    
    def add_documents(self, documents: List[Dict]):
        """
        Add financial documents to knowledge base
        
        Args:
            documents: List of documents with 'text' and 'metadata' fields
        """
        self.documents.extend(documents)
        
        # Encode documents
        texts = [doc['text'] for doc in documents]
        embeddings = self.encoder.encode(texts)
        
        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
        
        # Build/update FAISS index
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.astype('float32'))
        
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve most relevant documents for query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))
        
        return results
    
    def answer_question_with_context(self, question: str, top_k: int = 5) -> Dict:
        """
        Answer question using retrieved context
        
        Args:
            question: Financial question
            top_k: Number of context documents to retrieve
            
        Returns:
            Answer with sources
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question, top_k)
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for doc, score in relevant_docs:
            context_parts.append(f"Source: {doc['metadata'].get('source', 'Unknown')}")
            context_parts.append(doc['text'])
            context_parts.append("")  # Blank line
            sources.append(doc['metadata'].get('source', 'Unknown'))
        
        context = "\n".join(context_parts)
        
        # Create prompt with context
        prompt = f"""
        Based on the following context, answer the question. If the context doesn't contain 
        enough information, say so and use your general financial knowledge.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a detailed, accurate answer with specific references to the context where applicable.
        """
        
        # Get LLM response
        response = self.llm._query_llm(prompt)
        
        return {
            'question': question,
            'answer': response,
            'sources': sources,
            'num_sources_used': len(relevant_docs),
            'relevance_scores': [score for _, score in relevant_docs]
        }
    
    def analyze_company_with_rag(self, symbol: str) -> Dict:
        """
        Perform comprehensive company analysis using RAG
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Analysis with sourced information
        """
        # Questions to answer
        questions = [
            f"What are the key financial metrics for {symbol}?",
            f"What are the main growth drivers for {symbol}?",
            f"What are the primary risks facing {symbol}?",
            f"How does {symbol} compare to competitors?",
            f"What is the investment outlook for {symbol}?"
        ]
        
        # Get answers for each question
        analysis = {}
        for question in questions:
            answer_data = self.answer_question_with_context(question)
            analysis[question] = answer_data
        
        # Synthesize into report
        synthesis_prompt = f"""
        Based on the following Q&A analysis, create a comprehensive investment report for {symbol}:
        
        {json.dumps(analysis, indent=2)}
        
        Create a structured report with:
        1. Executive Summary
        2. Company Overview
        3. Financial Analysis
        4. Growth Prospects
        5. Risk Assessment
        6. Valuation and Recommendation
        """
        
        report = self.llm._query_llm(synthesis_prompt)
        
        return {
            'symbol': symbol,
            'detailed_qa': analysis,
            'comprehensive_report': report,
            'generated_at': datetime.now()
        }

# Example usage
# rag_system = FinancialRAGSystem(api_key="your-api-key")

# Add financial documents to knowledge base
financial_documents = [
    {
        'text': "Apple reported Q4 2023 earnings of $1.52 per share, beating estimates of $1.39. Revenue reached $89.5B, up 8% YoY...",
        'metadata': {'source': 'Apple Q4 2023 Earnings Release', 'date': '2023-11-02', 'type': 'earnings'}
    },
    {
        'text': "Microsoft Azure cloud revenue grew 29% in Q2 2024, driven by AI services and enterprise adoption...",
        'metadata': {'source': 'Microsoft Q2 2024 Report', 'date': '2024-01-15', 'type': 'earnings'}
    },
    {
        'text': "Federal Reserve maintains interest rates at 5.25-5.50%, citing progress on inflation but continued monitoring...",
        'metadata': {'source': 'Fed Statement March 2024', 'date': '2024-03-20', 'type': 'policy'}
    }
]

# rag_system.add_documents(financial_documents)

# Answer questions with context
# result = rag_system.answer_question_with_context("How is Apple performing financially?")
# print("Answer:", result['answer'])
# print("Sources:", result['sources'])

# Full company analysis
# analysis = rag_system.analyze_company_with_rag("AAPL")
# print(analysis['comprehensive_report'])
```

## Fine-Tuning LLMs for Finance

### Financial Model Fine-Tuning

```python
class FinancialLLMFineTuner:
    """Fine-tune LLMs on financial data"""
    
    def __init__(self, base_model: str = "gpt-3.5-turbo"):
        self.base_model = base_model
        self.training_data = []
    
    def prepare_training_data(self, examples: List[Dict]) -> List[Dict]:
        """
        Prepare financial training examples
        
        Args:
            examples: List of {'prompt': str, 'completion': str} dicts
            
        Returns:
            Formatted training data
        """
        formatted_data = []
        
        for example in examples:
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": example['prompt']},
                    {"role": "assistant", "content": example['completion']}
                ]
            })
        
        return formatted_data
    
    def create_financial_training_examples(self) -> List[Dict]:
        """Create domain-specific training examples"""
        examples = [
            {
                'prompt': "Analyze the P/E ratio of 25 for a tech company.",
                'completion': "A P/E ratio of 25 for a tech company is moderate. It's above the market average of ~20 but below high-growth tech stocks (30-40+). This suggests the market has moderate growth expectations. Compare to industry peers and historical P/E ranges for context."
            },
            {
                'prompt': "What does increasing debt-to-equity ratio indicate?",
                'completion': "An increasing debt-to-equity ratio indicates the company is taking on more debt relative to equity. This can signal: 1) Aggressive growth strategy 2) Declining equity (losses) 3) Leveraged buybacks 4) Increased financial risk. Context matters - some industries naturally have higher D/E ratios."
            },
            {
                'prompt': "Explain earnings surprise.",
                'completion': "Earnings surprise occurs when reported EPS differs from analyst consensus. Positive surprise (actual > expected) often drives stock price up as it signals better-than-anticipated performance. Negative surprise can lead to sell-offs. The magnitude and reason for the surprise matter significantly."
            }
        ]
        
        return examples

# Example of fine-tuning preparation
# fine_tuner = FinancialLLMFineTuner()
# training_examples = fine_tuner.create_financial_training_examples()
# formatted_data = fine_tuner.prepare_training_data(training_examples)
# Then use OpenAI's fine-tuning API with formatted_data
```

This module demonstrates cutting-edge LLM applications in finance, from basic analysis to advanced RAG systems and fine-tuning.

## Advanced LLM Techniques for Finance

### Tool-Use and Function Calling for Financial Analysis

Modern LLMs can invoke external tools and APIs, enabling real-time data retrieval and computation.

```python
import json
from typing import List, Dict, Callable, Any
from openai import OpenAI
from pydantic import BaseModel, Field

class FinancialTool(BaseModel):
    """Base class for financial analysis tools"""
    name: str
    description: str
    parameters: Dict[str, Any]

class ToolEnabledFinancialAgent:
    """Financial agent with tool-use capabilities"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = self._define_tools()
        self.tool_functions = self._register_tool_functions()
        
    def _define_tools(self) -> List[Dict]:
        """Define available tools for the LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get current stock price and key metrics for a ticker symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financial_statements",
                    "description": "Retrieve financial statements for a company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "statement_type": {
                                "type": "string",
                                "enum": ["income", "balance_sheet", "cash_flow"]
                            },
                            "period": {
                                "type": "string",
                                "enum": ["quarterly", "annual"]
                            }
                        },
                        "required": ["ticker", "statement_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_dcf_valuation",
                    "description": "Calculate DCF valuation for a company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "growth_rate": {"type": "number"},
                            "terminal_growth": {"type": "number"},
                            "discount_rate": {"type": "number"}
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_news_sentiment",
                    "description": "Get recent news and sentiment analysis for a company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "num_articles": {"type": "integer", "default": 5}
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "screen_stocks",
                    "description": "Screen stocks based on financial criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "market_cap_min": {"type": "number"},
                            "pe_max": {"type": "number"},
                            "dividend_yield_min": {"type": "number"},
                            "sector": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "backtest_strategy",
                    "description": "Backtest a trading strategy on historical data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strategy_type": {
                                "type": "string",
                                "enum": ["momentum", "mean_reversion", "value", "trend_following"]
                            },
                            "tickers": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"}
                        },
                        "required": ["strategy_type", "tickers"]
                    }
                }
            }
        ]
    
    def _register_tool_functions(self) -> Dict[str, Callable]:
        """Map tool names to actual functions"""
        return {
            "get_stock_price": self._get_stock_price,
            "get_financial_statements": self._get_financial_statements,
            "calculate_dcf_valuation": self._calculate_dcf_valuation,
            "get_news_sentiment": self._get_news_sentiment,
            "screen_stocks": self._screen_stocks,
            "backtest_strategy": self._backtest_strategy
        }
    
    def _get_stock_price(self, ticker: str) -> Dict:
        # In production, use real API like yfinance or Bloomberg
        return {
            "ticker": ticker,
            "price": 185.50,
            "change_percent": 1.25,
            "volume": 52_000_000,
            "market_cap": "2.85T",
            "pe_ratio": 28.5,
            "52_week_high": 199.62,
            "52_week_low": 143.90
        }
    
    def _get_financial_statements(self, ticker: str, statement_type: str, period: str = "quarterly") -> Dict:
        return {
            "ticker": ticker,
            "statement_type": statement_type,
            "period": period,
            "data": {
                "revenue": 94_836_000_000,
                "net_income": 22_956_000_000,
                "eps": 1.52,
                "gross_margin": 0.438
            }
        }
    
    def _calculate_dcf_valuation(
        self, ticker: str, growth_rate: float = 0.08, 
        terminal_growth: float = 0.025, discount_rate: float = 0.10
    ) -> Dict:
        return {
            "ticker": ticker,
            "fair_value": 195.50,
            "current_price": 185.50,
            "upside": "5.4%",
            "assumptions": {
                "growth_rate": growth_rate,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate
            }
        }
    
    def _get_news_sentiment(self, ticker: str, num_articles: int = 5) -> Dict:
        return {
            "ticker": ticker,
            "overall_sentiment": 0.72,
            "sentiment_label": "Bullish",
            "num_articles": num_articles,
            "key_topics": ["earnings beat", "AI expansion", "services growth"]
        }
    
    def _screen_stocks(self, **criteria) -> List[Dict]:
        return [
            {"ticker": "AAPL", "name": "Apple Inc.", "match_score": 0.95},
            {"ticker": "MSFT", "name": "Microsoft Corp.", "match_score": 0.92}
        ]
    
    def _backtest_strategy(
        self, strategy_type: str, tickers: List[str],
        start_date: str = "2020-01-01", end_date: str = "2024-01-01"
    ) -> Dict:
        return {
            "strategy": strategy_type,
            "total_return": 0.85,
            "annualized_return": 0.17,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.12
        }
    
    def chat(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """Process user message with tool calling support"""
        if conversation_history is None:
            conversation_history = []
        
        messages = [
            {"role": "system", "content": """You are an expert financial analyst AI assistant. 
            You have access to financial data tools. Use them to provide accurate, data-driven analysis.
            Always cite your data sources and explain your reasoning."""}
        ] + conversation_history + [{"role": "user", "content": user_message}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # Process tool calls if any
        if assistant_message.tool_calls:
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the tool
                if function_name in self.tool_functions:
                    result = self.tool_functions[function_name](**function_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            
            # Get final response with tool results
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        return assistant_message.content
```

### Multi-Agent Financial Analysis with LangGraph

```python
from typing import TypedDict, Annotated, Sequence
import operator

class FinancialAgentState(TypedDict):
    """State shared between agents"""
    messages: Annotated[Sequence[dict], operator.add]
    ticker: str
    analysis_results: dict
    final_recommendation: str

class MultiAgentFinancialSystem:
    """Multi-agent system using LangGraph-style architecture"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.agents = {
            "fundamental_analyst": self._create_fundamental_agent(),
            "technical_analyst": self._create_technical_agent(),
            "risk_analyst": self._create_risk_agent(),
            "portfolio_manager": self._create_portfolio_manager()
        }
        
    def _create_fundamental_agent(self) -> dict:
        return {
            "name": "Fundamental Analyst",
            "system_prompt": """You are a fundamental analyst specializing in:
            - Financial statement analysis
            - Valuation (DCF, comparable companies)
            - Industry and competitive analysis
            - Management quality assessment
            Provide detailed, quantitative analysis."""
        }
    
    def _create_technical_agent(self) -> dict:
        return {
            "name": "Technical Analyst", 
            "system_prompt": """You are a technical analyst specializing in:
            - Chart pattern recognition
            - Technical indicators (RSI, MACD, moving averages)
            - Support/resistance levels
            - Volume analysis
            Provide actionable trading signals."""
        }
    
    def _create_risk_agent(self) -> dict:
        return {
            "name": "Risk Analyst",
            "system_prompt": """You are a risk analyst specializing in:
            - Market risk assessment
            - Credit risk evaluation
            - Operational risk identification
            - Tail risk and stress testing
            Quantify risks and suggest mitigations."""
        }
    
    def _create_portfolio_manager(self) -> dict:
        return {
            "name": "Portfolio Manager",
            "system_prompt": """You are a senior portfolio manager who synthesizes inputs from:
            - Fundamental analysts
            - Technical analysts
            - Risk analysts
            Make final investment decisions with position sizing recommendations."""
        }
    
    def _invoke_agent(self, agent_key: str, context: str, query: str) -> str:
        agent = self.agents[agent_key]
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": agent["system_prompt"]},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{query}"}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def analyze_investment(self, ticker: str, context: dict) -> dict:
        """Run multi-agent analysis workflow"""
        
        context_str = json.dumps(context, indent=2)
        
        # Step 1: Fundamental analysis
        fundamental_analysis = self._invoke_agent(
            "fundamental_analyst",
            context_str,
            f"Provide comprehensive fundamental analysis for {ticker}"
        )
        
        # Step 2: Technical analysis (parallel with fundamental)
        technical_analysis = self._invoke_agent(
            "technical_analyst",
            context_str,
            f"Provide technical analysis and trading signals for {ticker}"
        )
        
        # Step 3: Risk analysis
        risk_analysis = self._invoke_agent(
            "risk_analyst",
            f"Fundamental: {fundamental_analysis}\n\nTechnical: {technical_analysis}",
            f"Assess key risks for investing in {ticker}"
        )
        
        # Step 4: Portfolio manager synthesizes
        combined_analysis = f"""
        FUNDAMENTAL ANALYSIS:
        {fundamental_analysis}
        
        TECHNICAL ANALYSIS:
        {technical_analysis}
        
        RISK ANALYSIS:
        {risk_analysis}
        """
        
        final_recommendation = self._invoke_agent(
            "portfolio_manager",
            combined_analysis,
            f"Synthesize all analyses and provide final recommendation for {ticker}"
        )
        
        return {
            "ticker": ticker,
            "fundamental_analysis": fundamental_analysis,
            "technical_analysis": technical_analysis,
            "risk_analysis": risk_analysis,
            "final_recommendation": final_recommendation
        }
```

### Structured Output and JSON Mode for Financial Data

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class InvestmentRecommendation(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"

class ValuationMetrics(BaseModel):
    pe_ratio: float = Field(description="Price to Earnings ratio")
    pb_ratio: float = Field(description="Price to Book ratio")
    ev_ebitda: float = Field(description="Enterprise Value to EBITDA")
    dcf_fair_value: float = Field(description="DCF-based fair value per share")
    upside_potential: float = Field(description="Percentage upside to fair value")

class RiskAssessment(BaseModel):
    overall_risk: Literal["Low", "Medium", "High", "Very High"]
    market_risk: float = Field(ge=0, le=10, description="Market risk score 0-10")
    company_risk: float = Field(ge=0, le=10, description="Company-specific risk score 0-10")
    key_risks: List[str] = Field(description="List of key risk factors")

class StructuredInvestmentAnalysis(BaseModel):
    ticker: str
    company_name: str
    sector: str
    recommendation: InvestmentRecommendation
    target_price: float
    current_price: float
    confidence: float = Field(ge=0, le=1, description="Confidence level 0-1")
    valuation: ValuationMetrics
    risk_assessment: RiskAssessment
    investment_thesis: str = Field(description="2-3 sentence investment thesis")
    key_catalysts: List[str]
    key_concerns: List[str]

class StructuredOutputFinancialAnalyst:
    """Financial analyst with structured JSON output"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def analyze(self, ticker: str, context: str) -> StructuredInvestmentAnalysis:
        """Generate structured investment analysis"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": """You are a financial analyst. 
                Generate structured investment analysis in the exact JSON format specified.
                Be precise with numbers and provide data-driven insights."""},
                {"role": "user", "content": f"Analyze {ticker}:\n{context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        return StructuredInvestmentAnalysis(**result)
```

### Advanced Prompt Engineering for Finance

```python
class AdvancedFinancialPromptEngineer:
    """Advanced prompt engineering techniques for financial analysis"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def chain_of_thought_analysis(self, ticker: str, data: dict) -> str:
        """Use chain-of-thought prompting for complex analysis"""
        
        prompt = f"""
        Analyze {ticker} step by step:
        
        Data provided:
        {json.dumps(data, indent=2)}
        
        Let's think through this systematically:
        
        Step 1: Business Quality Analysis
        - What is the company's competitive position?
        - What are the sustainable competitive advantages?
        - How strong is the management team?
        
        Step 2: Financial Analysis
        - Analyze revenue trends and growth drivers
        - Evaluate profitability metrics and trends
        - Assess cash flow quality and capital allocation
        - Review balance sheet strength
        
        Step 3: Valuation Assessment
        - Compare current multiples to historical averages
        - Compare to peer group valuations
        - Estimate intrinsic value using DCF
        
        Step 4: Risk Evaluation
        - Identify company-specific risks
        - Assess macro and industry risks
        - Evaluate risk/reward balance
        
        Step 5: Investment Conclusion
        - Synthesize all factors
        - Provide clear recommendation
        - Specify entry points and position sizing
        
        Provide your analysis for each step, showing your reasoning.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a meticulous financial analyst who shows all reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def few_shot_earnings_analysis(self, earnings_data: dict) -> str:
        """Few-shot prompting for earnings analysis"""
        
        examples = """
        Example 1:
        Input: AAPL Q4 2023 - Revenue $89.5B (est: $89.3B), EPS $1.46 (est: $1.39), iPhone revenue -1% YoY
        Analysis: Solid quarter with revenue/EPS beats. iPhone weakness offset by Services growth (+16% YoY). 
        Gross margin expansion to 45.2% shows pricing power. Slight concern on China weakness but manageable.
        Outlook: Cautiously optimistic, in-line guide suggests conservative management. Rating: BUY
        
        Example 2:
        Input: NVDA Q3 2024 - Revenue $18.1B (est: $16.1B), EPS $4.02 (est: $3.37), Data Center +279% YoY
        Analysis: Blowout quarter driven by AI demand. Data center segment now 80% of revenue. 
        Supply constraints easing. China export restrictions a headwind but manageable near-term.
        Outlook: Beat-and-raise cycle continues. AI demand secular not cyclical. Rating: STRONG BUY
        
        Example 3:
        Input: META Q4 2023 - Revenue $40.1B (est: $39.0B), EPS $5.33 (est: $4.96), Reality Labs loss $4.6B
        Analysis: Strong ad revenue recovery, +25% YoY driven by Reels monetization and AI ad targeting.
        Reality Labs losses remain elevated but company committing to efficiency. 
        Outlook: "Year of Efficiency" delivering results. Multiple still reasonable. Rating: BUY
        """
        
        prompt = f"""
        Based on these examples of earnings analysis:
        
        {examples}
        
        Now analyze this earnings report:
        Input: {json.dumps(earnings_data)}
        
        Provide a similar concise analysis with key metrics, insights, and rating.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def self_consistency_valuation(self, ticker: str, data: dict, n_samples: int = 5) -> dict:
        """Use self-consistency sampling for robust valuation"""
        
        valuations = []
        reasoning = []
        
        prompt = f"""
        Estimate the fair value for {ticker} based on:
        {json.dumps(data, indent=2)}
        
        Use your best judgment combining multiple valuation methods.
        Provide a single fair value estimate and brief reasoning.
        Format: Fair Value: $XXX | Reasoning: [brief explanation]
        """
        
        for _ in range(n_samples):
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a valuation expert. Provide precise estimates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7  # Higher temperature for diversity
            )
            
            result = response.choices[0].message.content
            reasoning.append(result)
            
            # Extract value (simplified parsing)
            if "Fair Value: $" in result:
                try:
                    value_str = result.split("Fair Value: $")[1].split()[0].replace(",", "")
                    valuations.append(float(value_str))
                except:
                    pass
        
        if valuations:
            return {
                "median_fair_value": np.median(valuations),
                "mean_fair_value": np.mean(valuations),
                "std_fair_value": np.std(valuations),
                "min_fair_value": min(valuations),
                "max_fair_value": max(valuations),
                "n_samples": len(valuations),
                "reasoning_samples": reasoning
            }
        
        return {"error": "Could not extract valuations"}
```

### Fine-Tuning with LoRA for Financial LLMs

```python
class FinancialLoRAFineTuner:
    """LoRA fine-tuning setup for financial domain LLMs"""
    
    def __init__(self, base_model: str = "meta-llama/Llama-2-7b-hf"):
        self.base_model = base_model
        
    def get_lora_config(self) -> dict:
        """LoRA configuration optimized for financial tasks"""
        return {
            "r": 16,  # Rank
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    
    def prepare_financial_dataset(self, data: List[dict]) -> List[dict]:
        """Prepare financial Q&A pairs for fine-tuning"""
        formatted = []
        
        for item in data:
            formatted.append({
                "instruction": item["question"],
                "input": item.get("context", ""),
                "output": item["answer"]
            })
        
        return formatted
    
    def get_training_config(self) -> dict:
        """Training configuration for financial fine-tuning"""
        return {
            "output_dir": "./financial_lora",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "fp16": True,
            "optim": "adamw_torch",
            "max_grad_norm": 0.3,
            "weight_decay": 0.001
        }
    
    def create_financial_training_examples(self) -> List[dict]:
        """Create domain-specific training examples"""
        return [
            {
                "question": "What does a PE ratio of 35 indicate for a software company?",
                "context": "The company is a fast-growing SaaS business with 40% revenue growth.",
                "answer": """A PE of 35 for a fast-growing SaaS company (40% growth) is reasonable given the PEG ratio of 0.875 (35/40). 
                High-growth software companies typically trade at 20-40x earnings due to high margins, recurring revenue, and scalability.
                Compare to peers: sector median ~30x for similar growth profiles. The premium is justified if growth is sustainable."""
            },
            {
                "question": "Analyze the working capital trend: AR +25%, Revenue +10%",
                "answer": """Red flag: Accounts receivable growing 2.5x faster than revenue indicates potential issues:
                1. Aggressive revenue recognition (channel stuffing)
                2. Deteriorating customer creditworthiness  
                3. Extended payment terms to drive sales
                Calculate DSO trend and compare to industry. If DSO increased >20%, investigate further.
                Check cash flow from operations vs. net income for confirmation."""
            },
            {
                "question": "Calculate enterprise value for: Market Cap $50B, Debt $10B, Cash $5B",
                "answer": """Enterprise Value = Market Cap + Total Debt - Cash
                EV = $50B + $10B - $5B = $55B
                
                This represents the theoretical takeover price. Key considerations:
                - Net debt position: $10B - $5B = $5B (manageable if EBITDA > $5B)
                - EV/Sales and EV/EBITDA multiples more relevant than P/E for this company
                - Cash could include restricted cash; verify in footnotes"""
            }
        ]
```

## Key Takeaways

- Modern LLMs with tool-use enable real-time financial data analysis
- Multi-agent architectures improve analysis quality through specialization
- Structured outputs ensure consistent, parseable financial recommendations
- Chain-of-thought prompting improves reasoning for complex analyses
- Fine-tuning with LoRA adapts general LLMs to financial domain efficiently
- Self-consistency sampling provides more robust estimates
- Always validate LLM outputs against authoritative data sources
- Consider hallucination risks especially for numerical data

## Next Steps

Explore the remaining modules to build complete AI-powered financial systems, including portfolio management, risk assessment, and production deployment strategies.

