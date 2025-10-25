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

## Key Takeaways

- LLMs can significantly enhance financial analysis capabilities
- Prompt engineering is crucial for quality financial insights
- RAG systems combine retrieval with generation for better accuracy
- Fine-tuning adapts LLMs to financial domain specifics
- Always validate LLM outputs against actual data
- Consider confidence scoring and uncertainty quantification
- Maintain source attribution and transparency

## Next Steps

Explore the remaining modules to build complete AI-powered financial systems, including portfolio management, risk assessment, and production deployment strategies.

