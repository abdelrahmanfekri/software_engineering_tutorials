# Finance AI Tutorial Module 2: Financial Text Analytics and NLP

## Learning Objectives
By the end of this module, you will be able to:
- Extract insights from financial documents
- Perform sentiment analysis on financial text
- Implement named entity recognition for finance
- Analyze earnings calls and SEC filings
- Build financial news analysis systems
- Create financial knowledge graphs

## Introduction to Financial Text Analytics

Financial markets generate massive amounts of textual data that can provide valuable trading signals and insights. This module covers how to process and analyze financial text using NLP techniques.

### Types of Financial Text Data

1. **News Articles**: Real-time market moving news
2. **SEC Filings**: 10-K, 10-Q, 8-K reports
3. **Earnings Call Transcripts**: Management commentary
4. **Analyst Reports**: Research and recommendations
5. **Social Media**: Twitter, Reddit, StockTwits
6. **Press Releases**: Corporate announcements
7. **Central Bank Communications**: Policy statements

## Financial Named Entity Recognition

### Finance-Specific NER

```python
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class FinancialNER:
    """Financial Named Entity Recognition system"""
    
    def __init__(self, model_name: str = 'dslim/bert-base-NER'):
        """
        Initialize Financial NER
        
        Entity types:
        - ORG: Organizations (companies, banks)
        - TICKER: Stock ticker symbols
        - MONEY: Monetary values
        - PERCENT: Percentage values
        - DATE: Temporal expressions
        - PRODUCT: Financial products
        - PERSON: Key personnel
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, 
                                     aggregation_strategy="simple")
        
        # Financial entity patterns
        self.ticker_pattern = r'\b[A-Z]{1,5}\b'  # Simple ticker pattern
        self.money_pattern = r'\$[\d,]+(?:\.\d{2})?[KMB]?'
        self.percent_pattern = r'\d+(?:\.\d+)?%'
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract financial entities from text"""
        # Use transformer-based NER
        entities = self.ner_pipeline(text)
        
        # Post-process and add financial-specific entities
        financial_entities = []
        
        for entity in entities:
            financial_entities.append({
                'text': entity['word'],
                'type': entity['entity_group'],
                'score': entity['score'],
                'start': entity['start'],
                'end': entity['end']
            })
        
        # Add regex-based financial entities
        import re
        
        # Extract ticker symbols
        tickers = re.finditer(self.ticker_pattern, text)
        for match in tickers:
            if self._is_likely_ticker(match.group()):
                financial_entities.append({
                    'text': match.group(),
                    'type': 'TICKER',
                    'score': 0.9,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Extract money amounts
        money_matches = re.finditer(self.money_pattern, text)
        for match in money_matches:
            financial_entities.append({
                'text': match.group(),
                'type': 'MONEY',
                'score': 0.95,
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract percentages
        percent_matches = re.finditer(self.percent_pattern, text)
        for match in percent_matches:
            financial_entities.append({
                'text': match.group(),
                'type': 'PERCENT',
                'score': 0.95,
                'start': match.start(),
                'end': match.end()
            })
        
        # Sort by position
        financial_entities.sort(key=lambda x: x['start'])
        
        return financial_entities
    
    def _is_likely_ticker(self, text: str) -> bool:
        """Heuristic to identify if text is likely a stock ticker"""
        # Common words to exclude
        exclude = {'THE', 'AND', 'FOR', 'ARE', 'WAS', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN'}
        return text not in exclude and len(text) <= 5 and text.isupper()
    
    def extract_company_mentions(self, text: str) -> List[Dict]:
        """Extract company mentions with context"""
        entities = self.extract_entities(text)
        companies = []
        
        for i, entity in enumerate(entities):
            if entity['type'] in ['ORG', 'TICKER']:
                # Get context window
                context_start = max(0, entity['start'] - 100)
                context_end = min(len(text), entity['end'] + 100)
                context = text[context_start:context_end]
                
                companies.append({
                    'name': entity['text'],
                    'type': entity['type'],
                    'context': context,
                    'position': entity['start'],
                    'confidence': entity['score']
                })
        
        return companies

# Example usage
financial_ner = FinancialNER()

sample_text = """
Apple Inc. (AAPL) reported quarterly earnings of $1.52 per share, beating 
analyst estimates by 15%. The company's revenue reached $97.3B, up 8.6% 
year-over-year. CEO Tim Cook highlighted strong iPhone sales in China.
Microsoft (MSFT) also announced solid results, with Azure cloud revenue 
growing 29%.
"""

entities = financial_ner.extract_entities(sample_text)
print("Extracted Entities:")
for entity in entities:
    print(f"  {entity['text']:15} | Type: {entity['type']:10} | Confidence: {entity['score']:.2f}")

companies = financial_ner.extract_company_mentions(sample_text)
print("\nCompany Mentions:")
for company in companies:
    print(f"  {company['name']}: {company['context'][:50]}...")
```

## Financial Sentiment Analysis

### Market Sentiment Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import torch.nn as nn

class FinancialSentimentAnalyzer:
    """
    Financial sentiment analysis with market-specific understanding
    
    Sentiment classes:
    - Positive: Bullish, optimistic
    - Negative: Bearish, pessimistic  
    - Neutral: Factual, balanced
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        """Initialize with financial sentiment model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        # Financial sentiment lexicon
        self.positive_terms = {
            'beat', 'exceed', 'strong', 'growth', 'profit', 'gain', 'bull',
            'outperform', 'surge', 'rally', 'upgrade', 'breakout'
        }
        
        self.negative_terms = {
            'miss', 'weak', 'decline', 'loss', 'bear', 'underperform',
            'plunge', 'crash', 'downgrade', 'breakdown', 'recession'
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of financial text"""
        # Get model prediction
        result = self.sentiment_pipeline(text)[0]
        
        # Calculate lexicon-based sentiment
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_terms)
        negative_count = sum(1 for word in words if word in self.negative_terms)
        
        lexicon_score = (positive_count - negative_count) / (len(words) + 1)
        
        # Combine model and lexicon
        model_score = self._label_to_score(result['label'])
        combined_score = 0.7 * model_score + 0.3 * lexicon_score
        
        return {
            'text': text,
            'model_label': result['label'],
            'model_confidence': result['score'],
            'model_score': model_score,
            'lexicon_score': lexicon_score,
            'combined_score': combined_score,
            'sentiment': self._score_to_sentiment(combined_score)
        }
    
    def _label_to_score(self, label: str) -> float:
        """Convert sentiment label to numerical score"""
        label_map = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        return label_map.get(label.lower(), 0.0)
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score > 0.2:
            return 'bullish'
        elif score < -0.2:
            return 'bearish'
        else:
            return 'neutral'
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def calculate_aggregate_sentiment(self, texts: List[str], 
                                     weights: List[float] = None) -> Dict:
        """Calculate aggregate sentiment from multiple texts"""
        if weights is None:
            weights = [1.0] * len(texts)
        
        results = self.analyze_batch(texts)
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(r['combined_score'] * w 
                           for r, w in zip(results, weights)) / total_weight
        
        # Distribution
        sentiments = [r['sentiment'] for r in results]
        distribution = {
            'bullish': sentiments.count('bullish') / len(sentiments),
            'bearish': sentiments.count('bearish') / len(sentiments),
            'neutral': sentiments.count('neutral') / len(sentiments)
        }
        
        return {
            'aggregate_score': weighted_score,
            'aggregate_sentiment': self._score_to_sentiment(weighted_score),
            'distribution': distribution,
            'num_texts': len(texts),
            'individual_results': results
        }

# Example usage
sentiment_analyzer = FinancialSentimentAnalyzer()

news_articles = [
    "Apple beats earnings estimates with strong iPhone sales and expanding margins.",
    "Tech stocks plunge amid concerns about rising interest rates and inflation.",
    "The Federal Reserve maintains current interest rate policy as expected.",
    "Tesla stock surges 12% on record delivery numbers and positive outlook."
]

# Analyze individual articles
print("Individual Sentiment Analysis:")
for article in news_articles:
    result = sentiment_analyzer.analyze_sentiment(article)
    print(f"\nArticle: {article[:60]}...")
    print(f"  Sentiment: {result['sentiment'].upper()}")
    print(f"  Score: {result['combined_score']:.3f}")
    print(f"  Confidence: {result['model_confidence']:.3f}")

# Aggregate sentiment
aggregate = sentiment_analyzer.calculate_aggregate_sentiment(news_articles)
print(f"\nAggregate Market Sentiment: {aggregate['aggregate_sentiment'].upper()}")
print(f"Score: {aggregate['aggregate_score']:.3f}")
print(f"Distribution: {aggregate['distribution']}")
```

## SEC Filings Analysis

### 10-K and 10-Q Report Analyzer

```python
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

class SECFilingsAnalyzer:
    """Analyze SEC filings (10-K, 10-Q, 8-K)"""
    
    def __init__(self):
        self.sec_base_url = "https://www.sec.gov"
        self.headers = {
            'User-Agent': 'Your Name your@email.com'  # Required by SEC
        }
        
        # Key sections to analyze
        self.key_sections = {
            '10-K': [
                'Business',
                'Risk Factors',
                'Management Discussion and Analysis',
                'Financial Statements'
            ],
            '10-Q': [
                'Financial Statements',
                'Management Discussion and Analysis',
                'Risk Factors'
            ]
        }
    
    def search_company_filings(self, ticker: str, filing_type: str = '10-K',
                              count: int = 5) -> List[Dict]:
        """Search for company filings"""
        # This is a simplified version
        # In practice, use SEC EDGAR API or libraries like sec-edgar-downloader
        
        # Mock data for demonstration
        filings = [
            {
                'ticker': ticker,
                'filing_type': filing_type,
                'filing_date': '2023-02-15',
                'url': f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={filing_type}',
                'accession_number': '0000320193-23-000006'
            }
        ]
        
        return filings[:count]
    
    def extract_text_from_filing(self, filing_url: str) -> str:
        """Extract text from SEC filing"""
        # In practice, download and parse HTML/XBRL
        # This is a simplified placeholder
        return "Sample SEC filing text with financial information..."
    
    def extract_risk_factors(self, filing_text: str) -> List[str]:
        """Extract risk factors section"""
        # Find "Risk Factors" section
        risk_pattern = r'(?i)risk factors.*?(?=item \d|$)'
        risk_match = re.search(risk_pattern, filing_text, re.DOTALL)
        
        if risk_match:
            risk_text = risk_match.group()
            # Split into individual risks
            risks = re.split(r'\n\s*\n', risk_text)
            return [r.strip() for r in risks if len(r.strip()) > 100]
        
        return []
    
    def extract_md_and_a(self, filing_text: str) -> str:
        """Extract Management Discussion and Analysis section"""
        mda_pattern = r'(?i)management.?s discussion.*?(?=item \d|$)'
        mda_match = re.search(mda_pattern, filing_text, re.DOTALL)
        
        if mda_match:
            return mda_match.group().strip()
        
        return ""
    
    def analyze_filing_sentiment(self, filing_text: str) -> Dict:
        """Analyze overall sentiment of filing"""
        sentiment_analyzer = FinancialSentimentAnalyzer()
        
        # Split into chunks
        max_length = 512
        words = filing_text.split()
        chunks = [' '.join(words[i:i+max_length]) 
                 for i in range(0, len(words), max_length)]
        
        # Analyze each chunk
        chunk_sentiments = []
        for chunk in chunks[:20]:  # Limit to first 20 chunks
            result = sentiment_analyzer.analyze_sentiment(chunk)
            chunk_sentiments.append(result['combined_score'])
        
        return {
            'average_sentiment': np.mean(chunk_sentiments),
            'sentiment_volatility': np.std(chunk_sentiments),
            'sentiment_trend': np.polyfit(range(len(chunk_sentiments)), 
                                         chunk_sentiments, 1)[0],
            'num_chunks_analyzed': len(chunk_sentiments)
        }
    
    def compare_filings(self, filing1_text: str, filing2_text: str) -> Dict:
        """Compare two filings to detect changes"""
        from difflib import SequenceMatcher
        
        # Calculate similarity
        similarity = SequenceMatcher(None, filing1_text, filing2_text).ratio()
        
        # Identify major changes
        # In practice, use more sophisticated diff algorithms
        
        return {
            'similarity': similarity,
            'change_percentage': (1 - similarity) * 100,
            'significant_change': similarity < 0.8
        }
    
    def extract_financial_metrics(self, filing_text: str) -> Dict:
        """Extract key financial metrics from filing"""
        metrics = {}
        
        # Revenue pattern
        revenue_pattern = r'revenue[s]?\s+of\s+\$?([\d,]+(?:\.\d+)?)\s*([MB])'
        revenue_match = re.search(revenue_pattern, filing_text, re.IGNORECASE)
        if revenue_match:
            amount = float(revenue_match.group(1).replace(',', ''))
            multiplier = {'M': 1e6, 'B': 1e9}[revenue_match.group(2)]
            metrics['revenue'] = amount * multiplier
        
        # EPS pattern
        eps_pattern = r'earnings per share.*?\$?([\d.]+)'
        eps_match = re.search(eps_pattern, filing_text, re.IGNORECASE)
        if eps_match:
            metrics['eps'] = float(eps_match.group(1))
        
        # Debt pattern
        debt_pattern = r'total debt.*?\$?([\d,]+(?:\.\d+)?)\s*([MB])'
        debt_match = re.search(debt_pattern, filing_text, re.IGNORECASE)
        if debt_match:
            amount = float(debt_match.group(1).replace(',', ''))
            multiplier = {'M': 1e6, 'B': 1e9}[debt_match.group(2)]
            metrics['total_debt'] = amount * multiplier
        
        return metrics

# Example usage
sec_analyzer = SECFilingsAnalyzer()

# Search for filings
ticker = "AAPL"
filings = sec_analyzer.search_company_filings(ticker, filing_type='10-K', count=3)

print(f"Found {len(filings)} filings for {ticker}:")
for filing in filings:
    print(f"  {filing['filing_type']} - {filing['filing_date']}")
    
    # Extract and analyze (using sample text for demonstration)
    filing_text = sec_analyzer.extract_text_from_filing(filing['url'])
    
    # Extract sections
    risks = sec_analyzer.extract_risk_factors(filing_text)
    print(f"    Risk factors identified: {len(risks)}")
    
    # Sentiment analysis
    sentiment = sec_analyzer.analyze_filing_sentiment(filing_text)
    print(f"    Sentiment score: {sentiment['average_sentiment']:.3f}")
    
    # Financial metrics
    metrics = sec_analyzer.extract_financial_metrics(filing_text)
    print(f"    Extracted metrics: {list(metrics.keys())}")
```

## Earnings Call Transcript Analysis

### Earnings Call Analyzer

```python
class EarningsCallAnalyzer:
    """Analyze earnings call transcripts"""
    
    def __init__(self):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.ner = FinancialNER()
    
    def parse_transcript(self, transcript: str) -> Dict:
        """Parse earnings call transcript into structured format"""
        sections = {
            'prepared_remarks': '',
            'qa_session': '',
            'participants': []
        }
        
        # Split into sections (simplified)
        if 'question-and-answer' in transcript.lower():
            parts = re.split(r'question-and-answer', transcript, flags=re.IGNORECASE)
            sections['prepared_remarks'] = parts[0]
            sections['qa_session'] = parts[1] if len(parts) > 1 else ''
        else:
            sections['prepared_remarks'] = transcript
        
        # Extract participants
        participants = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)(?:,| --)', transcript)
        sections['participants'] = list(set(participants))
        
        return sections
    
    def analyze_management_tone(self, prepared_remarks: str) -> Dict:
        """Analyze management's tone in prepared remarks"""
        # Split by speaker (simplified)
        # In practice, use more sophisticated speaker identification
        
        sentiment = self.sentiment_analyzer.analyze_sentiment(prepared_remarks)
        
        # Extract key themes
        themes = self._extract_key_themes(prepared_remarks)
        
        # Identify forward-looking statements
        forward_looking = self._identify_forward_looking_statements(prepared_remarks)
        
        return {
            'sentiment': sentiment,
            'key_themes': themes,
            'forward_looking_statements': forward_looking,
            'confidence_level': self._assess_confidence(prepared_remarks)
        }
    
    def analyze_qa_session(self, qa_text: str) -> Dict:
        """Analyze Q&A session dynamics"""
        # Split into individual Q&A pairs
        qa_pairs = self._split_qa_pairs(qa_text)
        
        # Analyze each pair
        analysis = []
        for q, a in qa_pairs:
            analysis.append({
                'question': q,
                'answer': a,
                'question_sentiment': self.sentiment_analyzer.analyze_sentiment(q),
                'answer_sentiment': self.sentiment_analyzer.analyze_sentiment(a),
                'question_topics': self._extract_topics(q),
                'answer_length': len(a.split()),
                'evasiveness_score': self._measure_evasiveness(q, a)
            })
        
        return {
            'num_questions': len(qa_pairs),
            'qa_pairs': analysis,
            'common_topics': self._aggregate_topics(analysis),
            'average_evasiveness': np.mean([qa['evasiveness_score'] for qa in analysis])
        }
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes using keyword extraction"""
        # Simplified keyword extraction
        keywords = ['growth', 'revenue', 'margin', 'profit', 'expansion',
                   'innovation', 'market', 'product', 'customer', 'efficiency']
        
        themes = []
        for keyword in keywords:
            if keyword in text.lower():
                themes.append(keyword)
        
        return themes
    
    def _identify_forward_looking_statements(self, text: str) -> List[str]:
        """Identify forward-looking statements"""
        forward_indicators = [
            'expect', 'anticipate', 'believe', 'plan', 'intend',
            'project', 'forecast', 'estimate', 'will', 'should'
        ]
        
        sentences = re.split(r'[.!?]', text)
        forward_looking = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in forward_indicators):
                forward_looking.append(sentence.strip())
        
        return forward_looking[:10]  # Return top 10
    
    def _assess_confidence(self, text: str) -> float:
        """Assess management confidence level"""
        confidence_terms = ['confident', 'strong', 'optimistic', 'positive']
        uncertainty_terms = ['uncertain', 'challenging', 'difficult', 'volatile']
        
        words = text.lower().split()
        confidence_score = sum(1 for word in words if word in confidence_terms)
        uncertainty_score = sum(1 for word in words if word in uncertainty_terms)
        
        total_score = confidence_score - uncertainty_score
        normalized_score = total_score / (len(words) + 1)
        
        return max(0, min(1, normalized_score * 10 + 0.5))
    
    def _split_qa_pairs(self, qa_text: str) -> List[Tuple[str, str]]:
        """Split Q&A text into question-answer pairs"""
        # Simplified splitting
        # In practice, use more sophisticated parsing
        pairs = []
        
        # Mock implementation
        sections = re.split(r'\n\s*\n', qa_text)
        for i in range(0, len(sections) - 1, 2):
            if i + 1 < len(sections):
                pairs.append((sections[i], sections[i + 1]))
        
        return pairs
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Use NER to extract topics
        entities = self.ner.extract_entities(text)
        topics = [e['text'] for e in entities if e['type'] in ['ORG', 'PRODUCT']]
        return list(set(topics))
    
    def _measure_evasiveness(self, question: str, answer: str) -> float:
        """Measure how evasive an answer is"""
        # Simplified heuristic
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        
        # Overlap between question and answer keywords
        overlap = len(question_keywords & answer_keywords)
        evasiveness = 1 - (overlap / (len(question_keywords) + 1))
        
        # Check for deflection phrases
        deflection_phrases = ['as I mentioned', 'as we discussed', 'going forward',
                             'we don\'t provide guidance', 'can\'t comment']
        
        deflection_count = sum(1 for phrase in deflection_phrases 
                              if phrase in answer.lower())
        
        evasiveness += deflection_count * 0.1
        
        return min(1.0, evasiveness)
    
    def _aggregate_topics(self, analysis: List[Dict]) -> Dict:
        """Aggregate topics across all Q&A pairs"""
        all_topics = []
        for qa in analysis:
            all_topics.extend(qa['question_topics'])
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
    
    def generate_summary(self, transcript: str) -> Dict:
        """Generate comprehensive summary of earnings call"""
        sections = self.parse_transcript(transcript)
        
        management_analysis = self.analyze_management_tone(sections['prepared_remarks'])
        qa_analysis = self.analyze_qa_session(sections['qa_session']) if sections['qa_session'] else None
        
        return {
            'participants': sections['participants'],
            'management_sentiment': management_analysis['sentiment']['sentiment'],
            'confidence_level': management_analysis['confidence_level'],
            'key_themes': management_analysis['key_themes'],
            'forward_looking_count': len(management_analysis['forward_looking_statements']),
            'qa_summary': {
                'num_questions': qa_analysis['num_questions'] if qa_analysis else 0,
                'average_evasiveness': qa_analysis['average_evasiveness'] if qa_analysis else 0,
                'common_topics': qa_analysis['common_topics'] if qa_analysis else {}
            } if qa_analysis else None
        }

# Example usage
earnings_analyzer = EarningsCallAnalyzer()

sample_transcript = """
Apple Q4 2023 Earnings Call Transcript

Tim Cook -- Chief Executive Officer

Good afternoon. We're pleased to report strong Q4 results with revenue of $89.5 billion, 
up 8% year-over-year. We're confident in our product pipeline and expect continued growth 
in the coming quarters. Our Services business continues to be a bright spot with record 
revenues.

Question-and-Answer Session

Analyst: Can you provide more color on iPhone demand in China?

Tim Cook: We're pleased with our performance in China. We expect the market to remain 
competitive, but we're confident in our ability to execute. As I mentioned, we're 
optimistic about our product lineup.

Analyst: What are your expectations for margins in Q1?

CFO: We don't provide specific guidance on margins, but as we discussed, we're focused 
on operational efficiency and expect to maintain strong profitability going forward.
"""

# Parse and analyze
summary = earnings_analyzer.generate_summary(sample_transcript)
print("Earnings Call Summary:")
print(f"  Management Sentiment: {summary['management_sentiment']}")
print(f"  Confidence Level: {summary['confidence_level']:.2f}")
print(f"  Key Themes: {summary['key_themes']}")
print(f"  Q&A Questions: {summary['qa_summary']['num_questions'] if summary['qa_summary'] else 0}")
if summary['qa_summary']:
    print(f"  Average Evasiveness: {summary['qa_summary']['average_evasiveness']:.2f}")
    print(f"  Common Topics: {summary['qa_summary']['common_topics']}")
```

This module provides comprehensive tools for financial text analytics. Continue with the remaining modules to build complete financial AI systems!

## Key Takeaways

- Financial text requires domain-specific NLP techniques
- Sentiment analysis in finance differs from general sentiment
- SEC filings contain valuable structured and unstructured data
- Earnings calls provide insights beyond financial numbers
- Entity extraction is crucial for building financial knowledge graphs
- Combining multiple text sources improves signal quality

## Next Steps

Next, explore quantitative finance basics and machine learning applications to trading in the following modules.

