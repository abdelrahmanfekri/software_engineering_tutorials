# Module 6: Alternative Data and NLP

## Table of Contents
1. [News Sentiment Analysis](#news-sentiment-analysis)
2. [Social Media Analytics](#social-media-analytics)
3. [SEC Filings Analysis](#sec-filings-analysis)
4. [Earnings Call Transcripts](#earnings-call-transcripts)
5. [Alternative Data Sources](#alternative-data-sources)
6. [Advanced NLP Techniques](#advanced-nlp-techniques)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## News Sentiment Analysis

### Real-Time News Aggregation

```python
import requests
import feedparser
from datetime import datetime
from typing import List, Dict
import pandas as pd

class NewsAggregator:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.sources = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'finnhub': 'https://finnhub.io/api/v1/news',
            'alpha_vantage': 'https://www.alphavantage.co/query'
        }
        
    def fetch_newsapi(self, query: str, from_date: str, to_date: str) -> List[Dict]:
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'publishedAt',
            'apiKey': self.api_keys['newsapi'],
            'language': 'en'
        }
        
        response = requests.get(self.sources['newsapi'], params=params)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return [{
                'title': article['title'],
                'description': article['description'],
                'content': article['content'],
                'url': article['url'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name']
            } for article in articles]
        
        return []
    
    def fetch_finnhub(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.api_keys['finnhub']
        }
        
        response = requests.get(self.sources['finnhub'], params=params)
        
        if response.status_code == 200:
            news = response.json()
            return [{
                'title': item['headline'],
                'summary': item['summary'],
                'url': item['url'],
                'publishedAt': datetime.fromtimestamp(item['datetime']).isoformat(),
                'source': item['source']
            } for item in news]
        
        return []
    
    def aggregate_news(
        self,
        query: str,
        from_date: str,
        to_date: str
    ) -> pd.DataFrame:
        all_news = []
        
        newsapi_results = self.fetch_newsapi(query, from_date, to_date)
        all_news.extend(newsapi_results)
        
        df = pd.DataFrame(all_news)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df = df.sort_values('publishedAt', ascending=False)
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        return df
```

### Financial Sentiment with Loughran-McDonald Lexicon

```python
import re
from typing import Dict, List
import numpy as np

class LoughranMcDonaldSentiment:
    def __init__(self):
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.uncertainty_words = self._load_uncertainty_words()
        self.litigious_words = self._load_litigious_words()
        
    def _load_positive_words(self) -> set:
        positive = {
            'profit', 'profits', 'profitable', 'profitability',
            'gain', 'gains', 'growth', 'increase', 'increased',
            'strong', 'strength', 'strengthen', 'strengthened',
            'improve', 'improved', 'improvement', 'improvements',
            'success', 'successful', 'successfully',
            'excellent', 'outstanding', 'exceptional',
            'benefit', 'benefits', 'beneficial',
            'opportunity', 'opportunities',
            'achieve', 'achieved', 'achievement', 'achievements'
        }
        return positive
    
    def _load_negative_words(self) -> set:
        negative = {
            'loss', 'losses', 'lost', 'losing',
            'decline', 'declined', 'declining', 'decrease', 'decreased',
            'weak', 'weakness', 'weaken', 'weakened',
            'deteriorate', 'deteriorated', 'deterioration',
            'fail', 'failed', 'failure', 'failures',
            'risk', 'risks', 'risky',
            'adverse', 'adversely',
            'poor', 'poorly',
            'negative', 'negatively',
            'challenge', 'challenges', 'challenging',
            'difficult', 'difficulty', 'difficulties'
        }
        return negative
    
    def _load_uncertainty_words(self) -> set:
        uncertainty = {
            'uncertain', 'uncertainty', 'uncertainties',
            'may', 'might', 'could', 'possibly', 'perhaps',
            'unclear', 'ambiguous', 'variable', 'volatility',
            'unpredictable', 'unknown', 'approximate',
            'depend', 'depends', 'depending', 'dependent'
        }
        return uncertainty
    
    def _load_litigious_words(self) -> set:
        litigious = {
            'lawsuit', 'lawsuits', 'litigation',
            'sue', 'sued', 'suing',
            'allegation', 'allegations', 'alleged',
            'complaint', 'complaints',
            'regulation', 'regulations', 'regulatory',
            'violation', 'violations', 'violate', 'violated'
        }
        return litigious
    
    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return words
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        words = self.preprocess_text(text)
        total_words = len(words)
        
        if total_words == 0:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'uncertainty': 0.0,
                'litigious': 0.0,
                'sentiment_score': 0.0
            }
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        uncertainty_count = sum(1 for word in words if word in self.uncertainty_words)
        litigious_count = sum(1 for word in words if word in self.litigious_words)
        
        sentiment_score = (positive_count - negative_count) / total_words
        
        return {
            'positive': positive_count / total_words,
            'negative': negative_count / total_words,
            'uncertainty': uncertainty_count / total_words,
            'litigious': litigious_count / total_words,
            'sentiment_score': sentiment_score
        }
```

### BERT-based Sentiment Models

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

class FinBERTSentiment:
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.labels = ['positive', 'negative', 'neutral']
        
    def predict_sentiment(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        sentiment = {
            label: float(prob)
            for label, prob in zip(self.labels, probabilities)
        }
        
        dominant_sentiment = max(sentiment, key=sentiment.get)
        
        return {
            'sentiment': dominant_sentiment,
            'probabilities': sentiment,
            'score': sentiment['positive'] - sentiment['negative']
        }
    
    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            for probs in probabilities:
                sentiment = {
                    label: float(prob)
                    for label, prob in zip(self.labels, probs)
                }
                
                dominant_sentiment = max(sentiment, key=sentiment.get)
                
                results.append({
                    'sentiment': dominant_sentiment,
                    'probabilities': sentiment,
                    'score': sentiment['positive'] - sentiment['negative']
                })
        
        return results
```

### Event-Driven Trading Signals

```python
class EventDrivenSignals:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_history = []
        
    def generate_signal(
        self,
        news_df: pd.DataFrame,
        price_data: pd.DataFrame,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        recent_news = news_df[
            news_df['publishedAt'] >= (pd.Timestamp.now() - pd.Timedelta(hours=lookback_hours))
        ]
        
        if len(recent_news) == 0:
            return {'signal': 0.0, 'confidence': 0.0}
        
        sentiments = []
        for _, article in recent_news.iterrows():
            text = f"{article['title']} {article.get('description', '')}"
            sentiment = self.sentiment_analyzer.predict_sentiment(text)
            sentiments.append(sentiment['score'])
        
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        sentiment_trend = self._calculate_sentiment_trend(sentiments)
        
        price_momentum = self._calculate_price_momentum(price_data)
        
        signal_strength = (avg_sentiment + sentiment_trend * 0.3) * (1 + price_momentum * 0.2)
        
        confidence = 1.0 / (1.0 + sentiment_std)
        
        return {
            'signal': np.clip(signal_strength, -1.0, 1.0),
            'confidence': confidence,
            'avg_sentiment': avg_sentiment,
            'sentiment_trend': sentiment_trend,
            'num_articles': len(recent_news)
        }
    
    def _calculate_sentiment_trend(self, sentiments: List[float]) -> float:
        if len(sentiments) < 2:
            return 0.0
        
        x = np.arange(len(sentiments))
        slope = np.polyfit(x, sentiments, 1)[0]
        
        return slope
    
    def _calculate_price_momentum(self, price_data: pd.DataFrame, periods: int = 20) -> float:
        if len(price_data) < periods:
            return 0.0
        
        recent_prices = price_data['close'].tail(periods)
        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        return momentum
```

## Social Media Analytics

### Twitter Sentiment Tracking

```python
import tweepy
from textblob import TextBlob
import re

class TwitterSentimentAnalyzer:
    def __init__(self, api_credentials: Dict[str, str]):
        auth = tweepy.OAuthHandler(
            api_credentials['consumer_key'],
            api_credentials['consumer_secret']
        )
        auth.set_access_token(
            api_credentials['access_token'],
            api_credentials['access_token_secret']
        )
        
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        
    def clean_tweet(self, tweet: str) -> str:
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'#\w+', '', tweet)
        tweet = re.sub(r'RT[\s]+', '', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)
        
        return tweet.strip()
    
    def analyze_tweet_sentiment(self, tweet: str) -> Dict[str, float]:
        cleaned_tweet = self.clean_tweet(tweet)
        
        analysis = TextBlob(cleaned_tweet)
        
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def fetch_tweets(
        self,
        query: str,
        count: int = 100,
        lang: str = 'en'
    ) -> pd.DataFrame:
        tweets = tweepy.Cursor(
            self.api.search_tweets,
            q=query,
            lang=lang,
            tweet_mode='extended'
        ).items(count)
        
        tweet_data = []
        for tweet in tweets:
            text = tweet.full_text
            sentiment = self.analyze_tweet_sentiment(text)
            
            tweet_data.append({
                'created_at': tweet.created_at,
                'text': text,
                'user': tweet.user.screen_name,
                'followers': tweet.user.followers_count,
                'retweets': tweet.retweet_count,
                'likes': tweet.favorite_count,
                'sentiment': sentiment['sentiment'],
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity']
            })
        
        return pd.DataFrame(tweet_data)
    
    def calculate_social_sentiment_index(self, tweets_df: pd.DataFrame) -> float:
        if len(tweets_df) == 0:
            return 0.0
        
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for _, tweet in tweets_df.iterrows():
            weight = (
                np.log1p(tweet['followers']) * 0.4 +
                np.log1p(tweet['retweets']) * 0.3 +
                np.log1p(tweet['likes']) * 0.3
            )
            
            weighted_sentiment += tweet['polarity'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sentiment / total_weight
```

### Reddit Sentiment Analysis

```python
import praw
from datetime import datetime

class RedditSentimentAnalyzer:
    def __init__(self, credentials: Dict[str, str]):
        self.reddit = praw.Reddit(
            client_id=credentials['client_id'],
            client_secret=credentials['client_secret'],
            user_agent=credentials['user_agent']
        )
        
    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        time_filter: str = 'day'
    ) -> List[Dict]:
        subreddit = self.reddit.subreddit(subreddit_name)
        
        posts = []
        for submission in subreddit.hot(limit=limit):
            posts.append({
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'created_utc': datetime.fromtimestamp(submission.created_utc),
                'url': submission.url,
                'id': submission.id
            })
        
        return posts
    
    def fetch_post_comments(
        self,
        post_id: str,
        limit: int = 100
    ) -> List[Dict]:
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        
        comments = []
        for comment in submission.comments.list()[:limit]:
            if hasattr(comment, 'body'):
                comments.append({
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'author': str(comment.author) if comment.author else '[deleted]'
                })
        
        return comments
    
    def analyze_wallstreetbets_sentiment(
        self,
        ticker: str,
        limit: int = 100
    ) -> Dict[str, any]:
        posts = self.fetch_subreddit_posts('wallstreetbets', limit=limit)
        
        relevant_posts = [
            post for post in posts
            if ticker.upper() in post['title'].upper() or 
               ticker.upper() in post['selftext'].upper()
        ]
        
        if not relevant_posts:
            return {
                'ticker': ticker,
                'sentiment': 'neutral',
                'score': 0.0,
                'num_posts': 0,
                'avg_upvote_ratio': 0.0
            }
        
        total_score = sum(post['score'] for post in relevant_posts)
        avg_upvote_ratio = np.mean([post['upvote_ratio'] for post in relevant_posts])
        
        sentiment_score = (avg_upvote_ratio - 0.5) * 2
        
        return {
            'ticker': ticker,
            'sentiment': 'bullish' if sentiment_score > 0.2 else 'bearish' if sentiment_score < -0.2 else 'neutral',
            'score': sentiment_score,
            'num_posts': len(relevant_posts),
            'avg_upvote_ratio': avg_upvote_ratio,
            'total_engagement': total_score
        }
```

## SEC Filings Analysis

### 10-K/10-Q Automated Parsing

```python
import requests
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import re

class SECFilingsParser:
    def __init__(self, company_name: str, email: str):
        self.dl = Downloader(company_name, email)
        self.base_url = 'https://www.sec.gov'
        
    def download_filings(
        self,
        ticker: str,
        filing_type: str,
        num_filings: int = 5
    ):
        self.dl.get(filing_type, ticker, amount=num_filings)
    
    def parse_10k(self, file_path: str) -> Dict[str, str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        sections = {
            'item_1': self._extract_section(soup, 'Item 1', 'Business'),
            'item_1a': self._extract_section(soup, 'Item 1A', 'Risk Factors'),
            'item_7': self._extract_section(soup, 'Item 7', 'Management'),
            'item_8': self._extract_section(soup, 'Item 8', 'Financial Statements')
        }
        
        return sections
    
    def _extract_section(
        self,
        soup: BeautifulSoup,
        item_number: str,
        item_name: str
    ) -> str:
        patterns = [
            rf'{item_number}[\.\s]+{item_name}',
            rf'{item_number}[\s]*{item_name}',
        ]
        
        text = soup.get_text()
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_idx = match.start()
                
                next_item = re.search(r'Item\s+\d+[A-Z]?[\.\s]+', text[start_idx+len(match.group()):], re.IGNORECASE)
                
                if next_item:
                    end_idx = start_idx + len(match.group()) + next_item.start()
                    return text[start_idx:end_idx].strip()
                else:
                    return text[start_idx:start_idx+50000].strip()
        
        return ""
    
    def extract_risk_factors(self, file_path: str) -> List[str]:
        sections = self.parse_10k(file_path)
        risk_section = sections.get('item_1a', '')
        
        risk_paragraphs = re.split(r'\n\s*\n', risk_section)
        
        risks = [
            para.strip() for para in risk_paragraphs
            if len(para.strip()) > 100 and 'risk' in para.lower()
        ]
        
        return risks
```

### MD&A Sentiment Extraction

```python
class MDASentimentAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = LoughranMcDonaldSentiment()
        
    def extract_mda(self, filing_path: str) -> str:
        parser = SECFilingsParser("Company", "email@example.com")
        sections = parser.parse_10k(filing_path)
        
        return sections.get('item_7', '')
    
    def analyze_mda_sentiment(self, mda_text: str) -> Dict[str, any]:
        sentiment = self.sentiment_analyzer.analyze_sentiment(mda_text)
        
        forward_looking = self._extract_forward_looking_statements(mda_text)
        
        tone_change = self._compare_with_previous(mda_text)
        
        return {
            'sentiment': sentiment,
            'forward_looking_statements': len(forward_looking),
            'tone_change': tone_change,
            'risk_level': sentiment['uncertainty'] + sentiment['litigious']
        }
    
    def _extract_forward_looking_statements(self, text: str) -> List[str]:
        forward_keywords = [
            'will', 'expect', 'anticipate', 'believe', 'estimate',
            'intend', 'may', 'plan', 'project', 'could', 'should'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        
        forward_looking = [
            sent.strip() for sent in sentences
            if any(keyword in sent.lower() for keyword in forward_keywords) and
               len(sent.strip()) > 50
        ]
        
        return forward_looking[:10]
    
    def _compare_with_previous(self, current_text: str) -> float:
        return 0.0
```

## Earnings Call Transcripts

### Earnings Call Processor

```python
from transformers import pipeline

class EarningsCallAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        
    def parse_transcript(self, transcript: str) -> Dict[str, List[str]]:
        lines = transcript.split('\n')
        
        presentation = []
        qa = []
        
        in_qa = False
        
        for line in lines:
            if 'question-and-answer' in line.lower() or 'q&a' in line.lower():
                in_qa = True
                continue
            
            if in_qa:
                qa.append(line)
            else:
                presentation.append(line)
        
        return {
            'presentation': '\n'.join(presentation),
            'qa': '\n'.join(qa)
        }
    
    def analyze_management_tone(self, presentation_text: str) -> Dict[str, float]:
        sentences = re.split(r'[.!?]+', presentation_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        sentiments = []
        for sentence in sentences[:50]:
            try:
                result = self.sentiment_analyzer(sentence[:512])[0]
                score = result['score'] if result['label'] == 'positive' else -result['score']
                sentiments.append(score)
            except:
                continue
        
        return {
            'avg_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'sentiment_std': np.std(sentiments) if sentiments else 0.0,
            'positive_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments) if sentiments else 0.0
        }
    
    def extract_key_metrics(self, transcript: str) -> Dict[str, List[str]]:
        metric_patterns = {
            'revenue': r'revenue.*?\$?\d+\.?\d*\s*(?:million|billion)',
            'earnings': r'(?:eps|earnings per share).*?\$?\d+\.?\d*',
            'guidance': r'guidance.*?\$?\d+\.?\d*\s*(?:million|billion)',
            'margin': r'margin.*?\d+\.?\d*%'
        }
        
        metrics = {}
        
        for metric_name, pattern in metric_patterns.items():
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            metrics[metric_name] = matches
        
        return metrics
    
    def summarize_earnings_call(self, transcript: str, max_length: int = 200) -> str:
        parsed = self.parse_transcript(transcript)
        
        presentation_summary = self.summarizer(
            parsed['presentation'][:1024],
            max_length=max_length,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        
        return presentation_summary
```

## Alternative Data Sources

### Satellite Imagery Analysis

```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

class SatelliteImageryAnalyzer:
    def __init__(self):
        self.parking_lot_detector = None
        
    def analyze_parking_lot(self, image_path: str) -> Dict[str, any]:
        image = cv2.imread(image_path)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blur, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        car_contours = [
            cnt for cnt in contours
            if 1000 < cv2.contourArea(cnt) < 10000
        ]
        
        return {
            'estimated_cars': len(car_contours),
            'total_area': image.shape[0] * image.shape[1],
            'occupancy_rate': len(car_contours) / (image.shape[0] * image.shape[1] / 5000)
        }
    
    def track_construction_progress(
        self,
        image_paths: List[str]
    ) -> List[Dict[str, float]]:
        progress = []
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            lower_construction = np.array([0, 0, 100])
            upper_construction = np.array([180, 50, 255])
            
            mask = cv2.inRange(hsv, lower_construction, upper_construction)
            
            construction_ratio = np.sum(mask > 0) / mask.size
            
            progress.append({
                'image': image_path,
                'construction_ratio': construction_ratio
            })
        
        return progress
```

### Web Scraping Techniques

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class WebScraper:
    def __init__(self, headless: bool = True):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        
        self.driver = webdriver.Chrome(options=options)
        
    def scrape_product_prices(self, url: str) -> List[Dict[str, any]]:
        self.driver.get(url)
        
        time.sleep(2)
        
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "product")))
        
        products = self.driver.find_elements(By.CLASS_NAME, "product")
        
        product_data = []
        for product in products:
            try:
                name = product.find_element(By.CLASS_NAME, "product-name").text
                price = product.find_element(By.CLASS_NAME, "product-price").text
                
                price_value = float(re.sub(r'[^\d.]', '', price))
                
                product_data.append({
                    'name': name,
                    'price': price_value
                })
            except:
                continue
        
        return product_data
    
    def close(self):
        self.driver.quit()
```

## Advanced NLP Techniques

### Named Entity Recognition for Finance

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class FinancialNER:
    def __init__(self):
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple"
        )
        
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        entities = self.ner_pipeline(text)
        
        financial_entities = []
        
        for entity in entities:
            financial_entities.append({
                'text': entity['word'],
                'label': entity['entity_group'],
                'score': entity['score']
            })
        
        return financial_entities
    
    def extract_company_mentions(self, text: str) -> List[str]:
        entities = self.extract_entities(text)
        
        companies = [
            entity['text'] for entity in entities
            if entity['label'] == 'ORG'
        ]
        
        return list(set(companies))
```

### Knowledge Graph Construction

```python
import networkx as nx
from typing import List, Tuple

class FinancialKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.ner = FinancialNER()
        
    def add_document(self, text: str, doc_id: str):
        entities = self.ner.extract_entities(text)
        
        for entity in entities:
            entity_text = entity['text']
            entity_type = entity['label']
            
            if not self.graph.has_node(entity_text):
                self.graph.add_node(
                    entity_text,
                    type=entity_type,
                    documents={doc_id}
                )
            else:
                self.graph.nodes[entity_text]['documents'].add(doc_id)
        
        for i in range(len(entities) - 1):
            entity1 = entities[i]['text']
            entity2 = entities[i + 1]['text']
            
            if self.graph.has_edge(entity1, entity2):
                self.graph[entity1][entity2]['weight'] += 1
            else:
                self.graph.add_edge(entity1, entity2, weight=1, relation='mentioned_with')
    
    def find_related_entities(self, entity: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        if not self.graph.has_node(entity):
            return []
        
        related = {}
        
        for neighbor in nx.single_source_shortest_path_length(self.graph, entity, cutoff=max_distance):
            if neighbor != entity:
                distance = nx.shortest_path_length(self.graph, entity, neighbor)
                related[neighbor] = 1.0 / (distance + 1)
        
        return sorted(related.items(), key=lambda x: x[1], reverse=True)
    
    def get_entity_importance(self, entity: str) -> float:
        if not self.graph.has_node(entity):
            return 0.0
        
        pagerank = nx.pagerank(self.graph)
        
        return pagerank.get(entity, 0.0)
```

## PhD-Level Research Topics

### Multimodal Learning (Text + Time Series)

```python
import torch
import torch.nn as nn

class MultimodalFinanceModel(nn.Module):
    def __init__(
        self,
        text_embedding_dim: int,
        ts_input_size: int,
        ts_hidden_size: int,
        output_size: int
    ):
        super(MultimodalFinanceModel, self).__init__()
        
        from transformers import BertModel
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.ts_encoder = nn.LSTM(
            input_size=ts_input_size,
            hidden_size=ts_hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        combined_dim = 768 + ts_hidden_size
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_dim // 2, output_size)
        )
        
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        time_series: torch.Tensor
    ) -> torch.Tensor:
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_embedding = text_outputs.pooler_output
        
        _, (ts_hidden, _) = self.ts_encoder(time_series)
        ts_embedding = ts_hidden[-1]
        
        combined = torch.cat([text_embedding, ts_embedding], dim=1)
        
        output = self.fusion_layer(combined)
        
        return output
```

### Cross-Lingual Financial NLP

```python
from transformers import MarianMTModel, MarianTokenizer

class CrossLingualFinancialAnalysis:
    def __init__(self):
        self.translation_models = {}
        self.tokenizers = {}
        
        languages = ['es', 'fr', 'de', 'zh']
        for lang in languages:
            model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
            self.translation_models[lang] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[lang] = MarianTokenizer.from_pretrained(model_name)
        
        self.sentiment_analyzer = FinBERTSentiment()
        
    def translate_to_english(self, text: str, source_lang: str) -> str:
        if source_lang not in self.translation_models:
            return text
        
        tokenizer = self.tokenizers[source_lang]
        model = self.translation_models[source_lang]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        translated = model.generate(**inputs)
        
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return translated_text
    
    def analyze_multilingual_sentiment(
        self,
        texts: List[Dict[str, str]]
    ) -> List[Dict[str, any]]:
        results = []
        
        for item in texts:
            text = item['text']
            lang = item['language']
            
            if lang != 'en':
                text = self.translate_to_english(text, lang)
            
            sentiment = self.sentiment_analyzer.predict_sentiment(text)
            
            results.append({
                'original_text': item['text'],
                'language': lang,
                'translated_text': text if lang != 'en' else None,
                'sentiment': sentiment
            })
        
        return results
```

## Implementation Frameworks

### Complete NLP Pipeline

```python
class ComprehensiveNLPPipeline:
    def __init__(self):
        self.news_aggregator = None
        self.sentiment_analyzer = FinBERTSentiment()
        self.ner = FinancialNER()
        self.knowledge_graph = FinancialKnowledgeGraph()
        
    def process_news_article(self, article: Dict[str, str]) -> Dict[str, any]:
        text = f"{article['title']} {article.get('content', '')}"
        
        sentiment = self.sentiment_analyzer.predict_sentiment(text)
        
        entities = self.ner.extract_entities(text)
        
        self.knowledge_graph.add_document(text, article.get('url', ''))
        
        return {
            'article_id': article.get('url', ''),
            'sentiment': sentiment,
            'entities': entities,
            'published_at': article.get('publishedAt'),
            'source': article.get('source')
        }
    
    def generate_trading_signal(
        self,
        ticker: str,
        processed_articles: List[Dict[str, any]]
    ) -> Dict[str, float]:
        relevant_articles = [
            article for article in processed_articles
            if ticker.upper() in str(article.get('entities', []))
        ]
        
        if not relevant_articles:
            return {'signal': 0.0, 'confidence': 0.0}
        
        sentiments = [article['sentiment']['score'] for article in relevant_articles]
        
        avg_sentiment = np.mean(sentiments)
        sentiment_consistency = 1.0 - np.std(sentiments)
        
        entity_importance = self.knowledge_graph.get_entity_importance(ticker)
        
        signal = avg_sentiment * (1 + entity_importance)
        confidence = sentiment_consistency * min(len(relevant_articles) / 10, 1.0)
        
        return {
            'signal': np.clip(signal, -1.0, 1.0),
            'confidence': confidence,
            'num_articles': len(relevant_articles),
            'avg_sentiment': avg_sentiment
        }
```
