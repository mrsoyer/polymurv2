# Implementation Guide: Twitter Sentiment Trading Bot

> Step-by-step implementation guide with production-ready code examples for building a Twitter sentiment trading bot for Polymarket

## Quick Start

### Prerequisites

```bash
# System requirements
Python 3.11+
PostgreSQL 14+
Redis 7+

# AWS/Cloud (optional but recommended)
AWS Account (EC2, RDS, CloudWatch)
```

### Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt:**
```txt
# Twitter API
tweepy==4.14.0

# Sentiment Analysis
vaderSentiment==3.3.2
transformers==4.35.0
torch==2.1.0

# Machine Learning
xgboost==2.0.2
lightgbm==4.1.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2

# Polymarket
py-clob-client==0.18.0
web3==6.11.3

# Database
psycopg2-binary==2.9.9
redis==5.0.1

# Message Queue
kafka-python==2.0.2

# Monitoring
prometheus-client==0.19.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiohttp==3.9.1
```

### Configuration

**.env file:**
```bash
# Twitter API
TWITTER_BEARER_TOKEN=your_bearer_token_here

# Polymarket API
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_api_secret
POLYMARKET_API_PASSPHRASE=your_passphrase
POLYMARKET_PRIVATE_KEY=your_wallet_private_key

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_bot
POSTGRES_USER=trader
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka (optional)
KAFKA_BROKER=localhost:9092

# Trading Parameters
INITIAL_CAPITAL=10000.0
MAX_POSITION_SIZE=0.10  # 10% of capital
MAX_DRAWDOWN=0.20  # 20%
MIN_CONFIDENCE=0.65  # Minimum signal confidence to trade

# Risk Management
STOP_LOSS_PCT=0.05  # 5% stop loss
TAKE_PROFIT_RATIO=2.0  # 2:1 reward-risk

# Sentiment Thresholds
VADER_THRESHOLD=0.5
FINBERT_THRESHOLD=0.7
MIN_TWEET_VOLUME=10  # Minimum tweets per signal
```

---

## Component 1: Twitter Stream Ingestion

### Basic Streaming Client

**twitter_stream.py:**
```python
import os
import json
import time
import requests
from typing import Callable
from dotenv import load_dotenv

load_dotenv()

class TwitterStreamClient:
    """Real-time Twitter stream client with retry logic"""

    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2/tweets/search/stream"
        self.rules_url = f"{self.base_url}/rules"

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def set_rules(self, rules: list):
        """
        Set stream filtering rules
        rules = [
            {"value": "bitcoin lang:en -is:retweet", "tag": "bitcoin"},
            {"value": "ethereum lang:en -is:retweet", "tag": "ethereum"}
        ]
        """
        # Delete existing rules
        current_rules = self.get_rules()
        if current_rules and 'data' in current_rules:
            rule_ids = [rule['id'] for rule in current_rules['data']]
            if rule_ids:
                self.delete_rules(rule_ids)

        # Add new rules
        payload = {"add": rules}
        response = requests.post(
            self.rules_url,
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_rules(self):
        """Get current stream rules"""
        response = requests.get(self.rules_url, headers=self._get_headers())
        return response.json()

    def delete_rules(self, rule_ids: list):
        """Delete stream rules by ID"""
        payload = {"delete": {"ids": rule_ids}}
        response = requests.post(
            self.rules_url,
            headers=self._get_headers(),
            json=payload
        )
        return response.json()

    def stream(self, callback: Callable, retry_max=5):
        """
        Start streaming tweets. Calls callback(tweet_data) for each tweet.
        Automatically retries on connection failures.
        """
        params = {
            "tweet.fields": "created_at,author_id,public_metrics,entities,lang",
            "user.fields": "username,verified,public_metrics",
            "expansions": "author_id"
        }

        retry_count = 0
        backoff_time = 1  # Start with 1 second backoff

        while retry_count < retry_max:
            try:
                response = requests.get(
                    self.base_url,
                    headers=self._get_headers(),
                    params=params,
                    stream=True,
                    timeout=90
                )

                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                    retry_count += 1
                    continue

                # Reset on successful connection
                retry_count = 0
                backoff_time = 1

                for line in response.iter_lines():
                    if line:
                        try:
                            tweet_data = json.loads(line)
                            if 'data' in tweet_data:
                                callback(tweet_data)
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {line}")

            except requests.exceptions.ChunkedEncodingError:
                print("Connection interrupted. Reconnecting...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)  # Cap at 60s
                retry_count += 1

            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)
                retry_count += 1

        print("Max retries exceeded. Exiting.")


# Example usage
if __name__ == "__main__":
    def on_tweet(tweet_data):
        tweet = tweet_data['data']
        print(f"New tweet: {tweet['text'][:100]}...")

    client = TwitterStreamClient(os.getenv('TWITTER_BEARER_TOKEN'))

    # Set rules
    rules = [
        {"value": "(bitcoin OR btc) lang:en -is:retweet has:hashtags", "tag": "bitcoin"},
        {"value": "(trump OR biden) lang:en -is:retweet", "tag": "politics"}
    ]
    client.set_rules(rules)

    # Start streaming
    client.stream(on_tweet)
```

### Advanced: Kafka Integration

**twitter_kafka_producer.py:**
```python
from kafka import KafkaProducer
import json

class TwitterKafkaProducer:
    """Push tweets to Kafka for distributed processing"""

    def __init__(self, kafka_broker: str, topic: str = 'tweets'):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            batch_size=16384,  # Batch for efficiency
            linger_ms=10  # Wait 10ms to batch tweets
        )
        self.topic = topic

    def send_tweet(self, tweet_data: dict):
        """Send tweet to Kafka topic"""
        # Extract relevant fields to reduce message size
        processed = {
            'id': tweet_data['data']['id'],
            'text': tweet_data['data']['text'],
            'created_at': tweet_data['data']['created_at'],
            'author_id': tweet_data['data']['author_id'],
            'public_metrics': tweet_data['data'].get('public_metrics', {}),
            'entities': tweet_data['data'].get('entities', {}),
            'tag': tweet_data.get('matching_rules', [{}])[0].get('tag', 'unknown')
        }

        # Include user data if available
        if 'includes' in tweet_data and 'users' in tweet_data['includes']:
            user = tweet_data['includes']['users'][0]
            processed['user'] = {
                'username': user['username'],
                'verified': user.get('verified', False),
                'followers': user['public_metrics'].get('followers_count', 0)
            }

        self.producer.send(self.topic, value=processed)

    def close(self):
        self.producer.flush()
        self.producer.close()


# Usage
if __name__ == "__main__":
    from twitter_stream import TwitterStreamClient
    import os

    kafka_producer = TwitterKafkaProducer(os.getenv('KAFKA_BROKER', 'localhost:9092'))

    def on_tweet(tweet_data):
        kafka_producer.send_tweet(tweet_data)
        print(f"Sent to Kafka: {tweet_data['data']['id']}")

    client = TwitterStreamClient(os.getenv('TWITTER_BEARER_TOKEN'))
    rules = [{"value": "bitcoin lang:en -is:retweet", "tag": "bitcoin"}]
    client.set_rules(rules)

    try:
        client.stream(on_tweet)
    except KeyboardInterrupt:
        kafka_producer.close()
```

---

## Component 2: Sentiment Analysis Engine

### Multi-Model Ensemble

**sentiment_analyzer.py:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time
from typing import Dict

class SentimentAnalyzer:
    """Ensemble sentiment analyzer with VADER + FinBERT"""

    def __init__(self, use_finbert: bool = True, device: str = 'cpu'):
        # VADER (fast, lexicon-based)
        self.vader = SentimentIntensityAnalyzer()

        # FinBERT (accurate, transformer-based)
        self.use_finbert = use_finbert
        if use_finbert:
            self.finbert_tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
            self.finbert_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.device = torch.device(device)
            self.finbert_model.to(self.device)
            self.finbert_model.eval()

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize tweet text"""
        import re
        import emoji

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Normalize mentions
        text = re.sub(r'@\w+', '@USER', text)

        # Convert emojis to text
        text = emoji.demojize(text)

        return text.strip()

    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores"""
        text = self.preprocess_text(text)
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu']
        }

    def analyze_finbert(self, text: str) -> Dict[str, float]:
        """Get FinBERT sentiment probabilities"""
        text = self.preprocess_text(text)

        inputs = self.finbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.finbert_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        return {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2])
        }

    def analyze(self, text: str) -> Dict[str, any]:
        """
        Run ensemble analysis
        Returns: {
            'vader_compound': float,
            'finbert_positive': float,
            'finbert_negative': float,
            'ensemble_score': float (-1 to +1),
            'confidence': float (0 to 1),
            'latency_ms': float
        }
        """
        start_time = time.time()

        # VADER (always run, fast)
        vader_scores = self.analyze_vader(text)

        # FinBERT (optional, slower but more accurate)
        if self.use_finbert and abs(vader_scores['compound']) > 0.3:
            # Only run FinBERT on non-neutral signals
            finbert_scores = self.analyze_finbert(text)
        else:
            finbert_scores = None

        # Ensemble scoring
        if finbert_scores:
            # Weighted average (FinBERT 70%, VADER 30%)
            finbert_sentiment = finbert_scores['positive'] - finbert_scores['negative']
            ensemble_score = 0.3 * vader_scores['compound'] + 0.7 * finbert_sentiment

            # Confidence based on model agreement
            confidence = 1.0 - abs(vader_scores['compound'] - finbert_sentiment)
        else:
            # VADER only
            ensemble_score = vader_scores['compound']
            confidence = abs(vader_scores['compound'])  # Strong signals = higher confidence

        latency_ms = (time.time() - start_time) * 1000

        return {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'finbert_positive': finbert_scores['positive'] if finbert_scores else None,
            'finbert_negative': finbert_scores['negative'] if finbert_scores else None,
            'finbert_neutral': finbert_scores['neutral'] if finbert_scores else None,
            'ensemble_score': ensemble_score,
            'confidence': confidence,
            'latency_ms': latency_ms
        }


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(use_finbert=True, device='cpu')

    tweets = [
        "Bitcoin is crashing! Sell everything now!",
        "Steady gains for ETH. Looking bullish long-term.",
        "Market is sideways today. No clear direction."
    ]

    for tweet in tweets:
        result = analyzer.analyze(tweet)
        print(f"\nTweet: {tweet}")
        print(f"Ensemble Score: {result['ensemble_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
```

### Batch Processing for Efficiency

**batch_sentiment.py:**
```python
def analyze_batch(analyzer, tweets: list, batch_size: int = 32) -> list:
    """Process tweets in batches for 10x speedup with FinBERT"""
    results = []

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]

        # VADER batch (vectorized)
        vader_scores = [analyzer.analyze_vader(t) for t in batch]

        # FinBERT batch (GPU parallel)
        if analyzer.use_finbert:
            texts = [analyzer.preprocess_text(t) for t in batch]
            inputs = analyzer.finbert_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(analyzer.device)

            with torch.no_grad():
                outputs = analyzer.finbert_model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            for j, (tweet, vader, finbert_prob) in enumerate(zip(batch, vader_scores, probs)):
                finbert_sentiment = finbert_prob[0] - finbert_prob[1]
                ensemble_score = 0.3 * vader['compound'] + 0.7 * finbert_sentiment
                confidence = 1.0 - abs(vader['compound'] - finbert_sentiment)

                results.append({
                    'text': tweet,
                    'ensemble_score': ensemble_score,
                    'confidence': confidence
                })
        else:
            # VADER only
            for tweet, vader in zip(batch, vader_scores):
                results.append({
                    'text': tweet,
                    'ensemble_score': vader['compound'],
                    'confidence': abs(vader['compound'])
                })

    return results
```

---

## Component 3: Signal Generation

### Feature Engineering

**feature_engineering.py:**
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    """Extract features from tweet sentiment streams"""

    def __init__(self, window_sizes: list = [5, 15, 60]):
        """
        window_sizes: Time windows in minutes for rolling aggregations
        """
        self.window_sizes = window_sizes

    def calculate_features(self, tweets_df: pd.DataFrame) -> dict:
        """
        Input: DataFrame with columns [timestamp, sentiment, author_id, followers, verified]
        Output: Feature dictionary for ML model
        """
        if len(tweets_df) == 0:
            return None

        tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
        tweets_df = tweets_df.sort_values('timestamp')

        features = {}

        # Sentiment statistics
        features['sentiment_mean'] = tweets_df['sentiment'].mean()
        features['sentiment_std'] = tweets_df['sentiment'].std()
        features['sentiment_median'] = tweets_df['sentiment'].median()

        # Volume features
        features['tweet_count'] = len(tweets_df)

        # User credibility features
        features['avg_followers'] = tweets_df['followers'].mean()
        features['verified_ratio'] = tweets_df['verified'].mean()

        # Rolling sentiment momentum
        for window in self.window_sizes:
            window_start = tweets_df['timestamp'].max() - timedelta(minutes=window)
            window_df = tweets_df[tweets_df['timestamp'] >= window_start]

            features[f'sentiment_{window}m'] = window_df['sentiment'].mean()
            features[f'volume_{window}m'] = len(window_df)

        # Momentum (change in sentiment)
        if len(self.window_sizes) >= 2:
            features['sentiment_momentum'] = (
                features[f'sentiment_{self.window_sizes[0]}m'] -
                features[f'sentiment_{self.window_sizes[1]}m']
            )

        # Volume spike detection
        recent_volume = features[f'volume_{self.window_sizes[0]}m']
        historical_volume = features[f'volume_{self.window_sizes[-1]}m'] / len(self.window_sizes)

        if historical_volume > 0:
            features['volume_ratio'] = recent_volume / historical_volume
        else:
            features['volume_ratio'] = 1.0

        features['is_volume_spike'] = 1 if features['volume_ratio'] > 2.0 else 0

        # Temporal features
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0

        return features

    def aggregate_tweets(self, tweets: list, lookback_minutes: int = 15) -> pd.DataFrame:
        """Convert raw tweet list to DataFrame for feature extraction"""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)

        processed_tweets = []
        for tweet in tweets:
            if datetime.fromisoformat(tweet['timestamp'].replace('Z', '+00:00')) >= cutoff_time:
                processed_tweets.append({
                    'timestamp': tweet['timestamp'],
                    'sentiment': tweet['sentiment'],
                    'author_id': tweet['author_id'],
                    'followers': tweet.get('user', {}).get('followers', 0),
                    'verified': tweet.get('user', {}).get('verified', False)
                })

        return pd.DataFrame(processed_tweets)
```

### Machine Learning Signal Generator

**signal_generator.py:**
```python
import xgboost as xgb
import pickle
from typing import Optional

class MLSignalGenerator:
    """Generate trading signals using trained ML model"""

    def __init__(self, model_path: str = 'model.pkl', threshold: float = 0.65):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.threshold = threshold

    def generate_signal(self, features: dict) -> dict:
        """
        Returns: {
            'action': 'BUY' | 'SELL' | 'HOLD',
            'confidence': float (0-1),
            'expected_return': float,
            'features': dict
        }
        """
        # Convert features to model input format
        feature_vector = [
            features['sentiment_mean'],
            features['sentiment_std'],
            features['sentiment_5m'],
            features['sentiment_15m'],
            features['sentiment_momentum'],
            features['volume_ratio'],
            features['is_volume_spike'],
            features['avg_followers'],
            features['verified_ratio'],
            features['hour_of_day'],
            features['day_of_week']
        ]

        # Predict (binary classification: 1 = price up, 0 = price down)
        dmatrix = xgb.DMatrix([feature_vector])
        probability = self.model.predict(dmatrix)[0]

        # Determine action
        if probability > self.threshold:
            action = 'BUY'
            confidence = probability
        elif probability < (1 - self.threshold):
            action = 'SELL'
            confidence = 1 - probability
        else:
            action = 'HOLD'
            confidence = 0.5

        return {
            'action': action,
            'confidence': confidence,
            'predicted_probability': probability,
            'features': features
        }


class RuleBasedSignalGenerator:
    """Simple rule-based signal generator (no ML training needed)"""

    def __init__(self,
                 sentiment_threshold: float = 0.5,
                 min_confidence: float = 0.6,
                 min_volume: int = 10):
        self.sentiment_threshold = sentiment_threshold
        self.min_confidence = min_confidence
        self.min_volume = min_volume

    def generate_signal(self, features: dict) -> dict:
        """Generate signal based on sentiment thresholds"""

        # Check minimum requirements
        if features['tweet_count'] < self.min_volume:
            return {'action': 'HOLD', 'reason': 'Insufficient volume'}

        sentiment = features['sentiment_mean']
        momentum = features.get('sentiment_momentum', 0)

        # Strong positive sentiment + momentum
        if sentiment > self.sentiment_threshold and momentum > 0.2:
            confidence = min(abs(sentiment) + abs(momentum), 1.0)
            if confidence >= self.min_confidence:
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'sentiment': sentiment,
                    'momentum': momentum
                }

        # Strong negative sentiment + momentum
        elif sentiment < -self.sentiment_threshold and momentum < -0.2:
            confidence = min(abs(sentiment) + abs(momentum), 1.0)
            if confidence >= self.min_confidence:
                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'sentiment': sentiment,
                    'momentum': momentum
                }

        return {'action': 'HOLD', 'reason': 'No strong signal'}
```

---

## Component 4: Risk Management

**risk_manager.py:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    """Manage position sizing, stop-loss, and drawdown limits"""

    def __init__(self):
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', 10000))
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.10))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', 0.20))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', 0.05))
        self.take_profit_ratio = float(os.getenv('TAKE_PROFIT_RATIO', 2.0))

    def calculate_position_size(self, signal_confidence: float, volatility: float = 0.02) -> float:
        """
        Calculate optimal position size using Kelly Criterion (fractional)
        """
        # Assume win rate = signal confidence
        win_rate = signal_confidence

        # Assume 2:1 reward-risk ratio
        avg_win = 0.02 * self.take_profit_ratio  # 4% win
        avg_loss = 0.02  # 2% loss

        # Kelly fraction: (p*b - q) / b
        kelly_fraction = (win_rate * (avg_win/avg_loss) - (1 - win_rate)) / (avg_win/avg_loss)

        # Apply fractional Kelly (25% of full Kelly for safety)
        fractional_kelly = max(0, kelly_fraction * 0.25)

        # Adjust by confidence
        adjusted_fraction = fractional_kelly * signal_confidence

        # Cap at max position size
        position_fraction = min(adjusted_fraction, self.max_position_size)

        # Calculate dollar amount
        position_size = self.current_capital * position_fraction

        return round(position_size, 2)

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop-loss price"""
        if side == 'BUY':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # SELL
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, stop_loss: float, side: str) -> float:
        """Calculate take-profit price (2:1 reward-risk)"""
        risk_amount = abs(entry_price - stop_loss)

        if side == 'BUY':
            return entry_price + (risk_amount * self.take_profit_ratio)
        else:  # SELL
            return entry_price - (risk_amount * self.take_profit_ratio)

    def check_drawdown(self) -> dict:
        """Check if drawdown limit is breached"""
        self.peak_capital = max(self.peak_capital, self.current_capital)
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if drawdown >= self.max_drawdown:
            return {
                'status': 'HALT',
                'drawdown': drawdown,
                'message': f'Max drawdown reached: {drawdown*100:.1f}%'
            }
        elif drawdown >= self.max_drawdown * 0.75:
            return {
                'status': 'REDUCE',
                'drawdown': drawdown,
                'position_multiplier': 0.5,
                'message': f'Drawdown warning: {drawdown*100:.1f}%'
            }
        else:
            return {
                'status': 'OK',
                'drawdown': drawdown,
                'position_multiplier': 1.0
            }

    def validate_trade(self, signal: dict, current_price: float) -> dict:
        """Validate if trade should be executed"""
        # Check drawdown
        drawdown_check = self.check_drawdown()
        if drawdown_check['status'] == 'HALT':
            return {'valid': False, 'reason': drawdown_check['message']}

        # Check signal confidence
        min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.65))
        if signal['confidence'] < min_confidence:
            return {'valid': False, 'reason': f'Low confidence: {signal["confidence"]:.2f}'}

        # Calculate position size
        position_size = self.calculate_position_size(signal['confidence'])
        position_size *= drawdown_check['position_multiplier']  # Reduce if in drawdown

        if position_size < 10:  # Minimum $10 position
            return {'valid': False, 'reason': 'Position size too small'}

        # Calculate risk parameters
        stop_loss = self.calculate_stop_loss(current_price, signal['action'])
        take_profit = self.calculate_take_profit(current_price, stop_loss, signal['action'])

        return {
            'valid': True,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': abs(current_price - stop_loss) * (position_size / current_price)
        }

    def update_capital(self, pnl: float):
        """Update capital after trade closes"""
        self.current_capital += pnl
```

---

## Component 5: Polymarket Integration

**polymarket_client.py:**
```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
import os
from dotenv import load_dotenv

load_dotenv()

class PolymarketClient:
    """Polymarket trading client"""

    def __init__(self):
        self.host = "https://clob.polymarket.com"
        self.chain_id = 137  # Polygon

        creds = ApiCreds(
            api_key=os.getenv('POLYMARKET_API_KEY'),
            api_secret=os.getenv('POLYMARKET_API_SECRET'),
            api_passphrase=os.getenv('POLYMARKET_API_PASSPHRASE')
        )

        self.client = ClobClient(
            self.host,
            key=os.getenv('POLYMARKET_PRIVATE_KEY'),
            chain_id=self.chain_id,
            creds=creds
        )

    def search_markets(self, keyword: str) -> list:
        """Search for markets by keyword"""
        markets = self.client.get_markets()
        return [m for m in markets if keyword.lower() in m['question'].lower()]

    def get_market_price(self, market_id: str, outcome: str = 'YES') -> float:
        """Get current market price"""
        market = self.client.get_market(market_id)

        for token in market['tokens']:
            if token['outcome'] == outcome:
                return float(token['price'])

        return None

    def place_order(self, market_id: str, side: str, size: float, price: float, outcome: str = 'YES') -> dict:
        """
        Place limit order
        side: 'BUY' or 'SELL'
        size: Dollar amount
        price: Probability (0.01 to 0.99)
        """
        # Get token ID for outcome
        market = self.client.get_market(market_id)
        token_id = next(t['token_id'] for t in market['tokens'] if t['outcome'] == outcome)

        # Create order
        order = self.client.create_order(
            token_id=token_id,
            side=side,
            size=size,
            price=price
        )

        # Submit order
        response = self.client.post_order(order)

        return {
            'order_id': response['orderID'],
            'status': response['status'],
            'market_id': market_id,
            'side': side,
            'size': size,
            'price': price
        }

    def get_order_status(self, order_id: str) -> dict:
        """Check order status"""
        return self.client.get_order(order_id)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel pending order"""
        return self.client.cancel_order(order_id)
```

---

## Complete Trading Bot

**trading_bot.py:**
```python
import time
import json
from datetime import datetime
from twitter_stream import TwitterStreamClient
from sentiment_analyzer import SentimentAnalyzer
from feature_engineering import FeatureEngineer
from signal_generator import RuleBasedSignalGenerator
from risk_manager import RiskManager
from polymarket_client import PolymarketClient
from database import Database
import os

class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self):
        self.twitter_client = TwitterStreamClient(os.getenv('TWITTER_BEARER_TOKEN'))
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert=True)
        self.feature_engineer = FeatureEngineer()
        self.signal_generator = RuleBasedSignalGenerator()
        self.risk_manager = RiskManager()
        self.polymarket_client = PolymarketClient()
        self.db = Database()

        self.tweet_buffer = []
        self.last_signal_time = None
        self.active_positions = {}

    def on_tweet(self, tweet_data):
        """Callback for each incoming tweet"""
        tweet = tweet_data['data']

        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze(tweet['text'])

        # Store tweet with sentiment
        processed_tweet = {
            'id': tweet['id'],
            'text': tweet['text'],
            'timestamp': tweet['created_at'],
            'author_id': tweet['author_id'],
            'sentiment': sentiment_result['ensemble_score'],
            'confidence': sentiment_result['confidence']
        }

        # Add user data if available
        if 'includes' in tweet_data:
            user = tweet_data['includes']['users'][0]
            processed_tweet['user'] = {
                'followers': user['public_metrics']['followers_count'],
                'verified': user.get('verified', False)
            }

        # Add to buffer
        self.tweet_buffer.append(processed_tweet)

        # Save to database
        self.db.save_tweet(processed_tweet)

        # Generate signal every 5 minutes
        if self.should_generate_signal():
            self.generate_and_execute_signal()

    def should_generate_signal(self) -> bool:
        """Check if enough time has passed to generate new signal"""
        if self.last_signal_time is None:
            return len(self.tweet_buffer) >= 10

        time_elapsed = (datetime.now() - self.last_signal_time).total_seconds()
        return time_elapsed >= 300  # 5 minutes

    def generate_and_execute_signal(self):
        """Generate trading signal and execute if valid"""
        print(f"\n[{datetime.now()}] Generating signal...")

        # Aggregate tweets
        tweets_df = self.feature_engineer.aggregate_tweets(
            self.tweet_buffer,
            lookback_minutes=15
        )

        if tweets_df.empty:
            print("No tweets in lookback window")
            return

        # Extract features
        features = self.feature_engineer.calculate_features(tweets_df)

        if features is None:
            print("Failed to extract features")
            return

        # Generate signal
        signal = self.signal_generator.generate_signal(features)

        print(f"Signal: {signal['action']}")
        print(f"Confidence: {signal.get('confidence', 0):.2f}")
        print(f"Tweet count: {features['tweet_count']}")
        print(f"Sentiment: {features['sentiment_mean']:.3f}")

        # Execute trade if signal is BUY or SELL
        if signal['action'] in ['BUY', 'SELL']:
            self.execute_trade(signal, features)

        self.last_signal_time = datetime.now()

        # Clear old tweets from buffer (keep last 1 hour)
        self.tweet_buffer = [
            t for t in self.tweet_buffer
            if (datetime.now() - datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))).seconds < 3600
        ]

    def execute_trade(self, signal: dict, features: dict):
        """Execute trade on Polymarket"""
        try:
            # Find relevant market (simplified - should be more sophisticated)
            markets = self.polymarket_client.search_markets('bitcoin')

            if not markets:
                print("No matching market found")
                return

            market = markets[0]
            market_id = market['condition_id']

            # Get current price
            current_price = self.polymarket_client.get_market_price(market_id)

            # Validate trade
            validation = self.risk_manager.validate_trade(signal, current_price)

            if not validation['valid']:
                print(f"Trade rejected: {validation['reason']}")
                return

            # Place order
            order_price = current_price + 0.01 if signal['action'] == 'BUY' else current_price - 0.01

            order = self.polymarket_client.place_order(
                market_id=market_id,
                side=signal['action'],
                size=validation['position_size'],
                price=order_price
            )

            print(f"\n✅ Order placed:")
            print(f"  Market: {market['question']}")
            print(f"  Side: {signal['action']}")
            print(f"  Size: ${validation['position_size']:.2f}")
            print(f"  Price: {order_price:.2f}")
            print(f"  Stop Loss: {validation['stop_loss']:.2f}")
            print(f"  Take Profit: {validation['take_profit']:.2f}")

            # Save trade to database
            self.db.save_trade({
                'order_id': order['order_id'],
                'market_id': market_id,
                'side': signal['action'],
                'entry_price': order_price,
                'size': validation['position_size'],
                'stop_loss': validation['stop_loss'],
                'take_profit': validation['take_profit'],
                'entry_time': datetime.now(),
                'signal': signal,
                'features': features
            })

            # Track active position
            self.active_positions[order['order_id']] = {
                'market_id': market_id,
                'entry_price': order_price,
                'stop_loss': validation['stop_loss'],
                'take_profit': validation['take_profit']
            }

        except Exception as e:
            print(f"❌ Trade execution failed: {e}")

    def monitor_positions(self):
        """Check active positions and close if stop-loss/take-profit hit"""
        for order_id, position in list(self.active_positions.items()):
            try:
                # Get current price
                current_price = self.polymarket_client.get_market_price(position['market_id'])

                # Check stop-loss
                if (position['side'] == 'BUY' and current_price <= position['stop_loss']) or \
                   (position['side'] == 'SELL' and current_price >= position['stop_loss']):
                    print(f"\n🛑 Stop-loss hit for {order_id}")
                    self.close_position(order_id, current_price, 'STOP_LOSS')

                # Check take-profit
                elif (position['side'] == 'BUY' and current_price >= position['take_profit']) or \
                     (position['side'] == 'SELL' and current_price <= position['take_profit']):
                    print(f"\n💰 Take-profit hit for {order_id}")
                    self.close_position(order_id, current_price, 'TAKE_PROFIT')

            except Exception as e:
                print(f"Error monitoring position {order_id}: {e}")

    def close_position(self, order_id: str, exit_price: float, reason: str):
        """Close position and update capital"""
        position = self.active_positions.pop(order_id)

        # Calculate PnL
        entry_price = position['entry_price']
        if position['side'] == 'BUY':
            pnl = (exit_price - entry_price) * position['size'] / entry_price
        else:
            pnl = (entry_price - exit_price) * position['size'] / entry_price

        # Update capital
        self.risk_manager.update_capital(pnl)

        # Update database
        self.db.update_trade(order_id, {
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'close_reason': reason
        })

        print(f"Position closed: PnL = ${pnl:.2f}")
        print(f"Current capital: ${self.risk_manager.current_capital:.2f}")

    def run(self):
        """Start trading bot"""
        print("🚀 Starting trading bot...")

        # Set Twitter stream rules
        rules = [
            {"value": "(bitcoin OR btc) lang:en -is:retweet", "tag": "bitcoin"}
        ]
        self.twitter_client.set_rules(rules)

        print(f"✅ Stream rules set: {rules}")
        print(f"💰 Initial capital: ${self.risk_manager.initial_capital:.2f}\n")

        # Start streaming tweets
        try:
            self.twitter_client.stream(self.on_tweet)
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n\n❌ Bot crashed: {e}")


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
```

**database.py (simplified):**
```python
import psycopg2
import json
from datetime import datetime
import os

class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST'),
            port=os.getenv('POSTGRES_PORT'),
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )

    def save_tweet(self, tweet: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tweets (tweet_id, text, author_id, created_at, sentiment, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING
            """, (
                tweet['id'],
                tweet['text'],
                tweet['author_id'],
                tweet['timestamp'],
                tweet['sentiment'],
                tweet['confidence']
            ))
            self.conn.commit()

    def save_trade(self, trade: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (
                    order_id, market_id, side, entry_price, size,
                    stop_loss, take_profit, entry_time, signal, features
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                trade['order_id'],
                trade['market_id'],
                trade['side'],
                trade['entry_price'],
                trade['size'],
                trade['stop_loss'],
                trade['take_profit'],
                trade['entry_time'],
                json.dumps(trade['signal']),
                json.dumps(trade['features'])
            ))
            self.conn.commit()

    def update_trade(self, order_id: str, updates: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE trades
                SET exit_price = %s, exit_time = %s, pnl = %s, close_reason = %s
                WHERE order_id = %s
            """, (
                updates['exit_price'],
                updates['exit_time'],
                updates['pnl'],
                updates['close_reason'],
                order_id
            ))
            self.conn.commit()
```

---

## Deployment

### Docker Compose Setup

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  trading_bot:
    build: .
    env_file: .env
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run bot
CMD ["python", "trading_bot.py"]
```

### AWS Deployment

**1. Launch EC2 Instance:**
```bash
# t3.medium (2 vCPU, 4GB RAM) for basic bot
# g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU) for FinBERT

aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx
```

**2. Install Dependencies:**
```bash
ssh ec2-user@your-instance

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repo
git clone https://github.com/yourrepo/sentiment-trading-bot.git
cd sentiment-trading-bot

# Set environment variables
nano .env  # Add API keys

# Start bot
docker-compose up -d
```

### Monitoring Setup

**1. View Logs:**
```bash
docker-compose logs -f trading_bot
```

**2. Check Metrics (Grafana):**
- Open http://your-instance-ip:3000
- Login (admin/admin)
- Import dashboard (included in repo)

**3. Alerts (CloudWatch):**
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def send_alert(metric_name, value):
    cloudwatch.put_metric_data(
        Namespace='TradingBot',
        MetricData=[{
            'MetricName': metric_name,
            'Value': value,
            'Unit': 'None'
        }]
    )
```

---

## Conclusion

This implementation guide provides a production-ready foundation for a Twitter sentiment trading bot. Key takeaways:

1. **Start Simple**: Use VADER + rule-based signals before adding ML complexity
2. **Test Thoroughly**: Backtest extensively before risking real capital
3. **Monitor Religiously**: Track latency, accuracy, and P&L continuously
4. **Manage Risk**: Never skip stop-losses or position sizing
5. **Iterate**: A/B test improvements and track performance metrics

**Next Steps:**
1. Backtest strategy on historical data
2. Paper trade for 2-4 weeks
3. Start with small capital ($1k-$5k)
4. Gradually increase as confidence grows
5. Continuously monitor and optimize

**Remember:** Past performance doesn't guarantee future results. Twitter sentiment is just one signal among many. Always trade responsibly and never risk more than you can afford to lose.
