# APIs & Tools for Twitter Sentiment Trading

> Comprehensive guide to APIs, libraries, and tools for building Twitter sentiment trading bots in 2026

## Twitter/X API v2

### Overview

Twitter's API v2 is the primary data source for real-time tweet access. As of 2026, Twitter (now X) offers tiered access with significant pricing differences based on usage needs.

### Pricing Tiers (2026)

| Tier | Monthly Cost | Tweet Read Limit | Tweet Write Limit | Streaming | Use Case |
|------|--------------|------------------|-------------------|-----------|----------|
| **Free** | $0 | 0 (read disabled) | 1,500 tweets | No | Unusable for trading bots |
| **Basic** | $100 | 10,000 tweets | 50,000 tweets | Yes (filtered) | Small-scale testing |
| **Pro** | $5,000 | 2,000,000 tweets | Unlimited | Yes (full) | Medium-scale trading |
| **Enterprise** | $42,000+ | Unlimited | Unlimited | Yes (full) | High-frequency trading |

**Critical Note:** The Free tier no longer supports reading tweets (as of 2023), making it unusable for sentiment analysis. The Basic tier's 10,000 tweet limit can be exhausted in hours for active keywords, making Pro tier ($5k/mo) the practical minimum for serious trading operations.

### Rate Limits by Endpoint

#### Filtered Stream API (Real-Time)

**Endpoint:** `GET /2/tweets/search/stream`

**Rate Limits:**
- **Basic:** 50 requests/15-min window
- **Pro:** 300 requests/15-min window
- **Enterprise:** Custom (typically 1000+/15-min)

**Monthly Post Consumption:**
- **Basic:** 10,000 tweets/month
- **Pro:** 2,000,000 tweets/month

**Latency:** 1-5 seconds from tweet publication

**Use Case:** Optimal for real-time trading signals. Tweets matching your rules are pushed to your connection immediately.

**Example Setup:**
```python
import requests

BEARER_TOKEN = 'your_bearer_token'

def create_stream_rules():
    """Define which tweets to receive"""
    rules = [
        {"value": "(bitcoin OR btc) lang:en -is:retweet", "tag": "bitcoin"},
        {"value": "(ethereum OR eth) lang:en -is:retweet", "tag": "ethereum"},
        {"value": "from:elonmusk", "tag": "elon_tweets"}
    ]

    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json={"add": rules}
    )
    return response.json()

def connect_to_stream():
    """Maintain persistent connection to receive tweets"""
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    # Request tweet fields for sentiment analysis
    params = {
        "tweet.fields": "created_at,author_id,public_metrics,entities",
        "user.fields": "username,verified,public_metrics",
        "expansions": "author_id"
    }

    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream",
        headers=headers,
        params=params,
        stream=True
    )

    for line in response.iter_lines():
        if line:
            tweet_data = json.loads(line)
            process_tweet(tweet_data)  # Your sentiment analysis

# Start streaming
create_stream_rules()
connect_to_stream()
```

**Rule Syntax (Filtered Stream):**
- `bitcoin OR btc`: Match either term
- `-is:retweet`: Exclude retweets
- `lang:en`: English only
- `has:links`: Tweets with URLs
- `from:username`: Specific user
- Combine operators: `(crypto OR bitcoin) lang:en -is:retweet has:media`

**Best Practices:**
- Exclude retweets (avoid duplicate signals)
- Filter by language (sentiment models are language-specific)
- Use specific keywords (broad terms exhaust quota quickly)
- Monitor rule match count (adjust if quota drains too fast)

#### Recent Search API (Historical/Polling)

**Endpoint:** `GET /2/tweets/search/recent`

**Rate Limits:**
- **Basic:** 15 requests/15-min window (450 tweets max)
- **Pro:** 300 requests/15-min window (9,000 tweets max)

**Lookback Window:** 7 days maximum

**Latency:** Depends on polling frequency (typically 30-60 seconds)

**Use Case:** Fallback for lower-frequency trading, backtesting, or when streaming quota is exhausted.

**Example:**
```python
def search_recent_tweets(query, max_results=100):
    """Poll for recent tweets matching query"""
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    params = {
        "query": query,
        "max_results": max_results,  # 10-100 per request
        "tweet.fields": "created_at,author_id,public_metrics",
        "user.fields": "verified,public_metrics",
        "expansions": "author_id"
    }

    response = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        headers=headers,
        params=params
    )

    return response.json()

# Example: Poll every 60 seconds
import time
while True:
    tweets = search_recent_tweets("(trump OR biden) -is:retweet", max_results=100)
    for tweet in tweets['data']:
        process_tweet(tweet)
    time.sleep(60)
```

**Cost Comparison:**
- Streaming: Continuous connection, near-instant tweets, but counts against monthly quota
- Search: Polling-based, 30-60s delay, rate-limited per 15-min window

**Recommendation:** Use Streaming for real-time HFT, Search for medium-frequency strategies or as backup.

### Authentication

Twitter API v2 uses Bearer Token authentication (OAuth 2.0):

```python
import os

BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN')

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

# All API requests include this header
response = requests.get(url, headers=headers, params=params)
```

**Security Best Practices:**
- Store tokens in environment variables (never hardcode)
- Use AWS Secrets Manager or similar for production
- Rotate tokens quarterly
- Monitor for unauthorized usage (Twitter dashboard)

### Cost-Benefit Analysis

**Scenario 1: Day Trading ($10k capital)**

Assumptions:
- Basic tier ($100/mo)
- 10k tweets/mo = ~330 tweets/day
- Target: 5% monthly return = $500/mo profit
- **Result:** Profitable ($500 profit - $100 API cost = $400 net)

**Scenario 2: Active Trading ($50k capital)**

Assumptions:
- Pro tier ($5,000/mo)
- 2M tweets/mo = ~66k tweets/day
- Target: 8% monthly return = $4,000/mo profit
- **Result:** Unprofitable ($4k profit - $5k API cost = -$1k loss)
- **Required capital:** $62.5k to break even (8% of $62.5k = $5k)

**Scenario 3: High-Frequency ($200k capital)**

Assumptions:
- Enterprise tier ($42,000/mo)
- Unlimited tweets
- Target: 10% monthly return = $20,000/mo profit
- **Result:** Unprofitable ($20k profit - $42k API cost = -$22k loss)
- **Required capital:** $420k to break even (10% of $420k = $42k)

**Key Insight:** Twitter API costs create a high barrier to profitability. Most successful bots either:
1. Use Basic tier with selective filtering (< $100/mo cost)
2. Have large capital bases (>$500k to justify Enterprise tier)
3. Combine Twitter with other cheaper data sources

### API Alternatives

#### 1. RapidAPI / TwitterAPI.io

**Cost:** $99-$399/mo (cheaper than official Pro tier)

**Pros:**
- No tweet read limits
- Unified API for multiple platforms
- Often includes historical data access

**Cons:**
- Third-party reliability (not official Twitter)
- Potential ToS violations (gray area)
- May have higher latency

**Example:**
```python
import requests

rapidapi_key = 'your_rapidapi_key'
url = "https://twitter135.p.rapidapi.com/Search/"

querystring = {"q":"bitcoin","count":"20"}

headers = {
    "X-RapidAPI-Key": rapidapi_key,
    "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)
```

#### 2. Scraping (Use with Caution)

**Cost:** $0 (just compute)

**Legal Risk:** High - violates Twitter ToS, risk of IP ban

**Tools:**
- snscrape (Python library)
- nitter instances (unofficial Twitter frontend)

**Recommendation:** Only for research/backtesting, not production trading.

#### 3. Social Data Providers

**Companies:**
- StockTwits API (free tier available)
- Sentdex (sentiment data feed)
- LunarCrush (crypto social metrics)

**Pros:**
- Pre-processed sentiment scores
- Lower cost than raw Twitter API
- No need for NLP infrastructure

**Cons:**
- Limited customization
- Potential signal lag (pre-aggregated data)

---

## Sentiment Analysis Libraries

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Type:** Lexicon-based (rule-based)

**Speed:** 2-5ms per tweet (extremely fast)

**Accuracy:** 60-70% on financial tweets

**Best For:** Real-time trading (low latency), initial signal filtering

**Installation:**
```bash
pip install vaderSentiment
```

**Usage:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

tweet = "Bitcoin is mooning! 🚀 Best investment I've made this year!"
scores = analyzer.polarity_scores(tweet)

print(scores)
# {'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.8658}

# Use compound score for trading signal
if scores['compound'] > 0.5:
    signal = 'BULLISH'
elif scores['compound'] < -0.5:
    signal = 'BEARISH'
else:
    signal = 'NEUTRAL'
```

**Compound Score Interpretation:**
- `>= 0.05`: Positive
- `-0.05 to 0.05`: Neutral
- `<= -0.05`: Negative

**Advantages:**
- No training data required
- CPU-only (low compute cost)
- Handles emojis and capitalization (GREAT vs great)
- Understands negation ("not good" = negative)

**Limitations:**
- Context-insensitive ("crash" = negative, even in "crypto crashed through resistance")
- Financial jargon often misclassified
- Sarcasm not detected

**Performance Optimization:**
```python
# Pre-load analyzer (avoid reinitialization)
analyzer = SentimentIntensityAnalyzer()

# Batch processing for efficiency
def batch_analyze(tweets):
    return [analyzer.polarity_scores(t)['compound'] for t in tweets]

tweets = ["tweet1", "tweet2", "tweet3"]
sentiments = batch_analyze(tweets)
```

### 2. FinBERT

**Type:** Transformer-based (BERT fine-tuned on financial text)

**Speed:** 50-100ms per tweet (GPU), 200-500ms (CPU)

**Accuracy:** 85-92% on financial sentiment

**Best For:** High-accuracy validation of strong signals

**Installation:**
```bash
pip install transformers torch
```

**Usage:**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Labels: positive, negative, neutral
    labels = ['positive', 'negative', 'neutral']
    scores = {labels[i]: probs[0][i].item() for i in range(3)}

    return scores

tweet = "Fed rate hike could trigger market correction"
sentiment = predict_sentiment(tweet)
print(sentiment)
# {'positive': 0.05, 'negative': 0.82, 'neutral': 0.13}
```

**GPU Optimization:**
```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict_sentiment_gpu(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()[0]  # Move back to CPU for post-processing
```

**Batch Inference (10x speedup):**
```python
def predict_batch(tweets, batch_size=32):
    sentiments = []

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiments.extend(probs.cpu().numpy())

    return sentiments
```

**Cost Considerations:**
- CPU-only: ~$50/mo (t3.medium AWS EC2)
- GPU: ~$200/mo (g4dn.xlarge AWS EC2, T4 GPU)
- Inference time: 50ms (GPU) vs 300ms (CPU)

### 3. TextBlob

**Type:** Lexicon-based (simpler than VADER)

**Speed:** 5-10ms per tweet

**Accuracy:** 55-65% on financial tweets

**Best For:** Subjectivity analysis (fact vs opinion detection)

**Installation:**
```bash
pip install textblob
python -m textblob.download_corpora
```

**Usage:**
```python
from textblob import TextBlob

tweet = "I think Bitcoin will reach $100k by year end"
blob = TextBlob(tweet)

print(blob.sentiment)
# Sentiment(polarity=0.0, subjectivity=0.5)

# Polarity: -1 (negative) to +1 (positive)
# Subjectivity: 0 (objective) to 1 (subjective)

if blob.sentiment.subjectivity > 0.5:
    print("Opinion-based (less reliable)")
else:
    print("Fact-based (more reliable)")
```

**Use Case in Trading:**
- Filter out highly subjective tweets (opinion vs news)
- Prioritize objective statements for trading signals

**Limitations:**
- Lower accuracy than VADER/FinBERT
- Not specialized for financial text
- No emoji/capitalization handling

### 4. RoBERTa + Custom Fine-Tuning

**Type:** Transformer-based (state-of-the-art)

**Speed:** 80-150ms per tweet (GPU)

**Accuracy:** 88-95% on crypto tweets (when fine-tuned)

**Best For:** Maximum accuracy for well-funded operations

**Setup:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load pre-trained RoBERTa
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Fine-tune on your labeled tweets
def fine_tune(train_data):
    """
    train_data: DataFrame with 'text' and 'label' columns
    label: 0 (negative), 1 (neutral), 2 (positive)
    """

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_data = train_data.apply(tokenize_function, axis=1)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=500,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()

# Use fine-tuned model
def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()[0]
```

**Training Data Requirements:**
- Minimum: 5,000 labeled tweets
- Recommended: 20,000+ labeled tweets
- Labeling cost: $0.01-$0.05 per tweet (Amazon Mechanical Turk)

**Research Findings:**
- RoBERTa achieved 2.01% MAPE (vs 3.5% for VADER) in Bitcoin price prediction
- Fine-tuning on crypto-specific tweets improved accuracy by 15%

### 5. FinVADER (VADER for Finance)

**Type:** Enhanced lexicon-based (VADER + financial terms)

**Speed:** 5-10ms per tweet

**Accuracy:** 65-75% on financial tweets (10% better than vanilla VADER)

**Installation:**
```bash
pip install finvader
```

**Usage:**
```python
from finvader import finvader

tweet = "Market crash imminent. Hedge your positions."
scores = finvader(tweet, use_sentibignomics=True, use_henry=True, indicator='compound')

print(scores)
# 0.62 (more accurate for financial terms)
```

**Enhancements over VADER:**
- Added 3,500+ financial terms (e.g., "bearish", "overbought", "resistance")
- Updated weights for financial context ("crash" = very negative)
- Integration with SentiBigNomics financial lexicon

---

## Machine Learning Frameworks

### XGBoost

**Use Case:** Ensemble signal generation (combine sentiment + volume + technical indicators)

**Installation:**
```bash
pip install xgboost
```

**Example:**
```python
import xgboost as xgb
import pandas as pd

# Feature engineering
features = pd.DataFrame({
    'sentiment_5m': sentiment_5m,
    'sentiment_15m': sentiment_15m,
    'volume_z_score': volume_z_score,
    'hour_of_day': current_time.hour,
    'day_of_week': current_time.dayofweek
})

# Train model
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'binary:logistic',  # Predict up/down
    'max_depth': 6,
    'eta': 0.1,
    'eval_metric': 'auc'
}
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict
dpred = xgb.DMatrix(features)
prediction = model.predict(dpred)[0]

if prediction > 0.65:
    signal = 'BUY'
```

**Hyperparameter Tuning:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 500, 1000]
}

xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

### LightGBM

**Advantage:** Faster training than XGBoost (10x speedup on large datasets)

**Installation:**
```bash
pip install lightgbm
```

**Example:**
```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
}

model = lgb.train(params, train_data, num_boost_round=100)
prediction = model.predict(features)[0]
```

### LSTM (Long Short-Term Memory)

**Use Case:** Time-series prediction (capture temporal sentiment patterns)

**Installation:**
```bash
pip install tensorflow
```

**Example:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prepare data (sequences of sentiment scores)
X = sentiment_sequences  # Shape: (samples, timesteps, features)
y = price_directions  # 0/1 for down/up

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predict
prediction = model.predict(X_new)[0][0]
```

**Bi-LSTM (Bidirectional LSTM):**
- Processes sequences forward and backward
- Research shows 2.01% MAPE (best performer for Bitcoin prediction)

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, features)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

---

## Polymarket API Integration

### py-clob-client (Official Python Client)

**Installation:**
```bash
pip install py-clob-client
```

**Setup:**
```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

# Initialize client
host = "https://clob.polymarket.com"
chain_id = 137  # Polygon

# Load API credentials
creds = ApiCreds(
    api_key="your_api_key",
    api_secret="your_api_secret",
    api_passphrase="your_passphrase"
)

client = ClobClient(host, key=creds.api_key, chain_id=chain_id, creds=creds)
```

### Get Market Data

```python
# Get all active markets
markets = client.get_markets()

# Search for specific market
trump_markets = [m for m in markets if 'trump' in m['question'].lower()]

# Get market details
market = client.get_market(market_id="0xabc123...")
print(f"Current YES price: {market['outcomes'][0]['price']}")
```

### Place Orders

```python
# Buy YES shares
order = client.create_order(
    token_id=market['tokens'][0]['token_id'],  # YES token
    side='BUY',
    size=100,  # $100 USDC
    price=0.65  # 65% probability
)

# Submit order
response = client.post_order(order)
print(f"Order ID: {response['orderID']}")
```

### Monitor Order Status

```python
# Check order status
order_status = client.get_order(order_id="order_123")

if order_status['status'] == 'MATCHED':
    print("Trade executed!")
elif order_status['status'] == 'LIVE':
    print("Order pending...")
```

### Market Making (Advanced)

```python
def market_make(market_id, sentiment_score):
    """Place limit orders on both sides with spread"""

    # Calculate fair value from sentiment
    fair_value = 0.5 + (sentiment_score * 0.3)  # Scale sentiment to price

    spread = 0.02  # 2% spread
    buy_price = fair_value - spread
    sell_price = fair_value + spread

    # Place buy order (bid)
    buy_order = client.create_order(
        token_id=market['tokens'][0]['token_id'],
        side='BUY',
        size=50,
        price=buy_price
    )

    # Place sell order (ask)
    sell_order = client.create_order(
        token_id=market['tokens'][0]['token_id'],
        side='SELL',
        size=50,
        price=sell_price
    )

    client.post_order(buy_order)
    client.post_order(sell_order)
```

---

## Supporting Tools

### Data Storage

#### PostgreSQL (Recommended)

**Use Case:** Store tweets, sentiment scores, trades, backtest results

```sql
CREATE TABLE tweets (
    id BIGSERIAL PRIMARY KEY,
    tweet_id VARCHAR(50) UNIQUE,
    text TEXT,
    author_id VARCHAR(50),
    created_at TIMESTAMP,
    sentiment_vader FLOAT,
    sentiment_finbert FLOAT,
    entities JSONB,
    public_metrics JSONB,
    INDEX idx_created_at (created_at),
    INDEX idx_entities (entities)
);

CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(100),
    side VARCHAR(10),
    entry_price FLOAT,
    exit_price FLOAT,
    size FLOAT,
    pnl FLOAT,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    signal_tweets JSONB,
    INDEX idx_market_id (market_id),
    INDEX idx_entry_time (entry_time)
);
```

#### Redis (Caching)

**Use Case:** Cache user credibility scores, deduplication, rate limit tracking

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Cache user credibility
def get_user_credibility(user_id):
    cached = r.get(f"user:{user_id}:credibility")
    if cached:
        return float(cached)

    # Calculate and cache
    credibility = calculate_credibility(user_id)
    r.setex(f"user:{user_id}:credibility", 3600, credibility)  # Cache 1 hour
    return credibility

# Deduplicate tweets
def is_duplicate(tweet_id):
    return r.exists(f"tweet:{tweet_id}")

def mark_processed(tweet_id):
    r.setex(f"tweet:{tweet_id}", 86400, 1)  # Mark as processed for 24 hours
```

### Message Queues

#### Apache Kafka

**Use Case:** High-throughput tweet ingestion, decoupling components

```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer (Twitter stream)
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def stream_to_kafka(tweet):
    producer.send('tweets', tweet)

# Consumer (Sentiment analyzer)
consumer = KafkaConsumer(
    'tweets',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    tweet = message.value
    sentiment = analyze_sentiment(tweet)
    producer.send('signals', {'tweet': tweet, 'sentiment': sentiment})
```

### Monitoring

#### Prometheus + Grafana

**Metrics Collection:**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
tweets_processed = Counter('tweets_processed_total', 'Total tweets processed')
sentiment_score = Histogram('sentiment_score', 'Distribution of sentiment scores')
trade_execution_time = Histogram('trade_execution_seconds', 'Order execution time')
current_position = Gauge('current_position_usd', 'Current position size')

# Instrument code
@sentiment_score.time()
def analyze_sentiment(tweet):
    # ...
    tweets_processed.inc()
    return score

# Start metrics server
start_http_server(8000)
```

**Grafana Dashboard Queries:**
```promql
# Average sentiment over time
avg_over_time(sentiment_score_sum[5m]) / avg_over_time(sentiment_score_count[5m])

# Tweet processing rate
rate(tweets_processed_total[1m])

# 95th percentile execution time
histogram_quantile(0.95, rate(trade_execution_seconds_bucket[5m]))
```

---

## Cost Summary

### Monthly Operating Costs (Typical Setups)

**Small-Scale Bot ($10k capital):**
- Twitter API: $100 (Basic tier)
- Compute: $20 (t3.small EC2 + VADER)
- Database: $10 (RDS micro)
- Monitoring: $0 (free tier)
- **Total:** $130/mo
- **Break-even:** 1.3% monthly return

**Medium-Scale Bot ($100k capital):**
- Twitter API: $5,000 (Pro tier)
- Compute: $500 (g4dn.xlarge GPU + workers)
- Database: $50 (RDS small + Redis)
- Kafka: $100 (managed service)
- Monitoring: $20 (Grafana Cloud)
- **Total:** $5,670/mo
- **Break-even:** 5.67% monthly return

**High-Frequency Bot ($500k capital):**
- Twitter API: $42,000 (Enterprise)
- Compute: $2,000 (multiple GPU instances + co-location)
- Database: $200 (RDS large cluster)
- Message queue: $500 (Kafka cluster)
- Monitoring: $100 (premium tier)
- **Total:** $44,800/mo
- **Break-even:** 8.96% monthly return

**Key Takeaway:** Twitter API costs dominate. Consider API alternatives or selective filtering to reduce tier requirements.

---

## Recommended Tech Stacks

### Budget Stack (<$200/mo)

```
Twitter Basic API → VADER → XGBoost → Polymarket
                      ↓
           PostgreSQL (local or small RDS)
```

**Pros:** Low cost, simple to deploy
**Cons:** Lower accuracy, limited scalability

### Standard Stack ($500-$1000/mo)

```
Twitter Pro API → Apache Kafka → [VADER + FinBERT] → Ensemble ML → Polymarket
                                         ↓
                           PostgreSQL + Redis + Prometheus
```

**Pros:** Good accuracy, scalable, production-ready
**Cons:** Moderate cost, requires DevOps

### Advanced Stack ($5k+/mo)

```
Twitter Enterprise → Kafka Cluster → [FinBERT + RoBERTa] GPU → LSTM + XGBoost Ensemble → Polymarket
                                              ↓
                              PostgreSQL Cluster + Redis Cluster + Full Observability
```

**Pros:** Maximum accuracy, ultra-low latency, enterprise-grade
**Cons:** High cost, complex operations

---

## References

### APIs
- Twitter API v2 Documentation: https://developer.twitter.com/en/docs/twitter-api
- Twitter API Pricing: https://getlate.dev/blog/twitter-api-pricing
- Polymarket CLOB API: https://docs.polymarket.com/

### Sentiment Libraries
- VADER: https://github.com/cjhutto/vaderSentiment
- FinBERT: https://github.com/ProsusAI/finBERT
- FinVADER: https://github.com/PetrKorab/FinVADER
- TextBlob: https://textblob.readthedocs.io/

### Machine Learning
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- TensorFlow (LSTM): https://www.tensorflow.org/guide/keras/rnn

### Research
- QuantInsti VADER Guide: https://blog.quantinsti.com/vader-sentiment/
- Alpaca Algorithmic Trading: https://alpaca.markets/learn/algorithmic-trading-with-twitter-sentiment-analysis
- Academic: Twitter Sentiment & Crypto (567k tweets, LSTM + RoBERTa): https://www.mdpi.com/2227-9091/11/9/159
