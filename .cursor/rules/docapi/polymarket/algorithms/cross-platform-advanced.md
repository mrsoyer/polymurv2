# Advanced Cross-Platform Prediction Market Trading

> Comprehensive guide to multi-platform arbitrage strategies, bot architecture, regulatory compliance, and risk management across Polymarket, Kalshi, PredictIt, and other prediction market platforms.

---

## Table of Contents

1. [Platform Comparison](#platform-comparison)
2. [Arbitrage Strategies](#arbitrage-strategies)
3. [Bot Architecture](#bot-architecture)
4. [Capital Allocation](#capital-allocation)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Risk Management](#risk-management)
7. [Performance Optimization](#performance-optimization)
8. [Implementation Examples](#implementation-examples)

---

## Platform Comparison

### Market Share & Dominance (2026)

| Platform | Market Share | Regulatory Status | Key Advantage |
|----------|--------------|-------------------|---------------|
| **Kalshi** | 66.4% daily volume | CFTC regulated | Sports markets in all 50 states |
| **Polymarket** | 47% (predicted 2026) | Offshore, recently US DCM | Highest liquidity, crypto markets |
| **PredictIt** | Declining | CFTC no-action letter (expanded 2025) | Political markets, academic research |

**Source**: As of January 2026, Kalshi claimed 66.4% of daily volume, while meta-markets on Manifold predict Polymarket will lead 2026 at 47% vs Kalshi's 34%.

### Platform Characteristics Deep Dive

#### Kalshi

**Regulatory Advantages**:
- First CFTC-regulated event contract exchange
- Sports markets legal in all 50 states (bypasses state-by-state licensing)
- Attracts institutional capital (market makers, brokers)
- Full KYC/AML compliance required

**API Features**:
- REST API for data analysis and trading
- WebSocket for real-time market data
- FIX protocol for high-frequency trading
- Access to order books, portfolio data, market statistics

**Fee Structure**:
- Probability-weighted fee model (averages ~1.2% per trade)
- Higher fees on near-certain outcomes
- No maker rebates

**Unique Markets**:
- Financial indicators (Inflation, GDP, Debt Ceiling)
- Supreme Court decisions
- Weekend trading for Monday's closing prices
- Regulatory-compliant sports contracts

**Limitations**:
- January 2026: Massachusetts injunction halted sports contracts (regulatory uncertainty)
- Higher fees than competitors
- More conservative market offerings (regulatory constraints)

#### Polymarket

**Regulatory Status**:
- Offshore platform (historically blocked US users)
- 2025/2026: Launched Polymarket US (DCM) with CFTC compliance path
- No KYC required for non-US users (changing)

**Technical Architecture**:
- Polygon blockchain for settlement
- Off-chain order matching (CLOB)
- On-chain settlement (gas costs)
- USDC-based contracts

**Fee Structure**:
- Polymarket US: 0.10% (hyper-competitive)
- Dynamic taker fees in 15-minute crypto markets (up to 3%)
- Maker rebate program funded by taker fees

**Liquidity Advantages**:
- Highest liquidity among prediction markets
- Leads price discovery (minutes ahead of competitors)
- Deep order books for major events
- $2B+ in total trading volume

**Unique Markets**:
- Cryptocurrency price predictions (BTC, ETH, SOL)
- Global events (less US-centric)
- 15-minute crypto price markets
- Entertainment, sports, politics

**Liquidity Rewards**:
- Daily payouts for limit order placement
- Higher rewards for orders closer to mid-market
- Automatic USDC rebates at midnight UTC
- Maker rebates in volatile markets

**Limitations**:
- Geographic restrictions (33 countries blocked)
- OFAC sanctions compliance required
- Gas fees on Polygon (settlement costs)
- Resolution oracle risks

#### PredictIt

**Regulatory Status**:
- CFTC no-action letter (academic research exemption)
- September 2025: Won approval as regulated derivatives exchange
- Victoria University of Wellington owned

**Recent Changes**:
- Single contract cap raised from $850 to $3,500
- 5,000-person market cap removed
- Expanded operations as regulated exchange

**Fee Structure**:
- 10% fee on profits (highest among major platforms)
- 5% withdrawal fee
- Creates "PredictIt Premium" (contracts trade above fair value)

**Market Focus**:
- Political prediction markets (core competency)
- US elections, nominations, polling
- Weekly tweet markets
- Government decisions

**Trading Strategies**:
- Cross-market arbitrage with Polymarket/Kalshi
- Zero-risk opportunities in multi-outcome markets
- Low-volume market exploitation
- Premium pricing inefficiencies

**Limitations**:
- High fees reduce profitability
- Lower liquidity than competitors
- Fewer market categories
- US-centric focus

### Platform Fee Comparison

| Platform | Maker Fee | Taker Fee | Withdrawal Fee | Notes |
|----------|-----------|-----------|----------------|-------|
| Kalshi | ~1.2% (avg) | ~1.2% (avg) | Free | Probability-weighted |
| Polymarket US | -0.10% (rebate) | 0.10% | Gas fees | Competitive pricing |
| Polymarket (15-min crypto) | Rebate | Up to 3% | Gas fees | Dynamic fees |
| PredictIt | 10% on profits | 10% on profits | 5% | Highest fees |

### Price Discovery & Efficiency

**Polymarket leads Kalshi by minutes**: During 2024 election, Polymarket generally led price discovery due to higher liquidity, but Kalshi often lagged by minutes, creating exploitable arbitrage windows.

**Current Price Spreads (Example)**:
- Democratic "Blue Wave" 2026:
  - Kalshi: 42%
  - PredictIt: 38%
  - **Spread: 400 basis points** (4% arbitrage opportunity)

**Efficiency Factors**:
- Liquidity depth
- Participant sophistication
- Fee structures
- Geographic restrictions
- Settlement speed

---

## Arbitrage Strategies

### 1. Within-Market Arbitrage

**Concept**: Exploit YES + NO prices that don't sum to $1.00 on a single platform.

**Example**:
```
Market: "Will Bitcoin hit $120k by March 2026?"
YES shares: $0.48
NO shares: $0.49
Total cost: $0.97
Guaranteed payout: $1.00
Risk-free profit: $0.03 per share (3.09%)
```

**Execution Steps**:
1. Monitor YES + NO prices continuously
2. When sum < $1.00 (after fees), execute both legs
3. Wait for market resolution
4. Collect $1.00 payout

**Profitability After Fees**:
```python
# Polymarket example (0.10% fees)
yes_cost = 0.48 * 1.001  # 0.4805
no_cost = 0.49 * 1.001   # 0.4905
total_cost = 0.971
profit = 1.00 - 0.971 = 0.029 (2.9%)

# Kalshi example (1.2% fees)
yes_cost = 0.48 * 1.012  # 0.4858
no_cost = 0.49 * 1.012   # 0.4959
total_cost = 0.9817
profit = 1.00 - 0.9817 = 0.0183 (1.83%)
```

**Frequency**: Rare on liquid markets, more common on low-volume markets.

**Risk**: Execution risk if prices move between legs.

### 2. Cross-Platform Arbitrage (Simple)

**Concept**: Buy YES on one platform, NO on another when prices create profit opportunity.

**Example**:
```
Event: "Will S&P 500 close above 6000 on Friday?"
Polymarket YES: 60% ($0.60)
Kalshi YES: 55% ($0.55)

Strategy:
1. Buy YES on Kalshi @ $0.55
2. Buy NO on Polymarket @ $0.40 ($1 - $0.60)
3. Total cost: $0.95
4. Guaranteed payout: $1.00
5. Profit: $0.05 (5.26%)
```

**Execution Considerations**:
- Must execute both legs simultaneously (price risk)
- Account for fees on both platforms
- Consider withdrawal/transfer costs
- Monitor for resolution divergence risk

**Optimal Conditions**:
- Consistent 2-5% spread between platforms
- High-liquidity markets (fast execution)
- Binary outcomes (reduces resolution risk)

### 3. Cross-Platform Arbitrage (Advanced)

**Synthetic Arbitrage**: Create synthetic positions across platforms to exploit mispricing.

**Example 1: Three-Way Arbitrage**
```
Event: "2026 World Cup Winner"
Brazil odds:
- Polymarket: 25% ($0.25)
- Kalshi: 28% ($0.28)
- PredictIt: 22% ($0.22)

Strategy:
1. Buy Brazil on PredictIt @ $0.22
2. Sell Brazil (or buy field) on Kalshi @ $0.28
3. Net credit: $0.06
4. Risk: Exposure to other outcomes
```

**Example 2: Time Arbitrage**
```
Event: "Bitcoin above $120k by March 2026?"
January 2026 contract (Polymarket): 60%
February 2026 contract (Kalshi): 55%

Strategy:
1. Buy Feb contract on Kalshi @ $0.55
2. Sell Jan contract on Polymarket @ $0.60
3. If BTC hits $120k in February:
   - Kalshi pays $1.00
   - Polymarket costs $1.00 (early resolution)
   - Net: Small profit from timing
```

**Example 3: Correlated Markets**
```
Event A: "Trump wins 2026 primary" (Polymarket: 70%)
Event B: "GOP wins 2026 election" (Kalshi: 65%)

If P(B|A) ≈ 90%, arbitrage exists:
- Expected B given A: 0.70 × 0.90 = 63%
- Actual B pricing: 65%
- Slight overvaluation of Event B
```

### 4. Triangular Arbitrage

**Concept**: Exploit pricing inconsistencies across three related markets.

**Example**:
```
Market 1: Democrats win House (Polymarket: 60%)
Market 2: Democrats win Senate (Kalshi: 55%)
Market 3: Democrats win both (PredictIt: 40%)

Theoretical: P(both) = P(House) × P(Senate) = 0.60 × 0.55 = 0.33
Actual: P(both) = 0.40
Overvalued by 7 percentage points

Strategy:
1. Sell "Democrats win both" on PredictIt @ $0.40
2. Buy "Democrats win House" on Polymarket @ $0.60
3. Buy "Democrats win Senate" on Kalshi @ $0.55
4. Total cost: $1.15, revenue: $0.40
5. Net exposure: Complex (requires hedging)
```

**Complexity**: Requires correlation analysis and careful position sizing.

### 5. Statistical Arbitrage

**Concept**: Exploit temporary deviations from historical pricing relationships.

**Mean Reversion Strategy**:
```python
# Monitor Kalshi vs Polymarket spread
historical_spread_mean = 0.02  # 2%
historical_spread_std = 0.01   # 1%

current_spread = kalshi_price - polymarket_price

if current_spread > historical_spread_mean + 2 * historical_spread_std:
    # Spread too wide: expect convergence
    # Buy cheaper platform, sell expensive platform
    buy_polymarket()
    sell_kalshi()

elif current_spread < historical_spread_mean - 2 * historical_spread_std:
    # Spread too narrow: expect widening
    buy_kalshi()
    sell_polymarket()
```

**Data Requirements**:
- Historical price data (6+ months)
- Correlation matrices
- Volatility metrics
- Volume patterns

### 6. Liquidity Provision Arbitrage

**Concept**: Earn maker rebates on Polymarket while hedging on other platforms.

**Strategy**:
```
1. Place limit orders on Polymarket (earn rebates)
   - Buy YES @ $0.58 (below mid-market of $0.60)
   - Sell YES @ $0.62 (above mid-market)

2. When filled, immediately hedge on Kalshi
   - If Polymarket YES buy fills @ $0.58
   - Sell YES on Kalshi @ $0.60
   - Lock in $0.02 spread + maker rebate

3. Collect daily rebates from Polymarket
   - Paid at midnight UTC
   - Higher for orders near mid-market
```

**Profitability**:
- Maker rebates: 0.1-0.3% daily
- Spread capture: 1-3% per execution
- Combined: 10-20% annualized (high effort)

**Risks**:
- Execution risk on hedge leg
- Adverse selection (informed traders take your liquidity)
- Platform risk (resolution divergence)

### 7. News-Driven Arbitrage

**Concept**: Exploit latency in price updates across platforms after breaking news.

**Example**:
```
Breaking news: "Federal Reserve announces surprise rate cut"

Platform reaction times:
- Polymarket: 10 seconds (crypto traders react instantly)
- Kalshi: 45 seconds (slower user base)
- PredictIt: 2 minutes (less automated trading)

Strategy:
1. Monitor news feeds (Twitter, Bloomberg, Reuters)
2. When relevant news breaks, check Polymarket first
3. If Polymarket moves significantly, immediately trade Kalshi/PredictIt
4. Close positions when prices converge (30-120 seconds)
```

**Infrastructure Requirements**:
- Real-time news monitoring
- Low-latency API connections
- Automated order execution
- Millisecond response times

**Profitability**: High frequency, low per-trade profit (0.5-2%), high Sharpe ratio.

### 8. Volume-Weighted Arbitrage

**Concept**: Exploit thin order books with large positions.

**Strategy**:
```
Low-volume market:
- Bid: $0.45 (100 shares)
- Ask: $0.55 (100 shares)
- Spread: 10 cents

Action:
1. Buy 100 shares @ $0.55
2. Simultaneously sell 100 shares @ $0.45 on another platform
3. Slowly accumulate large position
4. Exit when prices converge or event resolves
```

**Risks**:
- Liquidity risk (can't exit position)
- Slippage on large orders
- Information asymmetry (insiders may know outcome)

---

## Bot Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Multi-Platform Arbitrage Bot            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Market Data Layer                       │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  • Polymarket WebSocket (real-time prices)      │   │
│  │  • Kalshi REST/WebSocket (order book updates)   │   │
│  │  • PredictIt REST polling (1-5 second interval) │   │
│  │  • Event matching engine (normalize market IDs) │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │      Opportunity Detection Engine                │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  • Calculate YES + NO sums (within-market arb)  │   │
│  │  • Compare cross-platform prices (2-5% spread)  │   │
│  │  • Score opportunities (profit - fees - risk)   │   │
│  │  • Filter by liquidity (min $1000 volume)       │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Risk Management Layer                    │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  • Position sizing (Kelly Criterion)            │   │
│  │  • Exposure limits per platform (max 20%)       │   │
│  │  • Execution risk scoring                       │   │
│  │  • Resolution divergence detection              │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Execution Engine                         │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  • Parallel order placement (both legs)         │   │
│  │  • Smart order routing (limit vs market orders) │   │
│  │  • Retry logic (3 attempts with backoff)        │   │
│  │  • Slippage monitoring and cancellation         │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Position Management                      │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  • Track open positions across platforms        │   │
│  │  • Monitor market resolution status              │   │
│  │  • Auto-settle resolved markets                  │   │
│  │  • P&L calculation and reporting                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack Recommendations

#### Language: Rust vs Python

**Rust** (Recommended for Production):
- Sub-millisecond execution latency
- Memory safety and concurrency
- Superior performance for high-frequency trading
- Example: [polymarket-kalshi-arbitrage-bot](https://github.com/TopTrenDev/polymarket-kalshi-arbitrage-bot)

**Python** (Recommended for Prototyping):
- Rapid development and iteration
- Rich ecosystem (pandas, numpy, asyncio)
- Easier debugging and testing
- Example: [prediction-market-arbitrage-bot](https://github.com/realfishsam/prediction-market-arbitrage-bot)

#### Infrastructure

**Server Requirements**:
- VPS with low-latency network (AWS us-east-1, GCP us-central1)
- Minimum: 2 vCPU, 4GB RAM, 20GB SSD
- Recommended: 4 vCPU, 8GB RAM, 50GB SSD
- Cost: $20-40/month

**Network Latency**:
- Polymarket: <50ms to Polygon RPC endpoints
- Kalshi: <30ms to US East Coast
- PredictIt: <50ms to US servers

**Uptime**: 99.5%+ required (arbitrage opportunities are time-sensitive)

### Event Matching Engine

**Challenge**: Different platforms use different market identifiers and descriptions.

**Solution**: Fuzzy matching + manual mapping

```python
import difflib
from dataclasses import dataclass
from typing import Optional

@dataclass
class UnifiedEvent:
    """Normalized event across platforms"""
    event_id: str
    description: str
    resolution_criteria: str
    polymarket_market_id: Optional[str] = None
    kalshi_market_id: Optional[str] = None
    predictit_market_id: Optional[str] = None

class EventMatcher:
    def __init__(self):
        self.manual_mappings = {
            # Known mappings for high-priority markets
            "btc_120k_march_2026": {
                "polymarket": "0x1234...",
                "kalshi": "BTC-120K-MAR26",
                "predictit": None
            }
        }

    def match_markets(
        self,
        polymarket_desc: str,
        kalshi_desc: str
    ) -> float:
        """
        Calculate similarity score between market descriptions

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize descriptions
        poly_norm = self._normalize(polymarket_desc)
        kalshi_norm = self._normalize(kalshi_desc)

        # Calculate similarity
        similarity = difflib.SequenceMatcher(
            None,
            poly_norm,
            kalshi_norm
        ).ratio()

        return similarity

    def _normalize(self, description: str) -> str:
        """Normalize market description for comparison"""
        import re

        # Lowercase
        desc = description.lower()

        # Remove common variations
        replacements = {
            "president": "pres",
            "democratic": "dem",
            "republican": "rep",
            "united states": "us",
            "will": "",
            "does": "",
            "?": "",
        }

        for old, new in replacements.items():
            desc = desc.replace(old, new)

        # Remove extra whitespace
        desc = re.sub(r'\s+', ' ', desc).strip()

        return desc

    def find_cross_platform_match(
        self,
        source_platform: str,
        source_market_id: str,
        source_description: str,
        target_platform: str,
        target_markets: list[dict]
    ) -> Optional[str]:
        """
        Find matching market on target platform

        Args:
            source_platform: "polymarket", "kalshi", "predictit"
            source_market_id: Market ID on source platform
            source_description: Market description
            target_platform: Target platform to search
            target_markets: List of target platform markets

        Returns:
            Matching market ID on target platform, or None
        """
        # Check manual mappings first
        for event_id, mappings in self.manual_mappings.items():
            if mappings.get(source_platform) == source_market_id:
                return mappings.get(target_platform)

        # Fuzzy match
        best_match = None
        best_score = 0.0

        for target_market in target_markets:
            score = self.match_markets(
                source_description,
                target_market['description']
            )

            if score > best_score and score > 0.85:  # 85% threshold
                best_score = score
                best_match = target_market['id']

        return best_match
```

### Market Data Collection

**Polymarket WebSocket**:
```python
import asyncio
import websockets
import json

class PolymarketWebSocket:
    def __init__(self, on_update_callback):
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.on_update = on_update_callback

    async def connect(self, market_ids: list[str]):
        """Subscribe to real-time price updates"""
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to markets
            subscribe_msg = {
                "type": "subscribe",
                "markets": market_ids
            }
            await websocket.send(json.dumps(subscribe_msg))

            # Listen for updates
            async for message in websocket:
                data = json.loads(message)
                await self.on_update(data)
```

**Kalshi WebSocket**:
```python
class KalshiWebSocket:
    def __init__(self, api_key: str, on_update_callback):
        self.ws_url = "wss://trading-api.kalshi.com/trade-api/ws/v2"
        self.api_key = api_key
        self.on_update = on_update_callback

    async def connect(self, market_tickers: list[str]):
        """Subscribe to order book updates"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with websockets.connect(
            self.ws_url,
            extra_headers=headers
        ) as websocket:
            # Subscribe to order books
            for ticker in market_tickers:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "orderbook_delta",
                    "params": {"ticker": ticker}
                }
                await websocket.send(json.dumps(subscribe_msg))

            # Listen for updates
            async for message in websocket:
                data = json.loads(message)
                await self.on_update(data)
```

### Opportunity Detection Engine

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    type: str  # "within_market" or "cross_platform"
    market_id: str
    description: str

    # Leg 1
    platform_1: str
    side_1: str  # "YES" or "NO"
    price_1: float
    size_1: int

    # Leg 2
    platform_2: str
    side_2: str
    price_2: float
    size_2: int

    # Profitability
    gross_profit: float  # Before fees
    net_profit: float    # After fees
    profit_pct: float
    roi_annualized: float  # If holding to expiry

    # Risk metrics
    execution_risk: float  # 0.0 to 1.0
    resolution_risk: float  # 0.0 to 1.0
    liquidity_score: float  # 0.0 to 1.0

    # Timing
    opportunity_score: float  # Combined metric
    detected_at: float  # Unix timestamp
    expires_at: Optional[float] = None

class OpportunityDetector:
    def __init__(self, fee_calculator):
        self.fee_calc = fee_calculator
        self.min_profit_pct = 0.5  # 0.5% minimum profit

    def detect_within_market_arb(
        self,
        market_id: str,
        description: str,
        platform: str,
        yes_price: float,
        no_price: float,
        yes_size: int,
        no_size: int
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect YES + NO arbitrage on single platform
        """
        # Calculate total cost
        yes_cost_with_fees = yes_price * (1 + self.fee_calc.get_fee(platform, "taker"))
        no_cost_with_fees = no_price * (1 + self.fee_calc.get_fee(platform, "taker"))
        total_cost = yes_cost_with_fees + no_cost_with_fees

        # Check profitability
        if total_cost >= 1.0:
            return None  # No arbitrage

        gross_profit = 1.0 - (yes_price + no_price)
        net_profit = 1.0 - total_cost
        profit_pct = (net_profit / total_cost) * 100

        if profit_pct < self.min_profit_pct:
            return None

        # Calculate risk metrics
        execution_risk = self._calculate_execution_risk(
            yes_size, no_size, platform
        )

        liquidity_score = min(yes_size, no_size) / 1000  # Normalize to $1000

        return ArbitrageOpportunity(
            type="within_market",
            market_id=market_id,
            description=description,
            platform_1=platform,
            side_1="YES",
            price_1=yes_price,
            size_1=yes_size,
            platform_2=platform,
            side_2="NO",
            price_2=no_price,
            size_2=no_size,
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            roi_annualized=self._annualize_roi(net_profit, days_to_expiry=7),
            execution_risk=execution_risk,
            resolution_risk=0.0,  # No resolution risk (same platform)
            liquidity_score=liquidity_score,
            opportunity_score=self._score_opportunity(
                profit_pct, execution_risk, liquidity_score
            ),
            detected_at=time.time()
        )

    def detect_cross_platform_arb(
        self,
        market_id: str,
        description: str,
        platform_1: str,
        platform_1_yes_price: float,
        platform_1_size: int,
        platform_2: str,
        platform_2_yes_price: float,
        platform_2_size: int
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect cross-platform arbitrage opportunity
        """
        # Determine which platform is cheaper
        if platform_1_yes_price < platform_2_yes_price:
            buy_platform = platform_1
            buy_price = platform_1_yes_price
            buy_size = platform_1_size
            buy_side = "YES"

            sell_platform = platform_2
            sell_price = 1.0 - platform_2_yes_price  # NO price
            sell_size = platform_2_size
            sell_side = "NO"
        else:
            buy_platform = platform_2
            buy_price = platform_2_yes_price
            buy_size = platform_2_size
            buy_side = "YES"

            sell_platform = platform_1
            sell_price = 1.0 - platform_1_yes_price
            sell_size = platform_1_size
            sell_side = "NO"

        # Calculate costs with fees
        buy_cost = buy_price * (1 + self.fee_calc.get_fee(buy_platform, "taker"))
        sell_cost = sell_price * (1 + self.fee_calc.get_fee(sell_platform, "taker"))
        total_cost = buy_cost + sell_cost

        # Check profitability
        if total_cost >= 1.0:
            return None

        gross_profit = 1.0 - (buy_price + sell_price)
        net_profit = 1.0 - total_cost
        profit_pct = (net_profit / total_cost) * 100

        if profit_pct < self.min_profit_pct:
            return None

        # Calculate risk metrics
        execution_risk = self._calculate_cross_platform_execution_risk(
            buy_platform, sell_platform, buy_size, sell_size
        )

        resolution_risk = self._calculate_resolution_risk(
            buy_platform, sell_platform
        )

        liquidity_score = min(buy_size, sell_size) / 1000

        return ArbitrageOpportunity(
            type="cross_platform",
            market_id=market_id,
            description=description,
            platform_1=buy_platform,
            side_1=buy_side,
            price_1=buy_price,
            size_1=buy_size,
            platform_2=sell_platform,
            side_2=sell_side,
            price_2=sell_price,
            size_2=sell_size,
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            roi_annualized=self._annualize_roi(net_profit, days_to_expiry=14),
            execution_risk=execution_risk,
            resolution_risk=resolution_risk,
            liquidity_score=liquidity_score,
            opportunity_score=self._score_opportunity(
                profit_pct, execution_risk, liquidity_score, resolution_risk
            ),
            detected_at=time.time()
        )

    def _calculate_execution_risk(
        self,
        yes_size: int,
        no_size: int,
        platform: str
    ) -> float:
        """
        Risk that price moves between placing both legs

        Returns:
            Risk score (0.0 = low risk, 1.0 = high risk)
        """
        # Lower liquidity = higher execution risk
        min_size = min(yes_size, no_size)

        if min_size > 5000:
            base_risk = 0.1
        elif min_size > 1000:
            base_risk = 0.3
        elif min_size > 500:
            base_risk = 0.5
        else:
            base_risk = 0.8

        # Adjust for platform latency
        latency_multiplier = {
            "polymarket": 1.0,  # Fastest (on-chain + CLOB)
            "kalshi": 1.2,      # Slightly slower
            "predictit": 1.5    # Slowest
        }

        return min(base_risk * latency_multiplier.get(platform, 1.0), 1.0)

    def _calculate_cross_platform_execution_risk(
        self,
        platform_1: str,
        platform_2: str,
        size_1: int,
        size_2: int
    ) -> float:
        """
        Risk that prices move between executing both legs across platforms
        """
        base_risk = self._calculate_execution_risk(size_1, size_2, platform_1)

        # Cross-platform adds latency risk
        cross_platform_penalty = 0.2

        return min(base_risk + cross_platform_penalty, 1.0)

    def _calculate_resolution_risk(
        self,
        platform_1: str,
        platform_2: str
    ) -> float:
        """
        Risk that platforms resolve market differently

        Historical incidents:
        - 2024 government shutdown: Polymarket YES, Kalshi NO

        Returns:
            Risk score (0.0 = low risk, 1.0 = high risk)
        """
        # Oracle quality scores
        oracle_scores = {
            "kalshi": 0.95,    # CFTC-regulated, clear resolution rules
            "polymarket": 0.90,  # UMA oracle, community voting
            "predictit": 0.85   # Academic oversight, sometimes subjective
        }

        score_1 = oracle_scores.get(platform_1, 0.8)
        score_2 = oracle_scores.get(platform_2, 0.8)

        # Risk = 1 - (average oracle quality)
        avg_quality = (score_1 + score_2) / 2

        return 1.0 - avg_quality

    def _annualize_roi(self, net_profit: float, days_to_expiry: int) -> float:
        """Calculate annualized ROI"""
        if days_to_expiry == 0:
            return 0.0

        daily_roi = net_profit
        annual_roi = daily_roi * (365 / days_to_expiry)

        return annual_roi * 100  # Percentage

    def _score_opportunity(
        self,
        profit_pct: float,
        execution_risk: float,
        liquidity_score: float,
        resolution_risk: float = 0.0
    ) -> float:
        """
        Combine metrics into single opportunity score

        Returns:
            Score (higher = better opportunity)
        """
        # Weighted combination
        score = (
            profit_pct * 0.4 +           # Profit is most important
            (1 - execution_risk) * 30 +  # Low execution risk
            liquidity_score * 20 +       # Good liquidity
            (1 - resolution_risk) * 10   # Low resolution risk
        )

        return score
```

### Execution Engine

```python
import asyncio
from typing import Optional

class ExecutionEngine:
    def __init__(self, polymarket_client, kalshi_client, predictit_client):
        self.poly = polymarket_client
        self.kalshi = kalshi_client
        self.predictit = predictit_client

        self.max_slippage = 0.02  # 2% max slippage
        self.max_retries = 3

    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity
    ) -> dict:
        """
        Execute both legs of arbitrage simultaneously

        Returns:
            Execution report with fills, slippage, P&L
        """
        # Determine position size
        max_position = min(
            opportunity.size_1,
            opportunity.size_2
        )

        position_size = int(max_position * 0.9)  # 90% of available liquidity

        # Execute both legs in parallel
        try:
            leg1_task = self._execute_leg(
                platform=opportunity.platform_1,
                market_id=opportunity.market_id,
                side=opportunity.side_1,
                price=opportunity.price_1,
                size=position_size
            )

            leg2_task = self._execute_leg(
                platform=opportunity.platform_2,
                market_id=opportunity.market_id,
                side=opportunity.side_2,
                price=opportunity.price_2,
                size=position_size
            )

            # Wait for both legs
            leg1_result, leg2_result = await asyncio.gather(
                leg1_task,
                leg2_task,
                return_exceptions=True
            )

            # Check for errors
            if isinstance(leg1_result, Exception) or isinstance(leg2_result, Exception):
                # Partial fill - need to reverse successful leg
                await self._reverse_partial_fill(leg1_result, leg2_result)

                return {
                    "success": False,
                    "error": "Partial fill - positions reversed"
                }

            # Calculate actual P&L
            total_cost = leg1_result['cost'] + leg2_result['cost']
            total_payout = position_size * 1.0  # $1 per share
            actual_profit = total_payout - total_cost
            actual_profit_pct = (actual_profit / total_cost) * 100

            # Calculate slippage
            expected_cost = (
                opportunity.price_1 + opportunity.price_2
            ) * position_size
            slippage = ((total_cost - expected_cost) / expected_cost) * 100

            return {
                "success": True,
                "opportunity_id": opportunity.market_id,
                "position_size": position_size,
                "leg1": leg1_result,
                "leg2": leg2_result,
                "total_cost": total_cost,
                "expected_profit": opportunity.net_profit * position_size,
                "actual_profit": actual_profit,
                "actual_profit_pct": actual_profit_pct,
                "slippage_pct": slippage,
                "executed_at": time.time()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_leg(
        self,
        platform: str,
        market_id: str,
        side: str,
        price: float,
        size: int
    ) -> dict:
        """
        Execute single leg of arbitrage
        """
        client = self._get_client(platform)

        # Place limit order (faster execution, better price)
        order = await client.place_order(
            market_id=market_id,
            side=side,
            order_type="limit",
            limit_price=price,
            size=size
        )

        # Wait for fill (with timeout)
        fill = await self._wait_for_fill(
            client=client,
            order_id=order['id'],
            timeout=5.0  # 5 second timeout
        )

        if not fill:
            # Cancel unfilled order
            await client.cancel_order(order['id'])
            raise Exception(f"Order not filled within timeout: {order['id']}")

        return {
            "platform": platform,
            "order_id": order['id'],
            "side": side,
            "price": fill['price'],
            "size": fill['size'],
            "cost": fill['price'] * fill['size'],
            "filled_at": fill['timestamp']
        }

    async def _wait_for_fill(
        self,
        client,
        order_id: str,
        timeout: float
    ) -> Optional[dict]:
        """
        Wait for order to fill, with timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            order_status = await client.get_order(order_id)

            if order_status['status'] == 'filled':
                return order_status

            await asyncio.sleep(0.1)  # Check every 100ms

        return None  # Timeout

    async def _reverse_partial_fill(self, leg1_result, leg2_result):
        """
        If one leg fills but other fails, reverse the successful leg
        """
        # Determine which leg succeeded
        successful_leg = None
        if not isinstance(leg1_result, Exception):
            successful_leg = leg1_result
        elif not isinstance(leg2_result, Exception):
            successful_leg = leg2_result
        else:
            return  # Both failed, nothing to reverse

        # Place opposite order to close position
        client = self._get_client(successful_leg['platform'])

        opposite_side = "NO" if successful_leg['side'] == "YES" else "YES"

        await client.place_order(
            market_id=successful_leg['market_id'],
            side=opposite_side,
            order_type="market",  # Exit quickly
            size=successful_leg['size']
        )

    def _get_client(self, platform: str):
        """Get API client for platform"""
        clients = {
            "polymarket": self.poly,
            "kalshi": self.kalshi,
            "predictit": self.predictit
        }
        return clients[platform]
```

---

## Capital Allocation

### Kelly Criterion for Position Sizing

**Formula**:
```
f* = (bp - q) / b

where:
- f* = fraction of capital to bet
- b = odds received on bet (decimal odds - 1)
- p = probability of winning
- q = probability of losing (1 - p)
```

**Example**:
```python
def calculate_kelly_fraction(
    win_probability: float,
    profit_if_win: float,
    loss_if_lose: float
) -> float:
    """
    Calculate optimal Kelly fraction

    Args:
        win_probability: 0.0 to 1.0
        profit_if_win: Expected profit per dollar bet
        loss_if_lose: Expected loss per dollar bet (usually 1.0)

    Returns:
        Fraction of capital to allocate (0.0 to 1.0)
    """
    if win_probability <= 0 or win_probability >= 1:
        return 0.0

    lose_probability = 1 - win_probability

    kelly_fraction = (
        win_probability * profit_if_win - lose_probability * loss_if_lose
    ) / profit_if_win

    # Clamp to [0, 1]
    return max(0.0, min(kelly_fraction, 1.0))

# Arbitrage example (virtually certain win)
kelly = calculate_kelly_fraction(
    win_probability=0.98,  # 98% chance (accounting for execution risk)
    profit_if_win=0.03,    # 3% profit
    loss_if_lose=1.0       # 100% loss if fails
)
# Result: kelly ≈ 0.60 (60% of capital)

# Fractional Kelly (more conservative)
fractional_kelly = kelly * 0.25  # Quarter Kelly
# Result: 15% of capital
```

### Multi-Platform Capital Allocation

**Objective**: Maximize expected returns while limiting exposure to any single platform.

**Constraints**:
- Max 20% of capital on any single platform
- Max 50% of capital in cross-platform arbitrage positions
- Min 20% in cash reserves (for opportunities and emergencies)

**Allocation Strategy**:

```python
from dataclasses import dataclass

@dataclass
class PlatformAllocation:
    """Capital allocation across platforms"""
    platform: str
    allocated_capital: float
    max_position_size: float
    current_positions: int
    available_capital: float

class CapitalAllocator:
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.cash_reserve_pct = 0.20
        self.max_platform_pct = 0.20
        self.max_arb_pct = 0.50

        # Initialize platform allocations
        self.platforms = {
            "polymarket": PlatformAllocation(
                platform="polymarket",
                allocated_capital=0.0,
                max_position_size=total_capital * self.max_platform_pct,
                current_positions=0,
                available_capital=total_capital * self.max_platform_pct
            ),
            "kalshi": PlatformAllocation(
                platform="kalshi",
                allocated_capital=0.0,
                max_position_size=total_capital * self.max_platform_pct,
                current_positions=0,
                available_capital=total_capital * self.max_platform_pct
            ),
            "predictit": PlatformAllocation(
                platform="predictit",
                allocated_capital=0.0,
                max_position_size=total_capital * self.max_platform_pct,
                current_positions=0,
                available_capital=total_capital * self.max_platform_pct
            )
        }

        self.total_arb_allocation = 0.0

    def calculate_position_size(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Optional[int]:
        """
        Calculate optimal position size given constraints

        Returns:
            Position size in shares, or None if constraints violated
        """
        # Get platform allocations
        platform_1_alloc = self.platforms[opportunity.platform_1]
        platform_2_alloc = self.platforms[opportunity.platform_2]

        # Check if we have available capital
        if (platform_1_alloc.available_capital <= 0 or
            platform_2_alloc.available_capital <= 0):
            return None

        # Kelly criterion (quarter Kelly for safety)
        kelly_fraction = calculate_kelly_fraction(
            win_probability=1.0 - opportunity.execution_risk - opportunity.resolution_risk,
            profit_if_win=opportunity.profit_pct / 100,
            loss_if_lose=1.0
        ) * 0.25

        # Calculate position size based on Kelly
        kelly_position_size = self.total_capital * kelly_fraction

        # Apply constraints
        max_position_size = min(
            platform_1_alloc.available_capital,
            platform_2_alloc.available_capital,
            kelly_position_size,
            opportunity.size_1,
            opportunity.size_2
        )

        # Check arbitrage allocation limit
        if self.total_arb_allocation + max_position_size > self.total_capital * self.max_arb_pct:
            max_position_size = self.total_capital * self.max_arb_pct - self.total_arb_allocation

        # Ensure we leave cash reserve
        cash_reserve = self.total_capital * self.cash_reserve_pct
        total_allocated = sum(p.allocated_capital for p in self.platforms.values())

        if total_allocated + max_position_size > self.total_capital - cash_reserve:
            max_position_size = self.total_capital - cash_reserve - total_allocated

        # Return integer share count
        return int(max_position_size) if max_position_size > 0 else None

    def allocate_capital(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: int
    ):
        """Update allocations after executing position"""
        cost_per_share = opportunity.price_1 + opportunity.price_2
        total_cost = cost_per_share * position_size

        # Update platform allocations
        platform_1_alloc = self.platforms[opportunity.platform_1]
        platform_2_alloc = self.platforms[opportunity.platform_2]

        platform_1_cost = opportunity.price_1 * position_size
        platform_2_cost = opportunity.price_2 * position_size

        platform_1_alloc.allocated_capital += platform_1_cost
        platform_1_alloc.available_capital -= platform_1_cost
        platform_1_alloc.current_positions += 1

        platform_2_alloc.allocated_capital += platform_2_cost
        platform_2_alloc.available_capital -= platform_2_cost
        platform_2_alloc.current_positions += 1

        # Update total arbitrage allocation
        self.total_arb_allocation += total_cost

    def free_capital(
        self,
        platform: str,
        amount: float
    ):
        """Free up capital when position closes"""
        platform_alloc = self.platforms[platform]

        platform_alloc.allocated_capital -= amount
        platform_alloc.available_capital += amount
        platform_alloc.current_positions -= 1

        self.total_arb_allocation -= amount

    def get_allocation_report(self) -> dict:
        """Generate allocation report"""
        total_allocated = sum(p.allocated_capital for p in self.platforms.values())
        cash_reserve = self.total_capital - total_allocated

        return {
            "total_capital": self.total_capital,
            "cash_reserve": cash_reserve,
            "cash_reserve_pct": cash_reserve / self.total_capital * 100,
            "total_allocated": total_allocated,
            "total_arb_allocation": self.total_arb_allocation,
            "platforms": {
                name: {
                    "allocated": alloc.allocated_capital,
                    "available": alloc.available_capital,
                    "utilization_pct": (alloc.allocated_capital / alloc.max_position_size) * 100,
                    "positions": alloc.current_positions
                }
                for name, alloc in self.platforms.items()
            }
        }
```

### Rebalancing Strategy

**Triggers**:
1. Platform allocation exceeds 25% (rebalance down to 20%)
2. Cash reserve drops below 15% (close lowest-performing positions)
3. Weekly review (Sundays at midnight UTC)

**Example**:
```python
class Rebalancer:
    def should_rebalance(self, allocator: CapitalAllocator) -> bool:
        """Check if rebalancing is needed"""
        report = allocator.get_allocation_report()

        # Check cash reserve
        if report['cash_reserve_pct'] < 15:
            return True

        # Check platform allocations
        for platform, stats in report['platforms'].items():
            if stats['utilization_pct'] > 25:
                return True

        return False

    async def rebalance(self, allocator: CapitalAllocator, position_manager):
        """Execute rebalancing"""
        report = allocator.get_allocation_report()

        # Identify positions to close
        positions_to_close = []

        for platform, stats in report['platforms'].items():
            if stats['utilization_pct'] > 25:
                # Close lowest-performing positions on this platform
                platform_positions = position_manager.get_positions(platform)
                sorted_positions = sorted(
                    platform_positions,
                    key=lambda p: p.roi
                )

                # Close bottom 20% of positions
                num_to_close = max(1, int(len(platform_positions) * 0.2))
                positions_to_close.extend(sorted_positions[:num_to_close])

        # Execute closes
        for position in positions_to_close:
            await position_manager.close_position(position)
```

---

## Regulatory Compliance

### Geographic Restrictions

| Platform | Restricted Regions | KYC Required | Notes |
|----------|-------------------|--------------|-------|
| **Kalshi** | None (US-based, 50 states) | Yes (full KYC/AML) | CFTC-regulated |
| **Polymarket** | 33 countries, including US (geo-fence) | No (historically) | Offshore, OFAC compliance |
| **Polymarket US (DCM)** | Non-US regions | Yes (full KYC/AML) | New CFTC-compliant entity |
| **PredictIt** | Limited (primarily US focus) | Yes (academic exemption) | CFTC no-action letter |

### Polymarket Restricted Countries

**OFAC Sanctioned**:
- Cuba, Iran, North Korea, Syria, Russia (Crimea, Donetsk, Luhansk)

**Other Restrictions (33 total)**:
- Belarus, Burma, Central African Republic, Democratic Republic of Congo, Iraq, Lebanon, Libya, Mali, Nicaragua, Somalia, South Sudan, Sudan, Venezuela, Yemen, Zimbabwe
- Specific regions: Crimea, Donetsk, Luhansk (Ukraine)

**Enforcement**: IP/device checks, sanctions screening, geolocation monitoring.

### KYC/AML Compliance Requirements

#### Kalshi (Full Compliance)

**Identity Verification**:
- Government-issued ID (passport, driver's license)
- Social Security Number (SSN) or Tax ID
- Proof of address (utility bill, bank statement)
- Selfie verification (liveness check)

**AML Monitoring**:
- Transaction monitoring for suspicious activity
- Large transaction reporting (>$10,000)
- Suspicious Activity Reports (SARs) to FinCEN
- Compliance with Bank Secrecy Act (BSA)

**Accredited Investor Status**:
- Required for certain high-risk markets
- Income verification ($200k+ annual) or net worth ($1M+)

#### Polymarket (Evolving)

**Historical** (Offshore):
- No KYC for non-US users
- Anonymous crypto wallet trading
- OFAC sanctions screening only

**Current** (Polymarket US DCM):
- Full KYC for US users (similar to Kalshi)
- Identity verification, SSN, address proof
- AML transaction monitoring

**Compliance Controls**:
- Geofencing (block US IPs for offshore platform)
- RegTech tools for real-time monitoring
- Sanctions screening (OFAC lists)

#### PredictIt (Academic Exemption)

**Basic Verification**:
- Email verification
- Identity confirmation (less stringent than Kalshi)
- No SSN required (historically)

**Limits**:
- $850 per contract (raised to $3,500 in 2025)
- 5,000 traders per market (removed in 2025)

**Reporting**:
- Academic research reports to CFTC
- Market manipulation monitoring

### Tax Implications

#### United States

**Classification**: Prediction market profits are taxed as **capital gains** (IRS guidance pending).

**Short-Term Capital Gains** (positions held < 1 year):
- Taxed as ordinary income
- Rates: 10%, 12%, 22%, 24%, 32%, 35%, 37% (based on income bracket)

**Long-Term Capital Gains** (positions held > 1 year):
- Preferential rates: 0%, 15%, 20%
- Most arbitrage positions are short-term (days to weeks)

**Reporting Requirements**:
- Form 8949 (Sales and Dispositions of Capital Assets)
- Schedule D (Capital Gains and Losses)
- Platforms may issue Form 1099-B (Proceeds from Broker Transactions)

**Wash Sale Rule**:
- Does NOT apply to prediction markets (only securities)
- Can immediately repurchase similar positions

**Trader Tax Status (TTS)**:
- If trading is substantial and continuous (>75 trades/week)
- Allows Section 475(f) mark-to-market election
- Deduct losses against ordinary income (not just capital gains)
- Deduct trading expenses (software, data, infrastructure)

**Example Tax Calculation**:
```
Annual arbitrage profits: $50,000
Short-term capital gains rate: 24%
Tax owed: $12,000

With TTS election:
Profits: $50,000
Expenses: $10,000 (software, VPS, data feeds)
Net income: $40,000
Tax owed: $9,600
Tax savings: $2,400
```

#### International

**United Kingdom**:
- Spread betting: Tax-free (gambling exemption)
- Contract for difference: Capital gains tax (annual allowance £12,300)

**European Union**:
- Varies by country
- Germany: Capital gains tax (25% + solidarity surcharge)
- France: 30% flat tax on capital gains

**Canada**:
- 50% of capital gains are taxable
- Included in ordinary income

**Australia**:
- Capital gains tax (50% discount if held > 12 months)
- Professional traders: Ordinary income

### Regulatory Risks

#### Platform Shutdowns

**Historical Examples**:
- 2024: Polymarket settled with CFTC for $1.4M, agreed to geo-block US users
- 2026: Kalshi sports contracts halted by Massachusetts injunction

**Mitigation**:
- Diversify across platforms (don't rely on single platform)
- Monitor regulatory news daily
- Have withdrawal plans ready (24-hour withdrawal capability)
- Keep detailed transaction records (in case platform shuts down)

#### Resolution Divergence

**2024 Government Shutdown Incident**:
- Event: "Will US government shut down by December 2024?"
- Polymarket resolution: YES (shutdown occurred)
- Kalshi resolution: NO (different definition in contract terms)
- Result: Cross-platform arbitrageurs lost money

**Mitigation**:
- Read resolution criteria carefully on both platforms
- Avoid ambiguous markets (subjective outcomes)
- Prefer binary outcomes with clear resolution sources
- Monitor UMA disputes on Polymarket (community votes)

#### Market Manipulation

**CFTC Enforcement**:
- Spoofing (placing orders without intent to fill)
- Wash trading (self-trading to inflate volume)
- Pump and dump schemes

**Penalties**:
- Civil penalties up to $1M per violation
- Criminal prosecution (imprisonment)
- Lifetime ban from commodity markets

**Compliance**:
- Avoid spoofing (don't rapidly cancel large orders)
- Don't trade between your own accounts
- Don't coordinate with other traders (collusion)
- Keep detailed records (prove legitimate strategy)

---

## Risk Management

### Execution Risk

**Definition**: Risk that prices move between placing both legs of arbitrage.

**Causes**:
- Market volatility (news events)
- Low liquidity (slippage)
- Network latency (slow order placement)
- Platform downtime (API errors)

**Mitigation Strategies**:

1. **Parallel Execution**
   ```python
   # Execute both legs simultaneously using asyncio.gather()
   leg1, leg2 = await asyncio.gather(
       execute_leg_1(),
       execute_leg_2()
   )
   ```

2. **Limit Orders** (faster than market orders)
   ```python
   # Place limit orders at current best bid/ask
   order_1 = await client_1.place_limit_order(
       price=best_ask,  # Buy at ask
       size=position_size
   )
   ```

3. **Timeout and Cancel**
   ```python
   # If order doesn't fill within 5 seconds, cancel
   try:
       fill = await asyncio.wait_for(
           wait_for_fill(order_id),
           timeout=5.0
       )
   except asyncio.TimeoutError:
       await cancel_order(order_id)
   ```

4. **Slippage Limits**
   ```python
   # Cancel if execution price deviates >2% from expected
   if abs(fill_price - expected_price) / expected_price > 0.02:
       await reverse_position()
   ```

5. **Liquidity Thresholds**
   ```python
   # Only trade markets with >$1000 liquidity per leg
   if min(yes_size, no_size) < 1000:
       skip_opportunity()
   ```

### Resolution Risk

**Definition**: Risk that platforms resolve market differently (oracle mismatch).

**Historical Frequency**: Rare but catastrophic (1-2 times per year on high-profile markets).

**Mitigation Strategies**:

1. **Read Resolution Criteria**
   ```python
   def check_resolution_alignment(
       polymarket_criteria: str,
       kalshi_criteria: str
   ) -> bool:
       """
       Compare resolution criteria for consistency
       """
       # Parse criteria
       poly_source = extract_resolution_source(polymarket_criteria)
       kalshi_source = extract_resolution_source(kalshi_criteria)

       # Check if sources match
       if poly_source != kalshi_source:
           return False  # Different sources = resolution risk

       # Check for ambiguity
       if "subjective" in polymarket_criteria.lower():
           return False

       return True
   ```

2. **Prefer Clear Binary Outcomes**
   - Good: "Will Bitcoin close above $120k on March 31, 2026?" (clear price and date)
   - Bad: "Will Bitcoin have a good year in 2026?" (subjective)

3. **Avoid Political Markets with Ambiguous Terms**
   - Example: "Will Trump be convicted?" (which charges? which jurisdiction?)

4. **Monitor UMA Disputes (Polymarket)**
   ```python
   async def monitor_uma_disputes(market_id: str):
       """
       Alert if market enters UMA dispute resolution
       """
       uma_api = UMAClient()

       while True:
           status = await uma_api.get_market_status(market_id)

           if status == "disputed":
               alert("UMA DISPUTE: " + market_id)
               # Consider exiting position

           await asyncio.sleep(3600)  # Check hourly
   ```

5. **Resolution Risk Premium**
   ```python
   # Require higher profit for cross-platform arbitrage
   min_profit_single_platform = 0.5%  # Low risk
   min_profit_cross_platform = 1.5%   # Resolution risk premium
   ```

### Platform Risk

**Definition**: Risk of platform downtime, hacks, insolvency, regulatory shutdown.

**Mitigation Strategies**:

1. **Capital Limits per Platform**
   ```python
   # Max 20% of capital on any single platform
   max_platform_allocation = total_capital * 0.20
   ```

2. **Rapid Withdrawal Capability**
   ```python
   # Maintain API credentials for instant withdrawals
   async def emergency_withdraw_all():
       """Withdraw all funds from all platforms"""
       tasks = [
           polymarket.withdraw_all(),
           kalshi.withdraw_all(),
           predictit.withdraw_all()
       ]
       await asyncio.gather(*tasks)
   ```

3. **Monitor Platform Health**
   ```python
   async def monitor_platform_health():
       """Check platform uptime and API responsiveness"""
       platforms = ["polymarket", "kalshi", "predictit"]

       for platform in platforms:
           try:
               response_time = await ping_api(platform)

               if response_time > 1000:  # >1 second latency
                   alert(f"{platform} SLOW: {response_time}ms")

               if response_time > 5000:  # >5 seconds
                   # Consider emergency withdrawal
                   await emergency_withdraw_all()

           except Exception as e:
               alert(f"{platform} DOWN: {e}")
   ```

4. **Diversify Across Platforms**
   - Don't rely on single platform for liquidity
   - Have backup platforms ready (Augur, Manifold, etc.)

5. **Insurance (if available)**
   - Some DeFi protocols offer smart contract insurance
   - Nexus Mutual, Unslashed Finance (for on-chain platforms)

### Liquidity Risk

**Definition**: Risk of inability to exit position at fair price.

**Causes**:
- Low trading volume
- Wide bid-ask spreads
- Market maker withdrawal (during crises)

**Mitigation Strategies**:

1. **Liquidity Score**
   ```python
   def calculate_liquidity_score(market_data: dict) -> float:
       """
       Score market liquidity (0.0 to 1.0)

       Factors:
       - Trading volume (24h)
       - Bid-ask spread
       - Order book depth
       - Number of unique traders
       """
       volume_score = min(market_data['volume_24h'] / 100000, 1.0)
       spread_score = 1.0 - min(market_data['spread'] / 0.10, 1.0)
       depth_score = min(market_data['book_depth'] / 10000, 1.0)
       trader_score = min(market_data['unique_traders'] / 100, 1.0)

       # Weighted average
       liquidity_score = (
           volume_score * 0.3 +
           spread_score * 0.3 +
           depth_score * 0.2 +
           trader_score * 0.2
       )

       return liquidity_score
   ```

2. **Minimum Liquidity Thresholds**
   ```python
   # Only trade markets with good liquidity
   min_liquidity_score = 0.6

   if calculate_liquidity_score(market_data) < min_liquidity_score:
       skip_opportunity()
   ```

3. **Position Size Limits**
   ```python
   # Don't exceed 10% of daily volume
   max_position_size = market_data['volume_24h'] * 0.10
   ```

4. **Gradual Entry/Exit**
   ```python
   async def gradual_exit(position_size: int, num_chunks: int = 5):
       """Exit large position in chunks to minimize slippage"""
       chunk_size = position_size // num_chunks

       for i in range(num_chunks):
           await sell_chunk(chunk_size)
           await asyncio.sleep(60)  # Wait 1 minute between chunks
   ```

### Correlation Risk

**Definition**: Multiple arbitrage positions becoming correlated (all move together).

**Example**:
```
Position 1: "Bitcoin >$120k by March" (Polymarket vs Kalshi)
Position 2: "Ethereum >$7k by March" (Polymarket vs Kalshi)
Position 3: "S&P 500 >6500 by March" (Polymarket vs Kalshi)

If crypto/equity markets crash, all three positions may fail simultaneously.
```

**Mitigation**:

1. **Correlation Analysis**
   ```python
   import numpy as np

   def calculate_portfolio_correlation(positions: list) -> float:
       """
       Calculate average pairwise correlation of positions
       """
       returns = np.array([pos.historical_returns for pos in positions])
       corr_matrix = np.corrcoef(returns)

       # Average off-diagonal correlations
       n = len(positions)
       avg_corr = (corr_matrix.sum() - n) / (n * (n - 1))

       return avg_corr
   ```

2. **Diversification Across Uncorrelated Markets**
   - Crypto, politics, finance, entertainment, weather, sports
   - Low correlation = better risk-adjusted returns

3. **Position Limits by Category**
   ```python
   # Max 30% of portfolio in crypto-related markets
   category_limits = {
       "crypto": 0.30,
       "politics": 0.30,
       "finance": 0.20,
       "sports": 0.10,
       "other": 0.10
   }
   ```

### Gas Fee Risk (Polymarket Only)

**Definition**: Polygon gas fees reduce profitability of small arbitrage trades.

**Typical Costs**:
- Polygon transaction: $0.01 - $0.05 (normal)
- Network congestion: $0.10 - $0.50 (high)
- Settlement per market: 1 transaction (buy) + 1 transaction (settle)

**Mitigation**:

1. **Minimum Profit After Gas**
   ```python
   def calculate_net_profit_with_gas(
       gross_profit: float,
       position_size: int,
       gas_price_gwei: float
   ) -> float:
       """
       Subtract estimated gas costs from profit
       """
       # Estimate gas cost (2 transactions: buy + settle)
       gas_limit = 200000  # ~200k gas per transaction
       gas_cost_eth = (gas_price_gwei * 1e-9) * gas_limit * 2
       gas_cost_usd = gas_cost_eth * get_eth_price()

       net_profit = gross_profit * position_size - gas_cost_usd

       return net_profit
   ```

2. **Batch Settlements**
   ```python
   # Wait for multiple markets to resolve, settle in single transaction
   async def batch_settle_markets(market_ids: list[str]):
       """Settle multiple markets in single transaction"""
       contract = get_polymarket_contract()

       tx = await contract.batch_settle(market_ids)
       await tx.wait()
   ```

3. **Monitor Gas Prices**
   ```python
   async def get_optimal_gas_price() -> float:
       """Get current gas price from Polygon"""
       polygon_rpc = PolygonRPCClient()
       gas_price = await polygon_rpc.eth_gasPrice()

       return gas_price / 1e9  # Convert to gwei
   ```

4. **Avoid Trading During Network Congestion**
   ```python
   # Skip opportunities when gas >10 gwei
   if await get_optimal_gas_price() > 10:
       skip_opportunity()
   ```

---

## Performance Optimization

### Latency Optimization

**Target**: <50ms from market data update to order placement.

**Techniques**:

1. **WebSocket Connections** (not REST polling)
   ```python
   # BAD: REST polling (200-500ms latency)
   while True:
       prices = await rest_client.get_prices()
       await asyncio.sleep(1)  # Poll every second

   # GOOD: WebSocket streaming (<10ms latency)
   async with websocket.connect(ws_url) as ws:
       async for message in ws:
           prices = parse_message(message)
           # Instant update
   ```

2. **Colocation / VPS Near Exchanges**
   - AWS us-east-1 (New York) for Kalshi
   - GCP us-central1 for Polymarket
   - Reduce network latency from 100ms → 10ms

3. **Connection Pooling**
   ```python
   # Reuse HTTP connections (avoid TCP handshake overhead)
   import aiohttp

   session = aiohttp.ClientSession(
       connector=aiohttp.TCPConnector(
           limit=100,  # Max 100 concurrent connections
           ttl_dns_cache=300  # Cache DNS for 5 minutes
       )
   )
   ```

4. **Async/Await Everywhere**
   ```python
   # BAD: Synchronous (blocks thread)
   response = requests.get(url)

   # GOOD: Asynchronous (non-blocking)
   async with session.get(url) as response:
       data = await response.json()
   ```

5. **Pre-Authenticate**
   ```python
   # Cache authentication tokens (avoid re-auth on every request)
   class AuthCache:
       def __init__(self):
           self.tokens = {}

       async def get_token(self, platform: str) -> str:
           if platform not in self.tokens or self._is_expired(self.tokens[platform]):
               self.tokens[platform] = await self._refresh_token(platform)

           return self.tokens[platform]
   ```

### Database Optimization

**Use Case**: Store historical prices, opportunity history, execution reports.

**Schema**:

```sql
-- Market data
CREATE TABLE markets (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    description TEXT NOT NULL,
    resolution_date TIMESTAMP,
    status TEXT NOT NULL, -- 'active', 'resolved', 'disputed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_markets_platform ON markets(platform);
CREATE INDEX idx_markets_status ON markets(status);

-- Price snapshots
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(id),
    timestamp TIMESTAMP NOT NULL,
    yes_price FLOAT NOT NULL,
    no_price FLOAT NOT NULL,
    yes_size INT NOT NULL,
    no_size INT NOT NULL,
    volume_24h FLOAT
);

CREATE INDEX idx_prices_market_timestamp ON prices(market_id, timestamp DESC);

-- Opportunities
CREATE TABLE opportunities (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(id),
    type TEXT NOT NULL, -- 'within_market', 'cross_platform'
    platform_1 TEXT NOT NULL,
    platform_2 TEXT NOT NULL,
    gross_profit FLOAT NOT NULL,
    net_profit FLOAT NOT NULL,
    profit_pct FLOAT NOT NULL,
    opportunity_score FLOAT NOT NULL,
    detected_at TIMESTAMP NOT NULL,
    expired_at TIMESTAMP
);

CREATE INDEX idx_opportunities_detected ON opportunities(detected_at DESC);
CREATE INDEX idx_opportunities_score ON opportunities(opportunity_score DESC);

-- Executions
CREATE TABLE executions (
    id SERIAL PRIMARY KEY,
    opportunity_id INT NOT NULL REFERENCES opportunities(id),
    position_size INT NOT NULL,
    total_cost FLOAT NOT NULL,
    actual_profit FLOAT NOT NULL,
    slippage_pct FLOAT NOT NULL,
    executed_at TIMESTAMP NOT NULL,
    settled_at TIMESTAMP,
    status TEXT NOT NULL -- 'open', 'settled', 'failed'
);

CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_settled ON executions(settled_at DESC);
```

**Query Optimization**:
```python
# Use connection pooling for database
import asyncpg

db_pool = await asyncpg.create_pool(
    host='localhost',
    database='arbitrage',
    user='postgres',
    password='password',
    min_size=10,
    max_size=50
)

# Batch inserts (1000x faster than individual inserts)
async def batch_insert_prices(prices: list[dict]):
    """Insert multiple price snapshots efficiently"""
    async with db_pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO prices (market_id, timestamp, yes_price, no_price, yes_size, no_size)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [
                (p['market_id'], p['timestamp'], p['yes_price'],
                 p['no_price'], p['yes_size'], p['no_size'])
                for p in prices
            ]
        )
```

### Monitoring and Alerting

**Key Metrics**:
- Opportunities detected per hour
- Execution success rate
- Average profit per trade
- Slippage (expected vs actual)
- Latency (data update → order placement)
- Platform uptime

**Dashboard**:
```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Bot performance metrics"""
    time_window: timedelta
    opportunities_detected: int
    opportunities_executed: int
    execution_rate: float
    total_profit: float
    avg_profit_per_trade: float
    avg_slippage: float
    avg_latency_ms: float
    platform_uptime: dict[str, float]

async def get_performance_metrics(
    time_window: timedelta = timedelta(hours=24)
) -> PerformanceMetrics:
    """Calculate performance metrics for dashboard"""
    cutoff = datetime.now() - time_window

    async with db_pool.acquire() as conn:
        # Opportunities detected
        opp_count = await conn.fetchval(
            "SELECT COUNT(*) FROM opportunities WHERE detected_at > $1",
            cutoff
        )

        # Executions
        exec_count = await conn.fetchval(
            "SELECT COUNT(*) FROM executions WHERE executed_at > $1",
            cutoff
        )

        # Total profit
        total_profit = await conn.fetchval(
            """
            SELECT COALESCE(SUM(actual_profit), 0)
            FROM executions
            WHERE executed_at > $1 AND status = 'settled'
            """,
            cutoff
        ) or 0.0

        # Average slippage
        avg_slippage = await conn.fetchval(
            """
            SELECT COALESCE(AVG(slippage_pct), 0)
            FROM executions
            WHERE executed_at > $1
            """,
            cutoff
        ) or 0.0

        # Platform uptime (stub - implement with monitoring service)
        platform_uptime = {
            "polymarket": 0.998,
            "kalshi": 0.995,
            "predictit": 0.990
        }

        return PerformanceMetrics(
            time_window=time_window,
            opportunities_detected=opp_count,
            opportunities_executed=exec_count,
            execution_rate=exec_count / opp_count if opp_count > 0 else 0.0,
            total_profit=total_profit,
            avg_profit_per_trade=total_profit / exec_count if exec_count > 0 else 0.0,
            avg_slippage=avg_slippage,
            avg_latency_ms=45.0,  # Stub - implement with timing logs
            platform_uptime=platform_uptime
        )
```

**Alerts**:
```python
class AlertManager:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url  # Slack, Discord, Telegram

    async def send_alert(self, message: str, level: str = "info"):
        """Send alert via webhook"""
        payload = {
            "text": f"[{level.upper()}] {message}",
            "timestamp": time.time()
        }

        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)

    async def monitor_performance(self):
        """Monitor and alert on performance issues"""
        while True:
            metrics = await get_performance_metrics(timedelta(hours=1))

            # Alert if execution rate drops
            if metrics.execution_rate < 0.5:
                await self.send_alert(
                    f"Low execution rate: {metrics.execution_rate:.1%}",
                    level="warning"
                )

            # Alert if slippage increases
            if metrics.avg_slippage > 2.0:
                await self.send_alert(
                    f"High slippage: {metrics.avg_slippage:.2%}",
                    level="warning"
                )

            # Alert if profit drops
            if metrics.avg_profit_per_trade < 1.0:
                await self.send_alert(
                    f"Low profit per trade: ${metrics.avg_profit_per_trade:.2f}",
                    level="info"
                )

            await asyncio.sleep(3600)  # Check hourly
```

---

## Implementation Examples

### Complete Bot Example (Python)

```python
import asyncio
import time
from typing import Optional
from dataclasses import dataclass

# Configuration
CONFIG = {
    "total_capital": 10000,  # $10,000
    "min_profit_pct": 1.5,   # 1.5% minimum profit for cross-platform
    "max_slippage_pct": 2.0, # 2% max slippage
    "min_liquidity": 1000,   # $1000 minimum per leg
}

@dataclass
class BotState:
    """Track bot state"""
    is_running: bool = False
    total_opportunities: int = 0
    total_executions: int = 0
    total_profit: float = 0.0
    started_at: Optional[float] = None

class CrossPlatformArbitrageBot:
    def __init__(self, config: dict):
        self.config = config
        self.state = BotState()

        # Initialize clients
        self.poly_client = PolymarketClient()
        self.kalshi_client = KalshiClient()
        self.predictit_client = PredictItClient()

        # Initialize components
        self.event_matcher = EventMatcher()
        self.opportunity_detector = OpportunityDetector(FeeCalculator())
        self.capital_allocator = CapitalAllocator(config['total_capital'])
        self.execution_engine = ExecutionEngine(
            self.poly_client,
            self.kalshi_client,
            self.predictit_client
        )
        self.alert_manager = AlertManager(webhook_url="https://hooks.slack.com/...")

        # Market data cache
        self.polymarket_markets = {}
        self.kalshi_markets = {}
        self.predictit_markets = {}

    async def start(self):
        """Start the arbitrage bot"""
        self.state.is_running = True
        self.state.started_at = time.time()

        await self.alert_manager.send_alert("Bot started")

        # Start concurrent tasks
        await asyncio.gather(
            self._stream_polymarket_data(),
            self._stream_kalshi_data(),
            self._poll_predictit_data(),
            self._detect_opportunities(),
            self._monitor_performance(),
            return_exceptions=True
        )

    async def stop(self):
        """Stop the bot gracefully"""
        self.state.is_running = False

        # Generate final report
        report = self.capital_allocator.get_allocation_report()

        await self.alert_manager.send_alert(
            f"Bot stopped. Total profit: ${self.state.total_profit:.2f}, "
            f"Executions: {self.state.total_executions}, "
            f"Win rate: {self.state.total_executions / self.state.total_opportunities:.1%}"
        )

    async def _stream_polymarket_data(self):
        """Stream real-time data from Polymarket"""
        while self.state.is_running:
            try:
                # Get active markets
                markets = await self.poly_client.get_markets(status='active')

                # Update cache
                for market in markets:
                    self.polymarket_markets[market['id']] = market

                # Subscribe to WebSocket updates
                market_ids = [m['id'] for m in markets]
                await self._subscribe_polymarket_ws(market_ids)

            except Exception as e:
                await self.alert_manager.send_alert(
                    f"Polymarket stream error: {e}",
                    level="error"
                )
                await asyncio.sleep(5)  # Retry after 5 seconds

    async def _subscribe_polymarket_ws(self, market_ids: list[str]):
        """Subscribe to Polymarket WebSocket"""
        ws = PolymarketWebSocket(on_update_callback=self._handle_polymarket_update)
        await ws.connect(market_ids)

    async def _handle_polymarket_update(self, data: dict):
        """Handle Polymarket price update"""
        market_id = data['market_id']

        # Update cache
        self.polymarket_markets[market_id] = data

        # Trigger opportunity detection
        await self._check_cross_platform_opportunities(market_id)

    async def _stream_kalshi_data(self):
        """Stream real-time data from Kalshi"""
        while self.state.is_running:
            try:
                # Get active markets
                markets = await self.kalshi_client.get_markets(status='open')

                # Update cache
                for market in markets:
                    self.kalshi_markets[market['ticker']] = market

                # Subscribe to WebSocket updates
                tickers = [m['ticker'] for m in markets]
                await self._subscribe_kalshi_ws(tickers)

            except Exception as e:
                await self.alert_manager.send_alert(
                    f"Kalshi stream error: {e}",
                    level="error"
                )
                await asyncio.sleep(5)

    async def _subscribe_kalshi_ws(self, tickers: list[str]):
        """Subscribe to Kalshi WebSocket"""
        ws = KalshiWebSocket(
            api_key=self.kalshi_client.api_key,
            on_update_callback=self._handle_kalshi_update
        )
        await ws.connect(tickers)

    async def _handle_kalshi_update(self, data: dict):
        """Handle Kalshi price update"""
        ticker = data['ticker']

        # Update cache
        self.kalshi_markets[ticker] = data

        # Trigger opportunity detection
        await self._check_cross_platform_opportunities(ticker)

    async def _poll_predictit_data(self):
        """Poll PredictIt data (no WebSocket available)"""
        while self.state.is_running:
            try:
                markets = await self.predictit_client.get_markets()

                # Update cache
                for market in markets:
                    self.predictit_markets[market['id']] = market

            except Exception as e:
                await self.alert_manager.send_alert(
                    f"PredictIt poll error: {e}",
                    level="error"
                )

            await asyncio.sleep(5)  # Poll every 5 seconds

    async def _detect_opportunities(self):
        """Main opportunity detection loop"""
        while self.state.is_running:
            try:
                # Check within-market arbitrage on each platform
                await self._check_within_market_arb()

                # Cross-platform opportunities are checked in real-time
                # (triggered by price updates)

            except Exception as e:
                await self.alert_manager.send_alert(
                    f"Opportunity detection error: {e}",
                    level="error"
                )

            await asyncio.sleep(1)  # Check every second

    async def _check_within_market_arb(self):
        """Check for within-market arbitrage opportunities"""
        for platform, markets in [
            ("polymarket", self.polymarket_markets),
            ("kalshi", self.kalshi_markets),
            ("predictit", self.predictit_markets)
        ]:
            for market_id, market_data in markets.items():
                opportunity = self.opportunity_detector.detect_within_market_arb(
                    market_id=market_id,
                    description=market_data['description'],
                    platform=platform,
                    yes_price=market_data['yes_price'],
                    no_price=market_data['no_price'],
                    yes_size=market_data['yes_size'],
                    no_size=market_data['no_size']
                )

                if opportunity:
                    await self._execute_opportunity(opportunity)

    async def _check_cross_platform_opportunities(self, market_id: str):
        """Check for cross-platform arbitrage opportunities"""
        # Find matching markets across platforms
        poly_market = self.polymarket_markets.get(market_id)
        if not poly_market:
            return

        # Find matching Kalshi market
        kalshi_match = self.event_matcher.find_cross_platform_match(
            source_platform="polymarket",
            source_market_id=market_id,
            source_description=poly_market['description'],
            target_platform="kalshi",
            target_markets=list(self.kalshi_markets.values())
        )

        if kalshi_match:
            kalshi_market = self.kalshi_markets[kalshi_match]

            opportunity = self.opportunity_detector.detect_cross_platform_arb(
                market_id=market_id,
                description=poly_market['description'],
                platform_1="polymarket",
                platform_1_yes_price=poly_market['yes_price'],
                platform_1_size=poly_market['yes_size'],
                platform_2="kalshi",
                platform_2_yes_price=kalshi_market['yes_price'],
                platform_2_size=kalshi_market['yes_size']
            )

            if opportunity:
                await self._execute_opportunity(opportunity)

    async def _execute_opportunity(self, opportunity: ArbitrageOpportunity):
        """Execute detected arbitrage opportunity"""
        # Check if we have capital
        position_size = self.capital_allocator.calculate_position_size(opportunity)

        if not position_size:
            return  # Skip - insufficient capital

        # Log opportunity
        self.state.total_opportunities += 1

        await self.alert_manager.send_alert(
            f"Opportunity detected: {opportunity.description} "
            f"({opportunity.profit_pct:.2f}% profit, size: {position_size})"
        )

        # Execute
        result = await self.execution_engine.execute_arbitrage(opportunity)

        if result['success']:
            # Update capital allocation
            self.capital_allocator.allocate_capital(opportunity, position_size)

            # Update state
            self.state.total_executions += 1
            self.state.total_profit += result['actual_profit']

            await self.alert_manager.send_alert(
                f"Execution SUCCESS: {opportunity.description} "
                f"(profit: ${result['actual_profit']:.2f}, "
                f"slippage: {result['slippage_pct']:.2f}%)"
            )
        else:
            await self.alert_manager.send_alert(
                f"Execution FAILED: {opportunity.description} "
                f"(error: {result['error']})",
                level="warning"
            )

    async def _monitor_performance(self):
        """Monitor bot performance and alert on issues"""
        while self.state.is_running:
            # Generate performance report
            metrics = await get_performance_metrics(timedelta(hours=1))

            # Check for issues
            if metrics.execution_rate < 0.5:
                await self.alert_manager.send_alert(
                    f"Low execution rate: {metrics.execution_rate:.1%}",
                    level="warning"
                )

            if metrics.avg_slippage > self.config['max_slippage_pct']:
                await self.alert_manager.send_alert(
                    f"High slippage: {metrics.avg_slippage:.2f}%",
                    level="warning"
                )

            # Log performance
            print(f"""
            === Performance Report ===
            Opportunities: {metrics.opportunities_detected}
            Executions: {metrics.opportunities_executed} ({metrics.execution_rate:.1%})
            Total Profit: ${metrics.total_profit:.2f}
            Avg Profit/Trade: ${metrics.avg_profit_per_trade:.2f}
            Avg Slippage: {metrics.avg_slippage:.2f}%
            """)

            await asyncio.sleep(3600)  # Report hourly

# Main entry point
async def main():
    bot = CrossPlatformArbitrageBot(CONFIG)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Conclusion

Cross-platform prediction market arbitrage offers significant profit opportunities in 2026, with consistent 2-5% spreads between platforms like Polymarket, Kalshi, and PredictIt. However, success requires:

1. **Technical Infrastructure**: Low-latency connections, parallel execution, robust error handling
2. **Capital Management**: Kelly criterion position sizing, platform diversification, cash reserves
3. **Risk Management**: Execution risk mitigation, resolution risk avoidance, platform health monitoring
4. **Regulatory Compliance**: KYC/AML adherence, geographic restrictions, tax reporting

**Expected Returns** (based on 2024-2026 data):
- Conservative strategy: 15-25% annualized (quarter Kelly, high-profit opportunities only)
- Aggressive strategy: 30-50% annualized (full Kelly, all opportunities, higher risk)

**Key Risks**:
- Platform shutdowns (regulatory)
- Resolution divergence (oracle mismatch)
- Liquidity dry-ups (market crises)
- Execution failures (network latency)

**Competitive Landscape**:
- $40M+ extracted by arbitrageurs in 2024-2025 (IMDEA study)
- Top 0.51% of users earn >$1,000
- High barriers to entry (technical complexity, capital requirements)

---

## Sources

- [The Accuracy War: PredictIt vs. Kalshi vs. Polymarket](https://markets.financialcontent.com/stocks/article/predictstreet-2026-1-19-the-accuracy-war-predictit-vs-kalshi-vs-polymarket)
- [The Great Prediction War: Polymarket and Kalshi Battle](https://markets.financialcontent.com/stocks/article/predictstreet-2026-2-1-the-great-prediction-war-polymarket-and-kalshi-battle-for-the-soul-of-information-finance)
- [Prediction Market Arbitrage Guide: Strategies for 2026](https://newyorkcityservers.com/blog/prediction-market-arbitrage-guide)
- [The Limits of Arbitrage - Rajiv Sethi](https://rajivsethi.substack.com/p/the-limits-of-arbitrage)
- [Top Polymarket Traders Show Exceptional Performance Metrics](https://phemex.com/news/article/top-polymarket-traders-show-exceptional-performance-metrics-57796)
- [OpenClaw Bot Nets $115K in a Week on Polymarket](https://phemex.com/news/article/openclaw-bot-generates-115k-in-a-week-on-polymarket-57582)
- [Best Prediction Market Bots & Tools for Automated Trading](https://newyorkcityservers.com/blog/best-prediction-market-bots-tools)
- [Automated Trading on Polymarket: Bots, Arbitrage & Execution Strategies](https://www.quantvps.com/blog/automated-trading-polymarket)
- [Arbitrage Bots Dominate Polymarket With Millions in Profits](https://finance.yahoo.com/news/arbitrage-bots-dominate-polymarket-millions-100000888.html)
- [Building a Prediction Market Arbitrage Bot: Technical Implementation](https://navnoorbawa.substack.com/p/building-a-prediction-market-arbitrage)
- [GitHub: prediction-market-arbitrage-bot](https://github.com/realfishsam/prediction-market-arbitrage-bot)
- [GitHub: polymarket-kalshi-arbitrage-bot](https://github.com/TopTrenDev/polymarket-kalshi-arbitrage-bot)
- [How I Built a "Risk-Free" Arbitrage Bot for Polymarket & Kalshi](https://dev.to/realfishsam/how-i-built-a-risk-free-arbitrage-bot-for-polymarket-kalshi-4f)
- [Kalshi API Documentation](https://docs.kalshi.com/welcome)
- [Kalshi API: The Complete Developer's Guide](https://zuplo.com/learning-center/kalshi-api)
- [Market Making on Prediction Markets: Complete 2026 Guide](https://newyorkcityservers.com/blog/prediction-market-making-guide)
- [Kalshi's Business Breakdown & Founding Story](https://research.contrary.com/company/kalshi)
- [Kalshi vs Polymarket: Which Is Superior?](https://rotogrinders.com/best-prediction-market-apps/kalshi-vs-polymarket)
- [Polymarket Liquidity Rewards](https://docs.polymarket.com/polymarket-learn/trading/liquidity-rewards)
- [Top 10 Polymarket Trading Strategies](https://www.datawallet.com/crypto/top-polymarket-trading-strategies)
- [How Polymarket Scales Profitability Through Transaction Fees](https://www.ainvest.com/news/polymarket-scales-profitability-transaction-fees-liquidity-incentives-decentralized-prediction-market-2601/)
- [PredictIt - Wikipedia](https://en.wikipedia.org/wiki/PredictIt)
- [How I Turned $400 into $400,000 Trading Political Futures](https://luckboxmagazine.com/trends/how-i-turned-400-into-400000-trading-political-futures/)
- [Political Prediction Markets: How To Bet On Politics](https://www.bettingusa.com/prediction-markets/politics/)
- [Polymarket Geographic Restrictions](https://docs.polymarket.com/polymarket-learn/FAQ/geoblocking)
- [Polymarket Supported and Restricted Countries (2026)](https://www.datawallet.com/crypto/polymarket-restricted-countries)
- [Prediction Market Regulation: Legal Compliance Guide](https://heitnerlegal.com/2025/10/22/prediction-market-regulation-legal-compliance-guide-for-polymarket-kalshi-and-event-contract-startups/)
- [Why Prediction Markets Matter in 2026: Regulation, Licensing & Compliance](https://aurum.law/newsroom/Why-Prediction-Markets-Matter-in-2026)
- [The Future of Prediction Markets and Federal vs. State Regulation](https://www.ainvest.com/news/future-prediction-markets-critical-role-federal-state-regulation-2512/)
- [Multi-Manager Hedge Funds and Modern Allocation Strategies](https://am.gs.com/en-us/advisors/insights/article/2024/multi-manager-hedge-funds-modern-allocation-strategies)
- [Principles of Asset Allocation - CFA Institute](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/principles-asset-allocation)
- [Systematic Edges in Prediction Markets - QuantPedia](https://quantpedia.com/systematic-edges-in-prediction-markets/)

---

**Document Version**: 1.0
**Last Updated**: February 4, 2026
**Research Conducted By**: sym-web-research agent
**Total Sources**: 40+
**Coverage**: Polymarket, Kalshi, PredictIt, Augur, Manifold Markets
