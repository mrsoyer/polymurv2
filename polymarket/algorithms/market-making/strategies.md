# Market Making Strategies for Prediction Markets

## Overview

Market making on prediction markets involves providing liquidity by simultaneously posting buy (bid) and sell (ask) orders, profiting from the bid-ask spread while managing inventory risk and market volatility. This guide covers institutional-grade strategies adapted for platforms like Polymarket.

## Table of Contents

1. [Fundamentals](#fundamentals)
2. [Spread Management](#spread-management)
3. [Order Book Strategies](#order-book-strategies)
4. [Competitive Dynamics](#competitive-dynamics)
5. [Implementation Examples](#implementation-examples)

---

## Fundamentals

### What is Market Making?

Market making is the practice of simultaneously providing bid (buy) and ask (sell) orders to facilitate trading and earn profit from the spread. Market makers:

- **Provide liquidity**: Enable traders to buy/sell immediately
- **Earn from spreads**: Profit from the difference between bid and ask prices
- **Manage inventory**: Balance holdings to avoid directional risk
- **Compete on speed**: Sub-millisecond execution critical for success

### Why Prediction Markets?

Prediction markets offer unique opportunities for market makers:

- **Binary outcomes**: Tokens resolve to $1 (win) or $0 (loss)
- **High volume**: $2B+ weekly trading on platforms like Kalshi
- **Reward programs**: Polymarket allocated $12M in LP rewards (2025)
- **API access**: Full programmatic trading via WebSocket/REST APIs

### 2026 Market Landscape

**Market Size & Growth:**
- $37B+ trading volumes (2026)
- $5.6B+ institutional investments
- Projected $10B industry value by 2030

**Key Players:**
- Polymarket (crypto, CLOB on Polygon)
- Kalshi (regulated US, CFTC-approved)
- Augur (AMM-based)

**Competitive Environment:**
- Algorithmic trading dominates
- 4-6 cent arbitrage opportunities (fleeting)
- AI/ML integration accelerating

---

## Spread Management

### Basic Spread Calculation

The bid-ask spread is the core profit mechanism:

```
Spread = Ask Price - Bid Price
Mid Price = (Bid + Ask) / 2
Spread (%) = Spread / Mid Price × 100
```

**Example:**
```javascript
// Cryptocurrency at $1000
const bidPrice = 999.6;
const askPrice = 1000.4;
const spread = askPrice - bidPrice; // $0.8
const midPrice = (bidPrice + askPrice) / 2; // $1000
const spreadPercent = (spread / midPrice) * 100; // 0.08%
```

### Dynamic Spread Adjustment

Spreads must adapt to market conditions in real-time:

#### Volatility-Based Spreads

Higher volatility → wider spreads (more risk):

```python
def calculate_spread(base_spread, volatility, volatility_threshold=0.02):
    """
    Adjust spread based on market volatility.

    Args:
        base_spread: Minimum spread (e.g., 0.005 = 0.5%)
        volatility: Current volatility (e.g., 0.03 = 3%)
        volatility_threshold: Baseline volatility

    Returns:
        Adjusted spread percentage
    """
    volatility_multiplier = max(1.0, volatility / volatility_threshold)
    return base_spread * volatility_multiplier

# Example
base_spread = 0.005  # 0.5%
current_volatility = 0.04  # 4%
spread = calculate_spread(base_spread, current_volatility)
# Result: 0.01 (1.0%)
```

#### Inventory-Adjusted Spreads

Skew spreads to encourage inventory rebalancing:

```python
def inventory_adjusted_spread(
    mid_price,
    base_spread,
    inventory,
    target_inventory,
    max_inventory
):
    """
    Adjust bid/ask prices based on inventory position.

    Args:
        mid_price: Market mid price
        base_spread: Base spread in dollars
        inventory: Current inventory (positive = long, negative = short)
        target_inventory: Desired inventory level
        max_inventory: Maximum allowed inventory deviation

    Returns:
        (bid_price, ask_price)
    """
    inventory_skew = (inventory - target_inventory) / max_inventory

    # If long (positive inventory), lower ask, raise bid to encourage selling
    # If short (negative inventory), raise ask, lower bid to encourage buying
    bid_skew = -inventory_skew * (base_spread / 2)
    ask_skew = inventory_skew * (base_spread / 2)

    bid_price = mid_price - (base_spread / 2) + bid_skew
    ask_price = mid_price + (base_spread / 2) + ask_skew

    return bid_price, ask_price

# Example: Long 100 tokens (target 0, max 200)
bid, ask = inventory_adjusted_spread(
    mid_price=0.50,
    base_spread=0.02,
    inventory=100,
    target_inventory=0,
    max_inventory=200
)
# Result: bid=0.485, ask=0.515 (tighter on ask side)
```

#### Competition-Based Spreads

Monitor competitor quotes and maintain competitiveness:

```javascript
function getCompetitiveSpread(orderBook, targetPosition) {
  // Get best competing quotes
  const bestBid = orderBook.bids[0].price;
  const bestAsk = orderBook.asks[0].price;
  const currentSpread = bestAsk - bestBid;

  // Position within spread based on aggressiveness
  // targetPosition: 0 = match best, 0.5 = middle, 1.0 = opposite
  const ourBid = bestBid + (currentSpread * targetPosition * 0.5);
  const ourAsk = bestAsk - (currentSpread * targetPosition * 0.5);

  return { bid: ourBid, ask: ourAsk };
}

// Example: Position at 30% into spread
const quotes = getCompetitiveSpread(orderBook, 0.3);
// If spread is 0.50-0.52, places orders at 0.503-0.517
```

### Spread Optimization: The Stoikov Model

The **Avellaneda-Stoikov model** optimizes spreads using stochastic calculus, balancing profit against inventory risk.

#### Key Components

1. **Reservation Price**: Internal valuation adjusted for inventory risk

```
r = s - q × γ × σ² × (T - t)
```

Where:
- `r` = reservation price
- `s` = market mid price
- `q` = inventory position (positive = long)
- `γ` = risk aversion parameter
- `σ` = volatility
- `T - t` = time remaining in session

2. **Optimal Spread**: Distance between bid/ask around reservation price

```
δ = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/κ)
```

Where:
- `δ` = optimal spread
- `κ` = order book liquidity parameter

#### Implementation

```python
import math

class StoikovMarketMaker:
    def __init__(self, risk_aversion=0.1, session_duration=3600):
        self.gamma = risk_aversion  # Risk aversion
        self.T = session_duration   # Session duration in seconds
        self.start_time = time.time()

    def calculate_reservation_price(
        self,
        mid_price,
        inventory,
        volatility
    ):
        """Calculate Stoikov reservation price."""
        time_remaining = self.T - (time.time() - self.start_time)
        time_remaining = max(time_remaining, 1)  # Avoid division by zero

        inventory_adjustment = (
            inventory * self.gamma * (volatility ** 2) * time_remaining
        )

        reservation_price = mid_price - inventory_adjustment
        return reservation_price

    def calculate_optimal_spread(
        self,
        volatility,
        order_book_liquidity=1.0
    ):
        """Calculate optimal bid-ask spread."""
        time_remaining = self.T - (time.time() - self.start_time)
        time_remaining = max(time_remaining, 1)

        kappa = order_book_liquidity

        spread = (
            self.gamma * (volatility ** 2) * time_remaining +
            (2 / self.gamma) * math.log(1 + self.gamma / kappa)
        )

        return spread

    def get_quotes(self, mid_price, inventory, volatility):
        """Get bid/ask quotes."""
        reservation = self.calculate_reservation_price(
            mid_price, inventory, volatility
        )
        spread = self.calculate_optimal_spread(volatility)

        bid = reservation - spread / 2
        ask = reservation + spread / 2

        return bid, ask

# Example usage
mm = StoikovMarketMaker(risk_aversion=0.1, session_duration=3600)

# Market conditions
mid_price = 0.50
inventory = 100  # Long 100 tokens
volatility = 0.02  # 2% volatility

bid, ask = mm.get_quotes(mid_price, inventory, volatility)
print(f"Bid: {bid:.4f}, Ask: {ask:.4f}")
# Output: Bid: 0.4800, Ask: 0.5200 (approximate)
```

#### Parameter Tuning

| Parameter | Low Value | High Value | Effect |
|-----------|-----------|------------|--------|
| **γ (risk_aversion)** | 0.01 | 1.0 | Higher = more aggressive inventory adjustment |
| **σ (volatility)** | 0.01 | 0.10 | Higher = wider spreads |
| **κ (liquidity)** | 0.1 | 10.0 | Higher = tighter spreads (more competition) |

**Recommended Starting Values:**
- γ = 0.1 (moderate risk aversion)
- Update volatility every 5-10 minutes
- Estimate κ from order book depth

---

## Order Book Strategies

### Central Limit Order Book (CLOB)

Polymarket uses a CLOB on Polygon, providing full API access:

```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('wss://ws-subscriptions-clob.polymarket.com/ws/market');

ws.on('message', (data) => {
  const update = JSON.parse(data);

  if (update.event_type === 'book') {
    // Order book snapshot
    processOrderBook(update.bids, update.asks);
  }

  if (update.event_type === 'last_trade_price') {
    // Trade execution
    handleTrade(update.price, update.size);
  }
});

// Latency typically < 50ms
```

### Order Placement Strategy

**Two-Sided Quoting:**

```python
class OrderBookMaker:
    def __init__(self, api_client, market_id):
        self.client = api_client
        self.market_id = market_id
        self.active_orders = {}

    async def place_two_sided_quote(self, bid_price, bid_size, ask_price, ask_size):
        """Place simultaneous bid and ask orders."""

        # Cancel existing orders
        await self.cancel_all_orders()

        # Place new orders in parallel
        bid_order = self.client.create_order(
            market_id=self.market_id,
            side='BUY',
            price=bid_price,
            size=bid_size,
            order_type='LIMIT'
        )

        ask_order = self.client.create_order(
            market_id=self.market_id,
            side='SELL',
            price=ask_price,
            size=ask_size,
            order_type='LIMIT'
        )

        # Execute in parallel
        results = await asyncio.gather(bid_order, ask_order)

        # Track active orders
        self.active_orders['bid'] = results[0]['order_id']
        self.active_orders['ask'] = results[1]['order_id']

        return results

    async def cancel_all_orders(self):
        """Cancel all active orders."""
        if not self.active_orders:
            return

        cancel_tasks = [
            self.client.cancel_order(order_id)
            for order_id in self.active_orders.values()
        ]

        await asyncio.gather(*cancel_tasks, return_exceptions=True)
        self.active_orders = {}
```

### Order Size Management

**Layered Liquidity:**

Provide depth with multiple order levels:

```python
def calculate_layered_orders(
    mid_price,
    base_spread,
    total_size,
    num_layers=5
):
    """
    Create layered orders at increasing distances from mid.

    Args:
        mid_price: Current market mid price
        base_spread: Spread for first layer
        total_size: Total size to deploy
        num_layers: Number of order layers

    Returns:
        List of (price, size) tuples for bids and asks
    """
    size_per_layer = total_size / num_layers

    bids = []
    asks = []

    for i in range(num_layers):
        # Exponentially increasing spread
        spread_multiplier = 1 + (i * 0.5)
        layer_spread = base_spread * spread_multiplier

        bid_price = mid_price - layer_spread
        ask_price = mid_price + layer_spread

        bids.append((bid_price, size_per_layer))
        asks.append((ask_price, size_per_layer))

    return bids, asks

# Example
bids, asks = calculate_layered_orders(
    mid_price=0.50,
    base_spread=0.01,
    total_size=1000,
    num_layers=5
)

# Result:
# Bids: [(0.49, 200), (0.485, 200), (0.4775, 200), ...]
# Asks: [(0.51, 200), (0.515, 200), (0.5225, 200), ...]
```

### Stale Order Management

Cancel orders when market conditions change:

```javascript
class StaleOrderManager {
  constructor(api, updateIntervalMs = 500) {
    this.api = api;
    this.interval = updateIntervalMs;
    this.lastMidPrice = null;
    this.priceThreshold = 0.005; // 0.5% movement triggers cancel
  }

  async monitorAndUpdate(market) {
    setInterval(async () => {
      const currentMid = await this.api.getMidPrice(market);

      if (!this.lastMidPrice) {
        this.lastMidPrice = currentMid;
        return;
      }

      const priceChange = Math.abs(currentMid - this.lastMidPrice) / this.lastMidPrice;

      if (priceChange > this.priceThreshold) {
        console.log(`Price moved ${(priceChange * 100).toFixed(2)}%, canceling orders`);
        await this.api.cancelAllOrders(market);
        this.lastMidPrice = currentMid;
      }
    }, this.interval);
  }
}

// Usage
const manager = new StaleOrderManager(apiClient);
manager.monitorAndUpdate('MARKET_ID');
```

---

## Competitive Dynamics

### Market Maker Competition

**2026 Landscape:**
- Algorithmic trading dominates
- Millisecond-level competition
- AI/ML agents replacing humans for pricing

**Success Factors:**
1. **Speed**: Sub-50ms order updates
2. **Information**: Real-time news feeds integration
3. **Capital**: Sufficient liquidity across multiple markets
4. **Risk Management**: Automated hedging and position limits

### Arbitrage Considerations

**Cross-Platform Arbitrage:**

Bots exploit 4-6 cent price gaps between platforms:

```python
def detect_arbitrage_opportunity(
    polymarket_bid,
    polymarket_ask,
    kalshi_bid,
    kalshi_ask,
    min_profit=0.02
):
    """
    Detect arbitrage between two platforms.

    Args:
        polymarket_bid/ask: Polymarket prices
        kalshi_bid/ask: Kalshi prices
        min_profit: Minimum profit threshold

    Returns:
        Arbitrage opportunity dict or None
    """
    # Buy on Polymarket, sell on Kalshi
    poly_to_kalshi_profit = kalshi_bid - polymarket_ask

    # Buy on Kalshi, sell on Polymarket
    kalshi_to_poly_profit = polymarket_bid - kalshi_ask

    if poly_to_kalshi_profit > min_profit:
        return {
            'direction': 'poly_to_kalshi',
            'buy_platform': 'polymarket',
            'sell_platform': 'kalshi',
            'buy_price': polymarket_ask,
            'sell_price': kalshi_bid,
            'profit': poly_to_kalshi_profit
        }

    if kalshi_to_poly_profit > min_profit:
        return {
            'direction': 'kalshi_to_poly',
            'buy_platform': 'kalshi',
            'sell_platform': 'polymarket',
            'buy_price': kalshi_ask,
            'sell_price': polymarket_bid,
            'profit': kalshi_to_poly_profit
        }

    return None

# Example
opportunity = detect_arbitrage_opportunity(
    polymarket_bid=0.48,
    polymarket_ask=0.52,
    kalshi_bid=0.55,
    kalshi_ask=0.57,
    min_profit=0.02
)
# Result: {'direction': 'poly_to_kalshi', 'profit': 0.03, ...}
```

**Note:** These opportunities are fleeting (seconds) and dominated by high-frequency bots.

### 15-Minute Crypto Markets

Polymarket's 15-minute BTC/ETH/SOL/XRP markets offer unique opportunities:

**Characteristics:**
- High volatility (crypto price resolution)
- Dynamic taker fees (up to 3%) to discourage HFT
- OpenClaw bot: $115K/week, 13,000 trades

**Strategy:**

```python
class FifteenMinuteMarketMaker:
    """
    Specialized MM for 15-minute crypto markets.
    """

    def __init__(self, crypto_feed):
        self.crypto_feed = crypto_feed  # External crypto price feed
        self.positions = {}

    def calculate_fair_value(self, strike_price, current_crypto_price, minutes_remaining):
        """
        Estimate fair value based on crypto price and time.

        Args:
            strike_price: Resolution price threshold
            current_crypto_price: Current crypto spot price
            minutes_remaining: Time until market resolution

        Returns:
            Estimated probability (0-1)
        """
        # Distance to strike
        distance = current_crypto_price - strike_price
        distance_pct = distance / strike_price

        # Time decay factor (less certainty with more time)
        time_factor = 1 - (minutes_remaining / 15)

        # Simple logistic probability
        z = distance_pct * 10 * (1 + time_factor)
        probability = 1 / (1 + math.exp(-z))

        return probability

    def get_quotes(self, market_info):
        """Generate bid/ask quotes."""
        fair_value = self.calculate_fair_value(
            strike_price=market_info['strike'],
            current_crypto_price=self.crypto_feed.get_price(market_info['asset']),
            minutes_remaining=market_info['minutes_remaining']
        )

        # Wide spreads due to volatility
        base_spread = 0.15  # 15 cents

        bid = fair_value - (base_spread / 2)
        ask = fair_value + (base_spread / 2)

        # Clamp to valid range
        bid = max(0.01, min(bid, 0.99))
        ask = max(0.01, min(ask, 0.99))

        return bid, ask

# Example
mm = FifteenMinuteMarketMaker(crypto_feed)
bid, ask = mm.get_quotes({
    'asset': 'BTC',
    'strike': 95000,
    'minutes_remaining': 10
})
```

**Key Insight from OpenClaw:**
- "Absorbs buy pressure at 80-83¢"
- "Sells back into momentum at 15-20¢ spreads"
- Position sizes: $10 to $60,000 (flexible)

---

## Implementation Examples

### Complete Market Maker Bot

```python
import asyncio
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MarketData:
    mid_price: float
    bid_price: float
    ask_price: float
    volatility: float
    timestamp: float

@dataclass
class Position:
    quantity: int
    avg_entry_price: float

class PolymarketMaker:
    """
    Complete market making bot for Polymarket.
    """

    def __init__(
        self,
        api_client,
        market_id: str,
        risk_aversion: float = 0.1,
        base_spread: float = 0.02,
        order_size: int = 100,
        max_inventory: int = 500,
        target_inventory: int = 0
    ):
        self.api = api_client
        self.market_id = market_id
        self.gamma = risk_aversion
        self.base_spread = base_spread
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.target_inventory = target_inventory

        self.position = Position(quantity=0, avg_entry_price=0)
        self.market_data_history: List[MarketData] = []
        self.active_orders: Dict[str, str] = {}

    async def run(self):
        """Main bot loop."""
        print(f"Starting market maker for {self.market_id}")

        while True:
            try:
                # Fetch market data
                market_data = await self.fetch_market_data()
                self.market_data_history.append(market_data)

                # Calculate volatility
                volatility = self.calculate_volatility()

                # Get optimal quotes
                bid, ask = self.calculate_quotes(
                    market_data.mid_price,
                    volatility
                )

                # Check inventory limits
                if abs(self.position.quantity) > self.max_inventory:
                    print(f"Inventory limit reached: {self.position.quantity}")
                    await self.reduce_inventory()
                    continue

                # Place orders
                await self.place_two_sided_orders(bid, ask)

                # Wait before next update
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    async def fetch_market_data(self) -> MarketData:
        """Fetch current market data from API."""
        order_book = await self.api.get_order_book(self.market_id)

        best_bid = order_book['bids'][0]['price'] if order_book['bids'] else 0
        best_ask = order_book['asks'][0]['price'] if order_book['asks'] else 1
        mid_price = (best_bid + best_ask) / 2

        # Calculate volatility from recent trades
        trades = await self.api.get_recent_trades(self.market_id, limit=100)
        prices = [t['price'] for t in trades]
        volatility = self.std_dev(prices) if len(prices) > 10 else 0.02

        return MarketData(
            mid_price=mid_price,
            bid_price=best_bid,
            ask_price=best_ask,
            volatility=volatility,
            timestamp=time.time()
        )

    def calculate_volatility(self, window=60) -> float:
        """Calculate rolling volatility."""
        if len(self.market_data_history) < 10:
            return 0.02  # Default

        recent_data = self.market_data_history[-window:]
        prices = [d.mid_price for d in recent_data]

        return self.std_dev(prices)

    @staticmethod
    def std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def calculate_quotes(self, mid_price: float, volatility: float):
        """Calculate bid/ask using Stoikov model + inventory adjustment."""

        # Inventory adjustment
        inventory_ratio = (self.position.quantity - self.target_inventory) / self.max_inventory
        inventory_adjustment = inventory_ratio * (self.base_spread / 2)

        # Volatility adjustment
        volatility_multiplier = max(1.0, volatility / 0.02)
        adjusted_spread = self.base_spread * volatility_multiplier

        # Apply adjustments
        bid = mid_price - (adjusted_spread / 2) - inventory_adjustment
        ask = mid_price + (adjusted_spread / 2) - inventory_adjustment

        # Clamp to valid range
        bid = max(0.01, min(bid, 0.99))
        ask = max(bid + 0.01, min(ask, 0.99))

        return bid, ask

    async def place_two_sided_orders(self, bid_price: float, ask_price: float):
        """Place or update bid and ask orders."""

        # Cancel existing orders if prices changed significantly
        await self.cancel_stale_orders(bid_price, ask_price)

        # Place new orders
        if 'bid' not in self.active_orders:
            bid_order = await self.api.create_order(
                market_id=self.market_id,
                side='BUY',
                price=bid_price,
                size=self.order_size
            )
            self.active_orders['bid'] = bid_order['order_id']

        if 'ask' not in self.active_orders:
            ask_order = await self.api.create_order(
                market_id=self.market_id,
                side='SELL',
                price=ask_price,
                size=self.order_size
            )
            self.active_orders['ask'] = ask_order['order_id']

    async def cancel_stale_orders(self, new_bid: float, new_ask: float):
        """Cancel orders if prices have moved significantly."""
        threshold = 0.005  # 0.5%

        for side in ['bid', 'ask']:
            if side not in self.active_orders:
                continue

            order_id = self.active_orders[side]
            order_info = await self.api.get_order(order_id)

            if order_info['status'] == 'FILLED':
                # Update position
                await self.handle_fill(order_info)
                del self.active_orders[side]
                continue

            current_price = order_info['price']
            target_price = new_bid if side == 'bid' else new_ask

            price_diff = abs(current_price - target_price) / current_price

            if price_diff > threshold:
                await self.api.cancel_order(order_id)
                del self.active_orders[side]

    async def handle_fill(self, order_info: Dict):
        """Update position when order fills."""
        filled_qty = order_info['filled_size']
        fill_price = order_info['price']

        if order_info['side'] == 'BUY':
            new_qty = self.position.quantity + filled_qty
            new_avg = (
                (self.position.quantity * self.position.avg_entry_price + filled_qty * fill_price)
                / new_qty
            ) if new_qty != 0 else 0

            self.position = Position(quantity=new_qty, avg_entry_price=new_avg)
            print(f"BUY filled: {filled_qty} @ {fill_price}, position: {new_qty}")

        else:  # SELL
            new_qty = self.position.quantity - filled_qty
            # Avg entry price stays same for sells
            self.position = Position(quantity=new_qty, avg_entry_price=self.position.avg_entry_price)
            print(f"SELL filled: {filled_qty} @ {fill_price}, position: {new_qty}")

    async def reduce_inventory(self):
        """Aggressively reduce inventory when at limits."""
        if self.position.quantity > self.max_inventory:
            # Too long, need to sell
            market_data = await self.fetch_market_data()
            aggressive_ask = market_data.bid_price + 0.001  # Take best bid

            await self.api.create_order(
                market_id=self.market_id,
                side='SELL',
                price=aggressive_ask,
                size=abs(self.position.quantity - self.target_inventory)
            )

        elif self.position.quantity < -self.max_inventory:
            # Too short, need to buy
            market_data = await self.fetch_market_data()
            aggressive_bid = market_data.ask_price - 0.001  # Take best ask

            await self.api.create_order(
                market_id=self.market_id,
                side='BUY',
                price=aggressive_bid,
                size=abs(self.position.quantity - self.target_inventory)
            )

# Usage
async def main():
    from polymarket_api import PolymarketAPI  # Hypothetical API client

    api = PolymarketAPI(api_key='YOUR_API_KEY')

    bot = PolymarketMaker(
        api_client=api,
        market_id='0x1234...',
        risk_aversion=0.1,
        base_spread=0.02,
        order_size=100,
        max_inventory=500,
        target_inventory=0
    )

    await bot.run()

if __name__ == '__main__':
    asyncio.run(main())
```

### Performance Monitoring

```python
class PerformanceTracker:
    """Track market maker performance metrics."""

    def __init__(self):
        self.trades: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}

    def record_trade(self, trade_info: Dict):
        """Record executed trade."""
        self.trades.append({
            'timestamp': time.time(),
            'side': trade_info['side'],
            'price': trade_info['price'],
            'size': trade_info['size'],
            'fee': trade_info.get('fee', 0)
        })

    def calculate_pnl(self) -> Dict:
        """Calculate P&L metrics."""
        if len(self.trades) < 2:
            return {'realized_pnl': 0, 'num_trades': 0}

        # Match buy/sell pairs
        buys = [t for t in self.trades if t['side'] == 'BUY']
        sells = [t for t in self.trades if t['side'] == 'SELL']

        total_pnl = 0
        matched_pairs = 0

        for sell in sells:
            if not buys:
                break

            buy = buys.pop(0)
            pnl = (sell['price'] - buy['price']) * min(buy['size'], sell['size'])
            fees = buy.get('fee', 0) + sell.get('fee', 0)
            total_pnl += (pnl - fees)
            matched_pairs += 1

        return {
            'realized_pnl': total_pnl,
            'num_trades': len(self.trades),
            'matched_pairs': matched_pairs,
            'avg_pnl_per_pair': total_pnl / matched_pairs if matched_pairs > 0 else 0
        }

    def get_daily_stats(self) -> Dict:
        """Get daily statistics."""
        today = time.strftime('%Y-%m-%d')
        today_trades = [
            t for t in self.trades
            if time.strftime('%Y-%m-%d', time.localtime(t['timestamp'])) == today
        ]

        if not today_trades:
            return {'volume': 0, 'trades': 0, 'pnl': 0}

        volume = sum(t['price'] * t['size'] for t in today_trades)

        return {
            'date': today,
            'volume': volume,
            'num_trades': len(today_trades),
            'pnl': self.daily_pnl.get(today, 0)
        }
```

---

## Summary & Best Practices

### Key Takeaways

1. **Spread Management is Critical**
   - Start with 1-2% base spreads
   - Adjust dynamically for volatility and inventory
   - Use Stoikov model for sophisticated optimization

2. **Speed Matters**
   - Sub-50ms order updates required
   - Cancel stale orders immediately on price moves
   - Use WebSocket for real-time data

3. **Inventory Risk is Real**
   - Set strict position limits
   - Aggressively unwind at limits
   - Use inventory-adjusted spreads

4. **Competition is Fierce**
   - 2026 market dominated by algos and AI
   - Fleeting arbitrage opportunities (seconds)
   - Need capital, speed, and sophistication

### Starting Parameters

For new market makers:

```python
RECOMMENDED_CONFIG = {
    'base_spread': 0.02,           # 2%
    'order_size': 100,             # 100 tokens
    'max_inventory': 500,          # ±500 tokens
    'risk_aversion': 0.1,          # Moderate
    'update_interval': 1.0,        # 1 second
    'volatility_window': 60,       # 60 data points
    'stale_threshold': 0.005,      # 0.5% price move
}
```

### Profitability Expectations

Based on 2026 data:

- **OpenClaw bot**: $115K/week, $1M total (15-min markets)
- **Typical LP**: $200-800/day with $10K capital (pre-competition spike)
- **Current environment**: Reduced profitability, only 0.51% of wallets > $1K profit
- **Capital requirement**: Minimum $5K-10K to be competitive

### Next Steps

1. **Start small**: Test strategies with minimal capital
2. **Monitor performance**: Track spreads, fills, inventory, P&L
3. **Iterate**: Adjust parameters based on market conditions
4. **Scale gradually**: Increase size as profitability proven

---

## Sources

- [Market Making on Prediction Markets: Complete 2026 Guide](https://newyorkcityservers.com/blog/prediction-market-making-guide)
- [Capitalizing on Prediction Markets in 2026](https://www.ainvest.com/news/capitalizing-prediction-markets-2026-institutional-grade-strategies-market-making-arbitrage-2601/)
- [How Kalshi and Polymarket prediction market traders make money - NPR](https://www.npr.org/2026/01/17/nx-s1-5672615/kalshi-polymarket-prediction-market-boom-traders-slang-glossary)
- [4 Predictions for Crypto Prediction Markets in 2026 - Nasdaq](https://www.nasdaq.com/articles/4-predictions-yep-crypto-prediction-markets-2026)
- [Mastering the Market Maker Trading Strategy - EPAM](https://solutionshub.epam.com/blog/post/market-maker-trading-strategy)
- [Crypto Trading 101: Bid Ask Spread - CryptoHopper](https://www.cryptohopper.com/blog/a-deep-dive-into-bid-ask-spread-and-slippage-4538)
- [Market Making Mechanics and Strategies - BlockApex](https://medium.com/blockapex/market-making-mechanics-and-strategies-4daf2122121c)
- [Crypto Market Making Guide - Shift Markets](https://www.shiftmarkets.com/blog/crypto-market-making-guide)
- [OpenClaw Bot Nets $115K in a Week on Polymarket](https://phemex.com/news/article/openclaw-bot-generates-115k-in-a-week-on-polymarket-57582)
- [Automated Market Making on Polymarket](https://news.polymarket.com/p/automated-market-making-on-polymarket)
- [Guide to Hedging Strategies of Crypto Market Makers - DWF Labs](https://www.dwf-labs.com/news/understanding-market-maker-hedging)
- [Guide to the Avellaneda-Stoikov Strategy - Hummingbot](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/)
- [Stoikov market-making algorithm - PLOS ONE](https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0277042)

---

**Last Updated:** 2026-02-04
**Version:** 1.0
**Maintained by:** SYM Web Research Agent
