# Risk Management for Market Making on Prediction Markets

## Overview

Risk management is the cornerstone of profitable market making. This guide covers inventory risk, hedging strategies, position limits, and practical techniques for managing risk on prediction market platforms like Polymarket.

## Table of Contents

1. [Inventory Risk](#inventory-risk)
2. [Hedging Strategies](#hedging-strategies)
3. [Position Limits](#position-limits)
4. [Risk Monitoring](#risk-monitoring)
5. [Crisis Management](#crisis-management)
6. [Implementation](#implementation)

---

## Inventory Risk

### What is Inventory Risk?

**Definition:**

> "Inventory risk is the probability a market maker can't find buyers for his inventory, resulting in the risk of holding more of an asset at exactly the wrong time, e.g. accumulating assets when prices are falling or selling too early when prices are rising."

**Core Problem:**

Market makers provide two-sided liquidity. In trending markets:
- **Downtrend**: Buy orders fill, sell orders don't → accumulate depreciating assets
- **Uptrend**: Sell orders fill, buy orders don't → miss appreciation

**Example:**

```python
# Market scenario: Price falling from 0.60 to 0.40

time_0 = {'price': 0.60, 'inventory': 0}
# Place bid at 0.59, ask at 0.61

time_1 = {'price': 0.58, 'inventory': 100}
# Bid filled at 0.59, ask unfilled

time_2 = {'price': 0.55, 'inventory': 300}
# More bids filled, no asks filled

time_3 = {'price': 0.50, 'inventory': 600}
# Heavily long, price dropped

# Unrealized loss: 600 * (0.50 - 0.59) = -$54
```

### Inventory Value

**Definition:**

Inventory value represents the current value of all assets held in a portfolio, quantified to some benchmark (e.g., USD).

**Calculation:**

```python
def calculate_inventory_value(positions, current_prices):
    """
    Calculate total inventory value.

    Args:
        positions: Dict of {asset: quantity}
        current_prices: Dict of {asset: price}

    Returns:
        Total value in USD
    """
    total_value = 0

    for asset, quantity in positions.items():
        price = current_prices.get(asset, 0)
        value = quantity * price
        total_value += value

    return total_value

# Example
positions = {
    'YES_TOKEN': 500,
    'NO_TOKEN': -200,
    'USDC': 10000
}

prices = {
    'YES_TOKEN': 0.55,
    'NO_TOKEN': 0.45,
    'USDC': 1.0
}

value = calculate_inventory_value(positions, prices)
# Result: 500*0.55 + (-200)*0.45 + 10000 = $10,185
```

### Inventory Risk Factors

**1. Position Size**

Larger positions = higher risk:

```python
def calculate_inventory_risk(
    position_size,
    volatility,
    time_horizon_days=1
):
    """
    Calculate inventory risk (Value at Risk).

    Args:
        position_size: Dollar value of position
        volatility: Daily volatility (e.g., 0.02 = 2%)
        time_horizon_days: Risk horizon

    Returns:
        Value at Risk (95% confidence)
    """
    # 95% confidence = 1.645 standard deviations
    z_score = 1.645

    var = position_size * volatility * z_score * (time_horizon_days ** 0.5)

    return var

# Example: $10K position, 2% volatility
risk = calculate_inventory_risk(10_000, 0.02, 1)
# Result: $329 at risk (95% confidence)
```

**2. Market Volatility**

Higher volatility = greater risk of adverse moves:

```python
class VolatilityCalculator:
    """Calculate market volatility."""

    @staticmethod
    def calculate_historical_volatility(prices, window=20):
        """
        Calculate historical volatility.

        Args:
            prices: List of historical prices
            window: Lookback window

        Returns:
            Annualized volatility
        """
        if len(prices) < window:
            return None

        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

        # Standard deviation
        mean_return = sum(returns[-window:]) / window
        variance = sum((r - mean_return)**2 for r in returns[-window:]) / window
        std_dev = variance ** 0.5

        # Annualize (assuming 365 days)
        annual_volatility = std_dev * (365 ** 0.5)

        return annual_volatility

# Example
prices = [0.50, 0.52, 0.51, 0.53, 0.49, 0.48, 0.51, 0.52, ...]
vol_calc = VolatilityCalculator()
volatility = vol_calc.calculate_historical_volatility(prices)
# Result: 0.35 (35% annual volatility)
```

**3. Time to Resolution**

More time = more uncertainty:

```python
def adjust_risk_for_time(
    base_risk,
    days_to_resolution
):
    """
    Adjust inventory risk based on time to resolution.

    Args:
        base_risk: Base daily risk
        days_to_resolution: Days until market resolves

    Returns:
        Time-adjusted risk
    """
    # Risk scales with square root of time
    time_adjusted_risk = base_risk * (days_to_resolution ** 0.5)

    return time_adjusted_risk

# Example
daily_risk = 100  # $100 daily risk
risk_30d = adjust_risk_for_time(daily_risk, 30)
# Result: $547 (30-day risk)
```

### Mitigation Techniques

**1. Inventory Skew**

Adjust order sizes to encourage rebalancing:

```python
def calculate_skewed_order_sizes(
    base_size,
    current_inventory,
    target_inventory,
    max_inventory,
    skew_intensity=0.5
):
    """
    Calculate inventory-skewed order sizes.

    Args:
        base_size: Base order size
        current_inventory: Current position
        target_inventory: Desired position
        max_inventory: Maximum allowed position
        skew_intensity: How aggressively to skew (0-1)

    Returns:
        (bid_size, ask_size)
    """
    inventory_deviation = current_inventory - target_inventory
    skew_ratio = inventory_deviation / max_inventory

    # If long (positive inventory), reduce buy size, increase sell size
    bid_skew = 1 - (skew_ratio * skew_intensity)
    ask_skew = 1 + (skew_ratio * skew_intensity)

    bid_size = base_size * max(0.1, bid_skew)  # Minimum 10% size
    ask_size = base_size * max(0.1, ask_skew)

    return bid_size, ask_size

# Example: Long 300 tokens (target 0, max 500)
bid, ask = calculate_skewed_order_sizes(
    base_size=100,
    current_inventory=300,
    target_inventory=0,
    max_inventory=500
)
# Result: bid_size=70, ask_size=130
```

**2. Filled Order Delay**

Introduce delay after fills to avoid rapid accumulation:

```python
class FilledOrderDelayManager:
    """Manage delays after order fills."""

    def __init__(self, base_delay_seconds=5):
        self.base_delay = base_delay_seconds
        self.last_fill_time = None

    def should_place_order(self):
        """Check if enough time has passed since last fill."""
        if self.last_fill_time is None:
            return True

        time_since_fill = time.time() - self.last_fill_time

        return time_since_fill >= self.base_delay

    def record_fill(self):
        """Record order fill timestamp."""
        self.last_fill_time = time.time()

    def get_dynamic_delay(self, inventory_ratio):
        """
        Calculate dynamic delay based on inventory.

        Args:
            inventory_ratio: Current inventory / max inventory

        Returns:
            Delay in seconds
        """
        # Higher inventory = longer delay
        dynamic_delay = self.base_delay * (1 + abs(inventory_ratio))

        return dynamic_delay

# Usage
delay_mgr = FilledOrderDelayManager(base_delay_seconds=5)

# After order fills
delay_mgr.record_fill()

# Before placing next order
if delay_mgr.should_place_order():
    place_new_order()
```

**3. Asymmetric Spreads**

Tighten spreads on one side to encourage fills:

```python
def calculate_asymmetric_spreads(
    mid_price,
    base_spread,
    inventory,
    target_inventory,
    max_inventory
):
    """
    Calculate asymmetric spreads based on inventory.

    Args:
        mid_price: Market mid price
        base_spread: Base spread amount
        inventory: Current inventory
        target_inventory: Target inventory
        max_inventory: Maximum inventory

    Returns:
        (bid_price, ask_price)
    """
    inventory_ratio = (inventory - target_inventory) / max_inventory

    # If long (positive), tighten ask, widen bid
    # If short (negative), tighten bid, widen ask
    bid_spread = base_spread * (1 + inventory_ratio * 0.5)
    ask_spread = base_spread * (1 - inventory_ratio * 0.5)

    bid_price = mid_price - bid_spread
    ask_price = mid_price + ask_spread

    return bid_price, ask_price

# Example: Long 200 (target 0, max 500)
bid, ask = calculate_asymmetric_spreads(
    mid_price=0.50,
    base_spread=0.02,
    inventory=200,
    target_inventory=0,
    max_inventory=500
)
# Result: bid=0.472, ask=0.512 (ask tighter to encourage selling)
```

**4. Hanging Orders**

Keep counter-orders open to complete pairs:

```python
class HangingOrderManager:
    """Manage hanging orders for inventory rebalancing."""

    def __init__(self):
        self.hanging_orders = []

    def add_hanging_order(self, order_id, side, price, size):
        """Add a hanging order to track."""
        self.hanging_orders.append({
            'order_id': order_id,
            'side': side,
            'price': price,
            'size': size,
            'created_at': time.time()
        })

    def should_cancel_hanging(self, order, current_price, max_age_seconds=3600):
        """
        Decide if hanging order should be cancelled.

        Args:
            order: Hanging order dict
            current_price: Current market price
            max_age_seconds: Maximum order age

        Returns:
            Boolean
        """
        # Cancel if too old
        age = time.time() - order['created_at']
        if age > max_age_seconds:
            return True

        # Cancel if price moved too far
        price_distance = abs(order['price'] - current_price) / current_price
        if price_distance > 0.10:  # 10%
            return True

        return False

# Usage
hanging_mgr = HangingOrderManager()

# After buy fills, keep sell order "hanging"
hanging_mgr.add_hanging_order(
    order_id='sell_123',
    side='SELL',
    price=0.52,
    size=100
)

# Later, check if should cancel
for order in hanging_mgr.hanging_orders:
    if hanging_mgr.should_cancel_hanging(order, current_price=0.48):
        api.cancel_order(order['order_id'])
```

**5. Ping Pong Strategy**

Only place orders on opposite side of filled orders:

```python
class PingPongStrategy:
    """Ping pong order strategy for tight inventory control."""

    def __init__(self):
        self.last_fill_side = None
        self.pending_offset_order = None

    def handle_fill(self, filled_order):
        """
        Handle order fill and determine next action.

        Args:
            filled_order: Dict with 'side', 'price', 'size'

        Returns:
            Next order to place (opposite side)
        """
        fill_side = filled_order['side']
        fill_price = filled_order['price']
        fill_size = filled_order['size']

        # Place offset order on opposite side
        if fill_side == 'BUY':
            # Just bought, now place SELL
            offset_side = 'SELL'
            offset_price = fill_price + 0.02  # Add spread
        else:
            # Just sold, now place BUY
            offset_side = 'BUY'
            offset_price = fill_price - 0.02  # Subtract spread

        self.last_fill_side = fill_side
        self.pending_offset_order = {
            'side': offset_side,
            'price': offset_price,
            'size': fill_size
        }

        return self.pending_offset_order

    def can_place_two_sided(self):
        """Check if can resume two-sided quoting."""
        # Only resume when no pending offset orders
        return self.pending_offset_order is None

# Usage
ping_pong = PingPongStrategy()

# After buy fills at 0.50
filled = {'side': 'BUY', 'price': 0.50, 'size': 100}
next_order = ping_pong.handle_fill(filled)
# Result: {'side': 'SELL', 'price': 0.52, 'size': 100}
```

---

## Hedging Strategies

### Delta Hedging

**Objective:** Maintain net-zero directional exposure.

**Concept:**

If you accumulate a long position in YES tokens, take an offsetting short position via futures, perpetual swaps, or opposite prediction market positions.

```python
class DeltaHedger:
    """Manage delta-neutral hedging."""

    def __init__(self, hedge_threshold=0.5):
        self.hedge_threshold = hedge_threshold  # Hedge when delta > threshold
        self.hedge_positions = {}

    def calculate_delta(self, positions):
        """
        Calculate portfolio delta.

        Args:
            positions: Dict of {asset: quantity}

        Returns:
            Net delta (positive = long, negative = short)
        """
        # For prediction markets: YES tokens = +1 delta, NO tokens = -1 delta
        delta = 0

        for asset, quantity in positions.items():
            if 'YES' in asset:
                delta += quantity
            elif 'NO' in asset:
                delta -= quantity

        return delta

    def should_hedge(self, delta, max_position=1000):
        """Check if hedging is needed."""
        delta_ratio = abs(delta) / max_position

        return delta_ratio > self.hedge_threshold

    def calculate_hedge_size(self, delta):
        """Calculate required hedge size."""
        # Hedge 100% of excess delta
        hedge_size = -delta  # Opposite direction

        return hedge_size

    def place_hedge(self, hedge_size, hedge_market):
        """
        Place hedge order.

        Args:
            hedge_size: Size to hedge (negative = short)
            hedge_market: Market to hedge in

        Returns:
            Hedge order details
        """
        if hedge_size > 0:
            side = 'BUY'
        else:
            side = 'SELL'
            hedge_size = abs(hedge_size)

        return {
            'market': hedge_market,
            'side': side,
            'size': hedge_size
        }

# Example
hedger = DeltaHedger(hedge_threshold=0.5)

positions = {
    'BTC_15MIN_YES': 600,  # Long 600
    'BTC_15MIN_NO': 100    # Short 100
}

delta = hedger.calculate_delta(positions)  # +500

if hedger.should_hedge(delta, max_position=1000):
    hedge_size = hedger.calculate_hedge_size(delta)
    hedge_order = hedger.place_hedge(hedge_size, 'BTC_PERP_OPPOSITE')
    # Result: SHORT 500 in opposite market
```

### Cross-Market Hedging

**Strategy:** Hedge prediction market positions with correlated markets.

**Example: BTC 15-Minute Markets**

```python
class CrossMarketHedger:
    """Hedge prediction markets with external markets."""

    def __init__(self, crypto_exchange_api):
        self.exchange = crypto_exchange_api

    def hedge_crypto_position(
        self,
        prediction_position,
        crypto_asset='BTC',
        hedge_ratio=1.0
    ):
        """
        Hedge crypto prediction market position.

        Args:
            prediction_position: Position in prediction market
            crypto_asset: Underlying crypto (BTC, ETH, etc.)
            hedge_ratio: Hedge size multiplier

        Returns:
            Hedge trade details
        """
        # Prediction market position = directional bet on crypto price
        # Hedge with opposite position in spot/futures

        hedge_size = abs(prediction_position) * hedge_ratio

        if prediction_position > 0:
            # Long prediction market = expecting price increase
            # Hedge: Short crypto
            hedge_side = 'SELL'
        else:
            # Short prediction market = expecting price decrease
            # Hedge: Long crypto
            hedge_side = 'BUY'

        # Place hedge order on crypto exchange
        hedge_order = self.exchange.create_order(
            symbol=f'{crypto_asset}/USDT',
            side=hedge_side,
            size=hedge_size
        )

        return hedge_order

# Example
hedger = CrossMarketHedger(binance_api)

# Long 500 YES tokens on "BTC > $95K in 15 min"
# Hedge by shorting $500 BTC
hedge = hedger.hedge_crypto_position(
    prediction_position=500,
    crypto_asset='BTC',
    hedge_ratio=1.0
)
```

**Considerations:**
- Hedging costs (fees, spreads)
- Basis risk (prediction market ≠ spot price)
- Capital requirements (need accounts on multiple platforms)

### Partial Hedging

**Strategy:** Hedge only portion of position to reduce cost.

```python
def calculate_partial_hedge(
    position,
    hedge_ratio=0.5,
    confidence=0.8
):
    """
    Calculate partial hedge size.

    Args:
        position: Current position
        hedge_ratio: % to hedge (0.5 = 50%)
        confidence: Confidence in position

    Returns:
        Optimal hedge size
    """
    # Higher confidence = less hedging needed
    adjusted_hedge_ratio = hedge_ratio * (1 - confidence)

    hedge_size = position * adjusted_hedge_ratio

    return hedge_size

# Example: 60% confident in long position
hedge = calculate_partial_hedge(
    position=1000,
    hedge_ratio=0.5,
    confidence=0.6
)
# Result: Hedge 200 (20% of position)
```

---

## Position Limits

### Setting Position Limits

**1. Capital-Based Limits**

```python
def calculate_position_limits(
    total_capital,
    risk_percentage=0.10,
    max_markets=10
):
    """
    Calculate position limits based on capital.

    Args:
        total_capital: Total trading capital
        risk_percentage: Max % of capital at risk per market
        max_markets: Number of markets trading

    Returns:
        Position limit per market
    """
    risk_per_market = total_capital * risk_percentage
    position_limit = risk_per_market / max_markets

    return {
        'total_risk_capital': total_capital * risk_percentage,
        'position_limit_per_market': position_limit,
        'max_loss_per_market': position_limit
    }

# Example: $50K capital
limits = calculate_position_limits(
    total_capital=50_000,
    risk_percentage=0.10,
    max_markets=10
)
# Result: $500 limit per market
```

**2. Volatility-Based Limits**

```python
def calculate_volatility_adjusted_limits(
    base_limit,
    volatility,
    base_volatility=0.02
):
    """
    Adjust position limits based on market volatility.

    Args:
        base_limit: Base position limit
        volatility: Current market volatility
        base_volatility: Baseline volatility

    Returns:
        Adjusted position limit
    """
    volatility_ratio = volatility / base_volatility

    # Reduce limits in high volatility
    adjusted_limit = base_limit / volatility_ratio

    return adjusted_limit

# Example: 5% volatility (high)
adjusted = calculate_volatility_adjusted_limits(
    base_limit=1000,
    volatility=0.05,
    base_volatility=0.02
)
# Result: 400 (60% reduction)
```

**3. Time-Based Limits**

```python
def calculate_time_decay_limits(
    base_limit,
    days_to_resolution,
    threshold_days=7
):
    """
    Reduce limits as market approaches resolution.

    Args:
        base_limit: Base position limit
        days_to_resolution: Days until market resolves
        threshold_days: Days below which to reduce limits

    Returns:
        Time-adjusted limit
    """
    if days_to_resolution >= threshold_days:
        return base_limit

    # Linear reduction as resolution approaches
    time_factor = days_to_resolution / threshold_days
    adjusted_limit = base_limit * time_factor

    return adjusted_limit

# Example: 2 days to resolution
adjusted = calculate_time_decay_limits(
    base_limit=1000,
    days_to_resolution=2,
    threshold_days=7
)
# Result: 285 (71% reduction)
```

### Enforcing Position Limits

```python
class PositionLimitEnforcer:
    """Enforce position limits with automatic actions."""

    def __init__(self, limits):
        self.limits = limits  # Dict of {market: limit}

    def check_position(self, market, current_position):
        """
        Check if position exceeds limit.

        Args:
            market: Market identifier
            current_position: Current position size

        Returns:
            (is_valid, action_required)
        """
        limit = self.limits.get(market, float('inf'))

        if abs(current_position) <= limit:
            return True, None

        # Exceeded limit
        excess = abs(current_position) - limit
        action = {
            'type': 'REDUCE',
            'amount': excess,
            'side': 'SELL' if current_position > 0 else 'BUY'
        }

        return False, action

    def enforce_limits(self, positions):
        """
        Enforce limits across all positions.

        Args:
            positions: Dict of {market: position}

        Returns:
            List of required actions
        """
        actions = []

        for market, position in positions.items():
            is_valid, action = self.check_position(market, position)

            if not is_valid:
                actions.append({
                    'market': market,
                    'action': action
                })

        return actions

# Usage
enforcer = PositionLimitEnforcer(limits={
    'BTC_15MIN': 500,
    'ETH_15MIN': 300,
    'ELECTION': 1000
})

positions = {
    'BTC_15MIN': 700,  # Exceeded
    'ETH_15MIN': 250,  # OK
    'ELECTION': 1200   # Exceeded
}

actions = enforcer.enforce_limits(positions)
# Result: [
#   {'market': 'BTC_15MIN', 'action': {'type': 'REDUCE', 'amount': 200, 'side': 'SELL'}},
#   {'market': 'ELECTION', 'action': {'type': 'REDUCE', 'amount': 200, 'side': 'SELL'}}
# ]
```

### Stop-Loss Mechanisms

```python
class StopLossManager:
    """Manage stop-loss orders for risk control."""

    def __init__(self, stop_loss_pct=0.05):
        self.stop_loss_pct = stop_loss_pct
        self.positions = {}

    def add_position(self, market, entry_price, quantity):
        """Add position with stop-loss level."""
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)

        self.positions[market] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss_price
        }

    def check_stop_loss(self, market, current_price):
        """
        Check if stop-loss triggered.

        Args:
            market: Market identifier
            current_price: Current market price

        Returns:
            (triggered, exit_order)
        """
        if market not in self.positions:
            return False, None

        position = self.positions[market]

        if current_price <= position['stop_loss']:
            # Stop-loss triggered
            exit_order = {
                'market': market,
                'side': 'SELL',
                'size': position['quantity'],
                'price': current_price,
                'reason': 'STOP_LOSS'
            }

            return True, exit_order

        return False, None

# Usage
stop_loss_mgr = StopLossManager(stop_loss_pct=0.05)

# Enter position
stop_loss_mgr.add_position(
    market='BTC_15MIN',
    entry_price=0.50,
    quantity=500
)
# Stop-loss set at 0.475 (5% below entry)

# Check on price update
triggered, exit_order = stop_loss_mgr.check_stop_loss('BTC_15MIN', 0.47)
# Result: triggered=True, exit_order={'side': 'SELL', 'size': 500, ...}
```

---

## Risk Monitoring

### Real-Time Risk Dashboard

```python
class RiskDashboard:
    """Real-time risk monitoring dashboard."""

    def __init__(self):
        self.positions = {}
        self.limits = {}
        self.alerts = []

    def update_position(self, market, quantity, price):
        """Update position."""
        self.positions[market] = {
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'timestamp': time.time()
        }

    def calculate_metrics(self):
        """Calculate risk metrics."""
        total_exposure = sum(
            abs(p['value']) for p in self.positions.values()
        )

        net_delta = sum(
            p['quantity'] for p in self.positions.values()
        )

        num_markets = len(self.positions)

        return {
            'total_exposure': total_exposure,
            'net_delta': net_delta,
            'num_markets': num_markets,
            'avg_position_size': total_exposure / num_markets if num_markets > 0 else 0
        }

    def check_alerts(self):
        """Check for risk alerts."""
        alerts = []
        metrics = self.calculate_metrics()

        # Alert: High total exposure
        if metrics['total_exposure'] > 50_000:
            alerts.append({
                'level': 'HIGH',
                'message': f"Total exposure ${metrics['total_exposure']:.0f} exceeds $50K"
            })

        # Alert: Large net delta
        if abs(metrics['net_delta']) > 1000:
            alerts.append({
                'level': 'MEDIUM',
                'message': f"Net delta {metrics['net_delta']} exceeds ±1000"
            })

        # Alert: Single market concentration
        for market, position in self.positions.items():
            position_pct = abs(position['value']) / metrics['total_exposure']

            if position_pct > 0.30:
                alerts.append({
                    'level': 'MEDIUM',
                    'message': f"Market {market} is {position_pct*100:.0f}% of exposure"
                })

        self.alerts = alerts
        return alerts

    def print_dashboard(self):
        """Print risk dashboard."""
        metrics = self.calculate_metrics()

        print("=== RISK DASHBOARD ===")
        print(f"Total Exposure: ${metrics['total_exposure']:.2f}")
        print(f"Net Delta: {metrics['net_delta']:.0f}")
        print(f"Markets: {metrics['num_markets']}")
        print(f"Avg Position: ${metrics['avg_position_size']:.2f}")
        print("\nPositions:")
        for market, pos in self.positions.items():
            print(f"  {market}: {pos['quantity']} @ {pos['price']} = ${pos['value']:.2f}")

        alerts = self.check_alerts()
        if alerts:
            print("\n⚠️ ALERTS:")
            for alert in alerts:
                print(f"  [{alert['level']}] {alert['message']}")

# Usage
dashboard = RiskDashboard()

dashboard.update_position('BTC_15MIN', 500, 0.52)
dashboard.update_position('ETH_15MIN', -200, 0.48)
dashboard.update_position('ELECTION', 800, 0.55)

dashboard.print_dashboard()
```

### Value at Risk (VaR)

```python
def calculate_var(
    positions,
    prices,
    volatilities,
    confidence=0.95,
    time_horizon_days=1
):
    """
    Calculate portfolio Value at Risk.

    Args:
        positions: Dict of {market: quantity}
        prices: Dict of {market: price}
        volatilities: Dict of {market: volatility}
        confidence: VaR confidence level (0.95 = 95%)
        time_horizon_days: Risk time horizon

    Returns:
        Value at Risk
    """
    from scipy.stats import norm

    # Z-score for confidence level
    z_score = norm.ppf(confidence)

    total_var = 0

    for market, quantity in positions.items():
        price = prices[market]
        volatility = volatilities[market]

        position_value = abs(quantity * price)
        position_var = position_value * volatility * z_score * (time_horizon_days ** 0.5)

        total_var += position_var ** 2  # Sum of variances

    # Portfolio VaR (assuming independence)
    portfolio_var = total_var ** 0.5

    return portfolio_var

# Example
positions = {'BTC_15MIN': 500, 'ETH_15MIN': 300}
prices = {'BTC_15MIN': 0.52, 'ETH_15MIN': 0.48}
volatilities = {'BTC_15MIN': 0.03, 'ETH_15MIN': 0.025}

var = calculate_var(positions, prices, volatilities)
# Result: $15.23 at risk (95% confidence, 1 day)
```

---

## Crisis Management

### Market Crash Protocols

**Scenario:** Market experiences sudden, large move against your position.

```python
class CrisisManager:
    """Manage crisis situations."""

    def __init__(self, api):
        self.api = api
        self.crisis_mode = False

    def detect_crisis(self, price_change_pct, volume_spike_ratio):
        """
        Detect crisis conditions.

        Args:
            price_change_pct: % price change
            volume_spike_ratio: Volume vs avg

        Returns:
            Boolean crisis detected
        """
        # Triggers:
        # - Price move > 20%
        # - Volume > 3x average

        price_crisis = abs(price_change_pct) > 0.20
        volume_crisis = volume_spike_ratio > 3.0

        return price_crisis or volume_crisis

    async def execute_crisis_protocol(self, positions):
        """
        Execute crisis management protocol.

        Args:
            positions: Current positions

        Returns:
            List of crisis actions taken
        """
        print("🚨 CRISIS MODE ACTIVATED")

        actions = []

        # 1. Cancel all open orders
        print("Step 1: Cancelling all open orders...")
        await self.api.cancel_all_orders()
        actions.append('CANCELLED_ALL_ORDERS')

        # 2. Assess positions
        print("Step 2: Assessing positions...")
        high_risk_positions = [
            (market, qty) for market, qty in positions.items()
            if abs(qty) > 500  # Threshold
        ]

        # 3. Reduce high-risk positions
        if high_risk_positions:
            print("Step 3: Reducing high-risk positions...")
            for market, qty in high_risk_positions:
                # Market order to reduce position immediately
                reduce_size = abs(qty) * 0.50  # Reduce 50%
                side = 'SELL' if qty > 0 else 'BUY'

                await self.api.create_market_order(
                    market=market,
                    side=side,
                    size=reduce_size
                )

                actions.append(f'REDUCED_{market}_BY_50PCT')

        # 4. Halt trading
        print("Step 4: Halting new trades...")
        self.crisis_mode = True
        actions.append('TRADING_HALTED')

        return actions

# Usage
crisis_mgr = CrisisManager(api_client)

# Monitor for crisis
price_change = -0.25  # -25% drop
volume_spike = 5.0    # 5x volume

if crisis_mgr.detect_crisis(price_change, volume_spike):
    actions = await crisis_mgr.execute_crisis_protocol(positions)
    print(f"Crisis actions: {actions}")
```

### Manual Override

```python
class ManualOverride:
    """Allow manual intervention in automated systems."""

    def __init__(self):
        self.override_active = False
        self.override_reason = None

    def activate_override(self, reason):
        """Activate manual override."""
        self.override_active = True
        self.override_reason = reason
        print(f"⚠️ MANUAL OVERRIDE ACTIVATED: {reason}")

    def deactivate_override(self):
        """Deactivate manual override."""
        self.override_active = False
        self.override_reason = None
        print("✅ Manual override deactivated")

    def should_execute_trade(self):
        """Check if trading allowed."""
        if self.override_active:
            print(f"Trade blocked by override: {self.override_reason}")
            return False

        return True

# Usage
override = ManualOverride()

# Activate during unusual conditions
override.activate_override("Unusual market conditions - manual review required")

# Check before trades
if override.should_execute_trade():
    execute_trade()
else:
    print("Trade execution blocked")
```

---

## Implementation

### Complete Risk Management System

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class RiskConfig:
    max_position_per_market: int = 500
    max_total_exposure: float = 50_000
    stop_loss_pct: float = 0.05
    var_limit: float = 5_000
    crisis_price_threshold: float = 0.20
    crisis_volume_threshold: float = 3.0

class ComprehensiveRiskManager:
    """Comprehensive risk management system."""

    def __init__(self, api, config: RiskConfig):
        self.api = api
        self.config = config

        self.positions: Dict[str, int] = {}
        self.entry_prices: Dict[str, float] = {}

        self.dashboard = RiskDashboard()
        self.stop_loss_mgr = StopLossManager(config.stop_loss_pct)
        self.crisis_mgr = CrisisManager(api)
        self.override = ManualOverride()

    async def run(self):
        """Main risk management loop."""
        print("Starting risk management system...")

        while True:
            try:
                # Update positions
                await self.update_positions()

                # Check risk metrics
                await self.check_risk_limits()

                # Monitor for crisis
                await self.monitor_crisis_conditions()

                # Update dashboard
                self.dashboard.print_dashboard()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                print(f"Error in risk management: {e}")
                await asyncio.sleep(10)

    async def update_positions(self):
        """Update current positions."""
        # Fetch from API
        positions = await self.api.get_positions()

        for market, position in positions.items():
            self.positions[market] = position['quantity']

            if market not in self.entry_prices:
                self.entry_prices[market] = position['avg_price']

            self.dashboard.update_position(
                market,
                position['quantity'],
                position['current_price']
            )

    async def check_risk_limits(self):
        """Check and enforce risk limits."""

        # 1. Position limits
        for market, quantity in self.positions.items():
            if abs(quantity) > self.config.max_position_per_market:
                print(f"⚠️ Position limit exceeded for {market}")
                await self.reduce_position(market, quantity)

        # 2. Total exposure
        metrics = self.dashboard.calculate_metrics()
        if metrics['total_exposure'] > self.config.max_total_exposure:
            print(f"⚠️ Total exposure ${metrics['total_exposure']:.0f} exceeds limit")
            await self.reduce_largest_positions()

        # 3. Stop-losses
        for market, quantity in self.positions.items():
            current_price = await self.api.get_price(market)
            triggered, exit_order = self.stop_loss_mgr.check_stop_loss(market, current_price)

            if triggered:
                print(f"🛑 Stop-loss triggered for {market}")
                await self.execute_exit(exit_order)

    async def reduce_position(self, market, current_quantity):
        """Reduce position to within limits."""
        target_quantity = self.config.max_position_per_market * 0.8  # 80% of limit
        reduce_amount = abs(current_quantity) - target_quantity

        side = 'SELL' if current_quantity > 0 else 'BUY'

        await self.api.create_order(
            market=market,
            side=side,
            size=reduce_amount,
            order_type='MARKET'
        )

        print(f"Reduced {market} by {reduce_amount}")

    async def reduce_largest_positions(self):
        """Reduce largest positions to lower total exposure."""
        # Sort by position size
        sorted_positions = sorted(
            self.positions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Reduce top 3 positions by 30%
        for market, quantity in sorted_positions[:3]:
            reduce_amount = abs(quantity) * 0.30
            side = 'SELL' if quantity > 0 else 'BUY'

            await self.api.create_order(
                market=market,
                side=side,
                size=reduce_amount,
                order_type='MARKET'
            )

            print(f"Reduced {market} by 30%")

    async def monitor_crisis_conditions(self):
        """Monitor for crisis situations."""
        # Check each market for crisis conditions
        for market in self.positions.keys():
            historical_prices = await self.api.get_historical_prices(market, window=60)
            recent_prices = historical_prices[-10:]

            # Calculate price change
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Calculate volume spike
            current_volume = await self.api.get_volume(market)
            avg_volume = await self.api.get_avg_volume(market, window=24)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            # Check crisis
            if self.crisis_mgr.detect_crisis(price_change, volume_ratio):
                actions = await self.crisis_mgr.execute_crisis_protocol(self.positions)
                print(f"Crisis protocol executed: {actions}")

                # Activate manual override
                self.override.activate_override(f"Crisis detected in {market}")
                break

    async def execute_exit(self, exit_order):
        """Execute exit order."""
        await self.api.create_order(
            market=exit_order['market'],
            side=exit_order['side'],
            size=exit_order['size'],
            order_type='MARKET'
        )

        print(f"Exit executed: {exit_order}")

# Usage
config = RiskConfig(
    max_position_per_market=500,
    max_total_exposure=50_000,
    stop_loss_pct=0.05,
    var_limit=5_000
)

risk_manager = ComprehensiveRiskManager(api_client, config)
await risk_manager.run()
```

---

## Summary

### Key Principles

1. **Inventory Risk is Primary**
   - Most significant risk for market makers
   - Mitigate with skewed orders, delays, asymmetric spreads

2. **Position Limits Essential**
   - Set and enforce strict limits
   - Adjust for volatility and time to resolution

3. **Hedging Optional but Valuable**
   - Delta hedging eliminates directional risk
   - Costs money but provides insurance

4. **Real-Time Monitoring Critical**
   - Automated dashboard and alerts
   - Manual override capabilities

5. **Crisis Protocols Mandatory**
   - Predefined actions for extreme events
   - Practice and test regularly

### Risk Management Checklist

- [ ] Define position limits per market
- [ ] Calculate and monitor total exposure
- [ ] Implement stop-loss mechanisms
- [ ] Set up inventory skew adjustments
- [ ] Configure asymmetric spreads
- [ ] Establish hedging strategy (if applicable)
- [ ] Create real-time risk dashboard
- [ ] Define crisis protocols
- [ ] Test manual override system
- [ ] Review and adjust limits weekly

---

## Sources

- [What is Inventory Risk? - Hummingbot](https://hummingbot.org/blog/what-is-inventory-risk/)
- [Guide to Hedging Strategies of Crypto Market Makers - DWF Labs](https://www.dwf-labs.com/news/understanding-market-maker-hedging)
- [Automated Market Making Bots in Cryptocurrency](https://madeinark.org/automated-market-making-bots-in-cryptocurrency-from-spread-capture-to-advanced-inventory-management/)
- [Market Making Mechanics and Strategies - BlockApex](https://medium.com/blockapex/market-making-mechanics-and-strategies-4daf2122121c)
- [Crypto Market Making Guide - Shift Markets](https://www.shiftmarkets.com/blog/crypto-market-making-guide)
- [Crypto Market Making Risk Management - Orcabay](https://orcabay.io/blog/crypto-market-making-risk-management/)
- [Guide to the Avellaneda-Stoikov Strategy - Hummingbot](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/)

---

**Last Updated:** 2026-02-04
**Version:** 1.0
**Maintained by:** SYM Web Research Agent
