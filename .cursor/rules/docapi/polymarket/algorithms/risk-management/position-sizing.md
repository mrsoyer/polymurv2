# Position Sizing Algorithms

> Systematic approaches to determine optimal bet size for prediction market trades

## Overview

Position sizing is the most critical component of risk management, contributing over 90% to overall risk-adjusted returns according to Van Tharp Institute research. This guide covers fixed fractional methods, dynamic sizing algorithms, and volatility-based adjustments.

---

## 1. Position Sizing Methods

### 1.1 Fixed Percentage Risk

Limit each trade to fixed percentage of total capital.

**Formula:**
```
Position Size = Total Capital × Risk Percentage
```

**Conservative Allocation:**
- Very Conservative: 1-2% per trade
- Conservative: 2-5% per trade
- Moderate: 5-10% per trade
- Aggressive: 10-20% per trade

**Python Implementation:**

```python
class FixedPercentageSizing:
    """Fixed percentage position sizing."""

    def __init__(self, capital, risk_pct=0.02):
        """
        Args:
            capital: Total capital available
            risk_pct: Risk percentage per trade (default: 2%)
        """
        self.capital = capital
        self.risk_pct = risk_pct

    def calculate_size(self, market_price):
        """Calculate position size."""
        max_risk = self.capital * self.risk_pct
        # Number of shares at market price
        shares = max_risk / market_price
        return min(shares, self.capital / market_price)

# Example
sizer = FixedPercentageSizing(capital=10000, risk_pct=0.02)
position = sizer.calculate_size(market_price=0.45)
print(f"Buy {position:.0f} shares at $0.45")
# Output: Buy 444 shares (risking $200)
```

**Pros:**
- Simple to implement
- Consistent risk per trade
- Easy to understand

**Cons:**
- Ignores edge/probability
- Doesn't scale with confidence
- May be too conservative with high-probability trades

---

### 1.2 Kelly Criterion (Optimal Growth)

Mathematically optimal sizing to maximize long-term growth.

**Full Kelly Formula:**
```
f* = (p × b - q) / b
```

**For Prediction Markets:**
```
f* = (your_prob - market_price) / (1 - market_price)
```

**Fractional Kelly:**
```
Position Size = f* × Fraction × Capital
```

Where Fraction is typically 0.25 (quarter-Kelly) or 0.5 (half-Kelly).

**Python Implementation:**

```python
class KellySizing:
    """Kelly criterion position sizing."""

    def __init__(self, capital, kelly_fraction=0.5):
        """
        Args:
            capital: Total capital
            kelly_fraction: Fractional Kelly (0.5 = half-Kelly recommended)
        """
        self.capital = capital
        self.kelly_fraction = kelly_fraction

    def calculate_size(self, your_prob, market_price):
        """
        Calculate Kelly position size.

        Args:
            your_prob: Your estimated probability (0-1)
            market_price: Current market price (0-1)

        Returns:
            Dollar amount to bet
        """
        # Edge calculation
        edge = your_prob - market_price

        # No edge, no bet
        if edge <= 0:
            return 0

        # Odds against winning
        odds_against = 1 - market_price

        # Kelly fraction
        kelly = edge / odds_against

        # Apply fractional Kelly
        position_fraction = kelly * self.kelly_fraction

        # Convert to dollars (cap at 100%)
        position_size = min(position_fraction, 1.0) * self.capital

        return position_size

    def calculate_with_confidence(self, your_prob, market_price, confidence=1.0):
        """
        Adjust Kelly sizing by confidence level.

        Args:
            confidence: How confident in your estimate (0-1)
        """
        base_size = self.calculate_size(your_prob, market_price)
        return base_size * confidence

# Example
kelly = KellySizing(capital=10000, kelly_fraction=0.5)

# High confidence trade
size1 = kelly.calculate_size(your_prob=0.65, market_price=0.50)
print(f"High edge trade: ${size1:.2f} ({size1/10000:.1%})")

# Low confidence trade
size2 = kelly.calculate_with_confidence(
    your_prob=0.55,
    market_price=0.50,
    confidence=0.5  # Only 50% confident
)
print(f"Low confidence trade: ${size2:.2f} ({size2/10000:.1%})")
```

**Pros:**
- Mathematically optimal for long-term growth
- Scales position with edge
- Prevents overbetting

**Cons:**
- Can be volatile with full Kelly
- Sensitive to probability estimation errors
- Requires accurate probability estimates

---

### 1.3 Volatility-Based Sizing

Adjust position size based on market volatility.

**Formula:**
```
Position Size = Base Size × (Target Volatility / Current Volatility)
```

**ATR-Based Sizing:**
```python
import pandas as pd
import numpy as np

class VolatilitySizing:
    """Volatility-adjusted position sizing."""

    def __init__(self, capital, target_volatility=0.02):
        """
        Args:
            capital: Total capital
            target_volatility: Target daily volatility (2% default)
        """
        self.capital = capital
        self.target_volatility = target_volatility

    def calculate_atr(self, prices, period=14):
        """
        Calculate Average True Range.

        Args:
            prices: DataFrame with 'high', 'low', 'close'
            period: ATR period (14 default)
        """
        high_low = prices['high'] - prices['low']
        high_close = np.abs(prices['high'] - prices['close'].shift())
        low_close = np.abs(prices['low'] - prices['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(period).mean().iloc[-1]
        return atr

    def calculate_size(self, current_price, atr):
        """
        Calculate position size based on ATR.

        Args:
            current_price: Current market price
            atr: Average True Range
        """
        # Calculate current volatility as percentage
        current_volatility = atr / current_price

        # Volatility adjustment factor
        vol_factor = self.target_volatility / current_volatility

        # Base position (10% of capital)
        base_position = 0.10 * self.capital

        # Adjusted position
        adjusted_position = base_position * vol_factor

        # Cap at 25% of capital
        return min(adjusted_position, 0.25 * self.capital)

# Example
vol_sizer = VolatilitySizing(capital=10000, target_volatility=0.02)

# High volatility market (reduce size)
size_high_vol = vol_sizer.calculate_size(current_price=0.50, atr=0.05)
print(f"High volatility: ${size_high_vol:.2f}")

# Low volatility market (increase size)
size_low_vol = vol_sizer.calculate_size(current_price=0.50, atr=0.01)
print(f"Low volatility: ${size_low_vol:.2f}")
```

**Bollinger Band Width Sizing:**
```python
def calculate_bb_width(prices, period=20, num_std=2):
    """Calculate Bollinger Band width as volatility measure."""
    rolling_mean = prices.rolling(period).mean()
    rolling_std = prices.rolling(period).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    bb_width = (upper_band - lower_band) / rolling_mean
    return bb_width.iloc[-1]

def volatility_adjusted_size(capital, bb_width, base_pct=0.10):
    """Adjust position size based on Bollinger Band width."""
    # Normalize: typical BB width is 0.10-0.30
    normalized_vol = max(min(bb_width / 0.20, 2.0), 0.5)

    # Inverse relationship: higher volatility = smaller position
    vol_factor = 1 / normalized_vol

    position_size = capital * base_pct * vol_factor
    return min(position_size, capital * 0.25)
```

**Pros:**
- Reduces risk during volatile periods
- Increases position size in stable markets
- Adapts to changing market conditions

**Cons:**
- Requires historical price data
- May reduce size during high-opportunity volatility
- Complex to implement

---

### 1.4 Confidence-Based Sizing

Scale position size by confidence in your probability estimate.

```python
class ConfidenceSizing:
    """Position sizing based on prediction confidence."""

    def __init__(self, capital, base_pct=0.10):
        self.capital = capital
        self.base_pct = base_pct

    def calculate_size(self, your_prob, market_price, confidence_score):
        """
        Args:
            your_prob: Your estimated probability
            market_price: Market price
            confidence_score: Confidence in estimate (0-1)
                - 1.0 = very confident (lots of data)
                - 0.5 = moderate confidence
                - 0.0 = no confidence (skip trade)
        """
        # Edge calculation
        edge = your_prob - market_price

        if edge <= 0 or confidence_score < 0.3:
            return 0

        # Base position size
        base_size = self.capital * self.base_pct

        # Scale by edge magnitude
        edge_factor = abs(edge) / 0.10  # Normalize to 10% edge

        # Scale by confidence
        position_size = base_size * edge_factor * confidence_score

        # Cap at 20% of capital
        return min(position_size, self.capital * 0.20)

    def calculate_confidence_from_sample_size(self, sample_size):
        """
        Estimate confidence based on data sample size.

        Args:
            sample_size: Number of data points used in estimate
        """
        if sample_size < 10:
            return 0.2  # Low confidence
        elif sample_size < 50:
            return 0.5  # Moderate confidence
        elif sample_size < 200:
            return 0.7  # Good confidence
        else:
            return 0.9  # High confidence

# Example
conf_sizer = ConfidenceSizing(capital=10000, base_pct=0.10)

# High confidence, large edge
size1 = conf_sizer.calculate_size(
    your_prob=0.70,
    market_price=0.50,
    confidence_score=0.9  # Very confident
)
print(f"High confidence, large edge: ${size1:.2f}")

# Low confidence, same edge
size2 = conf_sizer.calculate_size(
    your_prob=0.70,
    market_price=0.50,
    confidence_score=0.3  # Low confidence
)
print(f"Low confidence, same edge: ${size2:.2f}")

# Confidence from sample size
sample_sizes = [5, 25, 100, 500]
for n in sample_sizes:
    conf = conf_sizer.calculate_confidence_from_sample_size(n)
    print(f"Sample size {n}: confidence = {conf:.1f}")
```

---

## 2. Dynamic Position Sizing

### 2.1 Capital-Based Scaling

Adjust all positions as capital grows or shrinks.

```python
class DynamicSizer:
    """Dynamic position sizing that scales with capital."""

    def __init__(self, initial_capital, risk_pct=0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_pct = risk_pct

    def update_capital(self, new_capital):
        """Update current capital after wins/losses."""
        self.current_capital = new_capital

    def calculate_size(self, market_price):
        """Calculate position based on CURRENT capital."""
        max_risk = self.current_capital * self.risk_pct
        return max_risk / market_price

    def get_growth_factor(self):
        """Calculate capital growth since start."""
        return self.current_capital / self.initial_capital

# Example
sizer = DynamicSizer(initial_capital=10000, risk_pct=0.02)

# Initial trade
size1 = sizer.calculate_size(market_price=0.50)
print(f"Initial: {size1:.0f} shares")

# After winning +$2000
sizer.update_capital(12000)
size2 = sizer.calculate_size(market_price=0.50)
print(f"After win: {size2:.0f} shares (+{(size2-size1)/size1:.0%})")

# After losing -$3000
sizer.update_capital(9000)
size3 = sizer.calculate_size(market_price=0.50)
print(f"After loss: {size3:.0f} shares")
```

### 2.2 Win/Loss Streak Adjustment

Reduce size after losses, increase cautiously after wins.

```python
class StreakAdjustedSizer:
    """Adjust sizing based on win/loss streaks."""

    def __init__(self, capital, base_pct=0.05):
        self.capital = capital
        self.base_pct = base_pct
        self.streak = 0  # Positive = wins, negative = losses

    def record_result(self, won):
        """Record trade result and update streak."""
        if won:
            self.streak = max(self.streak + 1, 1)
        else:
            self.streak = min(self.streak - 1, -1)

    def calculate_size(self):
        """Calculate size with streak adjustment."""
        base_size = self.capital * self.base_pct

        # Reduce size after losses
        if self.streak < 0:
            reduction_factor = 0.5 ** abs(self.streak)  # 50% per loss
            adjusted_size = base_size * reduction_factor
        # Slight increase after wins (conservative)
        elif self.streak > 0:
            increase_factor = 1 + (0.1 * min(self.streak, 3))  # Max 30% increase
            adjusted_size = base_size * increase_factor
        else:
            adjusted_size = base_size

        return min(adjusted_size, self.capital * 0.20)

# Example
streak_sizer = StreakAdjustedSizer(capital=10000, base_pct=0.05)

print(f"Initial: ${streak_sizer.calculate_size():.2f}")

# After 2 losses
streak_sizer.record_result(won=False)
streak_sizer.record_result(won=False)
print(f"After 2 losses: ${streak_sizer.calculate_size():.2f}")

# After 3 wins
for _ in range(5):  # 5 wins to reverse streak
    streak_sizer.record_result(won=True)
print(f"After 3 wins: ${streak_sizer.calculate_size():.2f}")
```

---

## 3. Multi-Market Allocation

### 3.1 Portfolio Constraint Sizing

Ensure total exposure across all markets respects capital constraints.

```python
class PortfolioConstraintSizer:
    """Position sizing with portfolio-level constraints."""

    def __init__(self, total_capital, max_total_exposure=0.75):
        """
        Args:
            total_capital: Total capital available
            max_total_exposure: Maximum % of capital allocated (default 75%)
        """
        self.total_capital = total_capital
        self.max_total_exposure = max_total_exposure
        self.positions = []

    def add_position(self, market_name, unconstrained_size):
        """Add a position with its unconstrained optimal size."""
        self.positions.append({
            'market': market_name,
            'unconstrained_size': unconstrained_size
        })

    def calculate_constrained_sizes(self):
        """Calculate final sizes respecting portfolio constraint."""
        # Total unconstrained allocation
        total_unconstrained = sum(p['unconstrained_size'] for p in self.positions)

        # Maximum allowed allocation
        max_allocation = self.total_capital * self.max_total_exposure

        # If over limit, scale down proportionally
        if total_unconstrained > max_allocation:
            scale_factor = max_allocation / total_unconstrained

            for p in self.positions:
                p['final_size'] = p['unconstrained_size'] * scale_factor
        else:
            for p in self.positions:
                p['final_size'] = p['unconstrained_size']

        return self.positions

# Example
portfolio = PortfolioConstraintSizer(total_capital=10000, max_total_exposure=0.75)

# Add unconstrained Kelly sizes
portfolio.add_position("Market A", unconstrained_size=3000)
portfolio.add_position("Market B", unconstrained_size=2500)
portfolio.add_position("Market C", unconstrained_size=2000)
portfolio.add_position("Market D", unconstrained_size=1500)

# Calculate constrained sizes
positions = portfolio.calculate_constrained_sizes()

total_allocated = sum(p['final_size'] for p in positions)
print(f"Total allocated: ${total_allocated:.2f} ({total_allocated/10000:.1%})")

for p in positions:
    print(f"{p['market']}: ${p['final_size']:.2f} "
          f"(unconstrained: ${p['unconstrained_size']:.2f})")
```

### 3.2 Correlation-Adjusted Sizing

Reduce size when markets are highly correlated.

```python
import numpy as np

def correlation_adjusted_sizing(base_sizes, correlation_matrix, adjustment_factor=0.5):
    """
    Adjust position sizes based on correlation between markets.

    Args:
        base_sizes: Array of unconstrained position sizes
        correlation_matrix: NxN correlation matrix
        adjustment_factor: How much to reduce for correlation (0-1)

    Returns:
        Adjusted position sizes
    """
    n_markets = len(base_sizes)
    adjusted_sizes = base_sizes.copy()

    for i in range(n_markets):
        # Average correlation with other markets
        avg_correlation = np.mean([correlation_matrix[i][j]
                                   for j in range(n_markets) if i != j])

        # Reduction factor: higher correlation = more reduction
        reduction = 1 - (avg_correlation * adjustment_factor)
        adjusted_sizes[i] *= max(reduction, 0.5)  # Max 50% reduction

    return adjusted_sizes

# Example
base_sizes = np.array([2000, 2000, 2000])  # 3 markets
correlation_matrix = np.array([
    [1.0, 0.8, 0.3],  # Market 1 highly correlated with Market 2
    [0.8, 1.0, 0.4],  # Market 2
    [0.3, 0.4, 1.0]   # Market 3 less correlated
])

adjusted = correlation_adjusted_sizing(base_sizes, correlation_matrix)

for i, (base, adj) in enumerate(zip(base_sizes, adjusted)):
    print(f"Market {i+1}: ${base:.0f} → ${adj:.0f} ({adj/base:.0%})")
```

---

## 4. Automated Risk Controls

### 4.1 Circuit Breakers

Automatically stop trading if drawdown exceeds threshold.

```python
class CircuitBreaker:
    """Automatic trading halt on excessive drawdown."""

    def __init__(self, initial_capital, max_drawdown_pct=0.10):
        """
        Args:
            initial_capital: Starting capital
            max_drawdown_pct: Max drawdown before halt (default 10%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.trading_halted = False

    def update(self, new_capital):
        """Update capital and check circuit breaker."""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

        # Calculate current drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        # Trigger circuit breaker
        if drawdown >= self.max_drawdown_pct:
            self.trading_halted = True
            return True  # Breaker triggered

        return False

    def can_trade(self):
        """Check if trading is allowed."""
        return not self.trading_halted

    def reset(self):
        """Manually reset circuit breaker (use with caution)."""
        self.trading_halted = False

# Example
breaker = CircuitBreaker(initial_capital=10000, max_drawdown_pct=0.10)

# Simulate losses
breaker.update(9500)  # -5% drawdown
print(f"Can trade: {breaker.can_trade()}")

breaker.update(8900)  # -11% drawdown
triggered = breaker.update(8900)
if triggered:
    print("⚠️  CIRCUIT BREAKER TRIGGERED - Trading halted")
    print(f"Drawdown: {(10000-8900)/10000:.1%}")
```

### 4.2 Position Limits

Enforce maximum position sizes.

```python
class PositionLimiter:
    """Enforce position size limits."""

    def __init__(self, capital, max_single_position=0.20, max_total_exposure=0.80):
        """
        Args:
            capital: Total capital
            max_single_position: Max % in single market (default 20%)
            max_total_exposure: Max % total allocation (default 80%)
        """
        self.capital = capital
        self.max_single = capital * max_single_position
        self.max_total = capital * max_total_exposure
        self.current_positions = {}

    def check_position(self, market_name, proposed_size):
        """
        Check if proposed position is allowed.

        Returns:
            (allowed: bool, adjusted_size: float, reason: str)
        """
        # Check single position limit
        if proposed_size > self.max_single:
            return (False, self.max_single,
                    f"Exceeds single position limit (${self.max_single:.2f})")

        # Check total exposure limit
        current_total = sum(self.current_positions.values())
        if current_total + proposed_size > self.max_total:
            remaining = self.max_total - current_total
            if remaining <= 0:
                return (False, 0, "Portfolio fully allocated")
            return (False, remaining,
                    f"Reduced to stay within total exposure limit")

        return (True, proposed_size, "Approved")

    def add_position(self, market_name, size):
        """Record a new position."""
        self.current_positions[market_name] = size

# Example
limiter = PositionLimiter(capital=10000, max_single_position=0.20)

# Attempt large position
allowed, adjusted, reason = limiter.check_position("Market A", 3000)
print(f"$3000 position: {reason}")
print(f"Adjusted size: ${adjusted:.2f}")

# Add allowed positions
limiter.add_position("Market A", 2000)
limiter.add_position("Market B", 2000)
limiter.add_position("Market C", 2000)
limiter.add_position("Market D", 2000)

# Try to add another
allowed, adjusted, reason = limiter.check_position("Market E", 1000)
print(f"\nAfter 4 positions totaling $8000:")
print(f"Add $1000 more: {reason}")
```

---

## 5. Complete Position Sizing System

Integrating multiple methods:

```python
class ComprehensivePositionSizer:
    """Complete position sizing system integrating multiple methods."""

    def __init__(self, capital, config=None):
        self.capital = capital
        self.config = config or {
            'method': 'kelly',  # kelly, fixed, volatility, confidence
            'kelly_fraction': 0.5,
            'fixed_pct': 0.02,
            'max_single_position': 0.20,
            'max_total_exposure': 0.75,
            'use_volatility_adjustment': True,
            'use_correlation_adjustment': True
        }

        # Components
        self.kelly_sizer = KellySizing(capital, self.config['kelly_fraction'])
        self.circuit_breaker = CircuitBreaker(capital, max_drawdown_pct=0.10)
        self.limiter = PositionLimiter(
            capital,
            max_single_position=self.config['max_single_position'],
            max_total_exposure=self.config['max_total_exposure']
        )

    def calculate_position(self, market_data):
        """
        Calculate final position size.

        Args:
            market_data: dict with keys:
                - market_name
                - your_prob
                - market_price
                - confidence (optional)
                - volatility (optional)
                - correlation_to_portfolio (optional)
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            return 0, "Circuit breaker active"

        # Base size calculation
        if self.config['method'] == 'kelly':
            base_size = self.kelly_sizer.calculate_size(
                market_data['your_prob'],
                market_data['market_price']
            )
        elif self.config['method'] == 'fixed':
            base_size = self.capital * self.config['fixed_pct']
        else:
            base_size = self.capital * 0.05  # Default

        # Confidence adjustment
        if 'confidence' in market_data:
            base_size *= market_data['confidence']

        # Volatility adjustment
        if self.config['use_volatility_adjustment'] and 'volatility' in market_data:
            vol_factor = 0.02 / max(market_data['volatility'], 0.01)  # Target 2% vol
            base_size *= min(vol_factor, 2.0)  # Cap at 2x

        # Correlation adjustment
        if self.config['use_correlation_adjustment'] and 'correlation_to_portfolio' in market_data:
            corr = market_data['correlation_to_portfolio']
            corr_factor = 1 - (abs(corr) * 0.5)  # Reduce up to 50%
            base_size *= corr_factor

        # Apply limits
        allowed, final_size, reason = self.limiter.check_position(
            market_data['market_name'],
            base_size
        )

        return final_size, reason

# Example usage
sizer = ComprehensivePositionSizer(capital=10000)

market = {
    'market_name': 'Bitcoin > 100k',
    'your_prob': 0.65,
    'market_price': 0.50,
    'confidence': 0.8,
    'volatility': 0.15,
    'correlation_to_portfolio': 0.3
}

size, reason = sizer.calculate_position(market)
print(f"Final position size: ${size:.2f}")
print(f"Reason: {reason}")
print(f"% of capital: {size/10000:.1%}")
```

---

## Best Practices

1. **Start Conservative**: Use Quarter-Kelly or Half-Kelly, not Full Kelly
2. **Never Exceed 20-25%**: Single position should never be >25% of capital
3. **Use Circuit Breakers**: Auto-halt trading at 10% drawdown
4. **Adjust for Confidence**: Scale down when uncertain
5. **Account for Correlation**: Reduce size for correlated positions
6. **Rebalance Dynamically**: Update sizes as capital changes
7. **Track Performance**: Monitor which sizing method works best

---

## References

- [Position Sizing Impact Study](https://speedbot.tech/blog/algo-trading-4/how-position-sizing-can-make-or-break-your-trading-strategy-221)
- [Trading Bot Strategies 2026](https://www.quantvps.com/blog/trading-bot-strategies)
- [Risk Management in Algorithmic Trading](https://nurp.com/wisdom/7-risk-management-strategies-for-algorithmic-trading/)
- [Kelly Criterion Calculator](https://www.albionresearch.com/tools/kelly)
- [Van Tharp Institute - Position Sizing Research](https://www.vantharp.com/)
