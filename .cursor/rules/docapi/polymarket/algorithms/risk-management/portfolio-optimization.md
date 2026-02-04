# Portfolio Optimization for Prediction Markets

> Mathematical frameworks for optimal capital allocation across prediction market positions

## Overview

Portfolio optimization in prediction markets combines Modern Portfolio Theory (MPT) with Kelly criterion to maximize long-term growth while managing risk. This guide covers diversification strategies, correlation analysis, and the efficient frontier for multi-market position management.

---

## 1. Kelly Criterion

### 1.1 Core Formula

The Kelly criterion determines optimal position size to maximize long-run wealth growth:

**Basic Kelly Formula:**
```
f* = (bp - q) / b
```

Where:
- `f*` = optimal fraction of bankroll to bet
- `b` = net odds received on the wager (e.g., 1 for even money)
- `p` = probability of winning
- `q` = probability of losing (1 - p)

**Simplified for Binary Markets:**
```
f* = edge / odds = (p - market_price) / (1 - market_price)
```

### 1.2 Mathematical Derivation

The Kelly criterion maximizes the expected value of the logarithm of wealth:

**Growth Rate Formula:**
```
G = lim(n→∞) (1/n) · log(Vₙ/V₀)
```

For repeated betting with constant fraction `f`:
```
G = p · log(1 + f·b) + q · log(1 - f)
```

Taking the derivative and setting to zero:
```
dG/df = p·b/(1 + f·b) - q/(1 - f) = 0
```

Solving yields: `f* = (bp - q) / b`

**Maximum Growth Rate:**
```
G_max = 1 + p·log(p) + q·log(q)
```

### 1.3 Practical Examples

**Example 1: Biased Coin (60% win probability)**
- Win probability: p = 0.60
- Lose probability: q = 0.40
- Odds: 1-to-1 (b = 1)
- Optimal stake: f* = (1×0.6 - 0.4) / 1 = 0.20 (20% of bankroll)
- Long-run gain: 2.9% per bet

**Example 2: Prediction Market**
- Your estimated probability: 0.55
- Market price (implied probability): 0.45
- If market at 0.45, you buy at 45¢
- Edge = 0.55 - 0.45 = 0.10
- Odds against: 1 - 0.45 = 0.55
- Kelly fraction: f* = 0.10 / 0.55 ≈ 0.182 (18.2% of bankroll)

### 1.4 Fractional Kelly

Full Kelly can be volatile. Many traders use **Fractional Kelly** to reduce variance:

```
Position Size = Kelly_Fraction × (1/2 or 1/4)
```

**Benefits of Half-Kelly (1/2):**
- Reduces volatility by ~50%
- Still captures ~75% of Kelly growth rate
- More forgiving of estimation errors
- Industry standard for professional traders

**Python Implementation:**

```python
def kelly_criterion(win_prob: float, market_price: float, fraction: float = 1.0) -> float:
    """
    Calculate Kelly criterion position size for prediction market.

    Args:
        win_prob: Your estimated probability (0-1)
        market_price: Market price in dollars (0-1)
        fraction: Fractional Kelly (0.5 for half-Kelly, 0.25 for quarter-Kelly)

    Returns:
        Optimal fraction of bankroll to bet (0-1)
    """
    # Edge = difference between your probability and market price
    edge = win_prob - market_price

    # If no edge, don't bet
    if edge <= 0:
        return 0.0

    # Odds against winning
    odds_against = 1 - market_price

    # Kelly fraction
    kelly = edge / odds_against

    # Apply fractional Kelly
    position_size = kelly * fraction

    # Cap at 100% (never bet more than bankroll)
    return min(position_size, 1.0)

# Example usage
win_prob = 0.60  # You estimate 60% chance
market_price = 0.45  # Market at 45¢
full_kelly = kelly_criterion(win_prob, market_price, fraction=1.0)
half_kelly = kelly_criterion(win_prob, market_price, fraction=0.5)

print(f"Full Kelly: {full_kelly:.2%}")  # ~27%
print(f"Half Kelly: {half_kelly:.2%}")  # ~14%
```

---

## 2. Modern Portfolio Theory (MPT)

### 2.1 Core Concepts

MPT constructs portfolios that maximize expected return for a given risk level by diversifying across uncorrelated assets.

**Key Principles:**
1. **Expected Return**: E(R) = Σ(wᵢ · rᵢ)
2. **Portfolio Variance**: σ²ₚ = Σ Σ(wᵢ · wⱼ · σᵢⱼ)
3. **Sharpe Ratio**: (Rₚ - Rₓ) / σₚ

Where:
- `wᵢ` = weight of asset i
- `rᵢ` = expected return of asset i
- `σᵢⱼ` = covariance between assets i and j
- `Rₚ` = portfolio return
- `Rₓ` = risk-free rate
- `σₚ` = portfolio standard deviation

### 2.2 Efficient Frontier

The **efficient frontier** represents the set of optimal portfolios offering maximum expected return for each risk level.

**Optimization Problem:**

```
Maximize: E(Rₚ) = Σ(wᵢ · E(rᵢ))
Subject to:
  - σ²ₚ = target_variance
  - Σwᵢ = 1 (weights sum to 100%)
  - wᵢ ≥ 0 (no short selling, optional)
```

### 2.3 Correlation Analysis

**Correlation Matrix:**
```
ρᵢⱼ = Cov(rᵢ, rⱼ) / (σᵢ · σⱼ)
```

**Diversification Benefits:**
- Correlation = +1: No diversification benefit
- Correlation = 0: Maximum diversification benefit
- Correlation = -1: Perfect hedge

**For Prediction Markets:**
- Politics markets often highly correlated (party outcomes)
- Sports markets moderately correlated (same league/season)
- Crypto markets can be highly correlated
- Cross-category diversification (politics + sports + crypto) reduces correlation

### 2.4 Python Implementation

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_stats(weights, returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    portfolio_return = np.sum(weights * returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.0):
    """Calculate negative Sharpe ratio (for minimization)."""
    p_return, p_std = calculate_portfolio_stats(weights, returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

def optimize_portfolio(returns, cov_matrix, target='max_sharpe'):
    """
    Optimize portfolio weights.

    Args:
        returns: Expected returns array
        cov_matrix: Covariance matrix
        target: 'max_sharpe' or 'min_variance'

    Returns:
        Optimal weights array
    """
    n_assets = len(returns)

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Bounds: 0 <= weight <= 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)

    if target == 'max_sharpe':
        # Minimize negative Sharpe ratio
        result = minimize(
            sharpe_ratio,
            initial_weights,
            args=(returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    elif target == 'min_variance':
        # Minimize variance
        result = minimize(
            lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

    return result.x

# Example: 3 prediction markets
returns = np.array([0.15, 0.10, 0.20])  # Expected returns: 15%, 10%, 20%
cov_matrix = np.array([
    [0.04, 0.01, 0.02],  # Market 1
    [0.01, 0.02, 0.005], # Market 2
    [0.02, 0.005, 0.09]  # Market 3
])

# Optimize for maximum Sharpe ratio
optimal_weights = optimize_portfolio(returns, cov_matrix, target='max_sharpe')
print("Optimal Weights:", optimal_weights)

# Calculate portfolio metrics
p_return, p_std = calculate_portfolio_stats(optimal_weights, returns, cov_matrix)
sharpe = p_return / p_std
print(f"Expected Return: {p_return:.2%}")
print(f"Volatility (Std Dev): {p_std:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
```

### 2.5 Efficient Frontier Visualization

```python
import matplotlib.pyplot as plt

def plot_efficient_frontier(returns, cov_matrix, num_portfolios=10000):
    """Generate and plot the efficient frontier."""
    n_assets = len(returns)
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        # Calculate metrics
        p_return, p_std = calculate_portfolio_stats(weights, returns, cov_matrix)
        sharpe = p_return / p_std

        results[0, i] = p_return
        results[1, i] = p_std
        results[2, i] = sharpe

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :],
                cmap='viridis', marker='o', alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Std Dev)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier - Prediction Market Portfolio')

    # Mark optimal portfolio
    optimal_weights = optimize_portfolio(returns, cov_matrix)
    opt_return, opt_std = calculate_portfolio_stats(optimal_weights, returns, cov_matrix)
    plt.scatter(opt_std, opt_return, c='red', marker='*', s=500,
                label='Max Sharpe Portfolio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot efficient frontier
plot_efficient_frontier(returns, cov_matrix)
```

---

## 3. Diversification Strategies

### 3.1 Category Diversification

Spread positions across uncorrelated market categories:

| Category | Allocation | Correlation to Crypto |
|----------|------------|---------------------|
| Politics | 30% | Low (0.1-0.2) |
| Sports | 25% | Low (0.0-0.1) |
| Crypto | 25% | High (1.0) |
| Economics | 20% | Medium (0.4-0.6) |

### 3.2 Time Diversification

Spread positions across different time horizons:

```python
def time_diversification_weights(days_to_resolution):
    """
    Allocate more capital to longer-term markets.
    Reduces impact of short-term volatility.
    """
    if days_to_resolution < 7:
        return 0.10  # 10% to very short-term
    elif days_to_resolution < 30:
        return 0.25  # 25% to short-term
    elif days_to_resolution < 90:
        return 0.35  # 35% to medium-term
    else:
        return 0.30  # 30% to long-term
```

### 3.3 Outcome Diversification

For multi-outcome markets, consider:
- Spreading across multiple outcomes when probabilities are uncertain
- Arbitrage opportunities across related markets
- Hedging positions when new information emerges

---

## 4. Kelly + MPT Integration

Combine Kelly criterion (position sizing) with MPT (portfolio allocation):

```python
class PredictionMarketPortfolio:
    """Integrated Kelly + MPT portfolio manager."""

    def __init__(self, bankroll, kelly_fraction=0.5):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.positions = []

    def add_market(self, name, your_prob, market_price):
        """Add a market opportunity."""
        kelly_size = kelly_criterion(your_prob, market_price, self.kelly_fraction)
        self.positions.append({
            'name': name,
            'your_prob': your_prob,
            'market_price': market_price,
            'kelly_size': kelly_size,
            'edge': your_prob - market_price
        })

    def optimize_allocations(self):
        """Optimize using MPT while respecting Kelly constraints."""
        # Extract Kelly recommendations
        kelly_sizes = np.array([p['kelly_size'] for p in self.positions])

        # Expected returns (edges)
        returns = np.array([p['edge'] for p in self.positions])

        # Estimate covariance (simplified - should use historical data)
        n = len(self.positions)
        cov_matrix = np.eye(n) * 0.01  # Assume low correlation

        # Optimize with Kelly as upper bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, min(k, 1.0)) for k in kelly_sizes)

        initial_weights = kelly_sizes / np.sum(kelly_sizes)

        result = minimize(
            sharpe_ratio,
            initial_weights,
            args=(returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Calculate dollar allocations
        optimal_weights = result.x
        for i, pos in enumerate(self.positions):
            pos['allocation'] = optimal_weights[i] * self.bankroll

        return self.positions

# Example usage
portfolio = PredictionMarketPortfolio(bankroll=10000, kelly_fraction=0.5)

# Add market opportunities
portfolio.add_market("Trump 2024", your_prob=0.48, market_price=0.52)
portfolio.add_market("Bitcoin > 100k", your_prob=0.65, market_price=0.55)
portfolio.add_market("Lakers Win", your_prob=0.40, market_price=0.30)

# Optimize
positions = portfolio.optimize_allocations()

for p in positions:
    print(f"{p['name']}: ${p['allocation']:.2f} ({p['allocation']/10000:.1%})")
```

---

## 5. Rebalancing Strategies

### 5.1 Calendar Rebalancing

Rebalance at fixed intervals:

```python
def should_rebalance_calendar(last_rebalance_date, frequency_days=7):
    """Rebalance every N days."""
    days_since = (datetime.now() - last_rebalance_date).days
    return days_since >= frequency_days
```

### 5.2 Threshold Rebalancing

Rebalance when allocations drift beyond threshold:

```python
def should_rebalance_threshold(current_weights, target_weights, threshold=0.05):
    """Rebalance if any weight drifts >5% from target."""
    drift = np.abs(current_weights - target_weights)
    return np.any(drift > threshold)
```

### 5.3 Volatility-Based Rebalancing

Rebalance more frequently during high volatility:

```python
def rebalancing_frequency(volatility, base_days=7):
    """Adjust rebalancing frequency based on volatility."""
    if volatility > 0.30:  # High volatility
        return base_days // 2  # Rebalance more often
    elif volatility < 0.10:  # Low volatility
        return base_days * 2   # Rebalance less often
    return base_days
```

---

## 6. Backtesting Framework

```python
import pandas as pd

class PortfolioBacktest:
    """Backtest portfolio optimization strategies."""

    def __init__(self, initial_capital, strategy='kelly_mpt'):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy = strategy
        self.history = []

    def run(self, market_data):
        """
        Run backtest on historical market data.

        Args:
            market_data: DataFrame with columns:
                - date, market_name, market_price, resolution, payout
        """
        for date in market_data['date'].unique():
            day_data = market_data[market_data['date'] == date]

            # Calculate positions for this day
            positions = self._calculate_positions(day_data)

            # Execute trades
            for pos in positions:
                result = self._execute_trade(pos)
                self.history.append({
                    'date': date,
                    'market': pos['market_name'],
                    'allocation': pos['size'],
                    'return': result
                })

            # Update capital
            daily_return = sum([h['return'] for h in self.history if h['date'] == date])
            self.capital += daily_return

        return self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        df = pd.DataFrame(self.history)
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Calculate Sharpe ratio
        daily_returns = df.groupby('date')['return'].sum()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        return {
            'total_return': total_return,
            'final_capital': self.capital,
            'sharpe_ratio': sharpe,
            'num_trades': len(self.history)
        }
```

---

## Best Practices

1. **Use Fractional Kelly**: Half-Kelly (50%) or Quarter-Kelly (25%) to reduce volatility
2. **Diversify Across Categories**: Politics, sports, crypto, economics
3. **Rebalance Regularly**: Weekly or when volatility changes significantly
4. **Cap Position Sizes**: Never exceed 20-25% in single market
5. **Account for Correlation**: Use correlation matrix in optimization
6. **Stress Test**: Simulate worst-case scenarios
7. **Track Edge Accuracy**: Monitor your probability estimates vs. outcomes

---

## References

- [Kelly Criterion - Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Application of the Kelly Criterion to Prediction Markets (arXiv)](https://arxiv.org/abs/2412.14144)
- [Kelly Criterion in Portfolio Optimization](https://medium.com/@jatinnavani/the-kelly-criterion-and-its-application-to-portfolio-management-3490209df259)
- [Stanford Kelly Criterion Probability Theory](https://crypto.stanford.edu/~blynn/pr/kelly.html)
- [Modern Portfolio Theory for Crypto](https://coinbureau.com/education/modern-portfolio-theory-crypto/)
- [MPT in Crypto Trading](https://medium.com/thecapital/modern-portfolio-theory-in-crypto-balancing-risk-and-reward-77b2ed7667c3)
- [Efficient Frontier with Python](https://towardsdatascience.com/python-markowitz-optimization-b5e1623060f5)
