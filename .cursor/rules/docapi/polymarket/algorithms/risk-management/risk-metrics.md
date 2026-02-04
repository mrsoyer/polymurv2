# Risk Metrics and Measurement

> Quantitative measures to track portfolio risk, performance, and exposure

## Overview

Risk metrics enable systematic evaluation of trading performance and risk exposure. This guide covers Value at Risk (VaR), Sharpe ratio, maximum drawdown, and other key metrics for prediction market portfolios.

---

## 1. Value at Risk (VaR)

### 1.1 Overview

**Value at Risk (VaR)** estimates the maximum potential loss over a specified time period at a given confidence level under normal market conditions.

**Definition:**
"VaR answers the question: What is the worst loss we might expect over a set time period, given normal market conditions, with X% confidence?"

**Example:**
- 95% VaR of $500 means: "We're 95% confident that losses won't exceed $500 tomorrow"
- Or equivalently: "There's a 5% chance of losing more than $500 tomorrow"

### 1.2 Historical Simulation VaR

Uses actual historical returns to estimate future risk.

**Method:**
1. Collect historical returns
2. Sort returns from worst to best
3. Find percentile corresponding to confidence level
4. VaR is the return at that percentile

**Python Implementation:**

```python
import numpy as np
import pandas as pd

class HistoricalVaR:
    """Calculate VaR using historical simulation."""

    def __init__(self, returns, confidence_level=0.95):
        """
        Args:
            returns: Array or Series of historical returns
            confidence_level: Confidence level (0.95 = 95%)
        """
        self.returns = np.array(returns)
        self.confidence_level = confidence_level

    def calculate(self):
        """Calculate historical VaR."""
        # VaR is the percentile at (1 - confidence_level)
        var_percentile = 1 - self.confidence_level
        var = np.percentile(self.returns, var_percentile * 100)
        return abs(var)  # Return as positive number

    def calculate_cvar(self):
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall).
        Average of all losses worse than VaR.
        """
        var = -self.calculate()  # Negative for losses
        # Find all returns worse than VaR
        tail_losses = self.returns[self.returns <= var]

        if len(tail_losses) == 0:
            return abs(var)

        cvar = abs(np.mean(tail_losses))
        return cvar

# Example
# Historical daily returns (as decimals)
returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.02,
                    -0.03, 0.04, -0.01, 0.01, -0.04, 0.02, -0.02])

var_calculator = HistoricalVaR(returns, confidence_level=0.95)

var_95 = var_calculator.calculate()
cvar_95 = var_calculator.calculate_cvar()

print(f"95% VaR: {var_95:.2%}")    # Worst expected loss
print(f"95% CVaR: {cvar_95:.2%}")  # Average loss in worst 5% of cases

# Dollar terms for $10,000 portfolio
portfolio_value = 10000
print(f"\nFor ${portfolio_value:,} portfolio:")
print(f"95% VaR: ${var_95 * portfolio_value:,.2f}")
print(f"95% CVaR: ${cvar_95 * portfolio_value:,.2f}")
```

**Pros:**
- Simple to calculate
- No assumptions about return distribution
- Uses actual historical data

**Cons:**
- Assumes future will resemble past
- Limited by historical sample size
- Doesn't capture unprecedented events

---

### 1.3 Parametric VaR (Variance-Covariance)

Assumes returns follow normal distribution.

**Formula:**
```
VaR = Portfolio_Value × σ × Z_score
```

Where:
- `σ` = portfolio standard deviation
- `Z_score` = standard normal inverse (1.65 for 95%, 2.33 for 99%)

**Python Implementation:**

```python
from scipy import stats

class ParametricVaR:
    """Calculate VaR assuming normal distribution."""

    def __init__(self, returns, confidence_level=0.95):
        self.returns = np.array(returns)
        self.confidence_level = confidence_level

    def calculate(self):
        """Calculate parametric VaR."""
        # Calculate mean and std of returns
        mean = np.mean(self.returns)
        std = np.std(self.returns)

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # VaR = mean + (z_score × std)
        # For losses, use negative z_score
        var = abs(mean + z_score * std)
        return var

# Example
param_var = ParametricVaR(returns, confidence_level=0.95)
var_95_param = param_var.calculate()

print(f"Parametric 95% VaR: {var_95_param:.2%}")
print(f"Historical 95% VaR: {var_95:.2%}")
```

**Pros:**
- Fast to calculate
- Smooth estimates

**Cons:**
- Assumes normal distribution (rarely true)
- Underestimates tail risk
- Poor for fat-tailed distributions (common in crypto)

---

### 1.3 Monte Carlo VaR

Simulates thousands of potential future scenarios.

**Method:**
1. Calculate historical mean and covariance matrix
2. Generate correlated random returns using Cholesky decomposition
3. Simulate portfolio evolution
4. Calculate VaR from simulated distribution

**Python Implementation:**

```python
class MonteCarloVaR:
    """Calculate VaR using Monte Carlo simulation."""

    def __init__(self, portfolio_weights, expected_returns, cov_matrix,
                 initial_value=10000, num_simulations=10000):
        """
        Args:
            portfolio_weights: Array of position weights (sum to 1)
            expected_returns: Expected daily returns for each asset
            cov_matrix: Covariance matrix of returns
            initial_value: Starting portfolio value
            num_simulations: Number of Monte Carlo runs
        """
        self.weights = np.array(portfolio_weights)
        self.returns = np.array(expected_returns)
        self.cov_matrix = cov_matrix
        self.initial_value = initial_value
        self.num_simulations = num_simulations

    def simulate(self, time_horizon=1):
        """
        Run Monte Carlo simulation.

        Args:
            time_horizon: Number of days to simulate

        Returns:
            Array of final portfolio values
        """
        # Cholesky decomposition for correlated random variables
        L = np.linalg.cholesky(self.cov_matrix)

        final_values = []

        for _ in range(self.num_simulations):
            portfolio_value = self.initial_value

            for _ in range(time_horizon):
                # Generate correlated random returns
                Z = np.random.standard_normal(len(self.weights))
                random_returns = self.returns + np.dot(L, Z)

                # Calculate portfolio return
                portfolio_return = np.dot(self.weights, random_returns)

                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)

            final_values.append(portfolio_value)

        return np.array(final_values)

    def calculate_var(self, confidence_level=0.95, time_horizon=1):
        """Calculate VaR from Monte Carlo simulation."""
        final_values = self.simulate(time_horizon)

        # Calculate losses (negative returns)
        losses = self.initial_value - final_values

        # VaR is the percentile of losses
        var = np.percentile(losses, confidence_level * 100)
        return var

    def calculate_cvar(self, confidence_level=0.95, time_horizon=1):
        """Calculate CVaR (Expected Shortfall)."""
        final_values = self.simulate(time_horizon)
        losses = self.initial_value - final_values

        var = np.percentile(losses, confidence_level * 100)

        # Average of losses exceeding VaR
        tail_losses = losses[losses >= var]
        cvar = np.mean(tail_losses)

        return cvar

# Example: 3-asset portfolio
weights = np.array([0.4, 0.3, 0.3])
expected_returns = np.array([0.001, 0.0015, 0.002])  # Daily returns
cov_matrix = np.array([
    [0.0004, 0.0001, 0.0002],
    [0.0001, 0.0003, 0.00015],
    [0.0002, 0.00015, 0.0006]
])

mc_var = MonteCarloVaR(weights, expected_returns, cov_matrix,
                       initial_value=10000, num_simulations=10000)

# 1-day VaR
var_1d = mc_var.calculate_var(confidence_level=0.95, time_horizon=1)
cvar_1d = mc_var.calculate_cvar(confidence_level=0.95, time_horizon=1)

print(f"Monte Carlo 1-day 95% VaR: ${var_1d:.2f}")
print(f"Monte Carlo 1-day 95% CVaR: ${cvar_1d:.2f}")

# 7-day VaR
var_7d = mc_var.calculate_var(confidence_level=0.95, time_horizon=7)
print(f"Monte Carlo 7-day 95% VaR: ${var_7d:.2f}")
```

**Pros:**
- Most flexible method
- Captures complex distributions
- Can model various scenarios

**Cons:**
- Computationally intensive
- Requires accurate covariance estimates
- Results vary between runs

---

## 2. Sharpe Ratio

### 2.1 Definition

**Sharpe Ratio** measures risk-adjusted return: excess return per unit of risk.

**Formula:**
```
Sharpe Ratio = (Rₚ - Rₓ) / σₚ
```

Where:
- `Rₚ` = portfolio return
- `Rₓ` = risk-free rate (often 0 for crypto/prediction markets)
- `σₚ` = portfolio standard deviation

**Interpretation:**
- Sharpe > 1.0: Good risk-adjusted returns
- Sharpe > 2.0: Very good
- Sharpe > 3.0: Excellent
- Sharpe < 1.0: Poor risk-adjusted returns

### 2.2 Implementation

```python
class SharpeRatio:
    """Calculate Sharpe ratio for portfolio."""

    def __init__(self, returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Args:
            returns: Array of periodic returns
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Trading periods per year (252 for daily)
        """
        self.returns = np.array(returns)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(self):
        """Calculate annualized Sharpe ratio."""
        # Mean return
        mean_return = np.mean(self.returns)

        # Standard deviation
        std_return = np.std(self.returns, ddof=1)

        # Risk-free rate per period
        rf_per_period = self.risk_free_rate / self.periods_per_year

        # Sharpe ratio
        if std_return == 0:
            return 0

        sharpe = (mean_return - rf_per_period) / std_return

        # Annualize
        sharpe_annual = sharpe * np.sqrt(self.periods_per_year)

        return sharpe_annual

    def calculate_rolling(self, window=30):
        """Calculate rolling Sharpe ratio."""
        rolling_sharpe = []

        for i in range(window, len(self.returns) + 1):
            window_returns = self.returns[i-window:i]
            mean = np.mean(window_returns)
            std = np.std(window_returns, ddof=1)

            if std > 0:
                sharpe = (mean / std) * np.sqrt(self.periods_per_year)
            else:
                sharpe = 0

            rolling_sharpe.append(sharpe)

        return np.array(rolling_sharpe)

# Example
daily_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns

sharpe_calc = SharpeRatio(daily_returns, risk_free_rate=0.02, periods_per_year=252)
sharpe = sharpe_calc.calculate()

print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

# Interpretation
if sharpe > 2.0:
    print("Excellent risk-adjusted returns")
elif sharpe > 1.0:
    print("Good risk-adjusted returns")
else:
    print("Poor risk-adjusted returns")
```

---

## 3. Maximum Drawdown

### 3.1 Definition

**Maximum Drawdown (MDD)** measures the largest peak-to-trough decline in portfolio value.

**Formula:**
```
Drawdown = (Trough Value - Peak Value) / Peak Value
MDD = max(all drawdowns)
```

### 3.2 Implementation

```python
class MaxDrawdown:
    """Calculate maximum drawdown metrics."""

    def __init__(self, portfolio_values):
        """
        Args:
            portfolio_values: Array or Series of portfolio values over time
        """
        self.values = np.array(portfolio_values)

    def calculate_drawdowns(self):
        """Calculate drawdown at each point in time."""
        # Running maximum (peak)
        running_max = np.maximum.accumulate(self.values)

        # Drawdown at each point
        drawdowns = (self.values - running_max) / running_max

        return drawdowns

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown."""
        drawdowns = self.calculate_drawdowns()
        max_dd = np.min(drawdowns)  # Most negative value
        return abs(max_dd)

    def calculate_drawdown_duration(self):
        """Calculate longest drawdown duration (number of periods)."""
        drawdowns = self.calculate_drawdowns()

        # Find periods in drawdown
        in_drawdown = drawdowns < 0

        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def get_drawdown_statistics(self):
        """Get comprehensive drawdown statistics."""
        drawdowns = self.calculate_drawdowns()
        max_dd = self.calculate_max_drawdown()
        duration = self.calculate_drawdown_duration()

        # Find max drawdown period
        dd_index = np.argmin(drawdowns)
        peak_index = np.argmax(self.values[:dd_index+1]) if dd_index > 0 else 0

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'longest_duration': duration,
            'peak_value': self.values[peak_index],
            'trough_value': self.values[dd_index],
            'peak_date_index': peak_index,
            'trough_date_index': dd_index
        }

# Example
portfolio_values = [10000, 10200, 10500, 10300, 9800, 9500, 9700,
                    10100, 10400, 10200, 9900, 10300, 10800]

mdd_calc = MaxDrawdown(portfolio_values)

stats = mdd_calc.get_drawdown_statistics()
print(f"Maximum Drawdown: {stats['max_drawdown_pct']:.2f}%")
print(f"Peak Value: ${stats['peak_value']:,.2f}")
print(f"Trough Value: ${stats['trough_value']:,.2f}")
print(f"Longest Drawdown Duration: {stats['longest_duration']} periods")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(portfolio_values, label='Portfolio Value')
plt.axhline(y=stats['peak_value'], color='g', linestyle='--', label='Peak')
plt.axhline(y=stats['trough_value'], color='r', linestyle='--', label='Trough')
plt.legend()
plt.title('Portfolio Value Over Time')

plt.subplot(2, 1, 2)
drawdowns = mdd_calc.calculate_drawdowns() * 100
plt.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
plt.plot(drawdowns, color='red')
plt.title('Drawdown Over Time')
plt.ylabel('Drawdown %')
plt.xlabel('Period')
plt.tight_layout()
plt.show()
```

---

## 4. Calmar Ratio

**Calmar Ratio** measures return per unit of maximum drawdown.

**Formula:**
```
Calmar Ratio = Annual Return / |Maximum Drawdown|
```

Higher is better (more return per unit of drawdown risk).

```python
def calculate_calmar_ratio(returns, portfolio_values):
    """
    Calculate Calmar ratio.

    Args:
        returns: Array of periodic returns
        portfolio_values: Array of portfolio values

    Returns:
        Calmar ratio
    """
    # Annualized return
    total_periods = len(returns)
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    periods_per_year = 252  # Assume daily
    years = total_periods / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Maximum drawdown
    mdd_calc = MaxDrawdown(portfolio_values)
    max_dd = mdd_calc.calculate_max_drawdown()

    # Calmar ratio
    if max_dd == 0:
        return float('inf')

    calmar = annual_return / max_dd
    return calmar

# Example
returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02, 0.01, -0.015])
values = [10000]
for r in returns:
    values.append(values[-1] * (1 + r))

calmar = calculate_calmar_ratio(returns, values)
print(f"Calmar Ratio: {calmar:.2f}")
```

---

## 5. Sortino Ratio

**Sortino Ratio** is like Sharpe but only penalizes downside volatility.

**Formula:**
```
Sortino Ratio = (Rₚ - Rₓ) / σ_downside
```

Where `σ_downside` = standard deviation of negative returns only.

```python
class SortinoRatio:
    """Calculate Sortino ratio (downside deviation)."""

    def __init__(self, returns, risk_free_rate=0.0, periods_per_year=252):
        self.returns = np.array(returns)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(self, mar=0.0):
        """
        Calculate Sortino ratio.

        Args:
            mar: Minimum acceptable return (MAR)
        """
        # Mean return
        mean_return = np.mean(self.returns)

        # Downside deviation (only negative returns)
        downside_returns = self.returns[self.returns < mar]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return float('inf')

        # Sortino ratio
        sortino = (mean_return - mar) / downside_std

        # Annualize
        sortino_annual = sortino * np.sqrt(self.periods_per_year)

        return sortino_annual

# Example
returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.02])

sortino_calc = SortinoRatio(returns, periods_per_year=252)
sortino = sortino_calc.calculate(mar=0.0)
sharpe = SharpeRatio(returns, periods_per_year=252).calculate()

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Sortino typically higher (only penalizes downside)")
```

---

## 6. Beta and Correlation

### 6.1 Beta

**Beta** measures sensitivity to market movements.

**Formula:**
```
β = Cov(Rₚ, Rₘ) / Var(Rₘ)
```

Where:
- `Rₚ` = portfolio returns
- `Rₘ` = market returns

```python
def calculate_beta(portfolio_returns, market_returns):
    """Calculate portfolio beta relative to market."""
    # Covariance between portfolio and market
    covariance = np.cov(portfolio_returns, market_returns)[0, 1]

    # Variance of market
    market_variance = np.var(market_returns, ddof=1)

    # Beta
    beta = covariance / market_variance
    return beta

# Example
portfolio_returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
market_returns = np.array([0.015, 0.01, -0.02, 0.02, -0.015])

beta = calculate_beta(portfolio_returns, market_returns)
print(f"Portfolio Beta: {beta:.2f}")

if beta > 1.0:
    print("Portfolio more volatile than market")
elif beta < 1.0:
    print("Portfolio less volatile than market")
```

### 6.2 Correlation

```python
def calculate_correlation_matrix(returns_matrix):
    """
    Calculate correlation matrix for multiple assets.

    Args:
        returns_matrix: 2D array (assets × time periods)
    """
    correlation_matrix = np.corrcoef(returns_matrix)
    return correlation_matrix

# Example: 3 markets
market1 = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
market2 = np.array([0.015, 0.018, -0.012, 0.014, -0.006])
market3 = np.array([-0.005, 0.025, 0.01, -0.02, 0.015])

returns_matrix = np.array([market1, market2, market3])
corr_matrix = calculate_correlation_matrix(returns_matrix)

print("Correlation Matrix:")
print(corr_matrix)
```

---

## 7. Comprehensive Risk Dashboard

```python
class RiskDashboard:
    """Comprehensive risk metrics dashboard."""

    def __init__(self, returns, portfolio_values, market_returns=None):
        self.returns = np.array(returns)
        self.values = np.array(portfolio_values)
        self.market_returns = market_returns

    def calculate_all_metrics(self):
        """Calculate all risk metrics."""
        metrics = {}

        # Returns
        total_return = (self.values[-1] / self.values[0]) - 1
        metrics['total_return'] = total_return
        metrics['total_return_pct'] = total_return * 100

        # Volatility
        metrics['volatility'] = np.std(self.returns, ddof=1)
        metrics['volatility_annual'] = metrics['volatility'] * np.sqrt(252)

        # Sharpe
        sharpe_calc = SharpeRatio(self.returns, periods_per_year=252)
        metrics['sharpe_ratio'] = sharpe_calc.calculate()

        # Sortino
        sortino_calc = SortinoRatio(self.returns, periods_per_year=252)
        metrics['sortino_ratio'] = sortino_calc.calculate()

        # Max Drawdown
        mdd_calc = MaxDrawdown(self.values)
        mdd_stats = mdd_calc.get_drawdown_statistics()
        metrics['max_drawdown'] = mdd_stats['max_drawdown']
        metrics['max_drawdown_pct'] = mdd_stats['max_drawdown_pct']
        metrics['drawdown_duration'] = mdd_stats['longest_duration']

        # Calmar
        metrics['calmar_ratio'] = calculate_calmar_ratio(self.returns, self.values)

        # VaR
        var_calc = HistoricalVaR(self.returns, confidence_level=0.95)
        metrics['var_95'] = var_calc.calculate()
        metrics['cvar_95'] = var_calc.calculate_cvar()

        # Beta (if market returns provided)
        if self.market_returns is not None:
            metrics['beta'] = calculate_beta(self.returns, self.market_returns)

        return metrics

    def print_report(self):
        """Print formatted risk report."""
        metrics = self.calculate_all_metrics()

        print("=" * 50)
        print(" RISK METRICS DASHBOARD")
        print("=" * 50)
        print(f"\n📈 RETURNS")
        print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"\n📊 RISK MEASURES")
        print(f"   Volatility (Annual): {metrics['volatility_annual']:.2%}")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"   Drawdown Duration: {metrics['drawdown_duration']} periods")
        print(f"   95% VaR: {metrics['var_95']:.2%}")
        print(f"   95% CVaR: {metrics['cvar_95']:.2%}")
        print(f"\n🎯 RISK-ADJUSTED RETURNS")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio: {metrics['calmar_ratio']:.2f}")

        if 'beta' in metrics:
            print(f"\n📉 MARKET EXPOSURE")
            print(f"   Beta: {metrics['beta']:.2f}")

        print("=" * 50)

# Example usage
returns = np.random.normal(0.001, 0.02, 252)
values = [10000]
for r in returns:
    values.append(values[-1] * (1 + r))

dashboard = RiskDashboard(returns, values)
dashboard.print_report()
```

---

## Best Practices

1. **Monitor VaR Daily**: Track 95% and 99% VaR
2. **Set Drawdown Limits**: Halt trading at 10-15% drawdown
3. **Track Sharpe Ratio**: Aim for >1.5 annualized
4. **Use CVaR for Tail Risk**: More conservative than VaR
5. **Compare to Benchmarks**: Track beta vs. crypto market
6. **Calculate Rolling Metrics**: Detect regime changes
7. **Stress Test**: Simulate 2x and 3x historical volatility

---

## References

- [Monte Carlo VaR Implementation](https://www.pyquantnews.com/the-pyquant-newsletter/quickly-compute-value-at-risk-with-monte-carlo)
- [VaR Methods Comparison](https://medium.com/@serdarilarslan/value-at-risk-var-and-its-implementation-in-python-5c9150f73b0e)
- [Maximum Drawdown Calculation](https://portfolio-geek.com/posts/articles/tutorials/max-drawdown)
- [Sharpe Ratio Calculation](https://www.investopedia.com/terms/s/sharperatio.asp)
- [VaR for Cryptocurrencies (GRF Study)](https://www.sciencedirect.com/science/article/pii/S0169207024001304)
- [VaR Models During Market Stress](https://www.mdpi.com/2071-1050/15/5/4395)
- [Calmar Ratio Guide](https://www.investopedia.com/terms/c/calmarratio.asp)
