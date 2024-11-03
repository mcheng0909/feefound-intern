
"""
This program constructs and evaluates a portfolio using factor-based modeling and statistical techniques.
The code aims to create a robust model that dynamically adjusts portfolio weights based on factor performance,
style rotation, and sector adjustments. It integrates a variety of techniques:

1. **Data Generation**: Simulates stock factor scores, returns, sector classifications, and macroeconomic data.
2. **Factor Decay Application**: Applies exponential decay to factor scores for a more realistic representation of factor influence over time.
3. **Information Coefficient Calculation**: Calculates Information Coefficients (IC) to gauge the predictive power of factors.
4. **Decision Tree Model**: Uses a pruned decision tree classifier to capture non-linear relationships between factor scores and returns.
5. **Style Rotation with Linear Regression**: Projects factor returns based on exogenous data to capture style rotation.
6. **Panel Data Regression**: Fits a panel data model with dynamic weights, incorporating various features and sector adjustments.
7. **Portfolio Construction and Performance Evaluation**: Constructs a portfolio and calculates performance metrics such as the Sharpe ratio.
8. **Visualization**: Plots active return, active risk, and rolling IC trends for insights into portfolio behavior and factor stability.

The main performance goals include maximizing return, minimizing risk, and dynamically adjusting to shifting factor strengths.
The code is structured to be flexible and adaptable for different scenarios and adjustments, making it suitable for
backtesting and further enhancement with additional financial metrics.
"""




import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set seed for reproducibility
np.random.seed(42)

# Define factors and sectors
factors = ['value', 'growth', 'momentum', 'sentiment', 'quality', 'technical']
sectors_list = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Utilities', 'Real Estate', 'Industrials', 'Materials', 'Telecom']

# Configuration parameters
num_stocks = 100
num_periods = 12

# Generate data for sectors and factors
sectors = pd.Series(np.random.choice(sectors_list, num_stocks), index=[f'Stock_{i+1}' for i in range(num_stocks)])
factor_scores = {factor: pd.DataFrame(np.random.randn(num_periods, num_stocks), columns=[f"Stock_{i+1}" for i in range(num_stocks)]) for factor in factors}
stock_returns = pd.DataFrame(np.random.randn(num_periods, num_stocks), columns=[f"Stock_{i+1}" for i in range(num_stocks)])

# Macroeconomic and capital market data for additional features
macroeconomic_data = pd.DataFrame(np.random.randn(num_periods, 3), columns=['Inflation', 'GDP_Growth', 'Unemployment'])
capital_market_data = pd.DataFrame(np.random.randn(num_periods, 3), columns=['VIX', 'Yield_Spread', 'Commodity_Prices'])
seasonal_data = pd.DataFrame({'January_Effect': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'December_Effect': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]})

# Combined exogenous data for models
exogenous_data = pd.concat([macroeconomic_data, capital_market_data, seasonal_data], axis=1)

# Apply exponential decay to factor scores
def apply_decay(factor_scores, half_life):
    decay_factor = 0.5 ** (1 / half_life)
    decayed_scores = factor_scores.copy()
    for t in range(num_periods):
        decay_weight = decay_factor ** t
        decayed_scores.iloc[t] *= decay_weight
    return decayed_scores

# Reduce multicollinearity by removing highly correlated factors
def preprocess_factors(factor_scores, threshold=0.8):
    all_factors_df = pd.DataFrame({factor: factor_scores[factor].stack() for factor in factors})
    corr_matrix = all_factors_df.corr().abs()
    to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]
    selected_factors = [factor for factor in factors if factor not in to_drop]
    return selected_factors

# Calculate Information Coefficient (IC) using Spearman rank correlation
def calculate_ic(factor_scores, stock_returns):
    ics = {}
    next_period_returns = stock_returns.shift(-1)
    for factor in factors:
        ic_values = []
        for t in range(num_periods - 1):
            factor_values = factor_scores[factor].iloc[t]
            returns = next_period_returns.iloc[t]
            sector_adjusted_returns = returns.groupby(sectors).transform(lambda x: x - x.median())
            ic, _ = spearmanr(factor_values, sector_adjusted_returns, nan_policy='omit')
            ic_values.append(ic)
        ics[factor] = np.mean(ic_values)
    return ics

# Classify stock returns into outperform (1) or underperform (0) categories
def classify_returns(stock_returns):
    classified_returns = stock_returns.copy()
    for t in range(num_periods):
        median_return = stock_returns.iloc[t].median()
        classified_returns.iloc[t] = np.where(stock_returns.iloc[t] > median_return, 1, 0)
    return classified_returns

# Pruned decision tree model for non-linear factor-return relationships
def pruned_tree_model(factor_scores, classified_returns):
    tree_predictions = pd.DataFrame(0, index=range(num_periods), columns=classified_returns.columns)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for t in range(num_periods):
        X = pd.DataFrame({factor: factor_scores[factor].iloc[t] for factor in factors})
        y = classified_returns.iloc[t]
        tree = DecisionTreeClassifier(max_depth=5)
        tree.fit(X, y)
        predicted_class = tree.predict(X)
        tree_predictions.iloc[t] = predicted_class
    return tree_predictions

# Linear regression for style rotation prediction
def style_rotation_regression(factor_returns, exogenous_data):
    predicted_factor_returns = pd.DataFrame(index=factor_returns.index, columns=factor_returns.columns)
    for factor in factor_returns.columns:
        y = factor_returns[factor]
        X = exogenous_data.shift(1).fillna(0)
        model = LinearRegression()
        model.fit(X, y)
        predicted_factor_returns[factor] = model.predict(X)
    return predicted_factor_returns

# Panel data regression for dynamic factor modeling
def fit_panel_data_model(factor_scores, stock_returns, tree_predictions, sectors, predicted_factor_returns, exogenous_vars):
    portfolio_weights = pd.DataFrame(0.0, index=range(num_periods), columns=stock_returns.columns, dtype='float64')

    for t in range(num_periods):
        sector_dummies = pd.get_dummies(sectors).reindex(stock_returns.columns, axis=1, fill_value=0)
        tree_dummy_vars = pd.DataFrame(tree_predictions.iloc[t], columns=['TREE_Node']).reindex(stock_returns.columns, axis=0)

        selected_factors = preprocess_factors(factor_scores)

        X_factors = pd.DataFrame({factor: factor_scores[factor].iloc[t] for factor in selected_factors}).reindex(stock_returns.columns, axis=1)
        X_predicted_returns = pd.DataFrame(predicted_factor_returns.iloc[t]).reindex(stock_returns.columns, axis=0)
        X = pd.concat([X_factors, X_predicted_returns, sector_dummies, tree_dummy_vars, exogenous_vars.shift(1).fillna(0)], axis=1)

        y = stock_returns.iloc[t].reindex(X.index)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.to_numeric(y, errors='coerce').fillna(0)

        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()

        stock_coefficients = results.params[:len(stock_returns.columns)] 
        portfolio_weights.iloc[t] = stock_coefficients

    return portfolio_weights

# Construct portfolio returns using generated weights
def construct_portfolio_with_tree_and_regression(factor_scores, stock_returns, dynamic_weights, tree_predictions):
    portfolio_returns = (dynamic_weights * stock_returns).sum(axis=1)
    return portfolio_returns, dynamic_weights

# Calculate the Sharpe ratio to evaluate portfolio performance
def sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    return (mean_return - risk_free_rate) / std_dev

# Visualization of active return and active risk trends
def plot_active_return_and_risk(active_returns, active_risks):
    moving_avg = active_returns.rolling(window=12).mean()
    plt.figure(figsize=(10, 5))
    plt.bar(active_returns.index, active_returns, color='lightblue', label='Active return')
    plt.plot(moving_avg.index, moving_avg, color='red', label='12-month moving average')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Active Return')
    plt.ylabel('Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()

    rolling_active_risk = active_risks.rolling(window=36).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_active_risk.index, rolling_active_risk, color='blue', label='Active risk or tracking error (3-year rolling)')
    plt.title('Active Risk')
    plt.ylabel('Tracking Error')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()

# Apply decay to factor scores
half_life = 4
decayed_factor_scores = {factor: apply_decay(factor_scores[factor], half_life) for factor in factors}

# Classify returns and predict with decision tree
classified_returns = classify_returns(stock_returns)
tree_predictions = pruned_tree_model(decayed_factor_scores, classified_returns)

# Factor returns and predicted factor returns
factor_returns = pd.DataFrame({factor: stock_returns.mean(axis=1) for factor in factors})
predicted_factor_returns = style_rotation_regression(factor_returns, exogenous_data)

# Calculate dynamic portfolio weights
dynamic_weights_with_regression = fit_panel_data_model(
    decayed_factor_scores, stock_returns, tree_predictions, sectors, predicted_factor_returns, exogenous_data)

# Portfolio performance and Sharpe ratio
portfolio_returns_with_tree_and_regression, portfolio_weights_with_tree_and_regression = construct_portfolio_with_tree_and_regression(
    decayed_factor_scores, stock_returns, dynamic_weights_with_regression, tree_predictions)
portfolio_sharpe_ratio = sharpe_ratio(portfolio_returns_with_tree_and_regression)

# Output portfolio details
print("\nPortfolio Weights:")
print(portfolio_weights_with_tree_and_regression)
print("\nPortfolio Returns:")
print(portfolio_returns_with_tree_and_regression)
print(f"\nPortfolio Sharpe Ratio: {portfolio_sharpe_ratio}")

# Calculate and plot rolling IC trends
def rolling_ic(factor_scores, stock_returns, window=3):
    rolling_ics = {factor: [] for factor in factors}
    for t in range(num_periods - window + 1):
        for factor in factors:
            factor_values = factor_scores[factor].iloc[t:t + window].stack()
            returns = stock_returns.iloc[t:t + window].stack()
            ic, _ = spearmanr(factor_values, returns)
            rolling_ics[factor].append(ic)
    return pd.DataFrame(rolling_ics)

rolling_ic_values = rolling_ic(factor_scores, stock_returns, window=3)
plt.figure(figsize=(10, 6))
for factor in factors:
    plt.plot(rolling_ic_values.index, rolling_ic_values[factor], label=factor)
plt.title("Rolling IC Trends for Factors")
plt.xlabel("Time Periods")
plt.ylabel("Information Coefficient (IC)")
plt.legend(title="Factors")
plt.grid(True)
plt.show()
