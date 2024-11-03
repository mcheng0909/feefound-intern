import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

stock_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'IBM', 'APLE', 'TSLA', 'NVDA', 'ORCL',
    'JNJ', 'PFE', 'MRK', 'JPM', 'BAC', 'WFC', 'KO', 'PG', 'NKE',
    'XOM', 'CVX', 'AMZN', 'WMT', 'TGT'
]

data = yf.download(stock_symbols, start="2023-01-01", end="2024-01-01", threads=True)

stock_prices = data['Adj Close']
stock_returns = stock_prices.pct_change().dropna()

composite_scores = []

def calculate_composite_score(symbol_data, symbol_volume):
    df = symbol_data.copy()
    pvt = np.zeros(len(df))
    previous_close = None
    previous_pvt = 0.0

    for i in range(1, len(df)):
        current_close = df.iloc[i]
        current_volume = symbol_volume.iloc[i]
        
        if previous_close is not None:
            if previous_close != 0:
                pvt[i] = previous_pvt + ((current_close - previous_close) / previous_close) * current_volume
        previous_close = current_close
        previous_pvt = pvt[i]

    pvt_change = pvt[-1] - pvt[0]

    roc_period = 14
    roc = df.pct_change(periods=roc_period) * 100
    roc_avg = roc.mean()

    obv = np.zeros(len(df))
    previous_obv = 0.0

    for i in range(1, len(df)):
        current_close = df.iloc[i]
        previous_close = df.iloc[i - 1]
        current_volume = symbol_volume.iloc[i]

        if current_close > previous_close:
            obv[i] = previous_obv + current_volume
        elif current_close < previous_close:
            obv[i] = previous_obv - current_volume
        previous_obv = obv[i]

    obv_change = obv[-1] - obv[0]

    composite_score = pvt_change + roc_avg + obv_change
    return composite_score

for symbol in stock_symbols:
    symbol_data = stock_prices[symbol]
    symbol_volume = data['Volume'][symbol]
    score = calculate_composite_score(symbol_data, symbol_volume)
    composite_scores.append({'Symbol': symbol, 'Score': score})

results_df = pd.DataFrame(composite_scores)
results_df = results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

sector_mapping = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'IBM': 'Tech', 'APLE': 'Real Estate', 'TSLA': 'Tech',
    'NVDA': 'Tech', 'ORCL': 'Tech', 'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'KO': 'Consumer Goods',
    'PG': 'Consumer Goods', 'NKE': 'Consumer Goods', 'XOM': 'Energy', 'CVX': 'Energy',
    'AMZN': 'Retail', 'WMT': 'Retail', 'TGT': 'Retail'
}

results_df['Sector'] = results_df['Symbol'].map(sector_mapping)

sector_avg_scores = results_df.groupby('Sector')['Score'].mean().reset_index()
print("\nSector Average Scores:\n", sector_avg_scores)

def sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = returns.mean()
    std_dev = returns.std()
    return (mean_return - risk_free_rate) / std_dev

sharpe_ratios = {}
for symbol in stock_symbols:
    symbol_returns = stock_returns[symbol]
    sharpe_ratios[symbol] = sharpe_ratio(symbol_returns)

sharpe_ratios_df = pd.DataFrame(list(sharpe_ratios.items()), columns=['Symbol', 'Sharpe Ratio']).sort_values(by='Sharpe Ratio', ascending=False)
print("\nSharpe Ratios for Stocks:\n", sharpe_ratios_df)

def calculate_ic(stock_symbols, stock_returns):
    factor_scores = {}

    for symbol in stock_symbols:
        stock_data = stock_returns[symbol]

        roc = stock_data.rolling(window=14).mean()
        obv = stock_data.cumsum()

        factor_scores[symbol] = {
            'ROC': roc,
            'OBV': obv
        }

    ics = {}
    for symbol, factors in factor_scores.items():
        ic_values = {}
        for factor, values in factors.items():
            next_period_returns = stock_returns[symbol].shift(-1)
            valid_index = values.dropna().index.intersection(next_period_returns.dropna().index)
            aligned_returns = next_period_returns.loc[valid_index]
            aligned_values = values.loc[valid_index]
            if len(aligned_values) > 0 and len(aligned_returns) > 0:
                ic, _ = spearmanr(aligned_values, aligned_returns)
                ic_values[factor] = ic

        ics[symbol] = ic_values

    return ics

ic_values = calculate_ic(stock_symbols, stock_returns)

print("Composite Scores for Stocks:\n", results_df)
print("\nInformation Coefficient (IC) for Factors by Stock:\n", ic_values)
print("\nSharpe Ratios for Stocks:\n", sharpe_ratios_df)
