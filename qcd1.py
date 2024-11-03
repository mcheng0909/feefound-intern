

"""
This code analyzes and ranks stock symbols based on calculated composite scores derived from several technical indicators.
Using the Alpha Vantage API, it retrieves daily historical stock data for specified symbols, then calculates key metrics to evaluate
price and volume trends. The primary steps and goals of this code are as follows:

1. **Data Retrieval**: Fetches full historical daily data for each specified stock symbol via Alpha Vantage API calls.
2. **Indicator Calculation**:
    - **Price Volume Trend (PVT)**: Assesses cumulative price and volume changes to capture the price trend.
    - **Rate of Change (ROC)**: Measures the percentage change in stock price over a defined period, indicating momentum.
    - **Average True Range (ATR)**: Provides insight into market volatility by averaging the true range of prices.
    - **Volume Rate of Change (VROC)**: Tracks changes in trading volume, signaling potential shifts in market interest.
    - **On-Balance Volume (OBV)**: Reflects cumulative buying and selling pressure based on volume and price changes.
3. **Composite Score Computation**: Aggregates the calculated indicators into a single composite score for each stock symbol, summarizing its overall performance.
4. **Ranking**: Ranks the specified stock symbols based on their composite scores, enabling a comparative analysis of stocks based on technical performance indicators.

This modular approach provides flexibility to add or modify indicators easily, making it adaptable to various market analysis strategies.
The final ranked results offer a snapshot of the relative strength or weakness of each stock based on the chosen metrics.
"""



import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

# API configuration
API_KEY = 'D4P2FSZGDPOUFU2F'
BASE_URL = 'https://www.alphavantage.co/query'

# Symbols for analysis
symbols = ['IBM', 'APLE']

# Fetch symbol data from Alpha Vantage
def fetch_symbol_data(symbol):
    url = f"{BASE_URL}?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    return response.json()

# Analyze a single stock symbol
def analyze_stock(symbol):
    data = fetch_symbol_data(symbol)
    time_series = data.get('Time Series (Daily)', {})
    last_refreshed = data.get('Meta Data', {}).get('3. Last Refreshed', '')

    if not last_refreshed:
        print(f"Error: No data found for symbol {symbol}")
        return None
    
    start_date = datetime.strptime(last_refreshed, '%Y-%m-%d').date()
    end_date = start_date - timedelta(days=366)
    
    data_recent_year = {date: values for date, values in time_series.items()
                        if end_date <= datetime.strptime(date, '%Y-%m-%d').date() <= start_date}

    df = pd.DataFrame.from_dict(data_recent_year, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Ensure all columns are numeric for calculations
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Calculate indicators
    pvt_change = calculate_pvt(df)
    roc_avg = calculate_roc(df, period=14)
    atr_avg = calculate_atr(df, period=14)
    vroc_avg = calculate_vroc(df, period=14)
    obv_change = calculate_obv(df)

    # Calculate composite score based on calculated indicators
    composite_score = pvt_change + roc_avg + atr_avg + vroc_avg + obv_change
    return {'Symbol': symbol, 'Score': composite_score}

# Price Volume Trend (PVT)
def calculate_pvt(df):
    df['PVT'] = 0.0
    previous_close = None
    previous_pvt = 0.0

    for i in range(1, len(df)):
        current_close = df.iloc[i]['4. close']
        current_volume = df.iloc[i]['5. volume']
        
        if previous_close:
            df.iloc[i, df.columns.get_loc('PVT')] = previous_pvt + ((current_close - previous_close) / previous_close) * current_volume
        
        previous_close = current_close
        previous_pvt = df.iloc[i]['PVT']
    
    return df['PVT'].iloc[-1] - df['PVT'].iloc[0]

# Rate of Change (ROC)
def calculate_roc(df, period):
    df['ROC'] = df['4. close'].pct_change(periods=period) * 100
    return df['ROC'].mean()

# Average True Range (ATR)
def calculate_atr(df, period):
    df['High-Low'] = df['2. high'] - df['3. low']
    df['High-PreviousClose'] = np.abs(df['2. high'] - df['4. close'].shift(1))
    df['Low-PreviousClose'] = np.abs(df['3. low'] - df['4. close'].shift(1))
    df['True Range'] = df[['High-Low', 'High-PreviousClose', 'Low-PreviousClose']].max(axis=1)
    df['ATR'] = df['True Range'].rolling(window=period).mean()
    return df['ATR'].mean()

# Volume Rate of Change (VROC)
def calculate_vroc(df, period):
    df['VROC'] = df['5. volume'].pct_change(periods=period) * 100
    return df['VROC'].mean()

# On-Balance Volume (OBV)
def calculate_obv(df):
    df['OBV'] = 0.0
    previous_obv = 0.0
    
    for i in range(1, len(df)):
        current_close = df.iloc[i]['4. close']
        previous_close = df.iloc[i - 1]['4. close']
        current_volume = df.iloc[i]['5. volume']
        
        if current_close > previous_close:
            df.iloc[i, df.columns.get_loc('OBV')] = previous_obv + current_volume
        elif current_close < previous_close:
            df.iloc[i, df.columns.get_loc('OBV')] = previous_obv - current_volume
        else:
            df.iloc[i, df.columns.get_loc('OBV')] = previous_obv
        
        previous_obv = df.iloc[i]['OBV']
    
    return df['OBV'].iloc[-1] - df['OBV'].iloc[0]

# Calculate scores for all symbols and rank them
def rank_symbols(symbols):
    results = []
    
    for symbol in symbols:
        score_data = analyze_stock(symbol)
        if score_data:
            results.append(score_data)
    
    # Rank based on composite scores
    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Main execution to rank symbols
ranked_results = rank_symbols(symbols)
print(ranked_results)
