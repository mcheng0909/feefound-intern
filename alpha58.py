
"""
This code performs a comprehensive analysis of stock performance using a series of custom "alpha" factors. 
Each alpha function applies different financial metrics to derive insights based on stock price, volume, and other 
variables. The main objectives and steps of the code are as follows:

1. **Data Retrieval**: Using yfinance, the code downloads historical stock data for specified symbols within a given time range.
2. **Alpha Factor Calculation**: The code defines 58 alpha functions, each generating a unique factor by combining various 
   metrics such as returns, rolling statistics, correlations, ranks, and conditions based on price, volume, and sector.
3. **Metrics Calculation**: For each alpha factor, key financial metrics are calculated, including the Sharpe ratio, 
   turnover, and cents-per-share to evaluate the risk-adjusted return and trading impact.
4. **Sector Adjustment**: Stocks are categorized by sector to allow for group-based neutralization of certain metrics.
5. **Correlation Analysis**: The code computes a returns matrix and generates a correlation matrix to assess dependencies 
   between alpha factors.
6. **Output and Analysis**: The final dataset provides alpha-based returns, correlations, and other statistics that can be 
   used to evaluate the predictive power of each alpha factor.

Overall, this code aims to provide a robust, multi-dimensional view of stock performance, helping identify factors that may 
serve as indicators for future stock returns.
"""


import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata
import csv

def calculate_Sharpe_Turnover_Cents(Pi, Vi, Di, Ii, Qi):
    
    Vi_safe = Vi if Vi != 0 else np.nan  
    Ii_safe = Ii if Ii != 0 else np.nan  
    Qi_safe = Qi if Qi != 0 else np.nan 
    
   
    Si = np.sqrt(252) * (Pi / Vi_safe) 
    Ti = Di / Ii_safe                   
    Ci = 100 * (Pi / Qi_safe)          
    
    return Si, Ti, Ci




stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'NVDA', 'PG',
                 'DIS', 'MA', 'HD', 'PEP', 'KO', 'NFLX', 'PYPL', 'NKE', 'XOM', 'VZ',
                 'MRK', 'WMT', 'T', 'BA', 'MCD', 'INTC', 'ADBE', 'CSCO', 'ORCL', 'PFE',
                 'ABT', 'MDT', 'UNH', 'LLY', 'MMM', 'HON', 'CAT', 'UPS', 'COST', 'WBA',
                 'CVS', 'AMGN', 'BKNG', 'GS', 'MS', 'BAC', 'C', 'CL', 'IBM', 'TXN', 'QCOM',
                 'NEE', 'DHR', 'SPGI', 'BLK', 'SCHW', 'RTX', 'PLD', 'AVGO', 'TMO', 'LOW',
                 'MDLZ', 'MO', 'SBUX', 'F', 'GM', 'DE', 'GD', 'BDX', 'LMT', 'DUK', 'SO',
                 'SRE', 'D', 'KHC', 'GE', 'WFC', 'USB', 'PNC', 'BK', 'AXP', 'MET', 'PRU',
                 'AIG', 'ALL', 'TRV', 'STT', 'ETN', 'ITW', 'EMR', 'ROK', 'PH', 'CMI', 'DOV']



data = yf.download(stock_symbols, start="2022-01-01", end="2023-01-01", threads=True)


def get_sector_data(stock_symbols):
    sector_map = {}
    for symbol in stock_symbols:
        stock_info = yf.Ticker(symbol).info
        sector_map[symbol] = stock_info.get('sector', 'Unknown')  
    return sector_map


sector_map = get_sector_data(stock_symbols)



def rank(series):
    return series.rank(pct=True)

def decay_linear(values, window):
    weights = np.arange(1, window + 1)
    return np.dot(values, weights) / weights.sum()

def scale(series):
    return (series - series.mean()) / series.std()

def spearman_corr(x, y):
    return pd.Series(rankdata(x)).corr(pd.Series(rankdata(y)))

def indneutralize(series, groupby_column):
    return series - series.groupby(groupby_column).transform('mean')

def alpha_1(df):
    ts_argmax = df['open_today'].rolling(window=20, min_periods=1).apply(np.argmax, raw=True)
    condition = df['open_today'] < df['close_today']
    return (ts_argmax * df['close_today'].rolling(window=5).std() if condition.all() else df['close_today']) - 0.5

def alpha_2(df):
    
    delta_log_volume = np.log(df['volume']).diff(periods=2)
    rank_1 = delta_log_volume.rank()
    rank_2 = ((df['close_today'] - df['open_today']) / df['open_today']).rank()
    return -1 * rank_1.corr(rank_2)

def alpha_3(df):
    
    return -1 * df['open_today'].rank(axis=0).corr(df['volume'].rank(axis=0))

def alpha_4(df):
    return -1 * df['open_today'].rank(axis=0).rolling(window=9).mean()

def alpha_5(df):
   
    vwap = (df['close_today'] * df['volume']).cumsum() / df['volume'].cumsum()
    return (df['open_today'] - vwap.rolling(window=10).mean()).rank() * (-1 * abs((df['close_today'] - vwap).rank()))

def alpha_6(df):
    
    return -1 * df['open_today'].corr(df['volume'].rolling(window=10).mean())

def alpha_7(df):
    condition = df['volume'] < df['open_today']
    ts_rank = df['close_today'].diff(periods=7).abs().rolling(window=60).apply(lambda x: rankdata(x)[-1], raw=True)
    sign_delta_close = np.sign(df['close_today'].diff(periods=7))
    return np.where(condition, (-1 * ts_rank * sign_delta_close), -1)

def alpha_8(df):
    return (-1 * df['open_today'].rolling(window=5).sum().rank() * df['volume'].rolling(window=5).sum().rank()
            - df['open_today'].rolling(window=5).sum().rank().shift(10) * df['volume'].rolling(window=5).sum().rank())

def alpha_9(df):
    cond = (0 < df['close_today'].diff(1).rolling(window=5).min())
    return np.where(cond, df['close_today'].diff(1), np.where(df['close_today'].diff(1).rolling(window=5).max() < 0, -df['close_today'].diff(1), 0))

def alpha_10(df):
    cond = (0 < df['close_today'].diff(1).rolling(window=4).min())
    return np.where(cond, df['close_today'].diff(1), np.where(df['close_today'].diff(1).rolling(window=4).max() < 0, -df['close_today'].diff(1), 0))

def alpha_11(df):
    vwap = (df['close_today'] * df['volume']).cumsum() / df['volume'].cumsum()
    return (vwap.sub(df['close_today']).rolling(window=3).max().rank() 
            + vwap.sub(df['close_today']).rolling(window=3).min().rank() * df['volume'].diff(1).rank())

def alpha_12(df):
    return np.sign(df['volume'].diff(1)) * (-1 * df['close_today'].diff(1))

def alpha_13(df):
    return (-1 * df['close_today'].rank().rolling(window=5).cov(df['volume'].rank()))

def alpha_14(df):
    return (-1 * df['volume'].diff(3).rank() * df['open_today'].rolling(window=10).corr(df['volume']))

def alpha_15(df):
    return (-1 * df['high_today'].rank().rolling(window=3).corr(df['volume'].rank()).rank().rolling(window=3).sum())

def alpha_16(df):
    return (-1 * df['high_today'].rank().rolling(window=5).cov(df['volume'].rank()).rank())

def alpha_17(df):
    return (-1 * df['close_today'].rolling(window=10).rank() 
            * df['close_today'].diff(1).diff(1).rank() * df['volume'].rolling(window=5).rank())

def alpha_18(df):
    return (-1 * df['close_today'].sub(df['open_today']).abs().rolling(window=5).std().rank() 
            + df['close_today'].sub(df['open_today']) + df['close_today'].rolling(window=10).corr(df['open_today']))

def alpha_19(df):
    df['returns'] = df['close_today'].pct_change()
    return (-1 * np.sign(df['close_today'] - df['close_today'].shift(7)) 
            + df['close_today'].diff(7)) * (1 + (1 + df['returns'].rolling(window=250).sum()).rank())

def alpha_20(df):
    return (-1 * (df['open_today'] - df['high_today'].shift(1)).rank() 
            * (df['open_today'] - df['close_today'].shift(1)).rank() 
            * (df['open_today'] - df['low_today'].shift(1)).rank())


def alpha_21(data):
    return np.where(
        (data['close_today'].rolling(window=8).sum() / 8 + data['close_today'].rolling(window=8).std()) < (data['close_today'].rolling(window=2).sum() / 2),
        -1 * np.where(
            (data['volume'] / data['volume'].rolling(window=20).mean()) == 1,
            1,
            -1
        ),
        1
    )

def alpha_22(data):
    return -1 * (data['high_today'].rolling(window=5).corr(data['volume']) * data['close_today'].rolling(window=20).std())

def alpha_23(data):
    return np.where((data['high_today'].rolling(window=20).sum() / 20) < data['high_today'], -1 * (data['high_today'].diff(periods=2)), 0)

def alpha_24(data):
    return np.where(
        (data['close_today'].rolling(window=100).sum() / 100).diff() / data['close_today'].shift(100) < 0.05,
        -1 * (data['close_today'] - data['close_today'].rolling(window=100).min()),
        -1 * data['close_today'].diff(periods=3)
    )

def alpha_25(data):
    adv20 = data['volume'].rolling(window=20).mean()
    return rank(((adv20 * -1 * data['returns']) * data['vwap']) * (data['high_today'] - data['close_today']))

def alpha_26(data):
    return -1 * data['volume'].rolling(window=5).corr(data['volume']).rank() * data['high_today'].rolling(window=5).rank()

def alpha_27(data):
    return np.where(
        rank(data['volume'].rolling(window=6).corr(data['vwap'])) < 2 / 2.0,
        -1,
        1
    )


def alpha_28(data):
    adv20 = data['volume'].rolling(window=20).mean()
    return ((data['high_today'] + data['low_today']) / 2 - data['close_today']).rank()

def alpha_29(data):
    close_diff = data['close_today'].diff()  
    rolling_min = data['close_today'].rolling(window=5).min()  
    ranked_series = rank(np.log(rank(rank(rolling_min + close_diff)))) 
    return ranked_series + rank(data['returns'].shift(periods=6)) 


def alpha_30(data):
    return np.where(
        rank(
            (np.sign(data['close_today'].diff(periods=1)) + np.sign(data['close_today'].shift(periods=1))) +
            data['volume'] / data['volume'].rolling(window=20).mean()
        ) > 0,
        data['volume'] / data['volume'].rolling(window=5).sum(),
        np.nan
    )

def alpha_31(data):
    close_diff = data['close_today'].diff(periods=10)
    rolling_diff = close_diff.rolling(window=10).apply(lambda x: decay_linear(x, 10))
    return rank(rank(rolling_diff))
def alpha_32(data):
    return scale((data['close_today'].rolling(window=7).sum() / 7) - data['close_today'])

def alpha_33(data):
    return rank((-1 * (data['open_today'] / data['close_today']) ** 1))

def alpha_34(data):
    return rank((1 - rank(data['returns'].rolling(window=5).std())) + (1 - rank(data['close_today'].diff(periods=1))))

def alpha_35(data):
    return (data['volume'].rolling(window=32).rank() * (1 - data['close_today'].rank()))

def alpha_36(data):

    rolling_corr = data['close_today'].rolling(window=15).apply(
        lambda x: spearman_corr(x, data['volume'].loc[x.index]), raw=False
    )
    return rank(rolling_corr) + rank(data['open_today'].rolling(window=5).corr(data['vwap']))

def alpha_37(data):
    return rank(data['close_today'].diff(periods=10).rolling(window=10).corr(data['volume']))

def alpha_38(data):
    return (-1 * rank(data['close_today']).rolling(window=10).corr(data['vwap']))

def alpha_39(data):
    return rank(-1 * data['close_today'].diff(periods=7).rolling(window=30).apply(lambda x: decay_linear(x, 30)))

def alpha_40(data):
    return (-1 * rank(data['high_today'].rolling(window=10).std()) * data['high_today'].corr(data['volume']))

def alpha_41(data):
    return ((data['high_today'] * data['low_today']) ** 0.5 - data['vwap'])

def alpha_42(data):
    return rank(data['vwap'] - data['close_today']) / rank(data['vwap'] + data['close_today'])

def alpha_43(data):
    return rank(data['volume'].rolling(window=20).mean()) * rank(-1 * data['close_today'].diff(periods=7))

def alpha_44(data):
    return (-1 * data['high_today'].corr(rank(data['volume'])))

def alpha_45(data):
    return (-1 * rank((data['close_today'].shift(5) - data['close_today']) / 20).corr(data['volume']))

def alpha_46(data):
    return np.where((data['close_today'].shift(20) - data['close_today']) < 0, -1 * data['close_today'].diff(periods=10), -1)

def alpha_47(data):
    return rank((data['close_today'] * data['volume']) / data['volume'].rolling(window=20).mean())

def alpha_48(data):
    return indneutralize(data['close_today'], data['sector']).rank()

def alpha_49(data):
    return np.where(data['close_today'].shift(20) - data['close_today'] < 0.1, -1 * (data['close_today'] - data['close_today'].shift(10)), -1)

def alpha_50(data):
    return (-1 * rank(data['volume'].rolling(window=5).corr(data['vwap'])))

def alpha_51(data):
    return np.where(data['close_today'].shift(20) - data['close_today'] > 0.05, -1 * data['close_today'].diff(periods=10), 1)

def alpha_52(data):
    return (data['low_today'].rolling(window=5).min() + data['volume'].rank()).rank()

def alpha_53(data):
    return (-1 * data['close_today'].diff(periods=1) - data['close_today'].diff(periods=3))

def alpha_54(data):
    return (-1 * (data['low_today'] - data['close_today'] * data['open_today']))

def alpha_55(data):
    return (-1 * data['high_today'].rank())

def alpha_56(data):
    return np.where(data['returns'].sum() < 0, -1, 0)

def alpha_57(data):
    return (data['close_today'] - data['vwap'] * rank(data['close_today'].rolling(window=30).max()))

def alpha_58(data):
    return rank(data['vwap'] - data['close_today']).rank()



alpha_functions = [
    alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, 
    alpha_8, alpha_9, alpha_10, alpha_11, alpha_12, alpha_13, 
    alpha_14, alpha_15, alpha_16, alpha_17, alpha_18, alpha_19, alpha_20,
    alpha_21, alpha_22, alpha_23, alpha_24, alpha_25, alpha_26, alpha_27,
    alpha_28, alpha_29, alpha_30, alpha_31, alpha_32, alpha_33, alpha_34,
    alpha_35, alpha_36, alpha_37, alpha_38, alpha_39, alpha_40,
    alpha_41, alpha_42, alpha_43, alpha_44, alpha_45, alpha_46,
    alpha_47, alpha_48, alpha_49, alpha_50, alpha_51, alpha_52,
    alpha_53, alpha_54, alpha_55, alpha_56, alpha_57, alpha_58
]


def apply_alphas(df, alpha_functions):
    for i, alpha_func in enumerate(alpha_functions, start=1):
        alpha_name = f'alpha_{i}'
        df[alpha_name] = alpha_func(df)  
    return df


def generate_alpha_data_from_real_data(data, alpha_functions):
    stock_dfs = []
    for stock in stock_symbols:
        stock_df = pd.DataFrame({
            'open_today': data['Open'][stock],
            'close_today': data['Close'][stock],
            'high_today': data['High'][stock],
            'low_today': data['Low'][stock],
            'volume': data['Volume'][stock]
        })
        stock_df['vwap'] = (stock_df['close_today'] * stock_df['volume']).cumsum() / stock_df['volume'].cumsum()
        stock_df['stock'] = stock
        stock_df['sector'] = sector_map.get(stock, 'Unknown')

        stock_df = apply_alphas(stock_df, alpha_functions)

        alpha_columns = {}
        for i in range(1, 59):
            alpha_col = f'alpha_{i}'
            if alpha_col in stock_df.columns:
                alpha_columns[f'Pi_alpha_{i}'] = stock_df[alpha_col]
                alpha_columns[f'Vi_alpha_{i}'] = stock_df['volume'] * stock_df[alpha_col]
                alpha_columns[f'Di_alpha_{i}'] = alpha_columns[f'Vi_alpha_{i}'] * 100
                alpha_columns[f'Ii_alpha_{i}'] = alpha_columns[f'Vi_alpha_{i}'] * 200
                alpha_columns[f'Qi_alpha_{i}'] = alpha_columns[f'Vi_alpha_{i}'] * 50


        stock_df = pd.concat([stock_df, pd.DataFrame(alpha_columns)], axis=1)


       
        stock_dfs.append(stock_df)

    combined_df = pd.concat(stock_dfs)

    return combined_df

df = generate_alpha_data_from_real_data(data, alpha_functions)


df = df.groupby('stock', group_keys=False).apply(lambda stock_df: apply_alphas(stock_df, alpha_functions))
df = df.reset_index(drop=True)


def generate_alpha_based_returns(df, n_alphas):
    returns_matrix = np.zeros((len(df), n_alphas))

    for i in range(1, n_alphas + 1):
        alpha_col = f'alpha_{i}'
        returns_matrix[:, i - 1] = np.nan_to_num(df[alpha_col], nan=0.0) * 0.01

    cov_matrix = np.cov(returns_matrix.T)
    volatilities = np.diag(cov_matrix)

    sigma = np.sqrt(volatilities)
    sigma = np.where(sigma == 0, 1, sigma)
    correlation_matrix = np.zeros_like(cov_matrix)

    for i in range(n_alphas):
        for j in range(n_alphas):
            correlation_matrix[i, j] = cov_matrix[i, j] / (sigma[i] * sigma[j])

    np.fill_diagonal(correlation_matrix, 1)
    off_diagonal_elements = [correlation_matrix[i, j] for i in range(n_alphas) for j in range(i + 1, n_alphas)]
    Psi_a_values = np.array(off_diagonal_elements)

    Psi_a_values_clean = np.nan_to_num(Psi_a_values, nan=0.0)
    Psi_a = np.mean(Psi_a_values_clean)

    print(f"\nAverage correlation (Ψ_a): {Psi_a:.3f}")
    return returns_matrix, cov_matrix, correlation_matrix, Psi_a_values_clean



alpha_analysis_results = []

def calculate_empirical_properties(alpha_name, df, Psi_a_values, alpha_result):
    print(f"\nEmpirical Properties for {alpha_name}")

    new_columns = {}

    Si, Ti, Ci = zip(*df.apply(
        lambda row: calculate_Sharpe_Turnover_Cents(row[f'Pi_{alpha_name}'], row[f'Vi_{alpha_name}'], row[f'Di_{alpha_name}'], row[f'Ii_{alpha_name}'], row[f'Qi_{alpha_name}']), axis=1))

    new_columns[f'Si_{alpha_name}'] = Si
    new_columns[f'Ti_{alpha_name}'] = Ti
    new_columns[f'Ci_{alpha_name}'] = Ci

    with np.errstate(divide='ignore', invalid='ignore'):
        new_columns[f'ln_R_{alpha_name}'] = np.log(np.where(df[f'Ii_{alpha_name}'] > 0, df[f'Pi_{alpha_name}'] / df[f'Ii_{alpha_name}'], np.nan)) + alpha_result
        new_columns[f'ln_sigma_{alpha_name}'] = np.log(np.where(df[f'Vi_{alpha_name}'] > 0, df[f'Vi_{alpha_name}'], np.nan))

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    regression_data = df[[f'ln_sigma_{alpha_name}', f'ln_R_{alpha_name}']].dropna()

    if not regression_data.empty:
        X = regression_data[[f'ln_sigma_{alpha_name}']]
        y = regression_data[f'ln_R_{alpha_name}']

        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]  
        intercept = reg.intercept_
        r_squared = reg.score(X, y)

        print(f"Calculated X (slope): {slope:.3f}")
        print(f"Intercept: {intercept}, R-squared: {r_squared}")
    else:
        print(f"Skipped regression for {alpha_name} due to missing data.")
        slope, intercept, r_squared = np.nan, np.nan, np.nan

    N = 20  
    mu = df[f'Ti_{alpha_name}'].mean()
    tau_i = df[f'Ti_{alpha_name}'] / mu
    ln_tau_i = np.log(tau_i)

    x_ij = np.ones((N, N))  
    y_ij = np.array([[1 * ln_tau_i.iloc[j] + 1 * ln_tau_i.iloc[i] for j in range(N)] for i in range(N)])
    z_ij = np.outer(ln_tau_i, ln_tau_i)

    composite_index = [(i, j) for i in range(1, N) for j in range(i)]  
    x_a = np.array([x_ij[i, j] for i, j in composite_index])
    y_a = np.array([y_ij[i, j] for i, j in composite_index])
    z_a = np.array([z_ij[i, j] for i, j in composite_index])

    Psi_a_values_clean = np.nan_to_num(Psi_a_values, nan=0.0)
    X_psi_clean = np.nan_to_num(np.vstack([x_a, y_a, z_a]).T, nan=0.0)

    min_len = min(X_psi_clean.shape[0], len(Psi_a_values_clean))
    X_psi_clean = X_psi_clean[:min_len]
    Psi_a_values_clean = Psi_a_values_clean[:min_len]

    assert X_psi_clean.shape[0] == len(Psi_a_values_clean), f"Size mismatch: X_psi_clean {X_psi_clean.shape[0]} vs Psi_a_values_clean {len(Psi_a_values_clean)}"


    reg_psi = LinearRegression().fit(X_psi_clean, Psi_a_values_clean)
    intercept_psi = reg_psi.intercept_
    coefficients_psi = reg_psi.coef_
    r_squared_psi = reg_psi.score(X_psi_clean, Psi_a_values_clean)

    print(f"Regression Ψ_a ~ x_a + y_a + z_a:")
    print(f"Intercept: {intercept_psi}, Coefficients: {coefficients_psi}, R-squared: {r_squared_psi}")

    result = {
        'Alpha': alpha_name,
        'Average Return': df[f'ln_R_{alpha_name}'].mean(),
        'Sharpe Ratio': df[f'Si_{alpha_name}'].mean(),
        'Turnover': df[f'Ti_{alpha_name}'].mean(),
        'Intercept': intercept,
        'Slope': slope,
        'R-squared': r_squared,
        'Intercept (Ψ_a)': intercept_psi,
        'Coefficients (Ψ_a)': coefficients_psi,
        'R-squared (Ψ_a)': r_squared_psi
    }
    alpha_analysis_results.append(result)

def save_results_to_csv(filename, results):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Alpha', 'Average Return', 'Sharpe Ratio', 'Turnover', 'Intercept', 'Slope', 'R-squared',
                      'Intercept (Ψ_a)', 'Coefficients (Ψ_a)', 'R-squared (Ψ_a)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            result['Coefficients (Ψ_a)'] = ','.join(map(str, result['Coefficients (Ψ_a)']))
            writer.writerow(result)

def save_top_10_results_to_csv(filename, top_alphas, metric):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Alpha', 'Sharpe Ratio', 'R-squared']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in top_alphas:
            writer.writerow({
                'Alpha': result['Alpha'],
                'Sharpe Ratio': f"{result['Sharpe Ratio']:.10f}",
                'R-squared': f"{result['R-squared']:.10f}"
            })

def print_top_alphas_by_metrics(alpha_analysis_results):
    valid_sharpe_alphas = [res for res in alpha_analysis_results if res['Sharpe Ratio'] > 0]
    valid_r_squared_alphas = [res for res in alpha_analysis_results if not pd.isna(res['R-squared'])]

    sorted_sharpe = sorted(valid_sharpe_alphas, key=lambda x: x['Sharpe Ratio'], reverse=True)
    
    print("Top 10 Alphas by Sharpe Ratio (with more precision):")
    for idx, result in enumerate(sorted_sharpe[:10], start=1):
        print(f"{idx}. {result['Alpha']} - Sharpe Ratio: {result['Sharpe Ratio']:.10f}")

    sorted_r_squared = sorted(valid_r_squared_alphas, key=lambda x: x['R-squared'], reverse=True)
    
    print("\nTop 10 Alphas by R-squared (with more precision):")
    for idx, result in enumerate(sorted_r_squared[:10], start=1):
        print(f"{idx}. {result['Alpha']} - R-squared: {result['R-squared']:.10f}")

    save_top_10_results_to_csv('top_10_sharpe_alphas.csv', sorted_sharpe[:10], 'Sharpe Ratio')
    save_top_10_results_to_csv('top_10_r_squared_alphas.csv', sorted_r_squared[:10], 'R-squared')

print_top_alphas_by_metrics(alpha_analysis_results)

n_alphas = len(alpha_functions)
returns_matrix, cov_matrix, correlation_matrix, Psi_a_values = generate_alpha_based_returns(df, n_alphas)

for i in range(1, n_alphas + 1):
    alpha_name = f'alpha_{i}'
    calculate_empirical_properties(alpha_name, df, Psi_a_values, df[alpha_name])

save_results_to_csv('alpha_analysis_results_with_regression.csv', alpha_analysis_results)




