import numpy as np
import pandas as pd
from numba.typed import List
from math import ceil, sqrt, exp, log
from numba import njit, prange
import matplotlib.pyplot as plt
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import winsorize
import seaborn as sns
import statsmodels.api as sm
from pypfopt import EfficientFrontier, risk_models, expected_returns
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from pprint import pprint
from time import time 
from dateutil.relativedelta import relativedelta
from datetime import date


# Standard normal density function
@njit("(double(double))", cache=False, nogil=True)
def standard_normal_dist(x: float) -> float:
    return np.exp(-0.5 * x**2) / (np.sqrt(2 * np.pi))


# Compute coefficients for the Efficient Frontier (Modern Portfolio Theory)
@njit(cache=False, nogil=True)
def get_coefs_efficient_frontier(mean_returns_vector: np.ndarray, inv_cov_matrix: np.ndarray, cov_matrix: np.ndarray) -> tuple:
    o_vector = np.ones_like(mean_returns_vector)
    
    k = mean_returns_vector.T @ (inv_cov_matrix @ o_vector)
    l = mean_returns_vector.T @ (inv_cov_matrix @ mean_returns_vector)
    p = o_vector.T @ (inv_cov_matrix @ o_vector)
    
    denominator = l * p - k**2
    
    g_vector = (l * (inv_cov_matrix @ o_vector) - k * (inv_cov_matrix @ mean_returns_vector)) / denominator
    h_vector = (p * (inv_cov_matrix @ mean_returns_vector) - k * (inv_cov_matrix @ o_vector)) / denominator
    
    a = h_vector.T @ (cov_matrix @ h_vector)
    b = 2. * (g_vector.T @ (cov_matrix @ h_vector))
    c = g_vector.T @ (cov_matrix @ g_vector)
    

    return a, b, c, g_vector, h_vector


# Compute standard deviation for a given expected return on the efficient frontier
@njit("double(double, double, double, double)", cache=False, nogil=True)
def get_sigma(mu: float, a: float, b: float, c: float) -> float:
    return np.sqrt(a * mu**2 + b * mu + c)


def find_closest_index(array, value):
    return np.abs(array - value).argmin()


def simulate_regime_paths(T: int, num_paths: int, transition_matrix: np.ndarray, initial_regime: int = 0) -> np.ndarray:
    """
    Simulates multiple regime paths using the provided Markov transition matrix.

    Parameters:
    - T (int): Number of time steps
    - num_paths (int): Number of paths to simulate
    - transition_matrix (np.ndarray): 2x2 Markov transition matrix
    - initial_regime (int): 0 for Bull, 1 for Bear (default: 0)

    Returns:
    - np.ndarray: A (num_paths x T) array of regimes (0 or 1)
    """
    paths = np.zeros((num_paths, T), dtype=int)
    paths[:, 0] = initial_regime

    for path_idx in range(num_paths):
        for t in range(1, T):
            current_regime = paths[path_idx, t - 1]
            next_regime = np.random.choice(
                [0, 1],
                p=transition_matrix[current_regime].flatten()  # <-- Add `.flatten()` here
            )

            paths[path_idx, t] = next_regime

    return paths


def calculate_probabilities(initial_wealth, debug, grid_points):
    # Reverse debug list to align with time
    probs_hist = [v[np.abs(grid_points[t] - initial_wealth).argmin()] for t, v in enumerate(debug[::-1])]
    return probs_hist


def build_weight_table(weights_array, mus, sigmas, asset_names):
    table = []
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        row = {
            "Portfolio": f"{i + 1}",
            "μ (annualized)": round(mu * 12, 4),
            "σ (annualized)": round(sigma * np.sqrt(12), 4)
        }
        for asset, weight in zip(asset_names, weights_array[i]):
            row[asset] = round(weight, 4)
        table.append(row)
    return pd.DataFrame(table)


def calculate_portfolio_metrics(portfolio_value, frequency):
    # Calculate returns from wealth path
    portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]

    if frequency == 'monthly':
        day_counts = 12
    else:
        day_counts = 252

    # Performance Metrics
    final_wealth = portfolio_value[-1]
    mean_return = np.mean(portfolio_returns) * day_counts
    volatility = np.std(portfolio_returns) * np.sqrt(day_counts)


    sharpe_ratio = (mean_return) / (volatility) if volatility != 0 else np.nan


    # Max drawdown calculation
    cumulative_returns = np.array(portfolio_value)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)

    # cpi_series = df['CPI'].loc[historical_returns.index]
    # # Convert to real terms (adjust to base CPI at t=0)
    # cpi_base = cpi_series.iloc[0]
    # real_wealth = np.array(portfolio_value) * (cpi_base / cpi_series.values)
    # real_returns = np.diff(real_wealth) / real_wealth[:-1]
    # real_mean_return = np.mean(real_returns)
    # real_volatility = np.std(real_returns)
    # real_sharpe = (real_mean_return * 12) / (real_volatility * np.sqrt(12)) if real_volatility != 0 else np.nan
    # real_final_wealth = real_wealth[-1]
    # real_success = real_final_wealth >= goal

        
    # Output results
    print("\n====== Performance Metrics (Historical Regime Path) ======")
    print(f"Final Wealth: ${final_wealth:,.2f}")
    print(f"Mean Monthly Return: {mean_return:.4%}")
    print(f"Volatility (Monthly): {volatility:.4%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print("==========================================================")

    # print("\n== Inflation-Adjusted Metrics ==")
    # print(f"Final Real Wealth: ${real_final_wealth:,.2f}")
    # print(f"Annualized Real Sharpe Ratio: {real_sharpe:.2f}")
    # print(f"Goal Achieved (in Real Terms): {'Yes' if real_success else 'No'}")
    # print("==========================================================")

    return final_wealth, mean_return, volatility, sharpe_ratio, max_drawdown


def gbi_regime_switching_model(initial_value, mu_portfolios, grid_points, optimal_funds, debug, historical_returns, historical_regimes, bull_weights, bear_weights, all_assets):
    gbi_portfolio_value = [initial_value]
    fund_indexes = [None]
    probabilities_hist = []

    start_date = historical_returns.index[0]
    end_date = historical_returns.index[-1]

    # Generate ideal rebalance anchors from start date (monthly frequency)
    # Step 1: Generate desired monthly anchors (e.g., 1st of month or start_date + 1M, etc.)
    rebalance_targets = pd.date_range(start=start_date, end=end_date, freq='M')

    # Ensure start_date is included if not already
    if start_date not in rebalance_targets:
        rebalance_targets = pd.DatetimeIndex([start_date]).union(rebalance_targets)

    # Step 2: Align each target to the nearest *future or same* trading day
    # This assumes historical_returns.index is sorted and is a DatetimeIndex
    rebalance_dates = []
    for dt in rebalance_targets:
        next_trading_day = historical_returns.index[historical_returns.index >= dt]
        if not next_trading_day.empty:
            rebalance_dates.append(next_trading_day[0])

    # Convert to DatetimeIndex
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    # all_in_index = pd.Index(rebalance_dates).isin(historical_returns.index).all()

    # if all_in_index:
    #     print("✅ All rebalance_dates are in historical_returns index.")
    # else:
    #     print("❌ Some rebalance_dates are NOT in the historical_returns index.")
    
    historical_regime_names = historical_regimes.map({0: "Bull", 1: "Bear"})
    historical_regime_names = list(historical_regime_names)  # ensures alignment by position

    fund_index = None
    t = 0
    for i, row in historical_returns.iterrows():
        current_value = gbi_portfolio_value[-1]
        regime = historical_regimes.loc[i]

        if i in rebalance_dates and t < len(optimal_funds):
            # Dynamically select portfolio based on wealth and regime
            fund_index = get_optimal_portfolio_index(
                t=t,
                wealth=current_value,
                regime_path=historical_regimes.values,
                optimal_funds=optimal_funds,
                grid_points=grid_points,
                mu_portfolios=mu_portfolios
            )
            prob_success = get_goal_probability(t, current_value, debug, grid_points)
            # print(f"Rebalance on {i.date()} | Fund: {fund_index + 1} | Wealth: {current_value:.2f} | Prob. of success: {prob_success:.4f}")

            # print('Time:', t, ' | ', 'Fund:', fund_index + 1, ' | ', regime, ' | ', current_value)
            t += 1
        
        if regime == 0:
            total_return = (row['SPTR Index_Returns'] * bull_weights[fund_index][0]) + (row['VSMAX US Equity_Returns'] * bull_weights[fund_index][1]) 
            + (row['LUTLTRUU Index_Returns'] * bull_weights[fund_index][2]) + (row['IBOXHY Index_Returns'] * bull_weights[fund_index][3]) 
            + (row['BCOMTR Index_Returns'] * bull_weights[fund_index][4]) + (row['GOLDLNPM Index_Returns'] * bull_weights[fund_index][5]) 
            + (row['DJUSRET Index_Returns'] * bull_weights[fund_index][6])
        else:
            total_return = (row['SPTR Index_Returns'] * bear_weights[fund_index][0]) + (row['VSMAX US Equity_Returns'] * bear_weights[fund_index][1]) 
            + (row['LUTLTRUU Index_Returns'] * bear_weights[fund_index][2]) + (row['IBOXHY Index_Returns'] * bear_weights[fund_index][3]) 
            + (row['BCOMTR Index_Returns'] * bear_weights[fund_index][4]) + (row['GOLDLNPM Index_Returns'] * bear_weights[fund_index][5]) 
            + (row['DJUSRET Index_Returns'] * bear_weights[fund_index][6])
            
        # Update portfolio value
        new_value = current_value * (1 + total_return)
        gbi_portfolio_value.append(new_value)
        fund_indexes.append(fund_index)
        probabilities_hist.append(prob_success)

    
    # probabilities_hist = probabilities_hist[::-1]
        
    # Get current last date and go one month back
    latest_date = historical_returns.index[0]
    one_day_back = latest_date - pd.DateOffset(days=1)  # 2020-09-30

    portfolio_dates = [one_day_back] + list(historical_returns.index)
    probabilities_hist = [probabilities_hist[0]] + probabilities_hist
    historical_regime_names = [historical_regime_names[0]] + historical_regime_names

    gbi_results_df = pd.DataFrame({
        "Date": portfolio_dates,
        "Fund": fund_indexes,
        "Portfolio Value": gbi_portfolio_value,
        "Probability": probabilities_hist,
        "Regime": historical_regime_names,
    })

    # Step 1: Create mapping from fund/regime to weight vector
    bull_weights_dict = {i+1: dict(zip(all_assets, map(lambda x: round(x, 4), w))) for i, w in enumerate(bull_weights)}
    bear_weights_dict = {i+1: dict(zip(all_assets, map(lambda x: round(x, 4), w))) for i, w in enumerate(bear_weights)}

    # Step 2: Function to fetch the right dict based on fund and regime
    def get_weights(row):
        fund = row["Fund"]
        regime = row["Regime"]
        if pd.isna(fund) or fund == 0:
            return {asset: None for asset in all_assets}
        elif regime == "Bull":
            return bull_weights_dict.get(fund, {asset: None for asset in all_assets})
        elif regime == "Bear":
            return bear_weights_dict.get(fund, {asset: None for asset in all_assets})
        else:
            return {asset: None for asset in all_assets}

    # Step 3: Apply it
    gbi_results_df["Weights"] = gbi_results_df.apply(get_weights, axis=1)
    gbi_results_df["Fund"] = gbi_results_df["Fund"] + 1
    gbi_results_df.at[0, "Weights"] = gbi_results_df.loc[1, "Weights"]
    gbi_results_df.at[0, "Fund"] = gbi_results_df.loc[1, "Fund"]

    return gbi_results_df


def equal_weight_model(initial_value, historical_returns):
    equal_portfolio_value = [initial_value]

    for _, row in historical_returns.iterrows():
        current_value = equal_portfolio_value[-1]
        avg_return = row.mean(skipna=True)  # handles missing data

        # Update portfolio value
        new_value = current_value * (1 + avg_return)
        equal_portfolio_value.append(new_value)

    # Adjust for the fact that portfolio_value has one more entry than monthly_returns
    latest_date = historical_returns.index[0]
    one_day_back = latest_date - pd.DateOffset(days=1)  # 2020-09-30

    portfolio_dates = [one_day_back] + list(historical_returns.index)

    equal_results_df = pd.DataFrame({
        "Date": portfolio_dates,
        "Portfolio Value": equal_portfolio_value
    })
    # equal_results_df.set_index("Date", inplace=True)

    return equal_results_df


def backtest_portfolios(start_value, weights, returns_df):
    # Append "_Returns" to the weights' index to match returns_df columns
    # weights.index = [f"{asset}_Returns" for asset in weights.index]
    portfolio_dict = {}
    dates = returns_df.index
    returns_array = returns_df.values  # (T, 7)

    for i in range(15):
        portfolio_value = [start_value]
        for r in returns_array:
            current_value = portfolio_value[-1]
            total_return = (
                r[0] * weights[i][0] +
                r[1] * weights[i][1] +
                r[2] * weights[i][2] +
                r[3] * weights[i][3] +
                r[4] * weights[i][4] +
                r[5] * weights[i][5] +
                r[6] * weights[i][6]
            )

            new_value = current_value * (1 + total_return)
            portfolio_value.append(new_value)
        ind = i + 1
        portfolio_dict[ind] = portfolio_value
    
    latest_date = dates[0]
    one_day_back = latest_date - pd.DateOffset(days=1)  # 2020-09-30

    portfolio_dates = [one_day_back] + list(dates)

    backtest_portfolio_df = pd.DataFrame({
        "Date": portfolio_dates,
        "Portfolio Value": portfolio_value
    })
    # backtest_portfolio_df.set_index("Date", inplace=True)
    return backtest_portfolio_df


def backtest_regime_portfolios(initial_value, historical_returns, bull_weights, bear_weights, single_weights):
    bull_results = backtest_portfolios(initial_value, bull_weights, historical_returns)
    bear_results = backtest_portfolios(initial_value, bear_weights, historical_returns)
    single_results = backtest_portfolios(initial_value, single_weights, historical_returns)

    return bull_results, bear_results, single_results


def mean_variance_optimization_method(initial_value, daily_returns, monthly_returns, start_date, end_date):

    initial_value = 100_000
    portfolio_value = [initial_value]

    rebalance_weights = {}

    # Generate all rebalance dates
    rebalance_dates = monthly_returns.loc[start_date:end_date].index

    for rebalance_date in rebalance_dates:
        lookback_start = rebalance_date - relativedelta(years=1)
        
        # Filter lookback data for MVO
        lookback_data = monthly_returns.loc[lookback_start:rebalance_date]
        
        if lookback_data.shape[0] < 10:
            continue  # skip if not enough data points

        # Estimate expected returns and covariance
        # mu = mean_historical_return(lookback_data, frequency=12)
        # S = CovarianceShrinkage(lookback_data).ledoit_wolf()

        # Estimate expected returns & covariance matrix
        mu = lookback_data.mean() * 12  # annualized
        S = lookback_data.cov() * 12

        # Run MVO optimization
        ef = EfficientFrontier(mu, S)
        ef._solver = "SCS"  # or "ECOS" if you'd prefer
        weights = ef.max_sharpe()

        cleaned_weights = ef.clean_weights()

        # Store weights
        rebalance_weights[rebalance_date] = cleaned_weights
    
    i = 0
    returns_daily = daily_returns.loc[start_date:end_date]

    for date, row in returns_daily.iterrows():
        current_value = portfolio_value[-1]

        weights = rebalance_weights[rebalance_dates[i]]

        total_return = (
            row['SPTR Index_Returns'] * weights['SPTR Index_Returns'] +
            row['VSMAX US Equity_Returns'] * weights['VSMAX US Equity_Returns'] +
            row['LUTLTRUU Index_Returns'] * weights['LUTLTRUU Index_Returns'] +
            row['IBOXHY Index_Returns'] * weights['IBOXHY Index_Returns'] +
            row['BCOMTR Index_Returns'] * weights['BCOMTR Index_Returns'] +
            row['GOLDLNPM Index_Returns'] * weights['GOLDLNPM Index_Returns'] +
            row['DJUSRET Index_Returns'] * weights['DJUSRET Index_Returns']
        )


            # Update portfolio value
        new_value = current_value * (1 + total_return)
        portfolio_value.append(new_value)

        if date == rebalance_dates[i]:
            i += 1

    
    latest_date = returns_daily.index[0]
    one_day_back = latest_date - pd.DateOffset(days=1)  # 2020-09-30

    portfolio_dates = [one_day_back] + list(returns_daily.index)

    mvo_results_df = pd.DataFrame({
        "Date": portfolio_dates,
        "Portfolio Value": portfolio_value
    })

    
    # mvo_results_df.set_index("Date", inplace=True)

    return mvo_results_df


def get_goal_probability(t, wealth, debug, grid_points):
    """
    Returns the probability of reaching the goal given current wealth and time step.
    """
    if t >= len(debug):
        return None  # graceful fail
    
    grid = grid_points[t]
    idx_closest = np.abs(grid - wealth).argmin()
    return debug[-(t + 1)][idx_closest]  # reverse time order


def get_optimal_portfolio_index(
    t: int,
    wealth: float,
    regime_path: np.ndarray,
    optimal_funds: list,
    grid_points: list,
    mu_portfolios: np.ndarray
) -> int:
    """
    Returns the index (0-14) of the optimal portfolio at time `t` and wealth level `wealth`.

    Parameters:
    - t (int): Time step
    - wealth (float): Current wealth level
    - regime_path (np.ndarray): Regime at each time step
    - optimal_funds (list): Dynamic programming output [(t, mu_array, sigma_array)]
    - grid_points (list): List of wealth grid arrays over time
    - mu_portfolios (np.ndarray): Array of 15 pre-set portfolio expected returns

    Returns:
    - int: Optimal portfolio index (0 to 14)
    """

    # Sanity check
    if t >= len(grid_points):
        raise ValueError(f"Time step {t} exceeds time horizon.")
    
    # Get grid and closest wealth index
    grid = grid_points[t]
    idx_closest = np.abs(grid - wealth).argmin()

    # Get mu array at time t
    _, mu_array, _ = optimal_funds[-(t + 1)]  # reverse order
    mu_at_wealth = mu_array[idx_closest]

    # Map μ to closest portfolio index
    portfolio_idx = find_closest_index(mu_portfolios, mu_at_wealth)

    return portfolio_idx


# Dynamic Programming Algorithm for Wealth Optimization
@njit(cache=False, nogil=True, parallel=True)
def run_dynamic_programming(W_init: float, G: float, cashflows: np.ndarray,
    T: int, regime_path: np.ndarray,  # New: regime path of length T
    bull_mu: np.ndarray, bull_sigma: np.ndarray,
    bear_mu: np.ndarray, bear_sigma: np.ndarray, 
    mu_portfolios: np.ndarray,
    i_max_init: int, h: float):
   

    # Store selected funds
    optimal_funds = List()
    debug_info = List()

    # Generate state space gridpoints
    grid_points = List()
    grid_points.append(np.array([W_init]))


    sigma_max = max(bull_sigma[-1], bear_sigma[-1])
    mu_min, mu_max = mu_portfolios[0], mu_portfolios[-1]
    time_values = np.arange(0, T+1, 1)

    for tt in time_values[1:]:

        i_max_t = i_max_init * ceil(tt * h)
        i_array_t = np.arange(-i_max_t, i_max_t + 1, 1)

        W_minus_i_max_prev = grid_points[tt-1][0]
        W_i_max_prev = grid_points[tt-1][-1]
        cashflow_prev = cashflows[tt-1]

        if W_minus_i_max_prev + cashflow_prev <= 0.:
            W_i_pos_prev = grid_points[tt - 1][grid_points[tt-1] + cashflow_prev > 0.]
            assert len(W_i_pos_prev) != 0., 'Bankruptcy guaranteed'
            W_minus_i_max_prev = W_i_pos_prev[0]

        W_minus_i_max_t = (W_minus_i_max_prev + cashflow_prev) * exp(
            (mu_min - 0.5 * sigma_max**2) * h + sigma_max * sqrt(h) * (-3.5)
        )
        W_i_max_t = (W_i_max_prev + cashflow_prev) * exp(
            (mu_max - 0.5 * sigma_max**2) * h + sigma_max * sqrt(h) * 3.5
        )

        grid_points_t = np.exp(
            ((i_array_t - (-i_max_t)) / (2. * i_max_t)) *
            (log(W_i_max_t) - log(W_minus_i_max_t)) +
            log(W_minus_i_max_t)
        )

        grid_points.append(grid_points_t)

    # Solve Bellman equation by backward recursion
    value_i_t_plus_1 = np.where(grid_points[-1] >= G, 1., 0.)

    for tt in time_values[:-1][::-1]:
        regime = regime_path[tt]  # 0 for bull, 1 for bear
        # Select regime-specific μ and σ portfolios
        if regime == 0:
            mus = bull_mu
            sigmas = bull_sigma
        else:
            mus = bear_mu
            sigmas = bear_sigma

        transition_probabilities = np.zeros(
            shape=(grid_points[tt+1].shape[0], grid_points[tt].shape[0])
        )

        value_i_t = np.ones_like(grid_points[tt]) * -1.
        mu_i_t = np.zeros_like(grid_points[tt])
        sigma_i_t = np.zeros_like(grid_points[tt])

        for sigma, mu in zip(sigmas, mus):
            sigma_inv = 1. / sigma

            for j in prange(transition_probabilities.shape[0]):
                i_pos = np.flatnonzero(grid_points[tt] + cashflows[tt] > 0.)[0]

                for i in range(i_pos, transition_probabilities.shape[1]):
                    z = (sigma_inv) * (
                        log(grid_points[tt+1][j] / (grid_points[tt][i] + cashflows[tt])) -
                        (mu - 0.5 * sigma**2)
                    )

                    transition_probabilities[j, i] = standard_normal_dist(z)

            transition_probabilities = transition_probabilities / (transition_probabilities.sum(axis=0).reshape(1, -1) + 1e-8)
            value_i_mu = value_i_t_plus_1 @ transition_probabilities

            mask = value_i_mu > value_i_t
            value_i_t = np.where(mask, value_i_mu, value_i_t)
            mu_i_t = np.where(mask, mu, mu_i_t)
            sigma_i_t = np.where(mask, sigma, sigma_i_t)


        value_i_t_plus_1 = value_i_t

        # Store the optimal fund choice
        optimal_funds.append((tt, mu_i_t, sigma_i_t))
        debug_info.append((value_i_t)) # Store time step and probability
    
    return value_i_t, mu_i_t, sigma_i_t, optimal_funds, debug_info, grid_points




# Run the Model
def run_script(initial_wealth, goal, cashflow, frequency, start_date, end_date):
    # Calculate T, h, cashflows, frequency
    
    all_assets = [
        "SPTR Index", "VSMAX US Equity", "LUTLTRUU Index", "IBOXHY Index", 
        "BCOMTR Index", "GOLDLNPM Index", "DJUSRET Index"
    ]

    # Load dataset
    file_path = "FRE7043PORTFOLIOPIONEERSDATA.xlsx"  # Replace with actual file path
    df = pd.read_excel(file_path, parse_dates=["Date"])


    # Define key columns (Updated with new variables)
    key_columns = ["SPX Index", "SPTR Index", "VIX Index", "VBINX US Equity",
                "VLACX US Equity", "VSMAX US Equity", "VBMFX US Equity", "MOVE Index",
                "LBUSTRUU Index", "LUTLTRUU Index", "IBOXHY Index", "GT10 Govt",
                "GOLDLNPM Index", "DJUSRET Index", "BCOMTR Index", "SPGSCITR Index",
                "DBLCDBCE Index"]

    # Remove rows where ALL key columns are missing
    df = df.dropna(subset=key_columns, how="all")

    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Set Date as index
    df.set_index("Date", inplace=True)

    df_daily = df.copy()


    # Filter data from January 2004 onward
    df = df[df.index >= "2004-01-01"]
    df_daily = df_daily[df_daily.index >= "2004-01-01"]

    df = df.resample('M').last()

    # Reset index
    df.reset_index(inplace=True)
    df_daily.reset_index(inplace=True)

    # Save cleaned dataset
    # df.to_csv("cleaned_regime_data.csv", index=False)
    # df_daily.to_csv("cleaned_data.csv", index=False)

    # Display first few rows
    # print(df.head())

    df["VBINX_Returns"] = np.log(df['VBINX US Equity'] / df['VBINX US Equity'].shift(1))

    # Compute log returns for the selected assets
    for asset in all_assets:
        df[f"{asset}_Returns"] = df[asset].pct_change(1)

    # Drop only rows where ALL returns are missing (not just one)
    df.dropna(subset=[f"{asset}_Returns" for asset in all_assets], how="all", inplace=True)

    # Compute log returns for the selected assets
    for asset in all_assets:
        df_daily[f"{asset}_Returns"] = df_daily[asset].pct_change(1)

    # Drop only rows where ALL returns are missing (not just one)
    df_daily.dropna(subset=[f"{asset}_Returns" for asset in all_assets], how="all", inplace=True)

    df["VBINX_Returns STD"] = (df["VBINX_Returns"] - df["VBINX_Returns"].mean()) / df["VBINX_Returns"].std()

    adf_test = adfuller(df["VBINX_Returns STD"])
    # print(f"ADF Test Statistic: {adf_test[0]}")
    # print(f"P-Value: {adf_test[1]}")

    df["VBINX_Returns"] = winsorize(df["VBINX_Returns"], limits=[0.01, 0.01])

    # Fit a Two-Regime Markov Switching Model to VBINX returns
    model = MarkovRegression(df["VBINX_Returns"], k_regimes=2, trend="c", switching_variance=True)
    result = model.fit()

    # Extract Smoothed Probabilities for Each Regime
    df["Regime_0_Prob"] = result.smoothed_marginal_probabilities[0]  # Bull Market Probability
    df["Regime_1_Prob"] = result.smoothed_marginal_probabilities[1]  # Bear Market Probability

    # Assign Each Time Period to the Most Likely Regime
    df["Regime"] = np.where(df["Regime_0_Prob"] > df["Regime_1_Prob"], 0, 1)

    # Extract the Transition Probability Matrix
    transition_matrix = result.regime_transition
    # # Extract and reshape the transition probability matrix correctly
    transition_matrix = np.array(result.regime_transition).squeeze().T
    # print("Transition Probability Matrix:\n", transition_matrix)

    # Compute expected duration in each regime
    # if transition_matrix.ndim == 2:  # Ensure it's a 2D array
    #     expected_duration = 1 / (1 - np.diag(transition_matrix))
    #     print("Expected Duration of Regimes (Months):\n", expected_duration)
    # else:
    #     print("Error: Transition matrix is not in the expected 2D format.")

    # Convert monthly dates to the start of the month (ensures correct mapping)
    df["Month"] = df["Date"].dt.to_period("M")  # Convert to monthly period

    # Convert daily dates to their respective month
    df_daily["Month"] = df_daily["Date"].dt.to_period("M")  # Assign each daily date to its month

    # Merge the monthly regime values into the daily dataset
    df_daily = df_daily.merge(df[["Month", "Regime"]], on="Month", how="left")

    # Drop the temporary 'Month' column
    df_daily.drop(columns=["Month"], inplace=True)
    df.drop(columns=["Month"], inplace=True)

    df.set_index('Date', inplace=True)
    df_daily.set_index('Date', inplace=True)

    # Separate bull and bear markets for all assets
    bull_data = df_daily[df_daily["Regime"] == 0][[f"{asset}_Returns" for asset in all_assets]]
    bear_data = df_daily[df_daily["Regime"] == 1][[f"{asset}_Returns" for asset in all_assets]]
    single_data = df_daily[[f"{asset}_Returns" for asset in all_assets]]

    # Compute mean returns and covariance matrices for each regime
    bull_mean = bull_data.mean() * 252
    bear_mean = bear_data.mean() * 252

    bull_cov = bull_data.cov() * 252
    bear_cov = bear_data.cov() * 252

    single_mean = single_data.mean() * 252
    single_cov = single_data.cov() * 252

    bull_std = bull_data.std() * np.sqrt(252)
    bear_std = bear_data.std() * np.sqrt(252)
    single_std = single_data.std() * np.sqrt(252)

    # Display results
    # print("Bull Market Mean Returns:\n", bull_mean)
    # print("\nBear Market Mean Returns:\n", bear_mean)
    # print("\nSingle Regime Market Mean Returns:\n", single_mean)

    # print("Bull Market Standard Deviation of Returns:\n", bull_std)
    # print("\nBear Market Standard Deviation of Returns:\n", bear_std)
    # print("\nSingle Regime Market Standard Deviation of Returns:\n", single_std)

    # print("\nBull Market Covariance Matrix:\n", bull_cov)
    # print("\nBear Market Covariance Matrix:\n", bear_cov)
    # print("\nSingle Regime Market Covariance Matrix:\n", single_cov)

    single_mean = np.array(single_mean, dtype=np.float64) / 12
    single_cov = np.array(single_cov, dtype=np.float64)

    bull_mean = np.array(bull_mean, dtype=np.float64) / 12
    bull_cov = np.array(bull_cov, dtype=np.float64)

    bear_mean = np.array(bear_mean, dtype=np.float64) / 12
    bear_cov = np.array(bear_cov, dtype=np.float64)

   
    inv_cov_bull = np.linalg.inv(bull_cov)
    inv_cov_bear = np.linalg.inv(bear_cov)
    inv_cov_single = np.linalg.inv(single_cov)


    #########################################################################################################
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################


    num_simulations = 100
    
    
    if frequency == 'daily':
        h = 1/252
    elif frequency == 'monthly':
        h = 1/12


    delta = relativedelta(end_date, start_date)
    T = int(delta.years * 12 + delta.months)
    
    

    cashflows = np.array([0.] + [-cashflow * h for _ in range(T-1)])
    # initial_wealth = 100000.
    # goal = 200000.
    i_max_init = 100

    np.random.seed(42)

    mu_min = 0.055 * h
    mu_max = 0.15 * h
    m = 15
    mu_portfolios = np.linspace(mu_min, mu_max, m)


    bull_a, bull_b, bull_c, g_vector_bull, h_vector_bull = get_coefs_efficient_frontier(
        bull_mean, inv_cov_bull, bull_cov
    )
    sigma_portfolios_bull = np.array([get_sigma(mu, bull_a, bull_b, bull_c) for mu in mu_portfolios]) * np.sqrt(h)

    bear_a, bear_b, bear_c, g_vector_bear, h_vector_bear = get_coefs_efficient_frontier(
        bear_mean, inv_cov_bull, bear_cov
    )
    sigma_portfolios_bear = np.array([get_sigma(mu, bear_a, bear_b, bear_c) for mu in mu_portfolios]) * np.sqrt(h)
    
    single_a, single_b, single_c, g_vector_single, h_vector_single = get_coefs_efficient_frontier(
        single_mean, inv_cov_single, single_cov
    )
    sigma_portfolios_single = np.array([get_sigma(mu, single_a, single_b, single_c) for mu in mu_portfolios]) * np.sqrt(h)


    # print('---------------------------------------------------------------')


    # 🔹 ADD HERE: Compute asset weights for each portfolio
    
    asset_weights_list_bull = []
    asset_weights_list_bear = []
    asset_weights_list_single = []
    for mu in mu_portfolios:
        w_bull = g_vector_bull + h_vector_bull * mu  # Compute asset weights
        w_bear = g_vector_bear + h_vector_bear * mu  # Compute asset weights
        w_single = g_vector_single + h_vector_single * mu  # Compute asset weights
        asset_weights_list_bull.append(w_bull)
        asset_weights_list_bear.append(w_bear)
        asset_weights_list_single.append(w_single)


    # Convert to NumPy array
    asset_weights_bull = np.array(asset_weights_list_bull)
    asset_weights_bear = np.array(asset_weights_list_bear)
    asset_weights_single = np.array(asset_weights_list_single)


    weights_table_bull = build_weight_table(asset_weights_bull, mu_portfolios, sigma_portfolios_bull, all_assets)
    weights_table_bear = build_weight_table(asset_weights_bear, mu_portfolios, sigma_portfolios_bear, all_assets)
    weights_table_single = build_weight_table(asset_weights_single, mu_portfolios, sigma_portfolios_single, all_assets)



    start = time()


    # Backtest on historical regime path (last 120 months)
    
    historical_regimes = df_daily.loc[start_date:end_date, "Regime"]
    regimes_hist = historical_regimes.values
    historical_returns_monthly = df[[f"{asset}_Returns" for asset in all_assets]][start_date:end_date]
    historical_returns_daily = df_daily[[f"{asset}_Returns" for asset in all_assets]][start_date:end_date]
    df_daily_mvo = df_daily[[f"{asset}_Returns" for asset in all_assets]][:end_date]
    df_monthly_mvo = df[[f"{asset}_Returns" for asset in all_assets]][:end_date]

    # Run the DP algorithm on actual regime path
    prob, mu, sigma, optimal_funds, debug, grid_points = run_dynamic_programming(
        W_init=np.float64(initial_wealth),
        G=np.float64(goal),
        cashflows=cashflows,
        T=T,
        regime_path=regimes_hist,
        bull_mu=mu_portfolios,
        bull_sigma=sigma_portfolios_bull,
        bear_mu=mu_portfolios,
        bear_sigma=sigma_portfolios_bear,
        mu_portfolios=mu_portfolios,
        i_max_init=i_max_init,
        h=h
    )

    if frequency == 'monthly':
        main_set = historical_returns_monthly
    elif frequency == 'daily':
        main_set = historical_returns_daily
    
    gbi_portfolio = gbi_regime_switching_model(initial_wealth, mu_portfolios, grid_points, optimal_funds, debug, historical_returns_daily, historical_regimes, asset_weights_bull, asset_weights_bear, all_assets)
    gbi_metrics = calculate_portfolio_metrics(gbi_portfolio['Portfolio Value'].squeeze().values, 'daily')
    
    equal_portfolio = equal_weight_model(initial_wealth, historical_returns_daily)
    equal_metrics = calculate_portfolio_metrics(equal_portfolio['Portfolio Value'].squeeze().values, 'daily')

    individual_bull_portfolio, individual_bear_portfolio, individual_single_portfolio = backtest_regime_portfolios(initial_wealth, historical_returns_daily, asset_weights_bull, asset_weights_bear, asset_weights_single)
    
    mvo_portfolio = mean_variance_optimization_method(initial_wealth, df_daily_mvo, df_monthly_mvo, start_date, end_date)
    mvo_metrics = calculate_portfolio_metrics(mvo_portfolio['Portfolio Value'].squeeze().values, 'daily')
    end = time()

    print('---------------------------------------------------------------')
    pprint(end - start)

    return {
        "gbi": gbi_portfolio,
        "equal": equal_portfolio,
        "mvo": mvo_portfolio,
        "gbi_metrics": gbi_metrics,
        "equal_metrics": equal_metrics,
        "mvo_metrics": mvo_metrics,
        "bull_results": individual_bull_portfolio,     
        "bear_results": individual_bear_portfolio,
        "single_results": individual_single_portfolio,
        "weights_bull": weights_table_bull,
        "weights_bear": weights_table_bear,
        "weights_single": weights_table_single,
        "gbi_portfolio": gbi_portfolio,
        "data": df
    }

# if __name__ == '__main__':
#     start = pd.to_datetime("2015-01-01")
#     end = pd.to_datetime("2025-01-01")
#     stuff = run_script(100_000, 200_000, 50., 'monthly', start, end)