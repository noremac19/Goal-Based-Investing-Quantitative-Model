### Regime-Switching Goals-Based Investing Dashboard

This repository contains an interactive Shiny for Python web application for simulating and visualizing portfolio performance under different investment strategies — including Dynamic Programming with Regime Switching (GBI), Mean-Variance Optimization (MVO), and Equal Weighting.

The model dynamically adjusts portfolio weights based on Markov regime-switching logic, simulating transitions between bull and bear markets to optimize wealth relative to a defined goal.

## Features

### Core Models
- **GBI Regime-Switching Model**: Implements a dynamic programming approach with Markov regime switching to optimize investment strategies based on market states (bull vs. bear).
- **Mean-Variance Optimization (MVO)**: Classical optimization of the risk-return tradeoff using efficient frontier theory.
- **Equal Weight Portfolio**: Baseline benchmark model where all assets are weighted equally.

### Visualization Dashboards
- **Portfolio Comparison:** Interactive line plots of portfolio value and performance over time.
- **Goal Probability Tracking:** Dynamic probability plot of achieving wealth goals.
- **Regime Heatmaps and Transitions:** Visualize regime classification, transition probabilities, and market behavior.
- **Asset Weight Evolution:** Displays how optimal portfolio weights change through time under each regime.
- **QuantStats Metrics:** Calculates Sharpe ratio, Sortino ratio, drawdowns, win rate, skewness, kurtosis, Value-at-Risk (VaR), and more.

### Inputs
- Initial wealth
- Target goal wealth
- Monthly cashflow (contribution or withdrawal)
- Data frequency (daily or monthly)
- Customizable backtest period

## Project Structure

```
.
├── app.py                  # Shiny app (UI + server logic)
├── app_methods.py          # Core model implementations and helper functions
├── requirements.txt        # Dependencies
├── www/                    # (Optional) Folder for heatmap and matrix images
│   ├── bull.png
│   ├── bear.png
│   ├── single.png
│   └── transition_matrix.png
└── README.md
```

## Methodology

### 1. Regime Classification
- A **Markov Switching Model** (`statsmodels.MarkovRegression`) classifies market states into bull and bear regimes using VBINX (balanced index fund) returns.
- Transition probabilities define regime persistence and switching likelihood.

### 2. Efficient Frontier Construction
- Using **PyPortfolioOpt**, the model derives expected returns (μ), covariance matrices (Σ), and efficient frontier coefficients (a, b, c).

### 3. Dynamic Programming Optimization
- Solves for the optimal portfolio allocation path that maximizes the probability of reaching a specified goal:
  ```
  maximize P(W_T ≥ G)
  ```
  subject to wealth transitions and regime-dependent returns.

### 4. Backtesting
- Historical returns and simulated regime paths are used to evaluate the strategies.
- Compares results across GBI, MVO, and Equal-Weight benchmarks.

## Installation

Clone the repository:
```
git clone https://github.com/yourusername/regime-switching-gbi-dashboard.git
cd regime-switching-gbi-dashboard
```

Install dependencies:
```
pip install -r requirements.txt
```

## Running the App

To launch the Shiny app locally:
```
shiny run --reload app.py
```

Then visit:
```
http://localhost:8000
```

## Data Requirements

The app expects a dataset named `FRE7043PORTFOLIOPIONEERSDATA.xlsx` (used internally by `app_methods.py`). It should include daily or monthly prices for key indices:

| Asset Name | Description |
|-------------|-------------|
| SPTR Index | S&P 500 Total Return |
| VSMAX US Equity | Small Cap Equity Index |
| LUTLTRUU Index | Long-Term Treasury Bonds |
| IBOXHY Index | High Yield Corporate Bonds |
| BCOMTR Index | Bloomberg Commodity Index |
| GOLDLNPM Index | Gold Prices |
| DJUSRET Index | Real Estate Index |

## Example Outputs

- **Portfolio Value Over Time**: Comparative growth paths for GBI, Equal Weight, and MVO.
- **Heatmaps**: Optimal portfolio selection by regime and wealth level.
- **Probability Curves**: Goal achievement probability trajectories.
- **Metrics Tables**: Sharpe ratio, drawdown, CAGR, and risk analytics.

## Technologies Used

- Python (NumPy, Pandas, Numba, Matplotlib)
- Shiny for Python (shiny, shinywidgets)
- PyPortfolioOpt for efficient frontier optimization
- Statsmodels for Markov regime classification
- Plotly for dynamic visualizations
- QuantStats for performance analytics

## Authors

**Cameron Walcott**  
Master of Science in Financial Engineering, NYU Tandon  
Research areas: Goals-Based Investing, Regime-Switching Models, Portfolio Optimization, Dynamic Programming

## License

This project is licensed under the MIT License.  
Feel free to use, modify, and distribute with attribution.

<img width="432" height="645" alt="image" src="https://github.com/user-attachments/assets/b937c8a6-0ce7-409a-8b95-74c5a3d612d0" />
