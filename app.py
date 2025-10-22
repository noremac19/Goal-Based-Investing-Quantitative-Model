from shiny import App, ui, render, reactive
from app_methods import run_script
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from shinywidgets import render_plotly, output_widget, render_widget
import numpy as np
import quantstats as qs
import quantstats.stats as qs_stats
qs.extend_pandas()
from pandas_datareader import data as pdr

app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel("Simulation Inputs", 
            ui.hr(),
            ui.h1('Backtest Models'),
            ui.input_numeric("initial_wealth", "Initial Wealth ($)", value=100000),
            ui.input_numeric("goal", "Goal ($)", value=200000),
            ui.input_numeric("cashflow", "Monthly Cashflow ($)", value=0),
            ui.input_select("frequency", "Data Frequency", choices=["daily", "monthly", "yearly"], selected="monthly"),
            ui.input_date_range("date_range", "Backtest Period", 
                                start="2015-01-01", 
                                end="2025-01-01",
                                min="2000-01-01", 
                                max="2025-02-28"),
            ui.input_action_button("run", "Run Simulation"),
        ), 

        ui.nav_panel("Portfolio Performance",
            ui.hr(),
            ui.h1('Portfolio Backtest Performance'),
            ui.hr(),
            ui.h2('Porfolio Value Graphs'),
            ui.input_checkbox_group(
                "portfolio_selection", 
                "Select Portfolios to Display:",
                choices=["GBI", "Equal", "MVO"],
                selected=["GBI", "Equal", "MVO"]
            ),
            output_widget("combined_plot"),
            ui.hr(),
            ui.h2("Portfolio Weights Plotted Over Time"),
            output_widget("weight_over_time_plot"),
            ui.hr(),
            ui.h2("Probability of Goal Achievement"),
            output_widget("probability_plot"),
            ui.hr(),
            ui.h2('Portfolio Metrics Table'),
            ui.output_table("quantstats")
        ),

        ui.nav_panel("Portfolio Weights",
            ui.hr(),
            ui.h1('Portfolio Weights'),
            ui.hr(),
            ui.h5("Bull Regime Weights"),
            ui.output_table("weights_bull_table"),
            ui.hr(),
            ui.h5("Bear Regime Weights"),
            ui.output_table("weights_bear_table"),
            ui.hr(),
            ui.h5("Single Regime Weights"),
            ui.output_table("weights_single_table"),
            
            ui.hr(),
            ui.h4("Portfolio Weights by Regime"),
            output_widget("bull_plot"),
            output_widget("bear_plot"),
            output_widget("single_plot"),
        ),

        ui.nav_panel("Data Visualizations",
            ui.hr(),
            ui.h1('Data Visualization'),
            ui.input_select("viz_category", "Select Category", 
                choices={
                    "VBINX_Returns": "VBINX Market Regime",
                    "DJUSRET Index": "DJUSRET Index Market Regime",
                    "BCOMTR Index": "BCOMTR Index Market Regime",
                    "GOLDLNPM Index": "Gold Index Market Regime",
                    "IBOXHY Index": "IBOXHY Index Market Regime",
                    "SPTR Index": "SPTR Index Market Regime",
                    "VSMAX US Equity": "VSMAX Index Market Regime",
                    "LUTLTRUU Index": "LUTLTRUU Index Market Regime",
                    "Bull Regime Heatmap": "Bull Regime Heatmap",
                    "Bear Regime Heatmap": "Bear Regime Heatmap",
                    "Single Regime Heatmap": "Single Regime Heatmap",
                    "Regime Transition Matrix": "Regime Transition Probability Matrix"
                },
                selected="VBINX_Returns"
            ),
            output_widget("regime_output"),
            ui.output_image("image")

        ),

        id="main_tabs"
    )
)

def server(input, output, session):
    result = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.run)
    def _():
        start = pd.to_datetime(input.date_range()[0])
        end = pd.to_datetime(input.date_range()[1])

        # Check if date range is at least 1 year apart
        if (end - start).days < 365:
            ui.notification_show(
                "Please select a date range of at least one year.",
                type="error",
                duration=5000
            )
            return  # Exit without running the simulation
        
        result.set(run_script(
            initial_wealth=input.initial_wealth(),
            goal=input.goal(),
            cashflow=input.cashflow(),
            frequency=input.frequency(),
            start_date=start,
            end_date=end
        ))
    
    @output
    @render.table
    def weights_bull_table():
        if result.get() is None:
            return pd.DataFrame()
        return result.get()["weights_bull"]

    @output
    @render.table
    def weights_bear_table():
        if result.get() is None:
            return pd.DataFrame()
        return result.get()["weights_bear"]

    @output
    @render.table
    def weights_single_table():
        if result.get() is None:
            return pd.DataFrame()
        return result.get()["weights_single"]

    # @output
    # @render_widget
    # def gbi_plot():
    #     if result.get() is None:
    #         return
    #     gbi = result.get()["gbi"]
    #     fig = go.Figure(data=go.Scatter(x=gbi['Date'], y=gbi['Portfolio Value'], mode='lines+markers'))
    #     fig.update_layout(title='Regime Switching GBI Portfolio', xaxis_title='Date', yaxis_title='Portfolio Value')
        
    #     return fig



    # @output
    # @render_widget
    # def equal_plot():
    #     if result.get() is None:
    #         return
    #     equal = result.get()["equal"]
    #     fig = go.Figure(data=go.Scatter(x=equal['Date'], y=equal['Portfolio Value'], mode='lines+markers'))
    #     fig.update_layout(title='Equally Weighted Portfolio', xaxis_title='Date', yaxis_title='Portfolio Value')
        
    #     return fig

    # @output
    # @render_widget
    # def mvo_plot():
    #     if result.get() is None:
    #         return
    #     mvo = result.get()["mvo"]
    #     fig = go.Figure(data=go.Scatter(x=mvo['Date'], y=mvo['Portfolio Value'], mode='lines+markers'))
    #     fig.update_layout(title='Mean-Variance Optimized Portfolio', xaxis_title='Date', yaxis_title='Portfolio Value')
        
    #     return fig

    @output
    @render_widget
    def combined_plot():
        if result.get() is None:
            return

        selected = input.portfolio_selection()
        fig = go.Figure()

        fig.add_hline(y=input.goal(), line_dash="dot", line_color="black", annotation_text=f"Goal: ${input.goal():,.0f}")


        # GBI with weights in hover
        if "GBI" in selected:
            gbi = result.get()["gbi"]

            fig.add_trace(go.Scatter(
                x=gbi["Date"],
                y=gbi["Portfolio Value"],
                mode="lines",
                name="GBI",
                text = gbi["Weights"].apply(
                    lambda d: "<br>".join([
                        f"{k}: {round(v * 100, 2)}%" if isinstance(v, (int, float)) and np.isfinite(v) else f"{k}: N/A"
                        for k, v in d.items()
                    ]) if isinstance(d, dict) else "Weights: N/A"
                ),

                hovertemplate="<b>Date</b>: %{x}<br>" +
                            "<b>Portfolio Value</b>: $%{y:,.2f}<br>" +
                            "<b>Fund</b>: %{customdata[0]}<br>" +
                            "<b>Weights</b>:<br>%{text}",
                customdata=gbi[["Fund"]]
            ))

        # Equal-weighted plot
        if "Equal" in selected:
            equal = result.get()["equal"]
            fig.add_trace(go.Scatter(
                x=equal["Date"],
                y=equal["Portfolio Value"],
                mode="lines",
                name="Equal"
            ))

        # MVO plot
        if "MVO" in selected:
            mvo = result.get()["mvo"]
            fig.add_trace(go.Scatter(
                x=mvo["Date"],
                y=mvo["Portfolio Value"],
                mode="lines",
                name="MVO"
            ))

        fig.update_layout(
            title="Portfolio Comparison with GBI Weights",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            template="plotly_white"
        )

        return fig



    @output
    @render.table
    def metrics_table():
        if result.get() is None:
            return pd.DataFrame()

        gbi = result.get()["gbi_metrics"]
        equal = result.get()["equal_metrics"]
        mvo = result.get()["mvo_metrics"]

        return pd.DataFrame({
            "Metric": ["Final Wealth", "Mean Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
            "GBI": [f"${gbi[0]:,.2f}", f"{gbi[1]:.2%}", f"{gbi[2]:.2%}", f"{gbi[3]:.2f}", f"{gbi[4]:.2%}"],
            "Equal": [f"${equal[0]:,.2f}", f"{equal[1]:.2%}", f"{equal[2]:.2%}", f"{equal[3]:.2f}", f"{equal[4]:.2%}"],
            "MVO": [f"${mvo[0]:,.2f}", f"{mvo[1]:.2%}", f"{mvo[2]:.2%}", f"{mvo[3]:.2f}", f"{mvo[4]:.2%}"]
        })
    
    @output
    @render.table
    def gbi_table():
        if result.get() is None:
            return pd.DataFrame()
        gbi_data = result.get()["gbi_portfolio"]
        return gbi_data

    @output
    @render_widget
    def bull_plot():
        if result.get() is None:
            return
        bull = result.get()["bull_results"]
        fig = go.Figure(data=go.Scatter(x=bull['Date'], y=bull['Portfolio Value'], mode='lines+markers'))
        fig.update_layout(title='Bull Regime Portfolios', xaxis_title='Date', yaxis_title='Portfolio Value')
        
        return fig

    @output
    @render_widget
    def bear_plot():
        if result.get() is None:
            return
        bear = result.get()["bear_results"]
        fig = go.Figure(data=go.Scatter(x=bear['Date'], y=bear['Portfolio Value'], mode='lines+markers'))
        fig.update_layout(title='Bear Regime Portfolios', xaxis_title='Date', yaxis_title='Portfolio Value')
        
        return fig

    @output
    @render_widget
    def single_plot():
        if result.get() is None:
            return
        single = result.get()["single_results"]
        fig = go.Figure(data=go.Scatter(x=single['Date'], y=single['Portfolio Value'], mode='lines+markers'))
        fig.update_layout(title='Regime Switching GBI Portfolio', xaxis_title='Date', yaxis_title='Portfolio Value')
        
        return fig

    @output
    @render_widget
    def regime_output():
        if result.get() is None:
            return

        data = result.get()["data"]
        selected_col = input.viz_category()

        if selected_col not in data.columns:
            return

        return make_regime_plot(data, selected_col)

    def make_regime_plot(data, y_col, regime_col="Regime", title_prefix="Market Regime Transitions"):
        colors = np.where(data[regime_col] == 0, "blue", "red")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[y_col],
            mode="lines",
            name=f"{y_col} Returns",
            line=dict(color="gray", width=1),
            opacity=0.6
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[y_col],
            mode="markers",
            marker=dict(color=colors),
            name="Market Regime",
            opacity=0.8
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

        fig.update_layout(
            title=f"{title_prefix} ({y_col})",
            xaxis_title="Date",
            yaxis_title="Log Returns",
            template="plotly_white"
        )

        return fig
    
    @output
    @render_widget
    def probability_plot():
        if result.get() is None:
            return

        gbi = result.get()["gbi"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gbi["Date"],
            y=gbi["Probability"],
            mode="lines",
            name="Probability of Reaching Goal"
        ))

        fig.add_hline(y=0.8, line_dash="dot", line_color="gray", annotation_text="Target: 80%")

        fig.update_layout(
            title="Probability of Goal Achievement Over Time",
            xaxis_title="Date",
            yaxis_title="Probability",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            template="plotly_white"
        )

        return fig

    @output
    @render_widget
    def weight_over_time_plot():
        if result.get() is None:
            return

        gbi = result.get()["gbi"]

        # Create weights DataFrame
        weights_df = gbi["Weights"].apply(pd.Series)
        weights_df["Date"] = gbi["Date"]
        weights_df["Fund"] = gbi["Fund"]

        # Melt to long format
        melted_weights = weights_df.melt(
            id_vars=["Date", "Fund"], var_name="Asset", value_name="Weight"
        )
        melted_weights = melted_weights.dropna(subset=["Weight"])

        # Set up plotly figure
        fig = go.Figure()

        for asset in melted_weights["Asset"].unique():
            asset_data = melted_weights[melted_weights["Asset"] == asset]

            fig.add_trace(go.Scatter(
                x=asset_data["Date"],
                y=asset_data["Weight"],
                mode="lines",
                name=asset,
                customdata=asset_data[["Fund"]],
                hovertemplate=(
                    "<b>Fund</b>: %{customdata[0]}<br>" +  # shown only once in unified mode
                    "<b>%{fullData.name}</b><br>" +         # asset name
                    "Weight: %{y:.2%}<br>" +
                    "<extra></extra>"
                )
            ))

        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        fig.update_layout(
            title="Portfolio Weights Over Time (GBI Model)",
            xaxis_title="Date",
            yaxis_title="Weight",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified"  # this makes it appear once at top
        )

        return fig

    @output
    @render.table
    def quantstats():
        if result.get() is None:
            return ui.markdown("**Run a simulation to generate QuantStats reports.**")

        # Extract portfolios
        gbi_df = result.get()["gbi"]
        equal_df = result.get()["equal"]
        mvo_df = result.get()["mvo"]

        # Get 3M T-Bill from FRED (symbol: DTB3)
        rfr = pdr.DataReader("DTB3", "fred", start=input.date_range()[0], end=input.date_range()[1]).dropna()

        # Convert from percent to decimal
        rfr = rfr["DTB3"] / 100

        # Daily risk-free rate (annual to daily)
        avg_annual_rfr = rfr.mean()
        rf_daily = (1 + avg_annual_rfr) ** (1 / 252) - 1


        # Helper: compute daily returns with clean index
        def get_returns(df):
            returns = df.set_index("Date")["Portfolio Value"].pct_change().dropna()
            returns.index = pd.to_datetime(returns.index)
            returns = returns.asfreq("D")
            return returns

        gbi_returns = get_returns(gbi_df)
        equal_returns = get_returns(equal_df)
        mvo_returns = get_returns(mvo_df)
        
        def avg_drawdown(returns):
            dd = qs_stats.drawdown_details(returns)
            return np.mean(dd["max drawdown"].values) if not dd.empty else 0.0
        
        # Helper: compute metrics
        def compute_qs_metrics(returns, portfolio_df):
            final_value = portfolio_df["Portfolio Value"].iloc[-1]
            return {
                "Final Portfolio Value": final_value,
                "CAGR": qs_stats.cagr(returns),
                "Sharpe Ratio": qs_stats.sharpe(returns, rf=rf_daily),
                "Sortino Ratio": qs_stats.sortino(returns),
                "Volatility": qs_stats.volatility(returns),
                "Max Drawdown": qs_stats.max_drawdown(returns),
                "Best Day Return": qs_stats.best(returns),
                "Worst Day Return": qs_stats.worst(returns),
                "Average Daily Return": qs_stats.avg_return(returns),
                "Average Drawdown": avg_drawdown(returns),
                "Win Rate": qs_stats.win_rate(returns),
                "Expected Annual Return": qs_stats.expected_return(returns, aggregate="yearly"),
                "Skewness": qs_stats.skew(returns),
                "Kurtosis": qs_stats.kurtosis(returns),
                "VaR (95%)": qs_stats.var(returns, sigma=1.65),
                "CVaR (95%)": qs_stats.cvar(returns, sigma=1.65),
            }

        # Compute metrics
        gbi_metrics = compute_qs_metrics(gbi_returns, gbi_df)
        equal_metrics = compute_qs_metrics(equal_returns, equal_df)
        mvo_metrics = compute_qs_metrics(mvo_returns, mvo_df)

        # Convert to DataFrames for display
        metrics_df = pd.DataFrame({
            "GBI": gbi_metrics,
            "Equal Weight": equal_metrics,
            "MVO": mvo_metrics
        }).round(4)

        metrics_df.index.name = "Metric"
        # Format config
        percent_metrics = {
            "CAGR", "Volatility", "Max Drawdown", "Best Day Return", "Worst Day Return",
            "Average Daily Return", "Average Drawdown", "Win Rate",
            "Expected Annual Return", "VaR (95%)", "CVaR (95%)"
        }

        dollar_metrics = {"Final Portfolio Value"}


        # Format the DataFrame by row
        df_formatted = metrics_df.copy()
        for metric in metrics_df.index:
            if metric in dollar_metrics:
                df_formatted.loc[metric] = metrics_df.loc[metric].apply(lambda x: f"${x:,.2f}")
            elif metric in percent_metrics:
                df_formatted.loc[metric] = metrics_df.loc[metric].apply(lambda x: f"{x:.2%}")
            else:
                df_formatted.loc[metric] = metrics_df.loc[metric].apply(lambda x: f"{x:.2f}")

        # Reset index so "Metric" is shown in the table
        df_formatted = df_formatted.reset_index()

        return df_formatted
        
        


    
    @output
    @render.image
    def image():
        mapping = {
            "Bull Regime Heatmap": "bull.png",
            "Bear Regime Heatmap": "bear.png",
            "Single Regime Heatmap": "single.png",
            "Regime Transition Matrix": "transition_matrix.png"
        }
        
        if input.viz_category() in list(mapping.keys()):
            selected = input.viz_category()
            filename = mapping.get(selected)

            if filename:
                return {"src": str(Path("www") / filename), "width": "75%"}  
                
            else:
                return None

        else:
            return None
        
app = App(app_ui, server)
