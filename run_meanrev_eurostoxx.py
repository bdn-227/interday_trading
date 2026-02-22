
# libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
from itertools import combinations
from datetime import time, date, timedelta
import statsmodels.api as sm
from scipy.stats import spearmanr, false_discovery_control
import pandas_market_calendars as mcal
import plotly.graph_objects as go
from math import floor

# load my own package
from inertia_trading import MarketData, BacktestEngine, MeanReversion


# load the data and convert to class
csv = pd.read_csv("data/eurostoxx.csv")
price_data = MarketData(csv, market="XFRA")
price_data.df

# strategy parameters
length_rsi=14
rsi_entry=30
rsi_exit = 70
length_atr=14
atr_sl=3
atr_limit=0.5

# modelling parameters
n_simulations = 100

# calculate indicators
price_data.add_atr(length=length_atr)
price_data.add_rsi(column="close", length=length_atr)


# backtest
strategy = MeanReversion(length_rsi=length_rsi, rsi_entry=rsi_entry, rsi_exit=rsi_exit, length_atr=length_atr, atr_sl=atr_sl, atr_limit=atr_limit)
backtest = BacktestEngine(price_data, strategy)
equity_curve = backtest.run_etf(risk=0.01)
backtest.plot_equity_df(normalize=True, log_axis=False)

# model sensitivity test
# simulations = backtest.test_overfit(backtest_type = "etf", simulations=100, augmentation_size=0.5, normalized=True)
# backtest.plot_simulations(simulations, title="Parameter Sensitivity Test")

# # perform monte carlo
# simulations = backtest.monte_carlo(n_simulations=n_simulations*10, drawdown=0.5)
# backtest.plot_simulations(simulations, title="Monte Carlo (Risk of Ruin)")

# # perform the random monkey
# simulations = backtest.monkey_carlo(n_simulations=n_simulations)
# backtest.plot_simulations(simulations, title="Monte Carlo (Statistical significance of the trading parameters)")



