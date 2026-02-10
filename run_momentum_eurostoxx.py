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
from inertia_trading import MarketData, EmaCrossoverStrategy, BacktestEngine


# load the data and convert to class
csv = pd.read_csv("data/eurostoxx.csv")
price_data = MarketData(csv, market="XFRA")

# parameters
ema_short = 5
ema_long = 10
atr_length = 14
atr_multiplier = 3
n_simulations = 100

# calculate indicators
price_data.add_ema(ema_short)
price_data.add_ema(ema_long)
price_data.add_atr(length=atr_length)


# backtest
strategy = EmaCrossoverStrategy(ema_short=ema_short, ema_long=ema_long, length_atr=atr_length, atr_multiplier=atr_multiplier)
backtest = BacktestEngine(price_data, strategy)
equity_curve = backtest.run_etf(risk=0.01, capital=1000)
print(equity_curve)
backtest.plot_equity_df(normalize=True, log_axis=False)

# test if the strategy overfits
simulations = backtest.test_overfit(backtest_type = "etf", simulations=100, augmentation_size=0.2, normalized=False)
backtest.plot_simulations(simulations)

# perform monte carlo
#simulations = backtest.monte_carlo(n_simulations=n_simulations*10, drawdown=0.5)
#backtest.plot_simulations(simulations)

# perform the random monkey
#simulations = backtest.monkey_carlo(n_simulations=n_simulations)
#backtest.plot_simulations(simulations)