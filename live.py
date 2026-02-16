# initial script to connect to ib and get a feel for the api

# imports
from inertia_trading import LiveIBKREngine, EmaCrossoverStrategy


# create instance of live trading class
contract = {"contract_type": "Stock", "symbol": "VUSA", "exchange": "AEB", "currency": "EUR"}
strategy = EmaCrossoverStrategy(ema_short=50, ema_long=100, length_atr=14, atr_sl=3, atr_limit=0.5, crossover=False)
trader = LiveIBKREngine(strategy=strategy, contract=contract)


# 1. connect to IB gateway
trader.start_IB()


# 2. clean-up: cancel all open orders from yesterday (especially buy orders)
open_orders = trader.get_open_orders()


# 3. get account state (portfolio, positions, net worth)
cash = trader.get_cash()
portfolio = trader.get_portfolio()
total_balance = cash + portfolio["market_value"].sum()


# 4. fetch data and calculate indicators
strategy.get_indicators()

# 5. calculate signals using the strategy

# 6. execute the signals: exits first, new buys etc.

# 7. log results (trade log if trades happened/were scheduled; log trades what were submitted; target state; portfolio, cash balance, positions, equity curve)
    # status.json --> override daily, current positions, cash, worth, active 
    # equity.csv --> Date, NetLiquidity
    # trades.csv --> Date, Symbol, Action, Price, PnL


# 8. stop the interactive broker instance
trader.terminate_IB()

