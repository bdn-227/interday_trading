# initial script to connect to ib and get a feel for the api

# imports
import numpy as np
from ib_insync import *
from inertia_trading import LiveIBKREngine, EmaCrossoverStrategy, MarketData


# create instance of live trading class
risk = 0.02
contract = {"contract_type": "Forex", "symbol": "EURUSD"}
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


# 4. fetch data and calculate indicator
max_duration = np.max([int(e.split(".")[-1]) for e in strategy.get_indicators()])
price_data = MarketData(data_in=contract, market="XFRA", durationStr="110 D")
strategy.calc_indicators(price_data)

# 5. calculate signals using the strategy
target_state, price_sl, price_limit = strategy.on_bar(price_data.df.iloc[-1], current_position=0)

# 6. calculate number of units
risk_amt = total_balance * risk
dist_to_sl = abs(price_limit - price_sl)
new_units = risk_amt / dist_to_sl if dist_to_sl > 0 else 0


# 7. determine which contract to use
cont = Forex("EURUSD")
trader.ib.qualifyContracts(cont)
trader.ib.reqMarketDataType(3)
spot_ticker = trader.ib.reqMktData(cont, '', False, False)
spot_price = spot_ticker.marketPrice()
ko_contract = Contract(symbol='EUR', secType='WAR')
details = trader.ib.reqContractDetails(ko_contract)

# 4. get live data and calculate leverage
cert_list = []
for d in details:
    ticker = trader.ib.reqMktData(d.contract, '', False, False)
    cert_list.append({
        'contract': d.contract, 
        'ticker': ticker
    })

results = []
for item in cert_list:
    c = item['contract']
    t = item['ticker']
    
    # Grab the best available price (Ask is preferred if you are buying)
    cert_price = t.ask if t.ask == t.ask else t.marketPrice()
    
    # Extract the multiplier (IBKR returns this as a string, e.g., '100')
    try:
        multiplier = float(c.multiplier)
    except:
        multiplier = 1.0
        
    # Calculate Leverage
    if cert_price > 0 and spot_price > 0:
        leverage = spot_price / (cert_price * multiplier)
    else:
        leverage = 0
        
    results.append({
        'Local_Symbol': c.localSymbol,
        'ConID': c.conId,
        'Strike': c.strike,
        'Type': c.right,
        'Price': cert_price,
        'Leverage': round(leverage, 2)
    })


# 6. execute the signals: exits first, new buys etc.


# 7. log results (trade log if trades happened/were scheduled; log trades what were submitted; target state; portfolio, cash balance, positions, equity curve)
    # status.json --> override daily, current positions, cash, worth, active 
    # equity.csv --> Date, NetLiquidity
    # trades.csv --> Date, Symbol, Action, Price, PnL


# 8. stop the interactive broker instance
trader.terminate_IB()

