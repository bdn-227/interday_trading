# initial script to connect to ib and get a feel for the api

# imports
import numpy as np
from ib_insync import *
from inertia_trading import LiveIBKREngine, EmaCrossoverStrategy, MarketData


# create instance of live trading class
risk = 0.02
contract = {"contract_type": "Forex", "symbol": "EURUSD", "exchange": "FWB", "currency": "EUR"}
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
# 7.1 get current spot price
trader.ib.reqMarketDataType(3)
eur_usd = Forex('EURUSD')
spot_ticker = trader.ib.reqMktData(eur_usd, '', False, False)

# 7.2 get the certificates
contract = Contract(
    symbol='EUR', 
    secType='IOPT',
    exchange='FWB', 
    currency='EUR',
    right='C')
details = trader.ib.reqContractDetails(contract)

# 7.3 get market data for the certificantes
tickers = []
for d in details:
    tickers.append(trader.ib.reqMktData(d.contract, '', False, False))
trader.ib.sleep(4)
spot_price = spot_ticker.marketPrice()

# 7.4 but the data together
contract_l = []
for d, t in zip(details, tickers):
    prices = [t.ask, t.last, t.close, t.marketPrice()]
    cert_price = next((p for p in prices if p > 0 and p == p), 0)
    multiplier = float(d.contract.multiplier) if d.contract.multiplier else 1.0
    barrier = d.contract.strike
    current_spot = spot_price if spot_price == spot_price else spot_ticker.delayedLast
    leverage = current_spot / (cert_price * multiplier) if cert_price > 0 else 0
    dist_to_ko = abs(current_spot - barrier) / current_spot if current_spot > 0 else 0

    contract_l.append({
        "conId": d.contract.conId,
        "symbol": d.contract.localSymbol,
        "multiplier": multiplier,
        "ko_barrier": barrier,
        "longname": d.longName,
        "last_traded": d.contract.lastTradeDateOrContractMonth,
        "ordertypes": d.orderTypes,
        "price": cert_price,
        "leverage": round(leverage, 2),
        "dist_to_ko_pct": round(dist_to_ko * 100, 2)
    })


# 6. execute the signals: exits first, new buys etc.


# 7. log results (trade log if trades happened/were scheduled; log trades what were submitted; target state; portfolio, cash balance, positions, equity curve)
    # status.json --> override daily, current positions, cash, worth, active 
    # equity.csv --> Date, NetLiquidity
    # trades.csv --> Date, Symbol, Action, Price, PnL


# 8. stop the interactive broker instance
trader.terminate_IB()

