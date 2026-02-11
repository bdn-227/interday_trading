# initial script to connect to ib and get a feel for the api

# imports
from ib_insync import util, IB, Order, LimitOrder, MarketOrder, StopOrder, Stock
import random
import pandas as pd

# start IB
util.startLoop()
ib = IB()

# # connect to client
my_client_id = random.randint(1, 9999)
try:
    ib.connect('127.0.0.1', 4002, clientId=my_client_id)
except Exception as e:
    print(f"Error: {e}")


# get the account balance
account_summary = ib.accountSummary()
cash_balance = 0.0
for item in account_summary:
    # TotalCashValue is the raw cash --> we use this
    # BuyingPower is the leverage. i'll stay away from this. no need to go into debt for now..
    if item.tag == 'TotalCashValue' and item.currency == 'EUR':
        cash_balance = float(item.value)
        break


# get the positions
positions = ib.positions()
if positions:
    df_positions = pd.DataFrame([
        {
            "symbol": p.contract.symbol,
            "conId": p.contract.conId,
            "exchange": p.contract.exchange,
            "size": p.position,
            "avg_cost": p.avgCost,
            "market_price": 0.0,
            "market_value": 0.0,
        } 
        for p in positions])


# get the portfolio worth
portfolio_items = ib.portfolio()
if portfolio_items:
    df_portfolio = pd.DataFrame([
        {
            "symbol": p.contract.symbol,
            "size": p.position,
            "market_price": p.marketPrice,
            "market_value": p.marketValue,
            "average_cost": p.averageCost,
            "unrealized_pnl": p.unrealizedPNL,
            "realized_pnl": p.realizedPNL
        }
        for p in portfolio_items])


# get a list of open orders
open_orders = ib.openOrders()
for order in open_orders:
    print(f"Open Order: {order.action} {order.totalQuantity} @ {order.lmtPrice}")



# how to cancel orders --> of course, here need to be some conditionswhich to order
for trade in ib.trades():
    if trade.orderStatus.status in ('Submitted', 'PreSubmitted'):
        ib.cancelOrder(trade.order)



# issue market orders for quick exits
symbol = "VUSA"
exchange = "AEB"
currency = "EUR"
action = "BUY"
totalquantity=50
market_closed = True

contract = Stock(symbol=symbol, exchange=exchange, currency=currency)
exit_order = MarketOrder(action, totalquantity)
if market_closed:
    exit_order.tif = 'OPG'
ib.placeOrder(contract, exit_order)



# limit order for strategic entries and exits
symbol = "VUSA"
exchange = "AEB"
currency = "EUR"
action = "BUY"
totalquantity=50
lmtprice = 50
slprice = 30

contract = Stock(symbol=symbol, exchange=exchange, currency=currency)
entry_order = LimitOrder(action=action, totalQuantity=totalquantity, lmtPrice=lmtprice)
entry_order.tif = "DAY"
trade = ib.placeOrder(contract, entry_order)
print(trade.orderStatus.status)



# limit order with internal stop loss
parent = LimitOrder("BUY", totalquantity, lmtprice)
parent.tif = "DAY"
parent.orderId = ib.client.getReqId()
parent.transmit = False
stop_loss = StopOrder('SELL', totalquantity, slprice)
stop_loss.parentId = parent.orderId
stop_loss.transmit = True
stop_loss.tif = "GTC"
ib.placeOrder(contract, parent)
ib.placeOrder(contract, stop_loss)



# disconnect
ib.disconnect()

