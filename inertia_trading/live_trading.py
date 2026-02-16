from ib_insync import *
import pandas as pd
import random

class LiveIBKREngine:
    def __init__(self, strategy, contract):
        self.strategy = strategy
        self.contract, self.whatToShow = self.get_contract(**contract)


    def get_contract(self, contract_type, symbol, exchange, currency):
        if contract_type == "Forex":
            contract = Forex(pair=symbol)
            whatToShow = "MIDPOINT"
        elif contract_type == "Index":
            contract = Index(symbol, exchange, currency)
            whatToShow = "TRADES"
        elif contract_type == "Stock":
            contract = Stock(symbol, exchange, currency)
            whatToShow = "TRADES"
        return contract, whatToShow
    
    
    def start_IB(self, ports=[4002, 7497], host='127.0.0.1'):
        util.startLoop()
        self.ib = IB()
        my_client_id = random.randint(1, 9999)
        for port in ports:
            try:
                self.ib.connect(host, port, clientId=my_client_id)
                print(f"Sucess with port: {port}")
            except:
                print(f"FAILED TO CONNECT WITH PORT: {port}")




    def terminate_IB(self):
        self.ib.disconnect()



    def get_markget_data(self):
        pass



    def get_cash(self):
        account_summary = self.ib.accountSummary()
        cash_balance = 0.0
        for item in account_summary:
            if item.tag == 'TotalCashValue' and item.currency == 'EUR':
                cash_balance = float(item.value)
                break
        return cash_balance



    def get_positions(self):
        positions = self.ib.positions()
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
        return df_positions



    def get_portfolio(self):
        portfolio_items = self.ib.portfolio()
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
        return df_portfolio



    def get_open_orders(self):
        open_orders = self.ib.openOrders()
        for order in open_orders:
            print(f"Open Order: {order.action} {order.totalQuantity} @ {order.lmtPrice}")
        return open_orders



    def cancel_order(self, trade):
        self.ib.cancelOrder(trade.order)



    def market_order(self, action, totalquantity, market_closed):
        exit_order = MarketOrder(action, totalquantity)
        if market_closed:
            exit_order.tif = 'OPG'
        trade = self.ib.placeOrder(self.contract, exit_order)
        return trade



    def limit_order(self, action, totalquantity, lmtprice):
        entry_order = LimitOrder(action=action, totalQuantity=totalquantity, lmtPrice=lmtprice)
        entry_order.tif = "DAY"
        trade = self.ib.placeOrder(self.contract, entry_order)
        return trade


    def limit_order_stop(self, totalquantity, lmtprice, slprice):
        parent = LimitOrder("BUY", totalquantity, lmtprice)
        parent.tif = "DAY"
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False
        stop_loss = StopOrder('SELL', totalquantity, slprice)
        stop_loss.parentId = parent.orderId
        stop_loss.transmit = True
        stop_loss.tif = "GTC"
        buy = self.ib.placeOrder(self.contract, parent)
        sell = self.ib.placeOrder(self.contract, stop_loss)
        return buy, sell

