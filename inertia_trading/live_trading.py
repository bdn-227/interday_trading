from ib_insync import *
import pandas as pd

class LiveIBKREngine:
    def __init__(self, ib_instance, market_data, strategy, symbol='EUR', multiplier=1):
        self.ib = ib_instance
        self.market_data = market_data
        self.strategy = strategy
        self.multiplier = multiplier
        self.contract = Future('M6E', 'GLOBEX', 'USD')
        self.ib.qualifyContracts(self.contract)
        self.bars = self.ib.reqRealTimeBars(self.contract, 5, 'MIDPOINT', False)
        self.bars.updateEvent += self.on_bar_update


    def on_bar_update(self, bars, has_new_bar):
        if not has_new_bar:
            return
        latest_df = util.df(bars)
        processed_data = self.market_data.add_indicators(latest_df)
        last_row = processed_data.iloc[-1]
        pos_size = self.get_current_position()
        action, sl_price = self.strategy.on_bar(last_row, pos_size)
        if action != "HOLD":
            self.execute_trade(action)


    def get_current_position(self):
        positions = self.ib.positions()
        for p in positions:
            if p.contract.conId == self.contract.conId:
                return p.position
        return 0


    def execute_trade(self, action):
        pos = self.get_current_position()
        if action == "BUY" and pos <= 0:
            quantity = self.multiplier if pos == 0 else self.multiplier * 2
            order = MarketOrder('BUY', quantity)
            self.ib.placeOrder(self.contract, order)
            
        elif action == "SELL" and pos >= 0:
            quantity = self.multiplier if pos == 0 else self.multiplier * 2
            order = MarketOrder('SELL', quantity)
            self.ib.placeOrder(self.contract, order)
            
        elif action == "EXIT" and pos != 0:
            side = 'SELL' if pos > 0 else 'BUY'
            order = MarketOrder(side, abs(pos))
            self.ib.placeOrder(self.contract, order)