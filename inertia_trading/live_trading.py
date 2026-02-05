from ib_insync import *
import pandas as pd

class LiveIBKREngine:
    def __init__(self, ib_instance, market_data, strategy, symbol='EUR', multiplier=1):
        self.ib = ib_instance
        self.market_data = market_data
        self.strategy = strategy
        self.multiplier = multiplier
        
        # 1. Define the Contract (Micro Euro Futures)
        self.contract = Future('M6E', 'GLOBEX', 'USD')
        self.ib.qualifyContracts(self.contract)
        
        # 2. Setup Data Stream (4-hour bars or 1-day bars)
        self.bars = self.ib.reqRealTimeBars(self.contract, 5, 'MIDPOINT', False)
        
        # 3. Hook the "Brain" to the Event
        self.bars.updateEvent += self.on_bar_update


    def on_bar_update(self, bars, has_new_bar):
        """ This function runs every 5 seconds, but only acts when a bar closes. """
        if not has_new_bar:
            return
            
        print(f"🔔 New bar received for {self.contract.localSymbol}")
        
        # 1. Update MarketData with the latest bar
        latest_df = util.df(bars)
        processed_data = self.market_data.add_indicators(latest_df)
        last_row = processed_data.iloc[-1]
        
        # 2. Get Current Position from Broker (Safety first!)
        pos_size = self.get_current_position()
        
        # 3. Ask Strategy for decision
        action, sl_price = self.strategy.on_bar(last_row, pos_size)
        
        # 4. Execute (Order Management)
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
            # If short (-1), we need 2 units to go long (+1)
            quantity = self.multiplier if pos == 0 else self.multiplier * 2
            order = MarketOrder('BUY', quantity)
            self.ib.placeOrder(self.contract, order)
            print(f"🚀 Going LONG {quantity} contracts")
            
        elif action == "SELL" and pos >= 0:
            quantity = self.multiplier if pos == 0 else self.multiplier * 2
            order = MarketOrder('SELL', quantity)
            self.ib.placeOrder(self.contract, order)
            print(f"🔻 Going SHORT {quantity} contracts")
            
        elif action == "EXIT" and pos != 0:
            side = 'SELL' if pos > 0 else 'BUY'
            order = MarketOrder(side, abs(pos))
            self.ib.placeOrder(self.contract, order)
            print(f"⏹ EXITING all positions")