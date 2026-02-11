from ib_insync import *
import pandas as pd

class LiveIBKREngine:
    def __init__(self, strategy, contract=Stock(symbol="VUSA", exchange="AEB", currency="EUR")):
        self.strategy = strategy
        self.contract = contract
    

    def get_ib_isntance():
        pass


    def get_markget_data():
        pass


    def run():
        
        # 1. connect to IB gateway

        # 2. clean-up: cancel all open orders from yesterday (especially buy orders)

        # 3. get account state (portfolio, positions, net worth)

        # 4. fetch data and calculate indicators

        # 5. calculate signals using the strategy

        # 6. execute the signals: exits first, new buys etc.

        # 7. log results (trade log if trades happened/were scheduled; log trades what were submitted; target state; portfolio, cash balance, positions, equity curve)
            # status.json --> override daily, current positions, cash, worth, active 
            # equity.csv --> Date, NetLiquidity
            # trades.csv --> Date, Symbol, Action, Price, PnL
        
        pass

