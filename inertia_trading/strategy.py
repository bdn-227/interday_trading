
# import libraries
from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, row, current_position):
        """
        Called on every single candle.
        Returns: 
           target_state (str): -1, 0, 1
           stop_loss (float): Price for the SL 
        """
        pass

    @abstractmethod
    def get_indicators(self):
        """
        Called during initilization of backtest or live trading --> ensures all required indicators are present
        Returns: 
           indicator_ls (list): list of indicators
        """
        pass


class EmaCrossoverStrategy(Strategy):

    def __init__(self, ema_short=50, ema_long=100, length_atr=14, atr_multiplier=3):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.length_atr = length_atr
        self.atr_mult = atr_multiplier
        self.sl_price = None
        self.prev_ema_s = None
        self.prev_ema_l = None
        self.limit_price = None
        
    

    def get_indicators(self):
        return [f"ema.close.{self.ema_short}", f"ema.close.{self.ema_long}", f'atr.{self.length_atr}']


    def on_bar(self, row, current_position):

        # extract the data
        ema_s = row[f"ema.close.{self.ema_short}"]
        ema_l = row[f"ema.close.{self.ema_long}"]
        close = row['close']
        atr = row[f'atr.{self.length_atr}']

        # detect Crossover
        was_crossover = False
        if self.prev_ema_s is not None:
            if self.prev_ema_s < self.prev_ema_l and ema_s > ema_l:
                was_crossover = True
            elif self.prev_ema_s > self.prev_ema_l and ema_s < ema_l:
                was_crossover = True

        # default parameters
        target_state = current_position

        # go long
        if ema_s > ema_l and was_crossover:
            target_state = 1
            self.sl_price = close - (self.atr_mult * atr)
            self.limit_price = close - (atr * 0.5)


        # go short
        elif ema_s < ema_l and was_crossover:
            target_state = -1
            self.sl_price = close + (self.atr_mult * atr)
            self.limit_price = close + (atr * 0.5)
        

        # exit long
        elif current_position == 1:
            if close < self.sl_price:
                target_state = 0
                self.limit_price = close + (atr * 0.5)
            
            else:
                new_sl = close - (self.atr_mult * atr)
                if new_sl > self.sl_price:
                    self.sl_price = new_sl


        # exit short
        elif current_position == -1:
            if close > self.sl_price:
                target_state = 0
                self.limit_price = close - (atr * 0.5)

            else:
                new_sl = close + (self.atr_mult * atr)
                if new_sl < self.sl_price:
                    self.sl_price = new_sl
        

        # update memory for next iteration
        self.prev_ema_s = ema_s
        self.prev_ema_l = ema_l
                
        return target_state, self.sl_price, self.limit_price