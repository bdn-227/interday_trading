
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

    @abstractmethod
    def get_constraints(self):
        """
        This method returns the constrains that apply to the respective strategy. For instance, in a trend following strategy,
        the short term moving average should always be larger than the long-term moving average. This method is designed to test these constraints.
        However, it does not test these during inference, as the user is expected to know the assumed data inputs of the model.
        It is rather designed to ensure parameter-stability and overfitting.
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
    
    def get_constraints(self, ema_short, ema_long, length_atr):
        """
        This method tests, if the given strategy is correctly parameterized. It returns true if all necessary conditions are met.
        If this is not the case, a false is returned
        
        :param ema_short: short moving average period parameter to be tested
        :param ema_long: long moving average period parameter to be tested
        :param length_atr: atr period length to be tested
        """
        
        # now, test all necessary conditions
        ema_correct = ema_short < ema_long
        ema_length = (ema_short > 1) and (ema_long > 1)
        atr_length = length_atr > 1
        
        # now return the test results
        return ema_correct and ema_length and atr_length


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