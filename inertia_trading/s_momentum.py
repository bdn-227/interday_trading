from .strategy import Strategy

class EmaCrossoverStrategy(Strategy):

    def __init__(self, ema_short=50, ema_long=100, length_atr=14, atr_sl=3, atr_limit=0.5, crossover=True):
        
        # save parameters
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.length_atr = length_atr
        self.atr_sl = atr_sl
        self.atr_limit = atr_limit
        self.crossover = crossover
        self.argument_d = self.get_arguments()

        # actual execution
        self.sl_price = None
        self.prev_ema_s = None
        self.prev_ema_l = None
        self.limit_price = None

        # validate the strategy --> a bit confusion, but check_constraints returns true if everything is fine; hence we raise an error 
        # if NOT check_constraints
        if not self.check_constraints(ema_short=self.ema_short, 
                                      ema_long=self.ema_long, 
                                      length_atr=self.length_atr, 
                                      atr_sl=self.atr_sl,
                                      atr_limit=self.atr_limit):
            raise ValueError("Some parameters do not meet the strategy requirements")



    def get_arguments(self):
        arguments_d = {"ema_short": [self.ema_short, int],
                       "ema_long": [self.ema_long, int],
                       "length_atr": [self.length_atr, int],
                       "atr_sl": [self.atr_sl, float],
                       "atr_limit": [self.atr_limit, float],
                       "crossover": [self.crossover, bool]}
        return arguments_d


    def get_indicators(self):
        return [f"ema.close.{self.ema_short}", f"ema.close.{self.ema_long}", f'atr.{self.length_atr}']
    

    def check_constraints(self, ema_short, ema_long, length_atr, atr_sl, atr_limit):
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
        atr = atr_sl >= 0 and atr_limit >= 0
        
        # now return the test results
        return ema_correct and ema_length and atr_length and atr


    def on_bar(self, row, current_position):

        # extract the data
        ema_s = row[f"ema.close.{self.ema_short}"]
        ema_l = row[f"ema.close.{self.ema_long}"]
        close = row['close']
        atr = row[f'atr.{self.length_atr}']

        # detect Crossover
        if self.crossover:
            was_crossover = False
            if self.prev_ema_s is not None:
                if self.prev_ema_s < self.prev_ema_l and ema_s > ema_l:
                    was_crossover = True
                elif self.prev_ema_s > self.prev_ema_l and ema_s < ema_l:
                    was_crossover = True
        else:
            was_crossover = True

        # default parameters
        target_state = current_position

        # go long
        if ema_s > ema_l and was_crossover:
            target_state = 1
            self.sl_price = close - (self.atr_sl * atr)
            self.limit_price = close - (atr * self.atr_limit)


        # go short
        elif ema_s < ema_l and was_crossover:
            target_state = -1
            self.sl_price = close + (self.atr_sl * atr)
            self.limit_price = close + (atr * self.atr_limit)
        

        # exit long
        elif current_position == 1:
            if close < self.sl_price:
                target_state = 0
                self.limit_price = close + (atr * self.atr_limit)
            
            else:
                new_sl = close - (self.atr_sl * atr)
                if new_sl > self.sl_price:
                    self.sl_price = new_sl


        # exit short
        elif current_position == -1:
            if close > self.sl_price:
                target_state = 0
                self.limit_price = close - (atr * self.atr_limit)

            else:
                new_sl = close + (self.atr_sl * atr)
                if new_sl < self.sl_price:
                    self.sl_price = new_sl
        

        # update memory for next iteration
        self.prev_ema_s = ema_s
        self.prev_ema_l = ema_l
                
        return target_state, self.sl_price, self.limit_price