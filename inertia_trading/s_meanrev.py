from .strategy import Strategy

class MeanReversion(Strategy):

    def __init__(self, length_rsi=14, rsi_entry=30, rsi_exit = 70, length_atr=14, atr_sl=3, atr_limit=0.5):
        
        # save parameters
        self.length_rsi = length_rsi
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.length_atr = length_atr
        self.atr_sl = atr_sl
        self.atr_limit = atr_limit
        self.argument_d = self.get_arguments()

        # actual execution
        self.sl_price = None
        self.limit_price = None

        # validate the strategy --> a bit confusion, but check_constraints returns true if everything is fine; hence we raise an error 
        # if NOT check_constraints
        if not self.check_constraints(
                                      length_rsi=self.length_rsi, 
                                      length_atr=self.length_atr,
                                      rsi_entry=rsi_entry,
                                      rsi_exit=rsi_exit,
                                      atr_sl=self.atr_sl,
                                      atr_limit=self.atr_limit):
            raise ValueError("Some parameters do not meet the strategy requirements")



    def get_arguments(self):
        arguments_d = {
                       "length_rsi": [self.length_rsi, int],
                       "rsi_entry": [self.rsi_entry, float],
                       "rsi_exit": [self.rsi_exit, float],
                       "length_atr": [self.length_atr, int],
                       "atr_sl": [self.atr_sl, float],
                       "atr_limit": [self.atr_limit, float],
                       }
        return arguments_d


    def get_indicators(self):
        return [f"rsi.close.{self.length_rsi}", f'atr.{self.length_atr}']
    

    def check_constraints(self, length_rsi, rsi_entry, rsi_exit, length_atr, atr_sl, atr_limit):
        """
        This method tests, if the given strategy is correctly parameterized. It returns true if all necessary conditions are met.
        If this is not the case, a false is returned
        
        :param ema_short: short moving average period parameter to be tested
        :param ema_long: long moving average period parameter to be tested
        :param length_atr: atr period length to be tested
        """

        # check indicators
        rsi_correct = length_rsi > 1
        atr_length = length_atr > 1

        # check indicator cutoffs
        rsi_inbound = (0 < rsi_entry < 100) and (0 < rsi_exit < 100)

        # check limits
        atr = atr_sl >= 0 and atr_limit >= 0
        
        # now return the test results
        return rsi_correct and atr_length and rsi_inbound and atr


    def on_bar(self, row, current_position):

        # extract the data
        rsi = row[f"rsi.close.{self.length_rsi}"]
        close = row['close']
        atr = row[f'atr.{self.length_atr}']

        # defaults
        target_state = current_position


        # calculate entry
        if rsi < self.rsi_entry and current_position == 0:
            target_state = 1
            self.limit_price = close - (atr * self.atr_limit)
            self.sl_price = close - (atr * self.atr_sl)
    

        # calculate logic for holding --> this one is supposed to allow for riding the momentum and adjust stop losses over time
        elif rsi < self.rsi_exit and current_position == 1:
            target_state = 1
            new_sl = close - (self.atr_sl * atr)
            if new_sl > self.sl_price:
                self.sl_price = new_sl


        # calculate exit
        elif rsi > self.rsi_exit and current_position == 1:
            target_state = 0
            self.limit_price = close + (atr * self.atr_limit)
        

        return target_state, self.sl_price, self.limit_price