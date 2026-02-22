
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
    def check_constraints(self):
        """
        This method returns the constrains that apply to the respective strategy. For instance, in a trend following strategy,
        the short term moving average should always be larger than the long-term moving average. This method is designed to test these constraints.
        However, it does not test these during inference, as the user is expected to know the assumed data inputs of the model.
        It is rather designed to ensure parameter-stability and overfitting.
        """
        pass

    @abstractmethod
    def get_arguments(self):
        """
        This method returns a dict with the keys as arguments for the __init__ as well as the corresponding types
        """
        pass


    def calc_indicators(self, market_data):
        """
        This method accepts a instance of the MarketData class and adds the indictors to the dataframe. 
        
        :param market_data: a MarketData class instance. missing columns will be calculated
        """

        # add the indicators
        required_indicators = self.get_indicators()
        for indicator in required_indicators:
            if indicator not in market_data.df.columns:
                if indicator.split(".")[0] == "ema":
                    market_data.add_ema(length=int(indicator.split(".")[2]), column=indicator.split(".")[1])
                elif indicator.split(".")[0] == "atr":
                     market_data.add_atr(int(indicator.split(".")[-1]))
                elif indicator.split(".")[0] == "rsi":
                    market_data.add_rsi(length=int(indicator.split(".")[2]), column=indicator.split(".")[1])


