import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
import numpy as np
from IPython.display import display


class MarketData:
    def __init__(self, raw_df, market = "XFRA"):
        """
        Expects a DataFrame with 'open', 'high', 'low', 'close' columns.
        """
        self.df = raw_df.copy()
        self.format_df()
        self.check_integrity(market)
        self.add_next(column="open", shift=1)
        self.add_next(column="low", shift=1)
        self.add_next(column="high", shift=1)

    def format_df(self):
            """
            Cleans column names, converts types, and sorts data.
            """

            # standardize column names (lowercase)
            self.df.columns = [col.lower() for col in self.df.columns]

            # 1.1 ensure that all important columns are present
            cols_needed = ["close", "high", "low", "open", "volume", "symbol", "datetime"]
            cols_present = [col for col in cols_needed if col in self.df.columns]
            missing_cols = np.setxor1d(cols_needed, cols_present)
            if len(missing_cols):
                raise ValueError(f"Some columns are missing: {missing_cols}")
            
            # 2. convert to datetime
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
            
            # 3. add calendar features
            calendar = self.df["datetime"].dt.isocalendar()
            self.df[["year", "week", "day"]] = calendar[["year", "week", "day"]]
            
            # 4. cast types (ensuring no string-math errors)
            cols_to_fix = ["close", "high", "low", "open", "volume"]
            self.df[cols_to_fix] = self.df[cols_to_fix].astype(float)
            
            # 5. sort
            self.df = self.df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
            return self
    

    def check_integrity(self, market):
        """
        Docstring for check_integrity
        
        :param self: Description
        :param market: Str declaring which market the ticker is being traded on
        """

        # get the respective market
        xetra = mcal.get_calendar(market)

        # now check for completeness
        results_ls = []
        for symbol in self.df["symbol"].unique():

            # subset the data
            ticker_data = self.df[self.df["symbol"] == symbol]

            # now check date completeness
            start = ticker_data["datetime"].min().date()
            end = ticker_data["datetime"].max().date()
            sched = xetra.schedule(start_date=start, end_date=end)
            expected_days = sched.index.date
            observed_days = ticker_data["datetime"].dt.date
            missing_days = set(expected_days) - set(observed_days)
            extra_days   = set(observed_days) - set(expected_days)
            duplicates = ticker_data["datetime"][ticker_data["datetime"].duplicated()].unique()

            # return
            results = {
                        "symbol": symbol,
                        "start": start,
                        "end": end,
                        "n_expected": len(expected_days),
                        "n_observed": len(observed_days),
                        "n_missing": len(missing_days),
                        "n_extra": len(extra_days),
                        "n_duplicates": len(duplicates),
                        "missing_days": sorted(missing_days),
                    }
            results_ls.append(results)

        # put the data together
        results = pd.DataFrame(results_ls).sort_values("n_missing")
        self.integrity_report = results
        display(results)
        return self


    def add_ema(self, length=20, column="close"):
        """
        Docstring for add_ema
        
        :param self: Description
        :param column: column to perform the ema calculation on
        :param length: period length
        """

        self.df[f'ema.{column}.{length}'] = ta.ema(self.df['close'], length=length)
        return self


    def add_atr(self, length=14):
        """
        Docstring for add_atr
        
        :param self: Description
        :param length: period length for calculation of the average true range
        """

        self.df[f'atr.{length}'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=length)
        return self


    def add_next(self, column="close", shift=1):
        """
        Docstring for add_next
        
        :param column: str which column to shift
        :param shift: int how many offsets
        """
        self.df[f"future.{column}.{shift}"] = self.df[column].shift(shift)
