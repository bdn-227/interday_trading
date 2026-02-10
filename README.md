# Inter day trading
This repository contains some bundled classes that should help with the development and testing of different trading strategies.
As of now, the logic handles only future contracts and not individual stocks.
As backtesting using stocks always introduces a survivor-bias, i will stick with futures for now..
All i need is a GmbH.
The general assumption as of now is that the trading descision is based on the close.
Hence, this system is supposed to be executed after the market close to issue orders that should be filled at the market open.
Intraday volatility is not accounted for. Consequently, no automatic stop-losses should be added via the brooker. 

Included:
- a class to handle market data with some functions to ensure data integrity and completeness of timeseries
- a strategy class for prototyping of different strategies
- backtesting engine for backtest, monte carlo simulations and plotting

## TODO
- implement a class for live trading (paper and real money)
- account for slipage better
- compare the suggested trading volume to the actual trading volume
- cap maximum number of draw downs --> no more than 100%
- add a new backtest method for etfs: --> long only and intraday volatility does not kill us
