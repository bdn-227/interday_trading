# Inter day trading
This repository contains some bundled classes that should help with the development and testing of different trading strategies.
As of now, the logic handles only future contracts and not individual stocks

Included:
- a class to handle market data with some functions to ensure data integrity and completeness of timeseries
- a strategy class for prototyping of different strategies
- backtesting engine for backtest, monte carlo simulations and plotting

## TODO
- implement a class for live trading (paper and real money)
- some code hygene to ensure there is not look ahead bias etc.
- account for slipage better
- compare the suggested trading volume to the actual trading volume
- get rich