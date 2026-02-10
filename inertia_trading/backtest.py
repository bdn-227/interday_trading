import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.colors as pc
import random
import math


class BacktestEngine:
    def __init__(self, market_data_obj, strategy_obj):
        self.data = market_data_obj.df
        self.strategy = strategy_obj
        self.required_indicators = strategy_obj.get_indicators()
        self.position = 0
        self.equity_curve = []


    def run_future(self, capital=1_000, risk=0.01):
        """
        Backtesting method for futures. this features short, long and exits. sizing based in risk.
        """
        print(f"Running backtest (Basis Points Mode | Start: {capital})..")
        
        # initialize
        self.capital = capital
        self.risk = risk
        self.position = 0
        self.entry_price = 0.0
        self.entry_date = None
        self.units = 0.0
        self.sl_price = None
        
        # lists for logging
        self.equity_curve = []
        self.realized_curve = []
        self.position_ls = []
        self.trade_log = []
        self.datetime_ls = []
        self.close_ls = []
        self.units_ls = []
        
        # filtering of data
        required_cols = self.required_indicators + ["future.open.1", "future.low.1", "future.high.1"]
        clean_data = self.data.dropna(subset=required_cols).copy()

        # backtesting loop
        for idx, row in clean_data.iterrows():
            
            # extract the data, including tomorrows prices --> this is interday trading, hence buy descisions are based on today's
            # close and executed in the coming morning. we only trade limit orders and fixed stop losses. therefore,
            # we check of the orders would be filled the next morning
            current_date = row["datetime"]
            today_close = row['close']
            f_open = row["future.open.1"]
            f_low  = row["future.low.1"]
            f_high = row["future.high.1"]

            # strategy is being executed here --> as return we get -1, 0 or 1 --> short, exit, sell
            target_state, strat_sl, strat_limit = self.strategy.on_bar(row, self.position)

            # variable for todays profit and losses
            pnl = 0.0

            # 1.) check intraday stop losses. does intraday volatility kill us?
            stop_hit = False
            if self.position == 1 and self.sl_price:
                if f_low <= self.sl_price:
                    
                    # are our long positions killed?
                    exit_price = self.sl_price
                    pnl = (exit_price - self.entry_price) * self.units
                    self.capital += pnl
                    
                    self.trade_log.append({
                        "entry_date": self.entry_date,
                        'exit_date': current_date, 
                        'type': 'STOP_LOSS (L)', 
                        'entry': self.entry_price, 
                        'exit': exit_price, 
                        'pnl': pnl, 
                        'capital': self.capital,
                        'return': pnl_exit / (self.capital - pnl_exit),
                    })
                    
                    self.position = 0
                    self.units = 0
                    self.sl_price = None
                    stop_hit = True

            elif self.position == -1 and self.sl_price:
                if f_high >= self.sl_price:

                    # are our short positions killed
                    exit_price = self.sl_price
                    pnl = (self.entry_price - exit_price) * self.units
                    self.capital += pnl
                    
                    self.trade_log.append({
                        "entry_date": self.entry_date,
                        'exit_date': current_date, 
                        'type': 'STOP_LOSS (S)', 
                        'entry': self.entry_price, 
                        'exit': exit_price, 
                        'pnl': pnl, 
                        'capital': self.capital,
                        'return': pnl_exit / (self.capital - pnl_exit),
                    })
                    
                    self.position = 0
                    self.units = 0
                    self.sl_price = None
                    stop_hit = True

            
            # new entries
            if not stop_hit and target_state != self.position:
                
                # are the limit prices reachable
                is_reachable = (f_low <= strat_limit <= f_high)
                
                if is_reachable:
                    
                    # exit old position if reversal
                    if self.position != 0:

                        # close current trade if reversal
                        pnl_exit = 0
                        if self.position == 1:
                            pnl_exit = (strat_limit - self.entry_price) * self.units
                        else:
                            pnl_exit = (self.entry_price - strat_limit) * self.units
                        
                        self.capital += pnl_exit
                        self.trade_log.append({
                            "entry_date": self.entry_date,
                            'exit_date': current_date, 
                            'type': 'EXIT', 
                            'entry': self.entry_price, 
                            'exit': strat_limit, 
                            'pnl': pnl_exit, 
                            'capital': self.capital,
                        'return': pnl_exit / (self.capital - pnl_exit),
                        })
                        self.position = 0
                        self.units = 0

                    # enter new position
                    if target_state != 0:

                        # calculate risk-adjusted position size --> risk parameter determines the maximum % of 
                        # portfolio that we might loose of trade goes wrong, i.e. stop loss is hit
                        risk_amt = self.capital * self.risk
                        dist_to_sl = abs(strat_limit - strat_sl)
                        
                        if dist_to_sl > 0:
                            new_units = risk_amt / dist_to_sl
                        else:
                            new_units = 0
                        
                        # check if intra-day volatility kills us
                        instant_loss = False
                        if target_state == 1 and f_low <= strat_sl:
                            instant_loss = True
                        elif target_state == -1 and f_high >= strat_sl:
                            instant_loss = True
                            
                        if instant_loss:

                            # here, the new position is stopped instantly (note: this is a conservative assumptionm but we always trigger it)
                            loss_val = (strat_sl - strat_limit) if target_state == 1 else (strat_limit - strat_sl)
                            pnl_crash = loss_val * new_units
                            self.capital += pnl_crash
                            
                            self.trade_log.append({
                                "entry_date": self.entry_date,
                                'exit_date': current_date, 
                                'type': 'INSTANT_STOP', 
                                'entry': strat_limit, 
                                'exit': strat_sl, 
                                'pnl': pnl_crash, 
                                'capital': self.capital,
                                'return': pnl_exit / (self.capital - pnl_exit),
                            })

                            # reset parameters upon stop loss
                            self.position = 0
                            self.units = 0
                            self.sl_price = None
                            
                        else:
                            # here, we actually entered a trade and did not get killed instantly
                            self.position = target_state
                            self.entry_price = strat_limit
                            self.entry_date = current_date
                            self.sl_price = strat_sl
                            self.units = new_units


            # here we perform mark-to-market tracking of portfolio size --> for smooth equity curve
            floating_pnl = 0
            if self.position == 1:
                floating_pnl = (today_close - self.entry_price) * self.units
            elif self.position == -1:
                floating_pnl = (self.entry_price - today_close) * self.units
            total_equity = self.capital + floating_pnl
            
            self.equity_curve.append(total_equity)
            self.realized_curve.append(self.capital)
            self.datetime_ls.append(current_date)
            self.position_ls.append(self.position)
            self.close_ls.append(today_close)
            self.units_ls.append(self.units)

        # print final capital size
        print(f"Final capital: {self.equity_curve[-1]:.2f}")
        
        # save everything as dataframe for plotting
        self.equity_df = pd.DataFrame({
            "equity": self.equity_curve,
            "equity_norm": np.array(self.equity_curve) / self.equity_curve[0],
            "realized_equity": np.array(self.realized_curve),
            "realized_equity_norm": np.array(self.realized_curve) / self.realized_curve[0],
            "close": np.array(self.close_ls),
            "close_norm": np.array(self.close_ls) / self.close_ls[0],
            "position": self.position_ls,
            "units": self.units_ls,
        }, index=self.datetime_ls)
        return self.equity_df


    def run_etf(self, capital=1_000, risk=0.01):
        """
        Backtesting method for ETFs (Long Only).
        - No intraday liquidations (Cash/Asset based).
        - Stop Loss is checked at Close and executed at Open next day.
        - Entries/Exits use Limit Orders.
        """
        print(f"Running backtest (ETF Mode | Start: {capital})..")
        
        # initialize
        self.capital = capital
        self.risk = risk
        self.position = 0
        self.entry_price = 0.0
        self.entry_date = None
        self.units = 0.0
        self.sl_price = None
        
        # lists for logging
        self.equity_curve = []
        self.realized_curve = []
        self.position_ls = []
        self.trade_log = []
        self.datetime_ls = []
        self.close_ls = []
        self.units_ls = []
        
        # filter data
        required_cols = self.required_indicators + ["future.open.1", "future.low.1", "future.high.1"]
        clean_data = self.data.dropna(subset=required_cols).copy()

        # run backtesting loop
        for idx, row in clean_data.iterrows():
            
            # extract relevant data
            current_date = row["datetime"]
            today_close = row['close']
            f_open = row["future.open.1"]
            f_low  = row["future.low.1"]
            f_high = row["future.high.1"]

            # get the target state. here, -1 and 0 are considered exit signals, as we go long only
            target_state, strat_sl, strat_limit = self.strategy.on_bar(row, self.position)

            # manage stop logic
            stop_triggered = False

            # stop losses are only triggered at the end of the day, no intraday stop losses required as there
            # is no requirement for margins
            if self.position == 1 and self.sl_price:
                if today_close <= self.sl_price:
                    stop_triggered = True
                    
                    # sell at next market open --> maybe in future switch to limit order
                    exit_price = f_open
                    pnl = (exit_price - self.entry_price) * self.units
                    self.capital += pnl
                    
                    # log trade
                    prev_capital = self.capital - pnl
                    self.trade_log.append({
                        "entry_date": self.entry_date,
                        'exit_date': current_date, 
                        'type': 'STOP_LOSS (EOD)', 
                        'entry': self.entry_price, 
                        'exit': exit_price, 
                        'pnl': pnl, 
                        'return': pnl / prev_capital,
                        'capital': self.capital,
                    })
                    
                    # reset position
                    self.position = 0
                    self.units = 0
                    self.sl_price = None

            # limit order exit
            if not stop_triggered:
                
                # signal processing --> 0 and -1 are exit signals
                if self.position == 1 and target_state in [0, -1]:
                    
                    # here we do a limit exit
                    is_reachable = (f_low <= strat_limit <= f_high)
                    
                    if is_reachable:
                        pnl_exit = (strat_limit - self.entry_price) * self.units
                        self.capital += pnl_exit
                        
                        # log trade
                        prev_capital = self.capital - pnl_exit
                        self.trade_log.append({
                            "entry_date": self.entry_date,
                            'exit_date': current_date, 
                            'type': 'EXIT', 
                            'entry': self.entry_price, 
                            'exit': strat_limit, 
                            'pnl': pnl_exit, 
                            'return': pnl_exit / prev_capital,
                            'capital': self.capital,
                        })
                        
                        # reset positions
                        self.position = 0
                        self.units = 0

                # enter new positions
                elif self.position == 0 and target_state == 1:
                    
                    # according to limit price
                    is_reachable = (f_low <= strat_limit <= f_high)
                    
                    if is_reachable:
                        
                        # risk based position sizes --> if we trade etfs; 1 unit = 1 etf
                        risk_amt = self.capital * self.risk
                        dist_to_sl = abs(strat_limit - strat_sl)
                        
                        if dist_to_sl > 0:
                            new_units = risk_amt / dist_to_sl
                        else:
                            new_units = 0
                        
                        # update position
                        self.position = 1
                        self.units = new_units
                        self.entry_price = strat_limit
                        self.entry_date = current_date
                        self.sl_price = strat_sl


            # update equity curve
            floating_pnl = 0.0
            if self.position == 1:
                floating_pnl = (today_close - self.entry_price) * self.units
            total_equity = self.capital + floating_pnl
            
            # log everything
            self.equity_curve.append(total_equity)
            self.realized_curve.append(self.capital)
            self.datetime_ls.append(current_date)
            self.position_ls.append(self.position)
            self.close_ls.append(today_close)
            self.units_ls.append(self.units)

        # print final net worth
        print(f"Final capital: {self.equity_curve[-1]:.2f}")
        
        self.equity_df = pd.DataFrame({
            "equity": self.equity_curve,
            "equity_norm": np.array(self.equity_curve) / self.equity_curve[0],
            "realized_equity": np.array(self.realized_curve),
            "realized_equity_norm": np.array(self.realized_curve) / self.realized_curve[0],
            "close": np.array(self.close_ls),
            "close_norm": np.array(self.close_ls) / self.close_ls[0],
            "position": self.position_ls,
            "units": self.units_ls,
        }, index=self.datetime_ls)
        return self.equity_df



    def calc_performance_stats(self):
            df = self.equity_df.copy()
            
            # maximum draw down
            df['peak_equity'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['peak_equity']) / df['peak_equity']
            max_drawdown = df['drawdown'].min()
            
            
            # win rate
            trade_returns = df['realized_equity'].pct_change()
            closed_trades = trade_returns[trade_returns != 0].dropna()
            
            if len(closed_trades) > 0:
                win_rate = (closed_trades > 0).mean()
                avg_win = closed_trades[closed_trades > 0].mean()
                avg_loss = closed_trades[closed_trades < 0].mean()
            else:
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0

            # profit factor
            if avg_loss != 0:
                profit_factor = abs(avg_win * win_rate / (avg_loss * (1-win_rate)))
            else:
                profit_factor = 0
            
            # calculate kelly
            kelly = win_rate - ((1-win_rate) / (avg_win/avg_loss))

            
            # save variables
            self.kelly = kelly
            self.win_rate = win_rate
            self.avg_win = avg_win
            self.avg_loss = avg_loss
            self.profit_factor = profit_factor
            self.max_dd = max_drawdown
            self.max_dd_df = df
            return self
    

    def plot_equity_df(self, normalize=True, log_axis=False):

        # calculate stats
        self.calc_performance_stats()

        # initialize figure with secondary_y enabled for the second row
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )

        # Plot the lines
        col_suffix = "_norm" if normalize else ""
        
        # Check for column existence (handles raw or normalized)
        close_col = 'close' + col_suffix if 'close' + col_suffix in self.equity_df.columns else 'close'
        
        fig.add_trace(go.Scatter(x=self.equity_df.index, 
                                 y=self.equity_df[close_col],
                                 mode='lines', 
                                 name='Asset Price',
                                 line=dict(color='black', width=1)), row=1, col=1)

        # return realized returns
        fig.add_trace(go.Scatter(x=self.equity_df.index, 
                                 y=self.equity_df['realized_equity' + col_suffix],
                                 mode='lines', 
                                 name='Equity (realized)',
                                 line=dict(color='orange')), row=1, col=1)

        # plot marked-to-market returns
        fig.add_trace(go.Scatter(x=self.equity_df.index, 
                                 y=self.equity_df['equity' + col_suffix],
                                 mode='lines', 
                                 name='Equity (MTM)',
                                 line=dict(color='blue')), row=1, col=1)

        # signals (handling raw/norm columns)
        for pos, color, label in [(1, 'green', 'LONG'), (-1, 'red', 'SHORT')]:
            signals = self.equity_df.copy()
            signals["marker_y"] = np.nan
            mask = signals['position'] == pos
            signals.loc[mask, "marker_y"] = self.equity_df.loc[mask, close_col]
            fig.add_trace(go.Scatter(x=signals.index, 
                                     y=signals['marker_y'],
                                     mode='lines', 
                                     name=label,
                                     line=dict(color=color)), row=1, col=1)
        
        # add drawdown
        fig.add_trace(go.Scatter(
            x=self.max_dd_df.index,
            y=self.max_dd_df["drawdown"],
            mode='lines',
            name='Drawdown',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)',
        ), row=2, col=1, secondary_y=False)

        # add units
        fig.add_trace(go.Scatter(
                x=self.equity_df.index,
                y=self.equity_df['units'],
                mode='lines',
                name='Unit',
                line=dict(color='rgba(0, 128, 0, 0.5)', width=2, shape='hv'),
            ), row=2, col=1, secondary_y=True)

        # update layout
        strategy_name = str(type(self.strategy)).split("\'")[1].split(".")[-1]
        title_str = f"{strategy_name} ({', '.join(self.required_indicators)}) - {self.data['symbol'].unique()[0]}<br>"
        title_str += f"<span style='font-size: 12px; font-style: italic;'>"
        title_str += f"Win Rate: {self.win_rate:.1%} | Avg Win: {self.avg_win:.2f} | Avg Loss: {self.avg_loss:.2f} | "
        title_str += f"Max DD: {self.max_dd:.1%} | Profit Factor: {self.profit_factor:.2f} | Kelly: {self.kelly:.2f}"
        title_str += f"</span>"
        
        fig.update_layout(
            title=title_str,
            template="simple_white",
            height=900, width=1200,
            showlegend=True,
            legend=dict(orientation="v", y=1.02, x=0)
        )

        # axis Labels
        fig.update_yaxes(title_text="Equity" + (" (Norm)" if normalize else " ($)"), row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat='.0%', row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Units", row=2, col=1, secondary_y=True, showgrid=False)
        fig.update_xaxes(showgrid=False)
        if log_axis:
            fig.update_yaxes(type="log", row=1, col=1)

        fig.show()


    def monte_carlo(self, n_simulations=1000, drawdown=0.5):

        # get the returns of the strategy
        realized_returns = [t['return'] for t in self.trade_log]
        
        if not realized_returns:
            print("No trades to simulate!")
            return

        # convert to series
        returns_series = pd.Series(realized_returns)
        print(f"Simulating {n_simulations} scenarios based on {len(returns_series)} trades...")
        
        # for results
        simulated_curves = []
        for i in range(n_simulations):
            
            # the shuffle
            shuffled_returns = returns_series.sample(frac=1, replace=True, ignore_index=True)
            
            # calculate the equity curve
            sim_curve = (1 + shuffled_returns).cumprod()
            sim_curve = pd.concat([pd.Series([1.0]), sim_curve], ignore_index=True)
            simulated_curves.append(sim_curve)

        # calcualte some statistics
        final_values = [curve.iloc[-1] for curve in simulated_curves]
        loss_probability = sum(x < 1.0 for x in final_values) / len(final_values)
        ruin_probability = 0
        for curve in simulated_curves:
            if curve.min() < drawdown:
                ruin_probability += 1
        ruin_probability /= len(simulated_curves)
        print(f"Probability of Loss: {loss_probability:.1%}")
        print(f"Risk of Ruin (>{drawdown*100}% DD): {ruin_probability:.1%}")


        # convert to dict
        all_sim_curves = {f"sim_{i}": e for i, e in enumerate(simulated_curves)}
        all_sim_curves[f"actual"] = pd.concat([pd.Series([1.0]), (1 + returns_series).cumprod()], ignore_index=True)
        sim_df = pd.DataFrame(all_sim_curves)
        return sim_df


    def monkey_carlo(self, n_simulations=100):
        print(f"Running {n_simulations} monkey tests...")
        
        # get stuff from real backtest
        if not hasattr(self, 'trade_log') or not self.trade_log:
            print("MISSING TRADELOG")
            return None
        
        # same clean data as original
        clean_data = self.data.dropna(subset=self.required_indicators + ["future.open.1"]).copy()
        
        # entry probability
        n_days = len(clean_data)
        n_real_trades = len(self.trade_log)
        prob_entry = n_real_trades / n_days
        
        # exit probability
        durations = [(t['exit_date'] - t['entry_date']).days for t in self.trade_log]
        avg_duration = sum(durations) / len(durations) if durations else 1
        prob_exit = 1 / avg_duration if avg_duration > 0 else 0.5

        # plot some logs
        print(f"Calibration: Entry Prob {prob_entry:.2%} | Avg Hold {avg_duration:.1f} days (Exit Prob {prob_exit:.2%})")

        # store results
        all_sim_curves = {}

        # perform the backtest
        for i in range(n_simulations):
            #print(f"Starting: {round(100*i/n_simulations, 2)}%")
            
            # initial variables
            banked_equity = 1.0
            position = 0
            entry_price = 0.0
            sim_equity_curve = []
            
            # iterate through time
            for idx, row in clean_data.iterrows():
                
                today_close = row['close']
                tomorrow_open = row["future.open.1"]
                
                # default state
                target_state = position
                
                # rolling dive for entry
                if position == 0:
                    if random.random() < prob_entry:
                        target_state = random.choice([-1, 1])
                
                # rolling dice for exit
                else:
                    if random.random() < prob_exit:
                        target_state = 0

                
                # calculate mark-to-market return
                floating_pnl_pct = 0
                if position == 1:
                    floating_pnl_pct = (today_close - entry_price) / entry_price
                elif position == -1:
                    floating_pnl_pct = (entry_price - today_close) / entry_price
                current_total_equity = banked_equity * (1 + floating_pnl_pct)
                sim_equity_curve.append(current_total_equity)

                
                # exiting trades
                if position != 0 and (target_state != position):
                    trade_return = 0
                    if position == 1:
                        trade_return = (tomorrow_open - entry_price) / entry_price
                    elif position == -1:
                        trade_return = (entry_price - tomorrow_open) / entry_price
                    
                    # bank the money and reset
                    banked_equity = banked_equity * (1 + trade_return)
                    position = 0
                
                # entry logic
                if target_state != 0 and position == 0:
                    position = target_state
                    entry_price = tomorrow_open

            # store the equity curve for plotting
            all_sim_curves[f"sim_{i}"] = sim_equity_curve

        # add the actual data
        all_sim_curves[f"actual"] = np.array(self.equity_curve) / self.equity_curve[0]
        sim_df = pd.DataFrame(all_sim_curves, index=clean_data["datetime"])
        return sim_df


    def plot_simulations(self, sim_df, quantile=(0.75, 0.95, 0.99)):
        
        # calculate statistics
        mc_mean = sim_df.median(axis=1)

        # calculate the quantiles
        quantile_n = [e for e in quantile] + [1-e for e in quantile]
        quantile_n = sorted(quantile_n, reverse=True)
        quantile_v = []
        for q in quantile_n:
            quantile_df = sim_df.quantile(q, axis=1)
            quantile_v.append(quantile_df)
        
        # make the plot
        fig = go.Figure()

        # generate the colors
        colors = pc.qualitative.Plotly
        
        for i, col in enumerate(sim_df.columns):
            if col != "actual":
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=sim_df.index,
                    y=sim_df[col],
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=0.15,
                    showlegend=False,
                    name=f"Sim {i}"
                ))


        # plot the quantiles
        for v, n in zip(quantile_v, quantile_n):

            # plot
            fig.add_trace(go.Scatter(
                x=sim_df.index,
                y=v,
                mode='lines',
                name=f'{round(n*100, 2)}% Percentile',
                line=dict(color='black', width=1, dash='dot'),
                legendgroup="stats"
            ))

        # plot mean
        fig.add_trace(go.Scatter(
            x=sim_df.index,
            y=mc_mean,
            mode='lines',
            name='Monte Carlo Mean',
            line=dict(color='black', width=2, dash='dash'),
        ))

        # plot the actual
        fig.add_trace(go.Scatter(
            x=sim_df.index,
            y=sim_df["actual"],
            mode='lines',
            line=dict(color="black", width=3),
            opacity=1,
            showlegend=True,
            name=f"Actual"
        ))


        # define layout
        is_date = type(sim_df.index) == pd.DatetimeIndex
        xaxis = "Date" if is_date else "Trade Count"
        title = "Monte Carlo Simulation " + ("(Significance of edge via Monkey trader)" if is_date else "(Risk of ruin via shuffled returns)")
        fig.update_layout(
            title=title,
            xaxis_title=xaxis,
            yaxis_title="Equity Multiple (Starting at 1.0)",
            template="simple_white",
            width=1200,
            height=900,
        )
        
        fig.show()
    

    def test_overfit(simualations=100, augment=0.2):
        """
        This function aims to provide some feedback on whether a given strategy is overfit or reasonable in terms of 
        parameters. The idea is the following: all arguments of the strategy are augmented by the augment argument (default is 20%).
        For instance, a momentun strategy that uses a 100 day moving average might have random values ranging from 80 to 120. This is done with all
        arguments for the strategy class for a certain number of times ('simulations' parameter). To the end of this, the actual strategy will
        be plotted against all these augmented simulations. If the strategy is reasonably parameterized, we expect that the actual
        strategy falls within the mean +/- 1*std of these augmented runs. if the strategy lays far out this, i.e., mean + 2.5*std, we can safely
        assume that our strategy as been overfitted. All parameters are augmented at once. general logic of the strategies will be preserved.
        """

        # get the strategy object, determine the parameter space and augment
        self.strategy