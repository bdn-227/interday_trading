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


    def run(self, capital=None, contract_size=12500, risk=0.01, trade_cost=2.0, contract_price=0, max_leverage=10):
        print(f"Running backtest (Mode: {'USD Futures' if capital else 'Theoretical %'})..")
        
        # initial parameters
        self.risk = risk
        self.banked_equity = capital if capital else 1.0
        self.position = 0
        self.leverage = 0
        self.n_contracts = 0
        self.n_contracts_ls = []
        entry_price = 0
        self.trade_log = []
        self.equity_curve = []
        self.step_curve = []
        self.target_states = []
        self.close_l = []
        self.datetime_ls = []
        self.position_ls = []
        self.floating_returns = []
        self.entry_date = None

        # prefilter the data for backtest
        clean_data = self.data.dropna(subset=self.required_indicators + [f"future.open.1", f"future.low.1", f"future.high.1"]).copy()
        normval = clean_data["close"].iloc[0]


        # iterate through time
        for idx, row in clean_data.iterrows():
            
            # get the decision of the strategy
            target_state, sl_price, limit_price = self.strategy.on_bar(row, self.position)
            today_close = row['close']
            tomorrow_open = row[f"future.open.1"]
            current_date = row["datetime"]

            # mark-to-market tracking
            floating_pnl_pct = 0
            
            if self.position == 1:
                floating_pnl_pct = (today_close - entry_price) / entry_price
            elif self.position == -1:
                floating_pnl_pct = (entry_price - today_close) / entry_price
            
            # calculate the equity curve
            if capital:
                # In $$$
                price_diff_mtm = (today_close - entry_price) if self.position == 1 else (entry_price - today_close)
                current_total_equity = self.banked_equity + (price_diff_mtm * self.n_contracts * contract_size)
            else:
                # theoretically as return
                current_total_equity = self.banked_equity * (1 + (floating_pnl_pct * self.leverage))

            # populate lists (Record state BEFORE it changes in the blocks below)
            self.equity_curve.append(current_total_equity)
            self.step_curve.append(self.banked_equity)
            self.target_states.append(target_state)
            self.close_l.append(today_close)
            self.datetime_ls.append(current_date)
            self.position_ls.append(self.position)
            self.floating_returns.append(floating_pnl_pct * self.leverage)
            self.n_contracts_ls.append(self.n_contracts)


            # EXIT
            if self.position != 0 and (target_state != self.position):
                
                # close positions and calculate returns
                trade_return = 0

                # closing long
                if self.position == 1:
                    trade_return = (tomorrow_open - entry_price) / entry_price
                
                # closing short
                elif self.position == -1:
                    trade_return = (entry_price - tomorrow_open) / entry_price
                
                # count the money 
                if capital:
                    # U$$$ mode
                    price_diff_exit = (tomorrow_open - entry_price) if self.position == 1 else (entry_price - tomorrow_open)
                    realized_pnl_usd = (price_diff_exit * self.n_contracts * contract_size) - (self.n_contracts * trade_cost)
                    self.banked_equity += realized_pnl_usd
                else:
                    self.banked_equity = self.banked_equity * (1 + (trade_return * self.leverage))
                
                # logging trades
                self.trade_log.append({
                    'exit_date': row["datetime"],
                    "entry_date": self.entry_date,
                    'type': 'LONG' if self.position == 1 else 'SHORT',
                    'entry': entry_price,
                    'exit': tomorrow_open,
                    'return': trade_return * self.leverage,
                    'leverage': self.leverage,
                    'contracts': self.n_contracts,
                    'new_equity': self.banked_equity
                })
                
                # now reset position to zero
                self.position = 0 
                self.leverage = 0
                self.n_contracts = 0
                self.entry_date = None

            # open new positions
            if target_state != 0 and self.position == 0:
                # Temporary entry price to calculate sizing
                temp_entry_price = tomorrow_open
                
                # risk-aware position sizing
                if sl_price:
                    dist_to_stop_pct = abs(temp_entry_price - sl_price) / temp_entry_price
                    dist_to_stop_usd = abs(temp_entry_price - sl_price) * contract_size
                    
                    if capital:
                        # $$$ mode
                        dollar_amount_to_risk = self.banked_equity * self.risk
                        self.n_contracts = math.floor(dollar_amount_to_risk // dist_to_stop_usd) if dist_to_stop_usd > 0 else 0
                        if (contract_price is not None) and (self.n_contracts > 0):
                            self.n_contracts = math.floor(self.banked_equity / (contract_price * self.n_contracts))
                        
                        # only with full contracts
                        if self.n_contracts > 0:
                            self.position = target_state
                            entry_price = temp_entry_price
                            self.entry_date = row["datetime"]
                            self.leverage = max((self.n_contracts * contract_size * entry_price) / self.banked_equity, max_leverage)
                    
                    else:
                        # theory mode
                        if dist_to_stop_pct > 0:
                            self.leverage = max(risk / dist_to_stop_pct, max_leverage)
                        # $$$ mode
                        else:
                            self.leverage = 1.0
                        self.position = target_state
                        entry_price = temp_entry_price
                        self.entry_date = row["datetime"]
                else:
                    # fallback if no stop loss
                    self.position = target_state
                    entry_price = temp_entry_price
                    self.entry_date = row["datetime"]
                    self.leverage = 1.0
                    self.n_contracts = 1 if capital else 0
        
        # print logs
        print(f"Final Equity: {self.equity_curve[-1]:.4f}")
        equity_df = pd.DataFrame({"equity": self.equity_curve,
                                  "equity_norm": np.array(self.equity_curve)/self.equity_curve[0], 
                                  "realized_equity": np.array(self.step_curve),
                                  "realized_equity_norm": np.array(self.step_curve)/self.step_curve[0],
                                  "close": self.close_l,
                                  "close_norm": self.close_l/normval,
                                  "return": self.floating_returns,
                                  "target_states": self.target_states,
                                  "position": self.position_ls,
                                  "n_contracts": self.n_contracts_ls
                                  }, index=self.datetime_ls)
        self.equity_df = equity_df
        return equity_df


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

        # add number of contracts
        if 'n_contracts' in self.equity_df.columns:
            fig.add_trace(go.Scatter(
                x=self.equity_df.index,
                y=self.equity_df['n_contracts'],
                mode='lines',
                name='Contracts',
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
        fig.update_yaxes(title_text="Contracts", row=2, col=1, secondary_y=True, showgrid=False)
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


    def plot_monte_carlo(self, sim_df, quantile=(0.75, 0.95, 0.99)):
        
        # calculate statistics
        mc_mean = sim_df.mean(axis=1)

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