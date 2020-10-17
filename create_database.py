""" Script to read in all data in folder and create dataframe """

import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from finta import TA


# functions to be used below





# First I want to add to the initial csv files to make sure they are up to date




# Second I want to add additional columns to the csv files to track performance metrics
stocks = [stock.split('.')[0] for stock in sorted(os.listdir('./australian-historical-stock-prices'))]

for stock in stocks:
    prices = pd.read_csv('./australian-historical-stock-prices/' + stock + '.csv')
    prices['datetime'] = pd.to_datetime(prices['Date'])
    prices['adj'] = prices['Adj Close'] - prices['Close']
    prices['Adj Open'] = prices['Open'] + prices['adj']
    prices['Adj High'] = prices['High'] + prices['adj']
    prices['Adj Low'] = prices['Low'] + prices['adj']
    ohlc = prices.filter(['Adj Open', 'Adj High', 'Adj Low', 'Adj Close'])
    ohlc.columns = ["open", "high", "low", "close"]
    ohlcv = prices.filter(['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Volume'])
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    # Daily change
    prices["movement"] = (prices["Adj Close"] - prices["Adj Close"].shift(periods=-1)) / prices["Adj Close"].shift(periods=-1)
    # 1 month
    prices['20_Day_moving_av'] = prices["Adj Close"].rolling(window=20).mean()
    prices['20_Day_stddev'] = prices["Adj Close"].rolling(window=20).std()
    # 2 months
    prices['126_Day_moving_av'] = prices["Adj Close"].rolling(window=126).mean()
    prices['momentum_factor'] = prices["20_Day_moving_av"] / prices["126_Day_moving_av"]
    # Daily spread
    prices["daily_spread"] = (prices["Adj High"] - prices["Adj Low"]) / prices["Adj Close"]
    # Daily volume relative to 6 month average
    prices["vol_change"] = prices["Volume"] / prices["Volume"].rolling(window=126).mean()

    # Target - predict how well the share will go over the next week
    prices["Target-future_week_gain"] = (prices["Adj Close"].shift(periods=-5) - prices["Adj Close"]) / prices["Adj Close"]


    # bollinger bands
    prices["Bollinger_upper"] = prices['20_Day_moving_av'] + 1.96 * prices['20_Day_stddev']
    prices["Bollinger_lower"] = prices['20_Day_moving_av'] - 1.96 * prices['20_Day_stddev']
    # small differences between upper and lower bands indicate that a sharp change is likely (chaos)
    # if the movement goes outside the bands, it is likely to continue
    # if the market tops or bottoms, first outside the band and then
    prices["_bollinger_linear_metric"] = (prices["Adj Close"] - prices['20_Day_moving_av']) / prices['20_Day_stddev']
    prices["_bollinger_buy_flag"] = prices["_bollinger_linear_metric"] < -2
    prices["_bollinger_sell_flag"] =  prices["_bollinger_linear_metric"] > 2

    prices["BB_upp-finta"], prices["BB_mid-finta"], prices["BB_low-finta"] = TA.BBANDS(ohlc)
    # indicates the 0.8 std dev over 10 periods to give a similar band to bollinger
    prices["MOBO"] = TA.MOBO(ohlc)
    # relative position of price within bands used for stochastics
    prices["Percent_b"] = TA.PERCENT_B(ohlc)
    # if candle is completely above the upper band, it will usually keep going up
    # In a rising market, we are more likely to hit the upper band than the lower band
    #   this forces us to sell and makes us miss out on the gains
    # if it passes from the lower band across the middle it will keep going to hit the top band

    # Keltner channels are similar to BBands but with exponential moving averages and average true range
    # useful to identify overbought or oversold levels in a sideways market
    prices["Keltner_Channels_upp"], prices["Keltner_Channels_down"] = TA.KC(ohlc)

    # Donchian channels
    prices["DC_low"], prices["DC_mid"], prices["DC_upp"] = TA.DO(ohlc)

    # exponential bollinger (lends more weight to recent trends)
    prices["Exp_mav_sh"] = prices['Adj Close'].ewm(span=4).mean()
    prices["Exp_mav_med"] = prices['Adj Close'].ewm(span=9).mean()
    prices["Exp_mav_lon"] = prices['Adj Close'].ewm(span=14).mean()
    # Buy when the short crosses above the long term average a further indicator of good direction is when the medium
    # also crosses the long
    # Sell when the short crosses below the medium and really sell when it crosses below the long
    # So we have a number of things to implement (signal is +1 when short above medium, +2 short above long, +3 when
    # medium above long as well. Similarly in the opposite direction. Positive is buy, negative is sell)
    EMAV_conditions = [
        (prices["Exp_mav_sh"] > prices["Exp_mav_med"]) & (prices["Exp_mav_sh"] < prices["Exp_mav_lon"]),
        (prices["Exp_mav_sh"] > prices["Exp_mav_med"]) & (prices["Exp_mav_sh"] > prices["Exp_mav_lon"]),
        (prices["Exp_mav_sh"] > prices["Exp_mav_lon"]) & (prices["Exp_mav_med"] > prices["Exp_mav_lon"]),
        (prices["Exp_mav_sh"] < prices["Exp_mav_med"]) & (prices["Exp_mav_sh"] < prices["Exp_mav_lon"]),
        (prices["Exp_mav_sh"] < prices["Exp_mav_med"]) & (prices["Exp_mav_sh"] < prices["Exp_mav_lon"]),
        (prices["Exp_mav_sh"] < prices["Exp_mav_lon"]) & (prices["Exp_mav_med"] < prices["Exp_mav_lon"]),
    ]
    EMAV_choices = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]
    prices["Exp_mav-signal"] = np.select(EMAV_conditions, EMAV_choices, default=0.0)

    # Accumulation and Distribution - Marc Chaikin to determine flow in or out of a security
    prices["Money Flow Multiplier"] = TA.ADL(ohlcv)
    # MACD oscillator
    prices["MACDb-finta"] = TA.CHAIKIN(ohlcv)
    prices["MFI"] = TA.MFI(ohlc)


    prices["CMFV"] = prices['Volume'] * ((prices['Adj Close'] - prices['Adj Low']) -
                                         (prices['Adj High']-prices['Adj Close'])) / \
                     (prices['Adj High'] - prices['Adj Low'])
    # It shows how strong the trend is, if the price is rising but the indicator is falling, this indicates that
    #    buying or accumulation volume many not be enough to support the price rise and a decline may be forthcoming
    prices["CMFV-rolling"] = prices["CMFV"].rolling(window=20).sum()
    prices["CMFV-signal"] = prices["CMFV-rolling"] - prices["CMFV-rolling"].shift(periods=5)

    # Moving average convergence divergence
    # A divergence is where the price doesnt agree with the indicator
    prices["MACD"] = prices["Adj Close"].rolling(window=12).mean() - prices["Adj Close"].rolling(window=26).mean()
    # above zero indicates that it is bullish or an upward trend in supply/demand. Once it falls below zero implies bearish
    prices["MACD-signal"] = prices["MACD"].shift(periods=9)
    # where the MACD crosses the signal (to drop below) it is a sell indicator, whereas crossing above is a buy indicator
    # this works poorly during a sideways market and likes volatility
    # It can also be used to put a 90% confidence interval on the MACD range to indicate when it is overbought or oversold
    # this indicator is faster than the signal
    prices["_MACD_range"] = prices["MACD"] / prices["MACD"].rolling(window=180).std()
    # if _MACD_range is less than -1.5 buy, if it exceeds 2 sell if profits have been made

    prices["MACD_finta"], prices["MACD-signal_finta"] = TA.MACD(ohlc)
    prices["MACDvw_finta"], prices["MACDvw-signal_finta"] = TA.VW_MACD(ohlcv)
    prices["MACDev_finta"], prices["MACDev-signal_finta"] = TA.EV_MACD(ohlcv)
    # Historical price volatility

    # Relative Strength Indicator (0-100 range)
    # above 70-80 is overbought and below 20-30 is the oversold indicator
    prices["rel_strength_ind"] = TA.RSI(ohlc)
    # Dynamic momentum index (variable term RSI)
    prices["rel_strength_ind_dyn"] = TA.DYMI(ohlc)
    # buy with IFT over -0.5, and sell when crosses under 0.5 (or -0.5 if it has not previously crossed under +0.5)
    prices["IFTrel_strength_ind"] = TA.IFT_RSI(ohlc)

    # On balance volume
    prices["OBV"] = TA.OBV(ohlcv)
    prices["WOBV"] = TA.WOBV(ohlcv)

    # VZO uses price, previous price and moving averages to compute its value
    # Used by lots of people, 5-40% is bullish, -40-5 is bearing and above 40 is overbought with above 60 extreme
    # below -40 is oversold with -60 extreme
    prices["VZO"] = TA.VZO(ohlc)
    prices["PZO"] = TA.PZO(ohlc) # only looks at previous day to today

    # Elders force index to identify possible turning points
    prices["EFI"] = TA.EFI(ohlcv)
    prices["CFI"] = TA.CFI(ohlcv)

    # Bull and bear power
    prices["BullPower"], prices["BearPower"] = TA.EBBP(ohlc)

    # Ease of movement oscillator - distance from zero indicates ease (neg is down, positive is up)
    prices["EMI"] = TA.EMV(ohlcv)

    # Commodity Channel Index - above +100 is overbought, below -100 is oversold
    prices["CCI"] = TA.CCI(ohlc)

    # Coppock curve - momentum indicator (buy when it crosses from negative to positive
    prices["COPP"] = TA.COPP(ohlc)

    # Buying and selling pressure indicator
    prices["BASP_buy"], prices["BASP_sell"] = TA.BASP(ohlc)
    prices["BASPN_buy"], prices["BASPN_sell"] = TA.BASPN(ohlc)

    # Double exponential moving average - attempts to remove the moving average lag
    prices["DEMA"] = TA.DEMA(ohlc)

    # Triple exponential moving average - attempts to remove the moving average lag
    prices["TEMA"] = TA.TEMA(ohlc)
    # rate of change of TEMA - BUY/SELL triggered by zero crossings
    prices["TRIX"] = TA.TRIX(ohlc)
    prices["TRIX-signal"] = prices["TRIX"] / prices['TRIX'].ewm(span=9).mean()

    # Triangular moving average
    prices["TRIMA"] = TA.TRIMA(ohlc)

    # Volume adjusted moving average
    prices["VAMA"] = TA.VAMA(ohlcv)

    # Variable index dynamic average indicator
    # prices["VIDYA"] = TA.VIDYA(ohlcv)

    # Kaufman Efficiency indicator (oscillator indicator between -100 to +100 (+100 is upward trending, -100 downward)
    prices["kaufman_eff"] = TA.ER(ohlc)
    # Kaufman adaptive moving average
    prices["Kaufman_AMA"] = TA.KAMA(ohlc)

    # Chande momentum oscillator range bounded +/- 100
    prices["CMO"] = TA.CMO(ohlc)

    # Chandelier Exit sets a trailing stop loss based on ATR
    prices["C_Exit"] = TA.CHANDELIER(ohlc)

    # Qstick indicator - dominance of candlesticks
    prices["QSTICK"] = TA.QSTICK(ohlc)

    # One look equilibrium chart (ICHIMOKU)
    # indicator for support, resistance, trend direction, momentum and trading signals
    prices["tenkan_sen"], prices["kijun_sen"], prices["senkou_span_a"], prices["senkou_span_b"], prices["chikou_span"] = TA.ICHIMOKU(ohlc)

    # Adaptive price zone - help to find turning points in non-trending choppy markets
    prices["APZ"] = TA.APZ(ohlc)

    # Vector size indicator
    prices["VSI"] = TA.VR(ohlc)

    # Squeeze momentum indicator
    # Generally the market is in quiet consolidation or vertical price discovery, this helps to identify these calm periods
    prices["SQZMI"] = TA.SQZMI(ohlc)

    # Volume price trend
    prices["VPT"] = TA.VPT(ohlc)

    # Money flow indicator
    prices["FVE"] = TA.FVE(ohlc)

    # Volume based on the direction of price movements
    prices["VFI"] = TA.VFI(ohlc)

    # Schaff trend cycle indicates if the trend is moving upwards or downwards
    prices["STC"] = TA.STC(ohlc)

    # Zero lag exponential moving average
    prices["ZLEMA"] = TA.ZLEMA(ohlc)

    # Weighted moving average
    prices["WMA"] = TA.WMA(ohlc)

    # Hull moving average (better for middle/long term trading)
    prices["HMA"] = TA.HMA(ohlc)

    # eVWMA essentially the average price paid per share lately
    prices["EVWMA"] = TA.EVWMA(ohlcv)

    # percentage price oscillator - represents convergence and divergence of two moving averages
    prices["PPO"], prices["PPO_signal"], prices["PPO_histo"] = TA.PPO(ohlc)

    # Rate of change indicator
    prices["ROC-indicator"] = TA.ROC(ohlc)
    # volatility based ROC
    prices["VBROC-indicator"] = TA.VBM(ohlc)

    # True range gives a maximum of three measures of range
    prices["True Range"] = TA.TR(ohlc)
    prices["True Range Ave"] = TA.ATR(ohlc)

    # Stop and reverse indicator (trails prices and flips direction depending on market direction)
    # This is used to set suitable entry or exit points or trailing stop losses
    prices["SAR"] = TA.SAR(ohlc)
    prices["SAR-signal"] = prices["Adj Close"] - prices["SAR"]
    # Parabolic SAR indicator
    prices["PSAR"], prices["PSAR_bull"], prices["PSAR_bear"] = TA.PSAR(ohlc)

    # Directional movement indicator
    # Assesses price direction and strength - Allows the trader to differentiate between strong and weak trends
    prices["DMI_plus"], prices["DMI_minus"] = TA.DMI(ohlc)

    # Trend strength - below 20 is weak, above 40 is strong and above 50 is extremely strong
    prices["Trend_strength"] = TA.ADX(ohlc)

    # Stochastic oscillator
    prices["STOCH"] = TA.STOCH(ohlc)
    prices["STOCHD"] = TA.STOCHD(ohlc)
    prices["STOCHRSI"] = TA.STOCHRSI(ohlc)

    # Williams %R is a technical analysis oscillator
    prices["WR"] = TA.WILLIAMS(ohlc)

    # Awesome Oscillator - measures market momentum
    prices["awesome_oscil"] = TA.AO(ohlc)
    prices["ultimate_oscil"] = TA.UO(ohlc)

    # Mass index - measures high-low range expansion to identify trend reversals. Essentially volatility indicator
    prices["MI"] = TA.MI(ohlc)

    # Balance of power indicator
    prices["BOP"] = TA.BOP(ohlc)

    # Vortex indicator is two oscillating lines one for positive trend movement and one for negative trend movment
    prices["VIm"], prices["VIp"] = TA.VORTEX(ohlc)

    # Know sure thing momentum oscillator
    prices["KST_k"], prices["KST_signal"] = TA.KST(ohlc)

    # True strength index momentum oscillator
    prices["TSI"], prices["TSI_signal"] = TA.TSI(ohlc)

    # Typical price is average of high, low and close
    prices["True_Price"] = TA.TP(ohlc)




    # Pivot points are significant support and resistance level that can be used to determine potential trades
    # currently these are daily, but if we want to do weekly you just adjust the last value in.
    # Not sure what to do with this yet
    # prices["PP_s1"], prices["PP_s2"], prices["PP_s3"], prices["PP_s4"], prices["PP_r1"], prices["PP_r2"],\
    # prices["PP_r3"], prices["PP_r4"] = TA.PIVOT(ohlc)
    # prices["PP_fib_s1"], prices["PP_fib_s2"], prices["PP_fib_s3"], prices["PP_fib_s4"], prices["PP_fib_r1"],
    # prices["PP_fib_r2"], prices["PP_fib_r3"], prices["PP_fib_r4"] = TA.PIVOT_FIB(ohlc)






    prices.to_csv("./processed_prices/" + stock + ".csv")

# Third I want to add in sector / asx / global performance metrics

share_df = pd.read_csv('./processed_prices/' + 'TWE.csv', index_col=0)

fig = go.Figure()

fig.add_trace(go.Scatter(x=share_df.datetime, y=share_df["20_Day_moving_av"], line=dict(color='blue', width=0.7), name="middle"))
fig.add_trace(go.Scatter(x=share_df.datetime, y=share_df["Bollinger_upper"], line=dict(color='red', width=0.7), name="upper"))
fig.add_trace(go.Scatter(x=share_df.datetime, y=share_df["Bollinger_lower"], line=dict(color='green', width=0.7), name="lower"))
fig.add_trace(go.Candlestick(x=share_df.datetime,
                             open=share_df["Adj Open"],
                             high=share_df["Adj High"],
                             low=share_df["Adj Low"],
                             close=share_df["Adj Close"],
                             name='market data', ))

fig.update_layout(title="Bollinger Band Strategy", yaxis_title="A2 Milk Stock Price AUD")

fig.show()

# Strategy buy signal when the lower band it hit, sell when the top band is hit

# # Forth I want to combine the different stocks together and look at them closer.
#
# stocks = [stock.split('.')[0] for stock in sorted(os.listdir('./australian-historical-stock-prices'))]
# dates = pd.date_range('2000-01-01', '2020-03-31') # Create date range from 01-01-2000 to 31-03-2020
# data = pd.DataFrame({'Time': dates})              # Add dates to dataframe with column name
#
# # Append the adjusted closing price of each stock to a dataframe keyed on date
#
# for stock in stocks:
#     prices = pd.read_csv('./australian-historical-stock-prices/' + stock + '.csv', usecols=['Date', 'Adj Close'])
#     prices['Date'] = pd.to_datetime(prices['Date'])
#     prices.rename(columns={"Date": "Time", "Adj Close": stock}, inplace=True)
#     data = pd.merge(data, prices, how='left', on=['Time'], sort=False)
#
# # Remove non-trading days
#
# data = data[data['Time'].dt.weekday < 5] # Remove weekend dates
# data = data.dropna(axis=0, how='all') # Remove empty rows
#
# # Get last price for each stock
#
# p = data.drop(['Time'], axis=1).tail(1).to_numpy()
#
# # Calculate weekly returns from 1 January 2019 onwards
#
# r = data[(data['Time'].dt.weekday == 4) & (data['Time'] >= '2019-01-01')] \
#     .drop(['Time'], axis=1) \
#     .pct_change(fill_method='ffill')
#
# # Calculate expected return and covariance matrix
#
# sigma = r.cov().to_numpy()
# mu = r.mean().to_numpy()
#
# # Set optimisation variable and parameters