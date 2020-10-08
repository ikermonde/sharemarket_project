""" Script to read in all data in folder and create dataframe """

import os
import pandas as pd
import plotly.graph_objs as go

# First I want to add to the initial csv files to make sure they are up to date




# Second I want to add additional columns to the csv files to track performance metrics
stocks = [stock.split('.')[0] for stock in sorted(os.listdir('./australian-historical-stock-prices'))]

for stock in stocks:
    prices = pd.read_csv('./australian-historical-stock-prices/' + stock + '.csv')
    prices['datetime'] = pd.to_datetime(prices['Date'])
    # Daily change
    prices["movement"] = (prices["Adj Close"] - prices["Adj Close"].shift(periods=-1)) / prices["Adj Close"].shift(periods=-1)
    # 1 month
    prices['20_Day_moving_av'] = prices["Adj Close"].rolling(window=20).mean()
    prices['20_Day_stddev'] = prices["Adj Close"].rolling(window=20).std()
    # 2 months
    prices['126_Day_moving_av'] = prices["Adj Close"].rolling(window=126).mean()
    prices['momentum_factor'] = prices["21_Day_moving_av"] / prices["126_Day_moving_av"]
    # Daily spread
    prices["daily_spread"] = (prices["High"] - prices["Low"]) / prices["Close"]
    # Daily volume relative to 6 month average
    prices["vol_change"] = prices["Volume"] / prices["Volume"].rolling(window=126).mean()

    # Target - predict how well the share will go over the next week
    prices["Target-weekly_gain"] = (prices["Adj Close"].shift(periods=+5) - prices["Adj Close"]) / prices["Adj_Close"]
    # bollinger bands
    prices["Bollinger_upper"] = prices['20_Day_moving_av'] + 1.96 * prices['20_Day_stddev']
    prices["Bollinger_lower"] = prices['20_Day_moving_av'] - 1.96 * prices['20_Day_stddev']
    # small differences between upper and lower bands indicate that a sharp change is likely (chaos)
    # if the movement goes outside the bands, it is likely to continue
    # if the market tops or bottoms, first outside the band and then
    prices["_bollinger_buy_flag"]
    prices["_bollinger_sell_flag"]
    prices.to_csv("./processed_prices/" + stock + ".csv")

# Third I want to add in sector / asx / global performance metrics

a2m_df = pd.read_csv('./processed_prices/' + 'A2M.csv')

fig = go.Figure()

fig.add_traces(go.Scatter(x=a2m_df.index, y=a2m_df["20_Day_moving_av"]))

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