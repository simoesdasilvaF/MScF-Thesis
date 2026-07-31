# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 08:20:23 2025

@author: Flavio Simoes da Silva

Purpose: Master's Thesis 
    Portfolio Optimization and NLP:
    Algorithmic trading with Black-Litterman and views from stock market sentiment coming from StockTwits.
    
    Supervisor: M.-A Divernois, Ph.D
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#================= Load Sentiment Data from data_tokenizer.py =================

df_SCSSR = pd.concat([pd.read_pickle('df_SCSSR_1_with_sentiment.pkl'), 
                      pd.read_pickle('df_SCSSR_2_with_sentiment.pkl')])


#================ Groupby and aggregate to get Polarities P_i,t ===============

# get probability measures with a groupby and an aggregation
df_polarity = df_SCSSR.groupby(['day', 'ticker']).agg(
    bullish_sent_sum = ('bullish', 'sum'),
    bearish_sent_sum = ('bearish', 'sum'),
    total_sent = ('predicted_sentiment', 'count')
).reset_index()

# epsilon
df_polarity['eps'] = 1.0 / (df_polarity['total_sent'] + 1.0)

# compute Polarities P_{i,t} based on measures previously computed
df_polarity['polarity_P_it'] = (
    df_polarity['bullish_sent_sum'] - df_polarity['bearish_sent_sum']) / (
        df_polarity['bullish_sent_sum'] + df_polarity['bearish_sent_sum'] + df_polarity['eps']
)

# rename ticker FB to META for merging and future calculations
df_polarity['ticker'] = df_polarity['ticker'].replace('FB', 'META')

print('Sentiment polarity computation done.')


#================= Load Stock Prices from yfinance =================

# extract earliest and latest dates from df_polarity
earliest_date = pd.to_datetime(df_polarity["day"]).min()
latest_date = pd.to_datetime(df_polarity["day"]).max()

# time period from df_polarity -5 days, for ulterior Monte Carlo mu and sigma calculation
start_date = earliest_date - timedelta(days=8) #8 days due to closed markets on Friday, July 3, 2009 --> Independence Day. And week-end.
end_date = latest_date

# list of stocks
SCSSR_tickers = [
    "AAPL", "AMD", "AMRN", "AMZN", "BABA", "BAC", "BB", "META", "GLD",
    "IWM", "JNUG", "MNKD", "NFLX", "PLUG", "QQQ", "SPY", "TSLA", "UVXY"
]

# Download historical stock data
df_prices_no_TWTR = yf.download(SCSSR_tickers, start=start_date, end=end_date,
                                auto_adjust=False)['Adj Close']


# add TWTR (X, formerly Twitter) from excel file downloaded at CEDIF, data from Refinitiv
df_TWTR = pd.read_excel('TWTR.xlsx', sheet_name="Sheet1")
df_TWTR.columns = ["Date", "TWTR"] #rename columns to manipulate Date to datetime
df_TWTR["Date"] = pd.to_datetime(df_TWTR["Date"])
df_TWTR = df_TWTR.dropna(subset=["TWTR"]) #drop NaN

# convert index to Date column for merging
df_prices_no_TWTR = df_prices_no_TWTR.reset_index() 
df_prices_no_TWTR = pd.merge(df_prices_no_TWTR, df_TWTR, on="Date", how="left")

# create df_prices, the complete DataFrame starting when all tickers exist
df_prices_no_TWTR.set_index("Date", inplace=True)
df_prices = df_prices_no_TWTR.copy()
df_prices = df_prices.loc[df_prices.index >= "2014-09-22"]
df_prices = df_prices.sort_index(axis=1)

#============= Rolling GBM Monte Carlo Simulation of stock prices =============

# manipulation of dates
df_prices.reset_index(inplace=True)
df_prices['Date'] = pd.to_datetime(df_prices['Date'])

# parameters for rolling Monte Carlo simulation
num_simulations = 10000  # number of Monte Carlo simulations 5000
forecast_days = 5  # days to forecast
rolling_window = 6  # shift window for next simulation
rolling_mu_window = 5  # use last 5 trading days for rolling drift & volatility -->  shifts the starting day of Monte Carlo

# list of tickers
tickers = df_prices.columns[1:]

# dictionary to store results
monte_carlo_results = {}

# Monte Carlo simulation
for ticker in tickers:
    if df_prices[ticker].isna().all(): # skip tickers with only NaN values (normally None)
        continue
    
    # create list for results, and drop miscellaneous NaNs
    ticker_results = []
    prices = df_prices[['Date', ticker]].dropna().reset_index(drop=True)
    
    # set the indexing up
    for start_idx in range(rolling_mu_window, len(prices) - forecast_days, rolling_window):
        # ensure  Monte Carlo starts at the real price at time t
        start_price = prices[ticker][start_idx]
        
        # historical period for rolling mu and sigma
        historical_prices = prices[ticker][start_idx - rolling_mu_window:start_idx]
        
        # skip if not enough data
        if len(historical_prices) < 2:
            continue
        
        # rolling daily log returns
        log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
        
        # estimate mu and sigma
        mu = log_returns.mean()
        sigma = log_returns.std()
        
        ## Monte Carlo based on GBM
        dt = 1 # 1 day
        simulated_prices = np.zeros((forecast_days+1, num_simulations)) # start price
        simulated_prices[0,:] = start_price # all paths should start at the same initial price
        
        # prepare Wiener process
        for t in range(1, forecast_days+1):
            # generate some random standard normal values for the Wiener process
            Z = np.random.normal(0, 1, num_simulations)
            
            # GBM formula with rolling parameters
            simulated_prices[t, :] = simulated_prices[t - 1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
        # store results
        ticker_results.append({
            'start_date': prices['Date'][start_idx],
            'start_price': start_price,
            'simulated_prices': simulated_prices
            })
        
    monte_carlo_results[ticker] = ticker_results


#====================== go from dictionary to DataFrame =======================

# create a list that will be transformed into a DataFrame rather than having a dictionary
results_list = []

# iterate through each ticker's results
for ticker, simulations in monte_carlo_results.items():
    # iterate through every simulation
    for run in simulations:
        start_date = run["start_date"]
        start_price = run["start_price"]
        simulated_prices = run["simulated_prices"]

        # convert the simulated price matrix into a DataFrame
        for sim_num in range(simulated_prices.shape[1]):  # iterate over simulations
            results_list.append({
                "day": start_date,
                "ticker": ticker,
                "Simulation Number": sim_num,
                "Day 0": start_price,
                "Day 1": simulated_prices[1, sim_num],
                "Day 2": simulated_prices[2, sim_num],
                "Day 3": simulated_prices[3, sim_num],
                "Day 4": simulated_prices[4, sim_num],
                "Day 5": simulated_prices[5, sim_num],
            })

# create a DataFrame from the results list
df_monte_carlo_results_complete = pd.DataFrame(results_list)

#================== Plotting rolling Monte Carlo Simulation ===================

        
# # Plot Monte Carlo simulations for every ticker in the list
# for ticker in monte_carlo_results.keys():
#     if monte_carlo_results[ticker]:  # Ensure results exist for this ticker
#         plt.figure(figsize=(12, 6))

#         # Plot Monte Carlo simulations in blue
#         for run in monte_carlo_results[ticker]:
#             if not df_prices[df_prices["Date"] == run["start_date"]].empty:
#                 start_date_idx = df_prices[df_prices["Date"] == run["start_date"]].index[0]
#                 start_date = df_prices["Date"][start_date_idx : start_date_idx + forecast_days + 1]

#                 for sim in run["simulated_prices"].T[:300]:  # Limit to 100 simulations for clarity
#                     plt.plot(start_date, sim, color='blue', alpha=0.05)

#         # Plot actual historical prices in red
#         plt.plot(df_prices['Date'], df_prices[ticker], color='red', label="Real Price")

#         plt.title(f"Rolling GBM Monte Carlo Simulation for {ticker}")
#         plt.xlabel("Time")
#         plt.ylabel("Stock Price")
#         plt.legend()
#         plt.show()


#### Palette block ####
palette = {
    "sim"  : "#88C0D0",   # glacier blue – individual paths
    "mean" : "#A3BE8C",   # sage green   – MC mean
    "real" : "#BF616A",   # muted red    – actual price
}
sim_alpha = 0.05          # path transparency

##### Plotting loop ####
for ticker, runs in monte_carlo_results.items():
    plt.figure(figsize=(10, 5))

    ## plot each simulated path
    for r in runs:
        start_idx = df_prices.index[df_prices["Date"] == r["start_date"]]
        if start_idx.empty:
            continue
        start_idx = start_idx[0]

        sim_dates = df_prices["Date"].iloc[start_idx : start_idx + r["simulated_prices"].shape[0]]

        # individual paths
        plt.plot(sim_dates, r["simulated_prices"][:, :1000],  # first x
                  color=palette["sim"], alpha=sim_alpha)

    # stack all matrices then average along simulations axis
    all_sims = np.concatenate([r["simulated_prices"] for r in runs], axis=1)

    # actual price
    plt.plot(df_prices["Date"], df_prices[ticker],
              color=palette["real"], lw=1.4, label="Real price")

    # plot
    plt.title(f"Rolling GBM Monte‑Carlo simulation – {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=.3, linestyle="--")
    plt.tight_layout()
    plt.show()




#============= Setup a DataFrame for Polarity-adjusted price vizualisation =============

# add max and min of day 5 price predictions to the DataFrame
df_monte_carlo_results_complete["Max Day 5"] = df_monte_carlo_results_complete.groupby(["ticker", "day"])["Day 5"].transform("max")
df_monte_carlo_results_complete["Min Day 5"] = df_monte_carlo_results_complete.groupby(["ticker", "day"])["Day 5"].transform("min")


# filter the DataFrame down -> only the rows corresponding to the max and min values for Day 5 are retained
# intuition is desinflating the DataFrame -> reduce computing time
df_monte_carlo_results = df_monte_carlo_results_complete[
    (df_monte_carlo_results_complete["Day 5"] == df_monte_carlo_results_complete.groupby(["ticker", "day"])["Day 5"].transform("max")) | # bitwise "or" instead of logical)
    (df_monte_carlo_results_complete["Day 5"] == df_monte_carlo_results_complete.groupby(["ticker", "day"])["Day 5"].transform("min"))
]

# merge Monte Carlo and Polarity for a complete DataFrame
df_polarity['day'] = pd.to_datetime(df_polarity['day']) # make sure all DataFrames are on the datetime base before merging
df_monte_carlo_results = df_monte_carlo_results.merge(df_polarity, on=["ticker", "day"], how="left")


# apply the polarity-weighted price adjustment formula from the following paper
# BERT’s sentiment score for portfolio optimization: a fine-tuned view in Black and Litterman model
# /!\ I adapted the formula, adding a max() and abs()
df_monte_carlo_results["Predicted_S_t+5"] = df_monte_carlo_results.apply(lambda row: 
    row["Day 0"] + ((row["Max Day 5"] - row["Day 0"]) * row["polarity_P_it"]) if row["polarity_P_it"] >= 0 
    else row["Day 0"] - (max(row["Day 0"] - row["Min Day 5"], 0) * abs(row["polarity_P_it"])), axis=1) # the max () ensures that the sentiment-adjusted price does not increase when sentiment is negative (due to Randomness of MC), and the abs() is here because i think there is a typo in his paper

df_monte_carlo_results.info()


# df_monte_carlo_results.to_pickle('df_monte_carlo_results.pkl')



#==================== Setup a DataFrame for B-L utilization ===================

# convert 'Date' column to 'day', and melt for merging purpose 
df_prices.rename(columns={'Date': 'day'}, inplace=True)
df_prices_melted = df_prices.melt(id_vars=['day'], var_name='ticker', value_name='real_price_t')

# extract relevant columns from df_monte_carlo_results
df_black_litterman = df_monte_carlo_results[['day', 'ticker', 'Predicted_S_t+5']].copy()

# merge to get the real price at t, and drop duplicates 
df_black_litterman = df_black_litterman.merge(df_prices_melted, on=['day', 'ticker'], how='left')
df_black_litterman.drop_duplicates(inplace=True)

# create column to get the real price at t+5
df_prices_melted["real_price_t+5"] = df_prices_melted.groupby("ticker")["real_price_t"].shift(-forecast_days) # coherent with Monte Carlo Simulation

# merge DataFrames 
df_black_litterman = df_black_litterman.merge(
    df_prices_melted[["day", "ticker", "real_price_t+5"]],
    left_on=["day", "ticker"],
    right_on=["day", "ticker"],
    how="left"
)

# reorder and rename columns 
df_black_litterman = df_black_litterman[["day", "ticker", "real_price_t", "real_price_t+5", "Predicted_S_t+5"]]

# create log-returns columns
df_black_litterman["real_log_return"] = np.log(df_black_litterman["real_price_t+5"] / df_black_litterman["real_price_t"])
df_black_litterman["predicted_log_return"] = np.log(df_black_litterman["Predicted_S_t+5"] / df_black_litterman["real_price_t"])

# annualize those returns 
annualization_factor = 252 / 5
df_black_litterman["annualized_real_log_return"] = df_black_litterman["real_log_return"] * annualization_factor
df_black_litterman["annualized_predicted_log_return"] = df_black_litterman["predicted_log_return"] * annualization_factor

#============================ Save to pkl format ================================
# df_black_litterman.to_pickle('df_black_litterman.pkl')
# df_prices.to_pickle('df_prices.pkl')
# df_monte_carlo_results_complete.to_pickle('df_monte_carlo_results_complete.pkl')

