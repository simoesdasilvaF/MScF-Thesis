# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:55:07 2025

@author: Flavio Simoes da Silva

Purpose: Master's Thesis
    Portfolio Optimization and NLP:
    Algorithmic trading with Black-Litterman and views from stock market sentiment coming from StockTwits.
   
    Supervisor: M.-A Divernois, Ph.D
"""

import pandas as pd
import numpy as np

from pypfopt.black_litterman import market_implied_prior_returns, BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage


#===================== import necessary pkl files for B-L =====================

# read real prices file, DataFrame is already clean
df_prices = pd.read_pickle('df_prices.pkl')

# read Polarity-based prices file, from Monte Carlo sim.
df_black_litterman = pd.read_pickle('df_black_litterman.pkl')


#===================== structure the DataFrames properly ======================

# compute log daily returns from prices and store in DataFrame
df_returns = df_prices.set_index('day')
df_returns = df_returns.sort_index(axis=1)
df_returns = np.log(df_returns) - np.log(df_returns.shift(1)).dropna()

# pivot the annualized Polarity-based predicted returns so that it is consistent with PyPortfolioOpt
df_predicted = df_black_litterman.pivot(index="day",
                                        columns="ticker",
                                        values="annualized_predicted_log_return")



#======== compute the anchor dates t, the trading day to rebalance ============

# compute anchor dates (make sure both the returns and prediction dates are consistent)
anchor_dates = df_predicted.index.intersection(df_returns.index).sort_values()


#====================== compute market capitalizations ========================

# read DataFrame and set index, data from CIQ & prof's paper
df_mcaps = pd.read_excel('mcaps.xlsx', sheet_name='mcaps')
df_mcaps.rename(columns={'Unnamed: 0': 'day'}, inplace=True)
df_mcaps.set_index("day", inplace=True)

# forward fill based on anchor dates
df_mcaps_filled = df_mcaps.reindex(anchor_dates, method="ffill")


#=================== compute confidence, for the Omega matrix ====================

df_SCSSR = pd.concat([pd.read_pickle('df_SCSSR_1_with_sentiment.pkl'), 
                      pd.read_pickle('df_SCSSR_2_with_sentiment.pkl')])

# groupby day and ticker, aggregate by summing (same code block as in polarity_mc.py)
df_polarity = (df_SCSSR.groupby(["day", "ticker"]).agg(
    bullish_sent_sum=("bullish", "sum"),
    bearish_sent_sum=("bearish", "sum"),
    total_sent=("predicted_sentiment", "count"))
).reset_index()

df_polarity['ticker'] = df_polarity['ticker'].replace('FB', 'META')

# get the sum of probabilities (bullish + bearish) per day per ticker
df_polarity['sum_bull_bear'] = (df_polarity['bullish_sent_sum'] + df_polarity['bearish_sent_sum'])

# get the fraction of bullish and bearish percentages out of the sum of both
df_polarity["bullish_fraction"] = (
    df_polarity["bullish_sent_sum"] / df_polarity["sum_bull_bear"]
)

df_polarity["bearish_fraction"] = (
    df_polarity["bearish_sent_sum"] / df_polarity["sum_bull_bear"]
)

# compute the Euclidean distance from (0.5, 0.5), point of complete uncertainty
df_polarity['euclidean_dist'] = np.sqrt( 
    (df_polarity['bullish_fraction'] - 0.5)**2 + (df_polarity['bearish_fraction']-0.5)**2
)

# compute the maximum distance you can get, for nomalization
max_d = np.sqrt(1 - 0.5)  # ~0.70710

# normalize to get confidences between 0-1
df_polarity['norm_dist'] = (
    df_polarity['euclidean_dist'] / max_d
)

# pivot results
df_confidences = df_polarity.pivot(index = 'day',
                                   columns = 'ticker',
                                   values = 'norm_dist')

df_confidences.index = pd.to_datetime(df_confidences.index)
df_confidences = df_confidences.loc[df_confidences.index >= "2014-09-22"]  


#============================== Black-Litterman ===============================

## store results for different strategies
results_mv_long = []
results_mv_long_c0_75 = []
results_mv_long_c0_50 = []
results_mv_long_c0_25 = []
results_mv = []
results_mv_cm75_1 = []
results_mv_cm50_1 = []
results_mv_cm25_1 = []
results_mv_cm75_75 = []
results_mv_cm50_50 = []
results_mv_cm25_25 = []
results_quad_long = []
results_quad = []
results_ms_long = []
results_ms = []

results_prior_quad_long = []
results_prior_quad = []
results_prior_sharpe_long = []
results_prior_sharpe = []


## deterministic parameters
# risk-free, gamma, cov_window, tau
rf_annual = 0.02
cov_window = 180 # number of days used to compute covariance
gamma = 2.8 # risk aversion parameter
tau = 0.05

### B-L and optimization ###
# loop for rolling weights
for date in anchor_dates:

    ## covariances
    i = df_returns.index.get_loc(date)
    if i >= cov_window:
        returns_slice = df_returns.iloc[i - cov_window : i]
    else:
        continue
    
    cs = CovarianceShrinkage(returns_slice, returns_data=True) # shrinkage 
    Σ = cs.ledoit_wolf()*252 # annualize
   
    ## market capitalizations   
    market_caps = df_mcaps_filled.loc[date].to_dict() # transform df into dict
   
    ## market-implied priors
    pi = market_implied_prior_returns(market_caps = market_caps,
                                      cov_matrix = Σ,
                                      risk_aversion = gamma,
                                      risk_free_rate = rf_annual
                                      )
    
    ## views(Q)
    Q = (df_predicted.loc[date].values) - rf_annual # views where the risk-free is deducted at date t

    k = len(Q) # number of views
    P = np.eye(k) # identity matrix based on views length
    
    ## confidence
    tickers = df_predicted.columns
    view_confidences = df_confidences.loc[date, tickers].values
   
    ## build Black-Litterman model with PyPortfolioOpt
    bl = BlackLittermanModel(cov_matrix = Σ,
                              pi = pi,
                              P = P,
                              Q = Q,
                              omega = "idzorek",
                              view_confidences = view_confidences,
                              tau = tau,
                              risk_aversion = gamma,
                              risk_free_rate = rf_annual
                              )
   
    ## posterior returns and covariances
    posterior_rets = bl.bl_returns()
    posterior_cov = Σ # bl.bl_cov() if did not use Idzorek
    
    # ## debug
    # print(f"Date: {date}")
    # print("Posterior rets:\n", posterior_rets)
    # if all(r <= 0 for r in posterior_rets):
    #     print("All returns are non-positive!")
    
    
    ########## optimized portfolios ##########
    ##### Min Volatility #####

    #1) Min Volatility Long only
    ef_minvol_long = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0,1))
    w_minvol_long = ef_minvol_long.min_volatility()
    weights_minvol_long = ef_minvol_long.clean_weights()

    results_mv_long.append({"day": date, "weights": weights_minvol_long})
    
    #1.a) Min Volatility Long only constrained (0, .75))
    ef_minvol_long_c0_75 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, 0.75))
    w_minvol_long_c0_75 = ef_minvol_long_c0_75.min_volatility()
    weights_minvol_long_c0_75 = ef_minvol_long_c0_75.clean_weights()
    
    results_mv_long_c0_75.append({"day": date, "weights": weights_minvol_long_c0_75})
    
    #1.b) Min Volatility Long only constrained (0, .5))
    ef_minvol_long_c0_50 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, 0.5))
    w_minvol_long_c0_50 = ef_minvol_long_c0_50.min_volatility()
    weights_minvol_long_c0_50 = ef_minvol_long_c0_50.clean_weights()
    
    results_mv_long_c0_50.append({"day": date, "weights": weights_minvol_long_c0_50})
   
    #1.c) Min Volatility Long only constrained (0, .25))
    ef_minvol_long_c0_25 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, 0.25))
    w_minvol_long_c0_25 = ef_minvol_long_c0_25.min_volatility()
    weights_minvol_long_c0_25 = ef_minvol_long_c0_25.clean_weights()
    
    results_mv_long_c0_25.append({"day": date, "weights": weights_minvol_long_c0_25})
    
    
    
    #2) Min Volatility Shorting allowed
    ef_minvol = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-1,1)) 
    w_minvol = ef_minvol.min_volatility()
    weights_minvol = ef_minvol.clean_weights()
    
    results_mv.append({"day": date, "weights": weights_minvol})
    
    #2.a) Min Volatility Shorting allowed constrained (-.75, 1)
    ef_minvol_cm75_1 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-0.75,1)) 
    w_minvol_cm75_1 = ef_minvol_cm75_1.min_volatility()
    weights_minvol_cm75_1 = ef_minvol_cm75_1.clean_weights()
    
    results_mv_cm75_1.append({"day": date, "weights": weights_minvol_cm75_1})
    
    #2.b) Min Volatility Shorting allowed constrained (-.50, 1)
    ef_minvol_cm50_1 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-0.50,1)) 
    w_minvol_cm50_1 = ef_minvol_cm50_1.min_volatility()
    weights_minvol_cm50_1 = ef_minvol_cm50_1.clean_weights()
    
    results_mv_cm50_1.append({"day": date, "weights": weights_minvol_cm50_1})
    
    #2.c) Min Volatility Shorting allowed constrained (-.25, 1)
    ef_minvol_cm25_1 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-0.25,1)) 
    w_minvol_cm25_1 = ef_minvol_cm25_1.min_volatility()
    weights_minvol_cm25_1 = ef_minvol_cm25_1.clean_weights()
    
    results_mv_cm25_1.append({"day": date, "weights": weights_minvol_cm25_1})
    
    
    #2.d) Min Volatility Shorting allowed constrained (-.75, .75)
    ef_minvol_cm75_75 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-0.75,0.75)) 
    w_minvol_cm75_75 = ef_minvol_cm75_75.min_volatility()
    weights_minvol_cm75_75 = ef_minvol_cm75_75.clean_weights()
    
    results_mv_cm75_75.append({"day": date, "weights": weights_minvol_cm75_75})
    
    #2.e) Min Volatility Shorting allowed constrained (-.50, .50)
    ef_minvol_cm50_50 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-0.50,0.50)) 
    w_minvol_cm50_50 = ef_minvol_cm50_50.min_volatility()
    weights_minvol_cm50_50 = ef_minvol_cm50_50.clean_weights()
    
    results_mv_cm50_50.append({"day": date, "weights": weights_minvol_cm50_50})
    
    #2.f) Min Volatility Shorting allowed constrained (-.25, .25)
    ef_minvol_cm25_25 = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-0.25,0.25))
    w_minvol_cm25_25 = ef_minvol_cm25_25.min_volatility()
    weights_minvol_cm25_25 = ef_minvol_cm25_25.clean_weights()
    
    results_mv_cm25_25.append({"day": date, "weights": weights_minvol_cm25_25})    
    
    
    ##### Max Quadratic Utility #####
    
    # Quadratic utility Long-only 
    ef_quad_long = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0,1))
    we_quad_long = ef_quad_long.max_quadratic_utility(risk_aversion=gamma)
    w_quad_long = ef_quad_long.clean_weights()
    
    results_quad_long.append({"day": date, "weights": w_quad_long})
    
    
    # Quadratic utility Shorting allowed
    ef_quad = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(-1,1))
    we_quad = ef_quad.max_quadratic_utility(risk_aversion=gamma)
    w_quad = ef_quad.clean_weights()
    
    results_quad.append({"day": date, "weights": w_quad})
    
    
    ##### Max Sharpe #####
    
    # 3) Max Sharpe Long only
    ef_maxsha_long = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0,1), verbose=False, solver='SCS')
    #                                     weight_bounds=(0,1), verbose=False, solver='ECOS',
    #                                     solver_options={'max_iters':100,
    #                                                     "abstol": 1e-1,
    #                                                     "reltol": 1e-1,})
    try:
        w_maxsha_long = ef_maxsha_long.max_sharpe(risk_free_rate=rf_annual)
    except ValueError as e: 
        if "at least one of the assets must have an expected return exceeding the risk-free rate" in str(e):
            print(f"Skipping date {date} because no asset beats Rf")
            
            continue
        else:
            raise

    weights_maxsha_long = ef_maxsha_long.clean_weights()

    results_ms_long.append({"day": date, "weights": weights_maxsha_long})
    
    # 4) Max Sharpe Shorting allowed
    ef_maxsha = EfficientFrontier(posterior_rets, posterior_cov, 
                                        weight_bounds=(-1, 1), verbose=False, solver='SCS')
                                        # solver_options={'max_iter':100000,
                                        #                 "eps_abs": 1e-9,
                                        #                 "eps_rel": 1e-9,})
    try:
        w_maxsha = ef_maxsha.max_sharpe(risk_free_rate=rf_annual)
    except ValueError as e:
        if "at least one of the assets must have an expected return exceeding the risk-free rate" in str(e):
            print(f"Skipping date {date} because no asset beats Rf")
                
            continue
        else:
            raise
            
    weights_maxsha = ef_maxsha.clean_weights()

    results_ms.append({"day": date, "weights": weights_maxsha})


    ##### Priors #####
    # Max Sharpe long only
    ef_prior_ms_long = EfficientFrontier(pi, posterior_cov, weight_bounds=(0, 1))
    w_prior_ms_long = ef_prior_ms_long.max_sharpe(risk_free_rate=rf_annual)
    w_prior_ms_long = ef_prior_ms_long.clean_weights()
    
    results_prior_sharpe_long.append({"day": date, "weights": w_prior_ms_long})

    # Max Sharpe
    ef_prior_ms = EfficientFrontier(pi, posterior_cov, weight_bounds=(-1, 1))
    w_prior_ms = ef_prior_ms.max_sharpe(risk_free_rate=rf_annual)
    w_prior_ms = ef_prior_ms.clean_weights()
    
    results_prior_sharpe.append({"day": date, "weights": w_prior_ms})
    
    # Max quad l
    ef_prior_quad_long = EfficientFrontier(pi, posterior_cov, weight_bounds=(0, 1))
    w_prior_quad_long = ef_prior_quad_long.max_quadratic_utility(risk_aversion=gamma)
    w_prior_quad_long = ef_prior_quad_long.clean_weights()
    
    results_prior_quad_long.append({"day": date, "weights": w_prior_quad_long})
    
    # Max quad s
    ef_prior_quad = EfficientFrontier(pi, posterior_cov, weight_bounds=(-1, 1))
    w_prior_quad = ef_prior_quad.max_quadratic_utility(risk_aversion=gamma)
    w_prior_quad = ef_prior_quad.clean_weights()
    
    results_prior_quad.append({"day": date, "weights": w_prior_quad})

#=============================== save results =================================

# convert the dictionary to a dictionary-based DataFrame
# then convert dictionary-based to DataFrame (to have one column per ticker)

############### Min Volatility ###############
## Min Volatility Long only
df_weights_mv_long = pd.DataFrame(results_mv_long)
df_weights_mv_long = (df_weights_mv_long.assign(**pd.DataFrame(df_weights_mv_long["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Long only constrained (0, .75)
df_weights_mv_long_c0_75 = pd.DataFrame(results_mv_long_c0_75)
df_weights_mv_long_c0_75 = (df_weights_mv_long_c0_75.assign(**pd.DataFrame(df_weights_mv_long_c0_75["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Long only constrained (0, .5)
df_weights_mv_long_c0_50 = pd.DataFrame(results_mv_long_c0_50)
df_weights_mv_long_c0_50 = (df_weights_mv_long_c0_50.assign(**pd.DataFrame(df_weights_mv_long_c0_50["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Long only constrained (0, .25)
df_weights_mv_long_c0_25 = pd.DataFrame(results_mv_long_c0_25)
df_weights_mv_long_c0_25 = (df_weights_mv_long_c0_25.assign(**pd.DataFrame(df_weights_mv_long_c0_25["weights"].tolist())).drop(columns="weights")).set_index('day')



## Min Volatility Shorting allowed
df_weights_mv = pd.DataFrame(results_mv)
df_weights_mv = (df_weights_mv.assign(**pd.DataFrame(df_weights_mv["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Shorting allowed constrained (-.75, 1)
df_weights_mv_cm75_1 = pd.DataFrame(results_mv_cm75_1)
df_weights_mv_cm75_1 = (df_weights_mv_cm75_1.assign(**pd.DataFrame(df_weights_mv_cm75_1["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Shorting allowed constrained (-.50, 1)
df_weights_mv_cm50_1 = pd.DataFrame(results_mv_cm50_1)
df_weights_mv_cm50_1 = (df_weights_mv_cm50_1.assign(**pd.DataFrame(df_weights_mv_cm50_1["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Shorting allowed constrained (-.25, 1)
df_weights_mv_cm25_1 = pd.DataFrame(results_mv_cm25_1)
df_weights_mv_cm25_1 = (df_weights_mv_cm25_1.assign(**pd.DataFrame(df_weights_mv_cm25_1["weights"].tolist())).drop(columns="weights")).set_index('day')



## Min Volatility Shorting allowed constrained (-.75, .75)
df_weights_mv_cm75_75 = pd.DataFrame(results_mv_cm75_75)
df_weights_mv_cm75_75 = (df_weights_mv_cm75_75.assign(**pd.DataFrame(df_weights_mv_cm75_75["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Shorting allowed constrained (-.50, .50)
df_weights_mv_cm50_50 = pd.DataFrame(results_mv_cm50_50)
df_weights_mv_cm50_50 = (df_weights_mv_cm50_50.assign(**pd.DataFrame(df_weights_mv_cm50_50["weights"].tolist())).drop(columns="weights")).set_index('day')

## Min Volatility Shorting allowed constrained (-.25, .25)
df_weights_mv_cm25_25 = pd.DataFrame(results_mv_cm25_25)
df_weights_mv_cm25_25 = (df_weights_mv_cm25_25.assign(**pd.DataFrame(df_weights_mv_cm25_25["weights"].tolist())).drop(columns="weights")).set_index('day')


############## Max Quadratic utility ###############

# Max Quadratic Utility Long-only
df_weights_quad_long = pd.DataFrame(results_quad_long)
df_weights_quad_long = (df_weights_quad_long.assign(**pd.DataFrame(df_weights_quad_long["weights"].tolist())).drop(columns="weights")).set_index('day')

# Max Quadratic Utility Shorting allowed
df_weights_quad = pd.DataFrame(results_quad)
df_weights_quad = (df_weights_quad.assign(**pd.DataFrame(df_weights_quad["weights"].tolist())).drop(columns="weights")).set_index('day')


############## Max Sharpe ###############

## Max Sharpe Long only
df_weights_ms_long = pd.DataFrame(results_ms_long)
df_weights_ms_long = (df_weights_ms_long.assign(**pd.DataFrame(df_weights_ms_long["weights"].tolist())).drop(columns="weights")).set_index('day')

## Max Sharpe Shorting allowed
df_weights_ms = pd.DataFrame(results_ms)
df_weights_ms = (df_weights_ms.assign(**pd.DataFrame(df_weights_ms["weights"].tolist())).drop(columns="weights")).set_index('day')


############### Priors ###############
# Max Sharpe long-only
df_weights_prior_ms_long = pd.DataFrame(results_prior_sharpe_long)
df_weights_prior_ms_long = (df_weights_prior_ms_long.assign(**pd.DataFrame(df_weights_prior_ms_long["weights"].tolist())).drop(columns="weights")).set_index('day')

# Max Sharpe
df_weights_prior_ms = pd.DataFrame(results_prior_sharpe)
df_weights_prior_ms = (df_weights_prior_ms.assign(**pd.DataFrame(df_weights_prior_ms["weights"].tolist())).drop(columns="weights")).set_index('day')

# Max Quad long
df_weights_prior_quad_long = pd.DataFrame(results_prior_quad_long)
df_weights_prior_quad_long = (df_weights_prior_quad_long.assign(**pd.DataFrame(df_weights_prior_quad_long["weights"].tolist())).drop(columns="weights")).set_index('day')

# Max Quad short
df_weights_prior_quad = pd.DataFrame(results_prior_quad)
df_weights_prior_quad = (df_weights_prior_quad.assign(**pd.DataFrame(df_weights_prior_quad["weights"].tolist())).drop(columns="weights")).set_index('day')

#========================== save results to excel =============================
with pd.ExcelWriter('weights_all_strategies.xlsx') as writer:
    # min vol
    df_weights_mv_long.to_excel(writer, sheet_name='MinVol_Long')
    df_weights_mv_long_c0_75.to_excel(writer, sheet_name='MinVol_Long_c0_75')
    df_weights_mv_long_c0_50.to_excel(writer, sheet_name='MinVol_Long_c0_50')
    df_weights_mv_long_c0_25.to_excel(writer, sheet_name='MinVol_Long_c0_25')
    
    df_weights_mv.to_excel(writer, sheet_name='MinVol_Short')
    df_weights_mv_cm75_1.to_excel(writer, sheet_name='MinVol_Short_cm75_1')
    df_weights_mv_cm50_1.to_excel(writer, sheet_name='MinVol_Short_cm50_1')
    df_weights_mv_cm25_1.to_excel(writer, sheet_name='MinVol_Short_cm25_1')
    
    df_weights_mv_cm75_75.to_excel(writer, sheet_name='MinVol_Short_cm75_75')
    df_weights_mv_cm50_50.to_excel(writer, sheet_name='MinVol_Short_cm50_50')
    df_weights_mv_cm25_25.to_excel(writer, sheet_name='MinVol_Short_cm25_25')
    
    #quad
    df_weights_quad_long.to_excel(writer, sheet_name='quad_long')
    df_weights_quad.to_excel(writer, sheet_name='quad')

    # max sharpe
    df_weights_ms_long.to_excel(writer, sheet_name='MaxSharpe_Long')
    df_weights_ms.to_excel(writer, sheet_name='MaxSharpe_Short')
    
    # priors
    df_weights_prior_quad.to_excel(writer, sheet_name='Priors')

# df_confidences.to_pickle('df_confidences.pkl')