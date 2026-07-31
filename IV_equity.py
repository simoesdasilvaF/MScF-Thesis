# -*- coding: utf-8 -*-
"""
Created on Sat April 17 08:42:24 2025

@author: Flavio Simoes da Silva

Purpose: Master's Thesis
    Portfolio Optimization and NLP:
    Algorithmic trading with Black-Litterman and views from stock market sentiment coming from StockTwits.
   
    Supervisor: M.-A Divernois, Ph.D
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from cycler import cycler

#====================== Style of the plot(s) for LaTeX ========================
plt.rcParams.update({
    # latex
    "text.usetex"           : False,
    "text.latex.preamble"   : r"\usepackage{amsmath}\usepackage{siunitx}",

    # save
    "figure.dpi"            : 300,
    "savefig.dpi"           : 300,
    "savefig.format"        : "pdf",
    "pdf.fonttype"          : 42,            # embed Type 42 (TrueType) fonts

    # font
    "font.family"           : "serif",
    "font.serif"            : ['DejaVu Serif'],
    "font.size"             : 10,            # base font size
    "axes.titlesize"        : 12,
    "axes.labelsize"        : 10,
    "legend.fontsize"       : 9,
    "xtick.labelsize"       : 9,
    "ytick.labelsize"       : 9,

    # if maths
    "mathtext.fontset"      : "cm", 
    "mathtext.rm"           : "serif",
    "mathtext.it"           : "serif:italic",
    "mathtext.bf"           : "serif:bold",

    # lines
    "lines.linewidth"       : 1.2,
    "axes.linewidth"        : 0.8,
    "xtick.direction"       : "in",
    "ytick.direction"       : "in",
    "xtick.major.size"      : 4,
    "ytick.major.size"      : 4,
    "xtick.minor.visible"   : True,
    "ytick.minor.visible"   : True,

    # grid
    "axes.grid"             : True,
    "grid.color"            : "#BBBBBB",
    "grid.linestyle"        : "--",
    "grid.alpha"            : 0.3,
})

# extended color cycle
plt.rcParams["axes.prop_cycle"] = cycler(color=[
    "#5E81AC", "#A3BE8C", "#EBCB8B", "#BF616A", "#88C0D0",
    "#81A1C1", "#D08770", "#B48EAD", "#EBCB8B", "#A3BE8C"
])

# dates
mdates.YearLocator()



#================= import necessary files for Equity curve ====================

df_prices = pd.read_pickle('df_prices.pkl')

## returns
df_returns = df_prices.set_index('day')
df_returns = df_returns.sort_index(axis=1)

# compute daily log returns
df_returns = np.log(df_returns) - np.log(df_returns.shift(1))
df_returns = df_returns.dropna()


#================= Load strategies weights from excel file ====================

excel_file = 'weights_all_strategies.xlsx'

# all strategy sheets
all_sheets = ['MinVol_Long', 'MinVol_Long_c0_75', 'MinVol_Long_c0_50', 'MinVol_Long_c0_25',
              'MinVol_Short', 'MinVol_Short_cm75_1', 'MinVol_Short_cm50_1', 'MinVol_Short_cm25_1',
              'MinVol_Short_cm75_75', 'MinVol_Short_cm50_50', 'MinVol_Short_cm25_25',
              'quad_long' ,'quad','MaxSharpe_Long', 'MaxSharpe_Short', 'Priors'
              ]

# send all strategy weight sheets into a dictionary
strategy_weights_all = {}
for sheet in all_sheets:
    strategy_weights_all[sheet] = pd.read_excel(excel_file, sheet_name=sheet, 
                                                 index_col=0, parse_dates=True)

## subsets
# only the four MinVol_Long strategies
minvol_long_sheets = ['MinVol_Long', 'MinVol_Long_c0_75', 'MinVol_Long_c0_50', 'MinVol_Long_c0_25']
strategy_weights_minvol_long = {sheet: strategy_weights_all[sheet] for sheet in minvol_long_sheets} 

# only the MinVol_short strategies
minvol_short_sheets = ['MinVol_Short', 'MinVol_Short_cm75_1', 'MinVol_Short_cm50_1', 'MinVol_Short_cm25_1', 
                       'MinVol_Short_cm75_75', 'MinVol_Short_cm50_50', 'MinVol_Short_cm25_25']
strategy_weights_minvol_short = {sheet: strategy_weights_all[sheet] for sheet in minvol_short_sheets}


#================= Forward-fill weights to daily frequency ====================
# Forward-fill for all strategies
strategy_weights_all_daily = {}
for strat, df_w in strategy_weights_all.items():
    strategy_weights_all_daily[strat] = df_w.reindex(df_returns.index, method='ffill')

# Forward-fill for MinVol_Long strategies
strategy_weights_minvol_long_daily = {}
for strat, df_w in strategy_weights_minvol_long.items():
    strategy_weights_minvol_long_daily[strat] = df_w.reindex(df_returns.index, method='ffill')
    
# Forward-fill for MinVol_short strategies
strategy_weights_minvol_short_daily = {}
for strat, df_w in strategy_weights_minvol_short.items():
    strategy_weights_minvol_short_daily[strat] = df_w.reindex(df_returns.index, method='ffill')


#==================== Set benchmarks up (SPY and URTH) ========================

# SPY (S&P 500) and URTH (a proxy for MSCI World)
start_date = df_returns.index.min().strftime("%Y-%m-%d")
end_date = df_returns.index.max().strftime("%Y-%m-%d")

spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=False)['Adj Close']
urth = yf.download("URTH", start=start_date, end=end_date, auto_adjust=False)['Adj Close']

# benchmark daily returns and equity curves (starting at 1)
spy_returns = (np.log(spy) - np.log(spy.shift())).dropna()
spy_equity = np.exp(spy_returns.cumsum())
spy_ret = (np.exp(spy_returns) - 1).reindex(df_returns.index).fillna(0) # for performance measure

urth_returns = np.log(urth) - np.log(urth.shift()).dropna()
urth_equity = np.exp(urth_returns.cumsum())
    

#========================== Compute equity curves =============================

def compute_equity_curve(weights_df, returns_df):
    # daily portfolio log return = sum(weights * asset log returns)
    port_log_returns = (weights_df * returns_df).sum(axis=1)
    cum_log_returns = port_log_returns.cumsum()
    equity_curve = np.exp(cum_log_returns)  # starting value is 1
    return equity_curve

# equity curves for all strategies
equity_curves_all = {}
for strat, df_w in strategy_weights_all_daily.items():
    equity_curves_all[strat] = compute_equity_curve(df_w, df_returns)

# equity curves for the MinVol_Long subset
equity_curves_minvol_long = {}
for strat, df_w in strategy_weights_minvol_long_daily.items():
    equity_curves_minvol_long[strat] = compute_equity_curve(df_w, df_returns)

# equity curves for the MinVol_Long subset
equity_curves_minvol_short = {}
for strat, df_w in strategy_weights_minvol_short_daily.items():
    equity_curves_minvol_short[strat] = compute_equity_curve(df_w, df_returns)


#============================ Plot equity curves ==============================

#### unconstrained comparison to benchamrks
# Palette for global startegies
strategy_palette = {
    "MinVol_Short"     : "#5E81AC",
    "MinVol_Long"      : "#88C0D0",
    "quad_long"        : "#A3BE8C",
    "quad"             : "#EBCB8B",
    "MaxSharpe_Long"   : "#BF616A",
    "MaxSharpe_Short"  : "#8F3B35",
    "Priors"           : "#5D81DC"
}

## strategies vs SPY & URTH
fig, ax = plt.subplots(figsize=(10, 5))
# benchmarks
ax.plot(spy_equity.index, spy_equity, label="SPY (S&P 500)", color="black", lw=2.5)
ax.plot(urth_equity.index, urth_equity, label="URTH (MSCI World)", color="#777777", lw=2.5)

# unconstrained strategies
selected = ["MinVol_Long", "MinVol_Short", "quad_long", "quad", "MaxSharpe_Long", "MaxSharpe_Short", "Priors"]

for strat in selected:
    eq = equity_curves_all[strat]
    ax.plot(eq.index, eq,
            label=strat,
            color=strategy_palette[strat],
            lw=1.8, alpha=0.9)

# line at 1
ax.axhline(1.0, color="#555555", lw=0.8, linestyle="--")

# style
ax.set_title("Equity Curves: Selected Strategies vs. Benchmarks")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Start = 1)")
# date
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
# legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
# save & plot
plt.tight_layout()
# plt.savefig("equity_unconstrained_priors.pdf", bbox_inches="tight")
plt.show()





#### Min vol long-only
constraint_palette = {
    "MinVol_Long"       : {"color": "#88C0D0", "alpha": 0.9},  # unconstrained
    "MinVol_Long_c0_75" : {"color": "#88C0D0", "alpha": 0.7},  # ≤75%
    "MinVol_Long_c0_50" : {"color": "#88C0D0", "alpha": 0.5},  # ≤50%
    "MinVol_Long_c0_25" : {"color": "#88C0D0", "alpha": 0.3},  # ≤25%
}

## Min vol long comparing constraints
fig, ax = plt.subplots(figsize=(10, 5))

for strat, params in constraint_palette.items():
    eq = equity_curves_all[strat]
    ax.plot(eq.index, eq,
            label=strat,
            color=params["color"],
            lw=2.0,
            alpha=params["alpha"])

# line at 1
ax.axhline(1.0, color="#555555", lw=0.8, linestyle="--")

# style
ax.set_title("Equity Curves: MinVol_Long Variants (Different Constraints)")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Start = 1)")
# date
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
# legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
# save & plot
plt.tight_layout()
# plt.savefig("equity_minvollong_constraints.pdf", bbox_inches="tight")
plt.show()






#### Min vol short allowed
constraint_palette_short = {
    "MinVol_Short"        : {"color": "#5E81AC", "alpha": 0.9},  # unconstrained
    "MinVol_Short_cm75_1" : {"color": "#5E81AC", "alpha": 0.7},  # −75/+100%
    "MinVol_Short_cm50_1" : {"color": "#5E81AC", "alpha": 0.5},  # −50/+100%
    "MinVol_Short_cm25_1" : {"color": "#5E81AC", "alpha": 0.3},  # −25/+100%
    "MinVol_Short_cm75_75": {"color": "#5E81AC", "alpha": 0.6},  # −75/+75%
    "MinVol_Short_cm50_50": {"color": "#5E81AC", "alpha": 0.4},  # −50/+50%
    "MinVol_Short_cm25_25": {"color": "#5E81AC", "alpha": 0.2},  # −25/+25%
}

## Min vol comparing constraints
fig, ax = plt.subplots(figsize=(10, 5))

for strat, params in constraint_palette_short.items():
    eq = equity_curves_all[strat]
    ax.plot(
        eq.index,
        eq,
        label=strat,
        color=params["color"],
        lw=2.0,
        alpha=params["alpha"]
        )
    
# line at 1
ax.axhline(1.0, color="#555555", lw=0.8, linestyle="--")

# style
ax.set_title("Equity Curves: MinVol_Short Variants (Different Constraints)")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Start = 1)")
# date
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
# legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
# save & plot
plt.tight_layout()
# plt.savefig("equity_minvol_short_constraints.pdf", bbox_inches="tight")
plt.show()






#### MaxSharpe & quad
constraint_palette_ms = {
    "quad_long"        : {"color": "#A3BE8C", "alpha": 0.9},
    "quad"             : {"color": "#EBCB8B", "alpha": 0.9},  # flax (quad)
    "MaxSharpe_Long"   : {"color": "#BF616A", "alpha": 0.9},  # muted red
#    "MaxSharpe_Short"  : {"color": "#8F3B35", "alpha": 0.9},  # darker red
    "Priors"   : {"color": "#5D81DC", "alpha": 0.9},  # muted red
}

## Sharpes vs quad
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(spy_equity.index, spy_equity, label="SPY (S&P 500)", color="black", lw=2.5)

for strat, params in constraint_palette_ms.items():
    eq = equity_curves_all[strat]
    ax.plot(
        eq.index,
        eq,
        label=strat,
        color=params["color"],
        lw=2.0,
        alpha=params["alpha"]
        )

# line at 1
ax.axhline(1.0, color="#555555", lw=0.8, linestyle="--")

# style
ax.set_title("Equity Curves: Quad & MaxSharpe Strategies")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Start = 1)")
# date
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
# legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
# save & plot
plt.tight_layout()
# plt.savefig("equity_maxsharpe_quad.pdf", bbox_inches="tight")
plt.show()


#### MaxSharpe & quad
priors_vs_post = {
    "quad_long"        : {"color": "#A3BE8C", "alpha": 0.9},
    "quad"             : {"color": "#EBCB8B", "alpha": 0.9},  # flax (quad)
    "MaxSharpe_Long"   : {"color": "#BF616A", "alpha": 0.9},  # muted red
    "MaxSharpe_Short"  : {"color": "#8F3B35", "alpha": 0.9},  # darker red
    "Priors"           : {"color": "#5D81DC", "alpha": 0.9},  # blue
}

## Sharpes vs quad
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(spy_equity.index, spy_equity, label="SPY (S&P 500)", color="black", lw=2.5)

for strat, params in priors_vs_post.items():
    eq = equity_curves_all[strat]
    ax.plot(
        eq.index,
        eq,
        label=strat,
        color=params["color"],
        lw=2.0,
        alpha=params["alpha"]
        )

# line at 1
ax.axhline(1.0, color="#555555", lw=0.8, linestyle="--")

# style
ax.set_title("Equity Curves: Priors and Posteriors")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Start = 1)")
# date
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
# legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
# save & plot
plt.tight_layout()
# plt.savefig("prior_vs_posterior.pdf", bbox_inches="tight")
plt.show()

#============================ Performance measures ============================

# set index and parameters
df_prices.set_index('day', inplace=True)
rf_annual = 0.02
trading_days = 252

# create list for results per strategy
rows = []

# loop performance measure for each strategy
for strat in selected:
    weights = pd.read_excel(excel_file, sheet_name=strat, index_col=0, parse_dates=True)
    weights = weights.reindex(df_returns.index, method='ffill').fillna(0)
    
    # Portfolio daily log-ret & convert to arithmetic
    ptfl_log_rets = (weights * df_returns).sum(axis=1)
    ptfl_rets = np.exp(ptfl_log_rets) - 1
    
    # start when trading 
    start = "2015-06-17"
    spy_ret = spy_ret.squeeze() # 1-column DataFrame into a Series
    ptfl_rets=ptfl_rets.loc[start:]
    spy_ret=spy_ret.loc[start:]

    # Annualized return & volatility
    ann_ret = ptfl_rets.mean() * trading_days
    ann_vol = ptfl_rets.std() * np.sqrt(trading_days)
    
    # Sharpe ratio
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol != 0 else np.nan
    
    # Sortino ratio (downside deviation)
    neg = ptfl_rets[ptfl_rets < 0]
    down_std = neg.std() * np.sqrt(trading_days)
    sortino = (ann_ret - rf_annual) / down_std if down_std > 0 else np.nan
    
    # Equity curve and drawdowns
    equity = (1 + ptfl_rets).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity / rolling_max) - 1
    max_dd = drawdowns.min()
    
    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    
    # Information ratio (SPY)
    active_ret = ptfl_rets - spy_ret
    ann_active = active_ret.mean() * trading_days
    tracking_error = active_ret.std() * np.sqrt(trading_days)
    info_ratio = ann_active / tracking_error

    # performance metrics
    rows.append({
        "Strategy": strat,
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Info Ratio (vs SPY)":  info_ratio
        })
    
    # add SPY itself
    ann_ret_spy = spy_ret.mean() * trading_days
    ann_vol_spy = spy_ret.std()  * np.sqrt(trading_days)
    sharpe_spy  = (ann_ret_spy - rf_annual) / ann_vol_spy
    eq_spy      = (1 + spy_ret).cumprod()
    max_dd_spy  = (eq_spy / eq_spy.cummax() - 1).min()
    calmar_spy  = ann_ret_spy / abs(max_dd_spy)
    neg_spy = spy_ret[spy_ret < 0]
    down_std_spy = neg_spy.std() * np.sqrt(trading_days)
    sortino_spy = (ann_ret_spy - rf_annual) / down_std_spy if down_std_spy > 0 else np.nan

rows.append({
    "Strategy": "SPY (benchmark)",
    "Annualized Return": ann_ret_spy,
    "Annualized Volatility": ann_vol_spy,
    "Sharpe Ratio": sharpe_spy,
    "Sortino Ratio": sortino_spy,
    "Max Drawdown": max_dd_spy,
    "Calmar Ratio": calmar_spy,
    "Info Ratio (vs SPY)": np.nan
    })
    
# create DataFrame
df_perf = pd.DataFrame(rows).set_index("Strategy").round(3)

# rename
name_map = {
    "MinVol_Long"      : "Min Vol. (long-only)",
    "MinVol_Short"     : "Min Vol. (short-allowed)",
    "quad_long"        : "Max Quadratic utility (long-only)",
    "quad"             : "Max Quadratic utility (short-allowed)",
    "MaxSharpe_Long"   : "Max Sharpe (long-only)",
    "MaxSharpe_Short"  : "Max Sharpe (short-allowed)",
    "SPY (benchmark)"  : "SPY (benchmark)",
    "Priors"           : "Priors"
    }

df_perf = df_perf.rename(index=name_map)

# reorder
new_order = [
    "SPY (benchmark)",
    "Min Vol. (short-allowed)",
    "Min Vol. (long-only)",
    "Max Sharpe (short-allowed)",
    "Max Sharpe (long-only)",
    "Max Quadratic utility (short-allowed)",
    "Max Quadratic utility (long-only)",
    "Priors"
    ]
df_perf = df_perf.reindex(new_order)


################################ LaTeX ########################################
# I then changed manually some details to fit best

latex_str = df_perf.to_latex(
    caption="Performance measures: Strategies and SPY (Benchmark)",
    label="tab:risk_return_summary",
    index=True,
    na_rep="--",
    float_format="%.3f"
)

# encoding='utf-8'
with open('risk_return_summary.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)



