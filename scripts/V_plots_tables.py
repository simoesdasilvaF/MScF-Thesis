# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:57:41 2025

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import yfinance as yf
from cycler import cycler
from matplotlib.colors import ListedColormap
import seaborn as sns
import scipy.stats as st
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


#============================ load df =========================================
df_prices = pd.read_pickle('df_prices.pkl')
df_black_litterman = pd.read_pickle('df_black_litterman.pkl')
df_SCSSR = pd.concat([pd.read_pickle('df_SCSSR_1_with_sentiment.pkl'), 
                      pd.read_pickle('df_SCSSR_2_with_sentiment.pkl')])

# groupby day and ticker, aggregate by summing (same code block as in polarity_mc.py)
df_polarity = (df_SCSSR.groupby(["day", "ticker"]).agg(
    bullish_sent_sum=("bullish", "sum"),
    bearish_sent_sum=("bearish", "sum"),
    total_sent=("predicted_sentiment", "count"))
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

#======================= Polarity summary stats ===============================
## all days
# pivot so columns are tickers, rows are dates
df_pol_stats = df_polarity.pivot(index='day', columns='ticker', values='polarity_P_it')
df_msg = df_polarity.pivot(index='day', columns='ticker', values='total_sent')

df_msg.index = pd.to_datetime(df_msg.index)
df_msg = df_msg.loc[df_msg.index >= pd.Timestamp("2015-06-17")]

# compute sum. stats
desc = df_pol_stats.describe().T.loc[:, ['mean','std','min','25%','50%','75%','max']]
desc = desc.rename(columns={'50%':'median'})
# add skewness & kurtosis
desc['skew'] = df_pol_stats.skew()
desc['kurtosis'] = df_pol_stats.kurtosis()
# add % positive & negative
desc['% pos days'] = (df_pol_stats > 0).mean()*100
desc['% neg days'] = (df_pol_stats < 0).mean()*100


# compute stats for messages
mean_msgs = df_msg.mean().round(1)
median_msgs = df_msg.median().round(1)
std_msgs = df_msg.std().round(1)
min_msgs = df_msg.min()
max_msgs = df_msg.max()

# round and display
summary = desc.round(3)

#add
summary['mean # Tweets'] = mean_msgs
summary['median # Tweets'] = median_msgs
summary['std # Tweets'] = std_msgs
summary['min # Tweets'] = min_msgs
summary['max # Tweets'] = max_msgs


pol_cols = ['mean','std','min','25%','median','75%','max','skew','kurtosis','% pos days','% neg days']
msg_cols = ['mean # Tweets', 'median # Tweets', 'std # Tweets','min # Tweets','max # Tweets']

pol_tuples = [('Polarity', c) for c in pol_cols]
msg_tuples = [('Messages', c) for c in msg_cols]
new_cols = pd.MultiIndex.from_tuples(pol_tuples + msg_tuples)

summary.columns = new_cols

# round and display
print(summary)




## only trading days
# pivot so columns are tickers, rows are dates
df_pol_stats_tr = df_polarity.pivot(index='day', columns='ticker', values='polarity_P_it')
df_pol_stats_tr.index = pd.to_datetime(df_pol_stats_tr.index)
start_cut = pd.Timestamp("2015-06-17")
df_pol_stats_tr = df_pol_stats_tr.loc[df_pol_stats_tr.index >= start_cut]

df_msg_tr = df_polarity.pivot(index='day', columns='ticker', values='total_sent')
df_msg_tr.index = pd.to_datetime(df_msg_tr.index)
df_msg_tr = df_msg_tr.loc[df_msg_tr.index >= start_cut]


# compute sum. stats
desc_tr = df_pol_stats_tr.describe().T.loc[:, ['mean','std','min','25%','50%','75%','max']]
desc_tr = desc_tr.rename(columns={'50%':'median'})
# add skewness & kurtosis
desc_tr['skew'] = df_pol_stats_tr.skew()
desc_tr['kurtosis'] = df_pol_stats_tr.kurtosis()
# add % positive & negative
desc_tr['% pos days'] = (df_pol_stats_tr > 0).mean()*100
desc_tr['% neg days'] = (df_pol_stats_tr < 0).mean()*100

# compute stats for messages
mean_msgs_tr = df_msg_tr.mean().round(1)
median_msgs_tr = df_msg_tr.median().round(1)
std_msgs_tr = df_msg_tr.std().round(1)
min_msgs_tr = df_msg_tr.min()
max_msgs_tr = df_msg_tr.max()

# round and display
summary_tr = desc_tr.round(3)

#add
summary_tr['mean # Tweets'] = mean_msgs_tr
summary_tr['median # tweets'] = median_msgs_tr
summary_tr['std # Tweets'] = std_msgs_tr
summary_tr['min # Tweets'] = min_msgs_tr
summary_tr['max # Tweets'] = max_msgs_tr


pol_cols_tr = ['mean','std','min','25%','median','75%','max','skew','kurtosis','% pos days','% neg days']
msg_cols_tr = ['mean # Tweets','median # Tweets', 'std # Tweets','min # Tweets','max # Tweets']

pol_tuples_tr = [('Polarity', c) for c in pol_cols_tr]
msg_tuples_tr = [('Messages', c) for c in msg_cols_tr]
new_cols_tr = pd.MultiIndex.from_tuples(pol_tuples_tr + msg_tuples_tr)

summary_tr.columns = new_cols_tr


# generate the LaTeX table string
latex_str = summary_tr.to_latex(
    caption="Polarity Summary Statistics (Trading Sample)",
    label="tab:polarity_summary",
    index=True,
    na_rep="--",
    float_format="%.3f"
)

print(summary_tr)

# Utf-8
with open('polarity_summary.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)
print(latex_str)


#==================== Polarity distribution per ticker ========================
df_polarity["day"] = pd.to_datetime(df_polarity["day"])
df_prices = df_prices.set_index("day")
df_prices.index = pd.to_datetime(df_prices.index)

#### violin plots #####
# all data
tickers = sorted(df_polarity["ticker"].unique())
data = [df_polarity.loc[df_polarity["ticker"] == tic, "polarity_P_it"].dropna() for tic in tickers]

fig, ax = plt.subplots(figsize=(10, 5))
vp = ax.violinplot(data, showmedians=True, widths=.9)
bp = ax.boxplot(data, positions=np.arange(1, len(tickers) + 1),
                widths=0.25, patch_artist=False, showfliers=False)

ax.set_xticks(np.arange(1, len(tickers) + 1))
ax.set_xticklabels(tickers, rotation=45, ha="right")
ax.set_ylabel("Polarity $P_{i,t}$")
ax.set_title("Distribution of polarity $P_{i,t}$")
ax.axhline(0, ls="--", lw=0.8)
fig.tight_layout()
plt.savefig("pol_dist.pdf", bbox_inches="tight")
plt.show()

# starts when trading
start_cut = pd.Timestamp("2015-06-17")
df_pol = df_polarity.loc[df_polarity["day"] >= start_cut]
data = [df_pol.loc[df_pol["ticker"] == t, "polarity_P_it"].dropna()
        for t in tickers]
fig, ax = plt.subplots(figsize=(10, 5))
vp = ax.violinplot(data, showmedians=True, widths=.9)
bp = ax.boxplot(data, positions=np.arange(1, len(tickers) + 1),
                widths=0.25, patch_artist=False, showfliers=False)

ax.set_xticks(np.arange(1, len(tickers) + 1))
ax.set_xticklabels(tickers, rotation=45, ha="right")
ax.set_ylabel("Polarity $P_{i,t}$")
ax.set_title("Distribution of polarity $P_{i,t}$ from 2015-06-17")
ax.axhline(0, ls="--", lw=0.8)
fig.tight_layout()
plt.savefig("pol_dist_trading.pdf", bbox_inches="tight")
plt.show()

#### polarity vs price #####
tickers_pvp = ["AAPL", "TSLA", "GLD", "SPY", "AMD"]

for tic in tickers_pvp:
    price_series = df_prices[tic].dropna()
    pol_series = (df_polarity.query("ticker == @tic")
                  .set_index("day")["polarity_P_it"]
                  .reindex(price_series.index))
    
    vol_series = (df_polarity.query("ticker == @tic")
              .set_index("day")["total_sent"]
              .reindex(price_series.index))
    
    s = vol_series**0.5

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(price_series.index, price_series, lw=1.4, label="Adj. close")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{tic} adjusted close")

    ax2 = ax1.twinx()
    colors = np.where(pol_series >= 0, "#2E8B57", "#B22222")
    ax2.scatter(pol_series.index, pol_series, s=s, c=colors, alpha=0.2, edgecolors="none", label="Polarity $P_{i,t}$")
    ax2.set_ylabel("Polarity $P_{i,t}$")
    
    ax2.plot(pol_series.rolling(5).mean(), lw=0.8, color="#555555", alpha=0.7)

    ax1.set_title(f"{tic}: polarity vs. price level")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
    
tickers_pvp = ["AAPL", "TSLA", "GLD", "SPY", "AMD"]

for tic in tickers_pvp:
    price_series = df_prices[tic].dropna()
    pol_series = (df_polarity.query("ticker == @tic")
                  .set_index("day")["polarity_P_it"]
                  .reindex(price_series.index))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(price_series.index, price_series)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{tic} adjusted close")

    ax2 = ax1.twinx()
    ax2.plot(pol_series.index, pol_series, alpha=0.4)
    ax2.set_ylabel("Polarity $P_{i,t}$")

    ax1.set_title(f"{tic}: polarity vs. price level")
    fig.tight_layout()
    plt.show()
    
    

#=============================== weights ======================================
## stacked plot
excel_file = 'weights_all_strategies.xlsx'
sheets = pd.ExcelFile(excel_file).sheet_names
weights_raw = {s: pd.read_excel(excel_file, sheet_name=s, index_col=0, parse_dates=True).round(2) for s in sheets}

daily_index = df_prices.index[df_prices.index >= start_cut]
weights_daily = {s: w.reindex(daily_index, method="ffill") for s, w in weights_raw.items()}

strat_long = {
    "Min Vol (long‑only)": "MinVol_Long",
    "Max Quad. utility (long‑only)": "quad_long",
    "Max Sharpe (long‑only)":         "MaxSharpe_Long",
}

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

for ax, (title, sheet) in zip(axes, strat_long.items()):
    df = weights_daily[sheet]
    df.plot.area(ax=ax, stacked=True, linewidth=0, alpha=0.9, legend=False)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=.3, linestyle="--")

handles, labels = axes[-1].get_legend_handles_labels()

fig.legend(handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=8,
            frameon=False,
            title="Ticker")

plt.tight_layout(rect=[0, 0.002, 1, 1])
plt.savefig("weights_stacked.pdf", bbox_inches="tight")
plt.show()





## heatmaps
strats_long = ["MinVol_Long", "quad_long", "MaxSharpe_Long", "Priors"]
n_months = 36                                             # look‑back window

# palette
nord_seq = ListedColormap([
    "#5E81AC", "#81A1C1", "#88C0D0", "#8FBCBB", "#A3BE8C",
    "#BFD6A0", "#EBCB8B", "#D8BBA5", "#BF616A", "#B48EAD"
])

strat_long = {
    "MinVol_Long"    : "Min Vol. (long‑only)",
    "quad_long"      : "Max Quad. utility (long‑only)",
    "MaxSharpe_Long" : "Max Sharpe (long‑only)",
}


for strat in strats_long:
    df_w = weights_raw[strat]
    end_date = df_w.index.max()
    start_date = end_date - pd.DateOffset(months=n_months)
    df_last = df_w.loc[start_date:end_date]

    # tickers with the most magnitude on top
    df_last = df_last[df_last.mean().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        df_last.T, cmap=nord_seq,
        vmin=0, vmax=df_last.values.max(),
        cbar_kws=dict(label="Weight"),
        linewidths=.2, linecolor="#ECEFF4", ax=ax
    )

    ax.set_title(f"{strat_long.get(strat, strat)} – last {n_months} months", pad=8, fontsize=15)
    ax.set_xlabel(""); ax.set_ylabel("Ticker")

    ticks = pd.date_range(start=start_date, end=end_date, freq="2MS")
    ax.set_xticks(np.searchsorted(df_last.index, ticks))
    ax.set_xticklabels([d.strftime("%Y‑%m‑%d") for d in ticks],
                       rotation=45, ha="right")

    ax.grid(which="both", axis="both",
            color="#ECEFF4", linestyle="--", linewidth=.4)

    plt.tight_layout()
    plt.show()
    
    
    
    
    
strats_short = ["MinVol_Short", "quad", "MaxSharpe_Short", "Priors"]
n_months = 36

# palette
nord_seq = ListedColormap([
    "#5E81AC", "#81A1C1", "#88C0D0", "#8FBCBB", "#A3BE8C",
    "#BFD6A0", "#EBCB8B", "#D8BBA5", "#BF616A", "#B48EAD"
])

strat_short= {
    "MinVol_Short"    : "Min Vol (short-allowed)",
    "quad"      : "Max Quad. utility (short-allowed)",
    "MaxSharpe_Short" : "Max Sharpe (short-allowed)",
}


for strat in strats_short:
    df_w = weights_raw[strat]
    end_date = df_w.index.max()
    start_date = end_date - pd.DateOffset(months=n_months)
    df_last = df_w.loc[start_date:end_date]

    # tickers with the most magnitude on top
    df_last = df_last[df_last.mean().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        df_last.T, cmap=nord_seq,
        vmin=-.5, vmax=df_last.values.max(),
        cbar_kws=dict(label="Weight"),
        linewidths=.2, linecolor="#ECEFF4", ax=ax
    )

    ax.set_title(f"{strat_short.get(strat,strat)} – last {n_months} months", pad=8, fontsize=15)
    ax.set_xlabel(""); ax.set_ylabel("Ticker")

    ticks = pd.date_range(start=start_date, end=end_date, freq="2MS")
    ax.set_xticks(np.searchsorted(df_last.index, ticks))
    ax.set_xticklabels([d.strftime("%Y‑%m‑%d") for d in ticks],
                       rotation=45, ha="right")

    ax.grid(which="both", axis="both",
            color="#ECEFF4", linestyle="--", linewidth=.4)

    plt.tight_layout()
    plt.show()



#========================= Euclidean distance =================================

# uncertainty point
uncertainty_point = (0.5, 0.5)

# feasible line (bullish + bearish = 1)
t = np.linspace(0, 1, 200)
f_plus = t
f_minus = 1 - t

# random point on the line
np.random.seed(5)
rand_t = np.random.rand()
rand_point = (rand_t, 1 - rand_t)

# distance and confidence
dist = np.hypot(rand_point[0] - uncertainty_point[0], rand_point[1] - uncertainty_point[1])
max_dist = np.sqrt(0.5)
confidence = dist / max_dist

# figure
plt.figure(figsize=(10, 5))
# square even if the sum of f is not 1
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linestyle='--', linewidth=1)

# plottings (line, uncertainty, random, line random from unecrtainty)
plt.plot(f_plus, f_minus, linewidth=2, label='Line where $f^{+}_{v,t}$ $+$ $f^{+}_{v,t}$ = 1')
plt.scatter(*uncertainty_point, marker='h', s=100, label='Complete uncertainty (0.5, 0.5)')
plt.scatter(*rand_point, s=50, c='#D08770', label=f'Random point ({rand_point[0]:.2f}, {rand_point[1]:.2f})')
plt.plot([uncertainty_point[0], rand_point[0]], [uncertainty_point[1], rand_point[1]],
         linestyle='-', linewidth=3,  alpha=0.7)

# annotation of the distance (set in the middle of the line) & confidence
mid_x = (uncertainty_point[0] + rand_point[0]) / 2
mid_y = (uncertainty_point[1] + rand_point[1]) / 2
plt.text(mid_x, mid_y, f'd = {dist:.3f}', fontsize=8, ha='left', va='bottom')

plt.text(rand_point[0] + 0.22, rand_point[1] - 0.19, f'→ confidence = {confidence:.2f}', fontsize=7)

# circle
circle = Circle(uncertainty_point, max_dist, fill=False, linestyle='--', linewidth=2, label='max distance')
plt.gca().add_patch(circle)

# formatting
plt.title("Euclidean distance: confidence measure")
plt.xlabel("$f^{+}_{v,t}$")
plt.ylabel("$f^{-}_{v,t}$")
plt.axis('equal')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True)
plt.legend(loc='upper right')
plt.show()


#============================== Classifier ====================================

summary_scssr = (df_SCSSR.groupby("predicted_sentiment").agg(
          count=("predicted_sentiment", "size"),
          share_pct=("predicted_sentiment", lambda x: 100*len(x)/len(df_SCSSR)),
          mean_bull=("bullish", "mean"),
          std_bull=("bullish", "std"),
          mean_bear=("bearish", "mean"),
          std_bear=("bearish", "std"),
          mean_neut=("neutral", "mean"),
          std_neut=("neutral", "std"),
          ).reset_index()
)

summary_scssr.to_latex("sentiment_summary.tex", index=False,
                       caption="Classification: descriptive statistics",
                       label="tab:sentiment_stats",
                       column_format="lrrrrrrrr")




base_blue = "#5078B8"
cmap = sns.light_palette(base_blue, n_colors=256, as_cmap=True)


# mask non-labeled
mask_lbl = df_SCSSR["sent"].isin([-1, 1])
df_eval = df_SCSSR.loc[mask_lbl].copy()

mapping = {1: "bullish", -1: "bearish"}
y_true = df_eval["sent"].map(mapping)
y_pred = df_eval["predicted_sentiment"]

labels_true = ["bullish", "bearish"]
labels_pred = ["bullish", "bearish", "neutral"]

cm_2x3 = (pd.crosstab(y_true, y_pred, dropna=False)
            .reindex(index=labels_true, columns=labels_pred)
            .fillna(0).astype(int))

annot_str = (cm_2x3.applymap(lambda x: f"{x:,}".replace(",", "'")))


sns.heatmap(cm_2x3, annot=annot_str, fmt='', linewidths=0, cbar=False, cmap=cmap,
            xticklabels=['bull','bear','neut'],
            yticklabels=['bull','bear'])
plt.title("Classification: confusion matrix")
plt.xlabel("Model classification"); plt.ylabel("User-labeled")
plt.tight_layout(); plt.savefig("cm.pdf", dpi=300)
plt.show()

print("\n2×3 confusion matrix counts:\n", cm_2x3)


pct_zeros = (~mask_lbl).mean() * 100
print(f"{pct_zeros:.1f}% of messages have no user tag (sent == 0).")

print("\nSklearn classification_report:")
print(classification_report(y_true, y_pred, labels=labels_true, zero_division=0))

acc = accuracy_score(y_true, y_pred)
print(f"Overall accuracy: {acc:.3%}")   




#============================ Realized vs Expected ============================

df_bl = pd.read_pickle('df_black_litterman.pkl')

rows = []
for ticker, column in df_black_litterman.groupby('ticker'):
    pred = column['annualized_predicted_log_return']
    real = column['annualized_real_log_return']
    rows.append({
        "Ticker": ticker,
        "ρ(Pred, Real) (%)": pred.corr(real)*100,
        "MSE (%)": mean_squared_error(real, pred),
        "MAE (%)": mean_absolute_error(real, pred),
        "Hit Ratio (%)": (pred.mul(real).gt(0)).mean()*100
    })

summary_pvp = (pd.DataFrame(rows)
             .set_index("Ticker")
             .round(3)
             .sort_values("ρ(Pred, Real) (%)", ascending=False))

summary_pvp = summary_pvp.T

# print( summary_pvp['ρ(Pred, Real)'].median())


plt.figure(figsize=(10,5))
plt.scatter(df_bl['annualized_predicted_log_return'],
            df_bl['annualized_real_log_return'],
            alpha=0.4, s=10,
            color="#88C0D0", label='Intersections')
# 45° line
lims = np.array([df_bl[['annualized_predicted_log_return',
                        'annualized_real_log_return']].min().min(),
                  df_bl[['annualized_predicted_log_return',
                        'annualized_real_log_return']].max().max()])
plt.plot(lims, lims, '--', color="#BF616A", lw=1, label='45° line')

plt.title("Returns: predicted vs realized 5-Day annualized log-returns")
plt.xlabel("Predicted")
plt.ylabel("Realized")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pred_vs_real_returns.pdf", bbox_inches="tight")
plt.show()

# ============================ Confidences ====================================
df_confidences = pd.read_pickle('df_confidences.pkl')

df_confidences.describe().T
