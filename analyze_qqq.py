"""Yearly consistency + alpha/beta regression for the QQQ ORB backtest."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from orb_backtest import load_data, run_backtest, buy_hold_curve

df = load_data("QQQ")
trades_df, equity_df = run_backtest(df)
bh = buy_hold_curve(df)

# --- Yearly consistency ---
trades_df["year"] = pd.to_datetime(trades_df.date).dt.year
equity_df["year"] = equity_df.index.year

print("=== Yearly breakdown (QQQ ORB) ===")
rows = []
for yr, g in trades_df.groupby("year"):
    eq_yr = equity_df[equity_df.year == yr].equity
    if len(eq_yr) < 2:
        continue
    yr_return = eq_yr.iloc[-1] / eq_yr.iloc[0] - 1
    rets = eq_yr.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else np.nan
    running_max = eq_yr.cummax()
    mdd = ((eq_yr - running_max) / running_max).min()
    rows.append({
        "year": yr, "trades": len(g), "win_rate": (g.pnl > 0).mean(),
        "avg_R": g.pnl_R.mean(), "return": yr_return, "sharpe": sharpe, "mdd": mdd,
    })
yearly = pd.DataFrame(rows).set_index("year")
pd.set_option("display.float_format", lambda x: f"{x:.2f}")
print(yearly.to_string())
print(f"\nProfitable years: {(yearly['return'] > 0).sum()} / {len(yearly)}")

# --- Alpha / beta regression vs QQQ buy & hold ---
strat_by_date = equity_df.equity.copy()
strat_by_date.index = strat_by_date.index.date
strat_ret = strat_by_date.pct_change().dropna()
bh_ret = bh.pct_change().dropna()
joined = pd.DataFrame({"strat": strat_ret, "bh": bh_ret}).dropna()
X = sm.add_constant(joined["bh"])
model = sm.OLS(joined["strat"], X).fit()
alpha_daily = model.params["const"]
beta = model.params["bh"]
alpha_annual = (1 + alpha_daily) ** 252 - 1
print(f"\n=== Alpha/Beta regression (QQQ ORB vs QQQ Buy&Hold) ===")
print(f"Annualized alpha : {alpha_annual:.1%}  (p-value: {model.pvalues['const']:.4f})")
print(f"Beta             : {beta:.3f}  (p-value: {model.pvalues['bh']:.4f})")
print(f"R-squared        : {model.rsquared:.3f}")
