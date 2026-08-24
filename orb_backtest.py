"""
ORB (Opening Range Breakout) backtest, adapted from:
Zarattini & Aziz (2023/2025), "Can Day Trading Really Be Profitable?"

Runs on 5m data downloaded from MetaApi (data_<SYMBOL>.csv), for a CFD/Darwin
account instead of the paper's US-stock ETF account:
  - LEVERAGE default 20x (typical retail CFD index leverage) instead of the
    paper's 4x stock margin.
  - Cost model uses each trade's actual quoted spread (from the broker feed)
    instead of a flat per-share commission.
  - Same entry/stop/target mechanics as the paper: first 5-min candle of the
    9:30 ET session sets the range; enter at the open of the second candle in
    the direction of the first candle; stop at the extreme of the first
    candle; target 10R or liquidate at end of day (16:00 ET).
"""
import sys
import numpy as np
import pandas as pd
from orb_engine import simulate_day, RISK_PCT, LEVERAGE, TARGET_R

ACCOUNT_START = 25_000.0


def load_data(symbol):
    df = pd.read_csv(f"data_{symbol}.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index = df.index.tz_convert("America/New_York")
    df["date"] = df.index.date
    # point size heuristic from quoted price precision, to convert the
    # broker's 'spread' field (in points) into a price-unit cost. Flagged as
    # an approximation since symbol digits weren't independently confirmed.
    sample = df["close"].iloc[:200].astype(str)
    decimals = sample.str.split(".").str[1].str.len().median()
    point = 10 ** (-decimals) if pd.notna(decimals) else 0.01
    df["spread_price"] = df["spread"] * point
    return df


def run_backtest(df):
    capital = ACCOUNT_START
    equity_curve = []
    trades = []

    for day, g in df.groupby("date"):
        g = g.between_time("09:30", "16:00")
        if len(g) < 2:
            continue

        trade = simulate_day(g, capital)
        if trade is None:
            equity_curve.append((g.index[-1], capital))
            continue

        capital += trade["pnl"]
        trade["date"] = day
        trade["capital"] = capital
        trades.append(trade)
        equity_curve.append((g.index[-1], capital))

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
    return trades_df, equity_df


def buy_hold_curve(df):
    sess = df.between_time("09:30", "16:00")
    daily_close = sess.groupby("date").close.last()
    start_price = sess.groupby("date").open.first().iloc[0]
    shares = ACCOUNT_START / start_price
    return daily_close * shares


def stats(trades_df, equity_df, years):
    if trades_df.empty:
        print("No trades generated.")
        return
    n = len(trades_df)
    win_rate = (trades_df.pnl > 0).mean()
    avg_R = trades_df.pnl_R.mean()
    total_return = equity_df.equity.iloc[-1] / ACCOUNT_START - 1
    rets = equity_df.equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else float("nan")
    running_max = equity_df.equity.cummax()
    mdd = ((equity_df.equity - running_max) / running_max).min()
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else float("nan")

    print(f"Window                : {equity_df.index.min().date()} -> {equity_df.index.max().date()} (~{years:.1f} years)")
    print(f"Trades                : {n}  (long {(trades_df.long).mean():.0%}, short {(~trades_df.long).mean():.0%})")
    print(f"Win rate              : {win_rate:.1%}")
    print(f"Avg PnL per trade     : {avg_R:.2f} R")
    print(f"Total return          : {total_return:.1%}")
    print(f"Annualized return     : {ann_return:.1%}")
    print(f"Sharpe (annualized)   : {sharpe:.2f}")
    print(f"Max drawdown          : {mdd:.1%}")
    print(f"Final equity          : ${equity_df.equity.iloc[-1]:,.0f}  (start ${ACCOUNT_START:,.0f})")


def run_for_symbol(symbol):
    print(f"\n{'='*60}\n{symbol}\n{'='*60}")
    df = load_data(symbol)
    trades_df, equity_df = run_backtest(df)
    bh = buy_hold_curve(df)
    years = (df.index.max() - df.index.min()).days / 365.25

    print(f"--- ORB strategy ({symbol}, {LEVERAGE:.0f}x leverage, spread cost) ---")
    stats(trades_df, equity_df, years)

    print(f"\n--- Buy & Hold {symbol} ---")
    bh_return = bh.iloc[-1] / bh.iloc[0] - 1
    bh_ann = (1 + bh_return) ** (1 / years) - 1 if years > 0 else float("nan")
    print(f"Total return          : {bh_return:.1%}")
    print(f"Annualized return     : {bh_ann:.1%}")
    print(f"Final equity          : ${bh.iloc[-1]:,.0f}")

    if not trades_df.empty:
        trades_df.to_csv(f"trades_{symbol}.csv", index=False)
    equity_df.to_csv(f"equity_{symbol}.csv")


if __name__ == "__main__":
    symbols = sys.argv[1:] or ["NDX", "QQQ", "SP500", "SPY"]
    for sym in symbols:
        run_for_symbol(sym)
