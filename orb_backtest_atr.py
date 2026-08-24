"""
ATR-stop variant of the ORB backtest (paper Section 4): instead of a stop at
the first 5-min candle's extreme, use stop = ATR_PCT% of the prior day's
14-day ATR. Target = EoD only (the paper's best-performing combination).
Sweeps ATR_PCT to see sensitivity, same spirit as the paper's Figure 7.
"""
import sys
import numpy as np
import pandas as pd
from orb_backtest import load_data, ACCOUNT_START, RISK_PCT, LEVERAGE


def compute_daily_atr(df, period=14):
    daily = df.between_time("09:30", "16:00").groupby("date").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    # shift so today's trade uses ATR known as of yesterday's close (no lookahead)
    return atr.shift(1)


def run_atr_backtest(df, atr_by_date, atr_pct):
    capital = ACCOUNT_START
    equity_curve = []
    trades = []

    for day, g in df.groupby("date"):
        g = g.between_time("09:30", "16:00")
        if len(g) < 2 or day not in atr_by_date.index or pd.isna(atr_by_date.get(day)):
            if len(g):
                equity_curve.append((g.index[-1], capital))
            continue

        first = g.iloc[0]
        o1, c1 = first.open, first.close
        if o1 == c1:
            equity_curve.append((g.index[-1], capital))
            continue

        entry_bar = g.iloc[1]
        entry_price = entry_bar.open
        long = c1 > o1
        r_dollars = atr_by_date[day] * atr_pct
        if r_dollars <= 0 or pd.isna(r_dollars):
            equity_curve.append((g.index[-1], capital))
            continue
        stop_price = entry_price - r_dollars if long else entry_price + r_dollars

        risk_shares = (capital * RISK_PCT) / r_dollars
        max_shares = (LEVERAGE * capital) / entry_price
        shares = int(min(risk_shares, max_shares))
        if shares <= 0:
            equity_curve.append((g.index[-1], capital))
            continue

        rest = g.iloc[2:]
        exit_price = None
        for _, bar in rest.iterrows():
            if long and bar.low <= stop_price:
                exit_price = stop_price
                break
            if not long and bar.high >= stop_price:
                exit_price = stop_price
                break
        if exit_price is None:
            exit_price = g.iloc[-1].close

        gross_pnl = (exit_price - entry_price) * shares if long else (entry_price - exit_price) * shares
        cost = entry_bar.spread_price * shares
        pnl = gross_pnl - cost
        capital += pnl
        trades.append({
            "date": day, "pnl": pnl, "pnl_R": pnl / (r_dollars * shares) if shares else np.nan,
            "capital": capital, "stop_dist": r_dollars,
        })
        equity_curve.append((g.index[-1], capital))

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
    return trades_df, equity_df


def summarize(trades_df, equity_df, years):
    if trades_df.empty:
        return dict(trades=0)
    total_return = equity_df.equity.iloc[-1] / ACCOUNT_START - 1
    rets = equity_df.equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else np.nan
    running_max = equity_df.equity.cummax()
    mdd = ((equity_df.equity - running_max) / running_max).min()
    return dict(
        trades=len(trades_df), win_rate=(trades_df.pnl > 0).mean(),
        avg_R=trades_df.pnl_R.mean(), total_return=total_return,
        ann_return=(1 + total_return) ** (1 / years) - 1, sharpe=sharpe, mdd=mdd,
        avg_stop_dist=trades_df.stop_dist.mean(),
    )


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    df = load_data(symbol)
    atr_by_date = compute_daily_atr(df)
    years = (df.index.max() - df.index.min()).days / 365.25

    print(f"=== ATR-stop sweep, {symbol}, target=EoD only ===\n")
    rows = []
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
        trades_df, equity_df = run_atr_backtest(df, atr_by_date, pct)
        s = summarize(trades_df, equity_df, years)
        s["atr_pct"] = pct
        rows.append(s)
    res = pd.DataFrame(rows).set_index("atr_pct")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(res.to_string())
