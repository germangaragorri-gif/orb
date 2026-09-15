"""
ORB 5-minute entry with a WIDER stop than the range extreme, on 1m bars.

orb_range.py moved both the entry time and the stop together and found
nothing robust. This isolates the stop: entry stays at the 09:35 open in
the direction of the first 5-minute candle, exactly as the paper, but the
stop is placed further away.

  k  : stop = range extreme - k * R0           (k=0 is the paper's rule)
  atr: stop = ref - a * ATR14 (daily, prior day) -- the paper's section-4
       idea, but tested on the wide side instead of the tight side

Sizing uses the effective R (1% of equity over the real stop distance).
Target is 10 x effective R, or EoD only. Observed friction, stop armed
immediately, 5x. Grid on 2021-05..2023-12, top-3 on 2024-01..2026-09.
"""
import sys
import numpy as np
import pandas as pd

from orb_engine import RISK_PCT
from orb_backtest import ACCOUNT_START
from orb_friction import OBSERVED
from orb_limit import load_1m, walk_exit
from orb_range import summarize, FMT, IS_END

ENTRY_BPS, STOP_BPS, COMM_BPS = OBSERVED["entry_bps"], OBSERVED["stop_bps"], OBSERVED["comm_bps"]


def daily_atr(df, period=14):
    rth = df.between_time("09:30", "15:59")
    d = rth.groupby("date").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().shift(1)  # known at yesterday's close


def simulate(sess, capital, atr_today=None, stop_mode="k", k=0.0, a=0.5, target_R=10.0,
             entry_bps=ENTRY_BPS, stop_bps=STOP_BPS, comm_bps=COMM_BPS, leverage=5.0):
    rng = sess.between_time("09:30", "09:34")
    if len(rng) < 3:
        return None
    o1, h1, l1, c1 = rng.iloc[0].open, rng.high.max(), rng.low.min(), rng.iloc[-1].close
    if o1 == c1:
        return None
    after = sess[sess.index > rng.index[-1]]
    if len(after) < 5:
        return None

    ref = after.iloc[0].open
    long = c1 > o1
    sign = 1 if long else -1
    extreme = l1 if long else h1
    r0 = abs(ref - extreme)
    if r0 <= 0:
        return None

    if stop_mode == "k":
        stop = extreme - sign * k * r0
    else:
        if atr_today is None or np.isnan(atr_today):
            return None
        stop = ref - sign * a * atr_today
    r = abs(ref - stop)
    if r <= 0:
        return None

    shares = int(min((capital * RISK_PCT) / r, (leverage * capital) / ref))
    if shares <= 0:
        return None

    entry_adv, stop_slip, comm = (ref * x / 1e4 for x in (entry_bps, stop_bps, comm_bps))
    fill = ref + sign * entry_adv
    target = ref + sign * target_R * r if target_R else (np.inf if long else -np.inf)

    exit_price, reason = walk_exit(after.iloc[1:], long, fill, stop, target, stop_slip, entry_adv)
    pnl = (exit_price - fill) * shares * sign - comm * shares
    return dict(long=long, ref=ref, r=r, r_bps=r / ref * 1e4, shares=shares,
                exit_reason=reason, pnl=pnl, pnl_R=pnl / (r * shares))


def run(df, atr=None, **kw):
    capital = ACCOUNT_START
    trades, curve = [], []
    for day, g in df.groupby("date"):
        sess = g.between_time("09:30", "15:59")
        if len(sess) < 30:
            continue
        t = simulate(sess, capital, atr_today=(atr.get(day) if atr is not None else None), **kw)
        if t is not None:
            capital += t["pnl"]
            t["date"] = day
            trades.append(t)
        curve.append((sess.index[-1], capital))
    return pd.DataFrame(trades), pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    df = load_1m(symbol)
    atr = daily_atr(df)
    ins, oos = df[df.index <= IS_END], df[df.index > IS_END]
    pd.set_option("display.width", 220)

    configs = {}
    for k in [0.0, 0.5, 1.0, 2.0]:
        for tgt, lab in [(10.0, "10R"), (None, "EoD")]:
            configs[f"k={k:<3} {lab}"] = dict(stop_mode="k", k=k, target_R=tgt)
    for a in [0.25, 0.5, 1.0]:
        for tgt, lab in [(10.0, "10R"), (None, "EoD")]:
            configs[f"atr={a:<4} {lab}"] = dict(stop_mode="atr", a=a, target_R=tgt)

    print(f"IN-SAMPLE  {ins.index.min().date()} -> {ins.index.max().date()}   "
          f"(entrada 09:35, friccion observada, stop al instante, 5x)\n")
    g = pd.DataFrame({n: summarize(*run(ins, atr=atr, **kw)) for n, kw in configs.items()}).T
    print(g.to_string(formatters=FMT))

    best = list(g.sort_values("sharpe", ascending=False).head(3).index)
    print(f"\ntop-3 in-sample por Sharpe: {best}")

    print(f"\nOUT-OF-SAMPLE  {oos.index.min().date()} -> {oos.index.max().date()}\n")
    names = best + [n for n in ["k=0.0 10R"] if n not in best]
    o = pd.DataFrame({(n + ("  (baseline)" if n == "k=0.0 10R" and n not in best else "")):
                      summarize(*run(oos, atr=atr, **configs[n])) for n in names}).T
    print(o.to_string(formatters=FMT))
