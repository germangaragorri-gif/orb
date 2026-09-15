"""
Combined test of the user's hypothesis (2026-09-15): wider opening range
(enter after the opening noise) + wider stop + LOWER target, so the higher
win rate the wider stop buys can pay off before the move reverses.

Grid, pre-specified:
  range_min : 30, 60          (5 as reference, range stop only)
  stop      : range extreme | 0.5 x ATR14 (daily, prior day)
  target    : 1.5R 2R 3R 5R EoD

Scored in-sample 2021-05..2023-12; top configs then run on 2024-01..
2026-09. Decision rule stated up front: with ~25 configs, a single OOS
winner is expected by chance. Only a CLUSTER of neighbouring configs that
are positive out of sample counts as evidence.
"""
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from orb_engine import RISK_PCT
from orb_backtest import ACCOUNT_START
from orb_friction import OBSERVED
from orb_limit import load_1m, walk_exit
from orb_range import summarize, FMT, IS_END
from orb_stopbuf import daily_atr

ENTRY_BPS, STOP_BPS, COMM_BPS = OBSERVED["entry_bps"], OBSERVED["stop_bps"], OBSERVED["comm_bps"]


def simulate(sess, capital, atr_today=None, range_min=30, stop_mode="range", a=0.5,
             target_R=2.0, entry_bps=ENTRY_BPS, stop_bps=STOP_BPS, comm_bps=COMM_BPS, leverage=5.0):
    rng_end = (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=range_min - 1)).time()
    rng = sess.between_time("09:30", rng_end)
    if len(rng) < max(3, range_min // 2):
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
    if stop_mode == "range":
        stop = l1 if long else h1
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


def run(df, atr, **kw):
    capital = ACCOUNT_START
    trades, curve = [], []
    for day, g in df.groupby("date"):
        sess = g.between_time("09:30", "15:59")
        if len(sess) < 30:
            continue
        t = simulate(sess, capital, atr_today=atr.get(day), **kw)
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

    targets = [(1.5, "1.5R"), (2.0, "2R"), (3.0, "3R"), (5.0, "5R"), (None, "EoD")]
    configs = {}
    for tgt, tl in targets:
        configs[f" 5min range  {tl:4s}"] = dict(range_min=5, stop_mode="range", target_R=tgt)
    for n in [30, 60]:
        for sm, sl in [("range", "range"), ("atr", "atr.5")]:
            for tgt, tl in targets:
                configs[f"{n}min {sl:6s}{tl:4s}"] = dict(range_min=n, stop_mode=sm, a=0.5, target_R=tgt)

    print(f"IN-SAMPLE  {ins.index.min().date()} -> {ins.index.max().date()}   ({len(configs)} configs)\n")
    g = pd.DataFrame({n: summarize(*run(ins, atr, **kw)) for n, kw in configs.items()}).T
    print(g.to_string(formatters=FMT))

    print(f"\nOUT-OF-SAMPLE  {oos.index.min().date()} -> {oos.index.max().date()}   (TODAS, para ver clusters, no solo el top)\n")
    o = pd.DataFrame({n: summarize(*run(oos, atr, **kw)) for n, kw in configs.items()}).T
    o["IS_sharpe"] = g["sharpe"]
    print(o[["trades", "win", "avg_R", "pct_stop", "ann", "sharpe", "mdd", "IS_sharpe"]].to_string(
        formatters={**FMT, "IS_sharpe": "{:.2f}".format}))
    pos = o[o.avg_R > 0]
    print(f"\nconfigs con avg_R > 0 fuera de muestra: {len(pos)}/{len(o)}")
    if len(pos):
        print(pos[["avg_R", "sharpe", "IS_sharpe"]].to_string(formatters={"avg_R": "{:+.3f}".format, "sharpe": "{:.2f}".format, "IS_sharpe": "{:.2f}".format}))
