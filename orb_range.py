"""
ORB with a variable opening-range length, on 1-minute bars.

orb_limit.py showed the paper's 5-minute range is net negative once the
stop is armed immediately (as the broker does) instead of after a 5-minute
grace the 5m backtest silently grants. The diagnosis: a ~14 bps stop is
inside the noise of the first minutes. A wider range gives a wider stop.

This generalises the rule to range_min minutes and lets the target be 10R
or EoD-only (with a wide range, 10R is a multi-percent intraday move).
Everything else is unchanged: direction = sign of the range candle, entry
at market at the next bar's open with observed friction, stop at the range
extreme, no grace, EoD liquidation.

Discipline: the grid is scored IN-SAMPLE (2021-05..2023-12) only; the
chosen config is then reported OUT-OF-SAMPLE (2024-01..2026-09).
"""
import sys
from datetime import time as dtime, timedelta, datetime
import numpy as np
import pandas as pd

from orb_engine import RISK_PCT
from orb_backtest import ACCOUNT_START
from orb_friction import OBSERVED
from orb_limit import load_1m, walk_exit

ENTRY_BPS, STOP_BPS, COMM_BPS = OBSERVED["entry_bps"], OBSERVED["stop_bps"], OBSERVED["comm_bps"]
IS_END = "2023-12-31"


def simulate(sess, capital, range_min=5, target_R=10.0, max_r_bps=None,
             entry_bps=ENTRY_BPS, stop_bps=STOP_BPS, comm_bps=COMM_BPS, leverage=5.0):
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
    stop = l1 if long else h1
    r = abs(ref - stop)
    if r <= 0:
        return None
    if max_r_bps is not None and r / ref * 1e4 > max_r_bps:
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


def run(df, **kw):
    capital = ACCOUNT_START
    trades, curve = [], []
    for day, g in df.groupby("date"):
        sess = g.between_time("09:30", "15:59")
        if len(sess) < 30:
            continue
        t = simulate(sess, capital, **kw)
        if t is not None:
            capital += t["pnl"]
            t["date"] = day
            trades.append(t)
        curve.append((sess.index[-1], capital))
    return pd.DataFrame(trades), pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")


def summarize(tr, cv):
    if tr.empty or len(cv) < 2:
        return {}
    years = (cv.index[-1] - cv.index[0]).days / 365.25
    ret = cv.equity.iloc[-1] / ACCOUNT_START - 1
    d = cv.equity.pct_change().dropna()
    return dict(
        trades=len(tr), win=(tr.pnl > 0).mean(), avg_R=tr.pnl_R.mean(),
        r_bps_med=tr.r_bps.median(),
        pct_stop=(tr.exit_reason == "stop").mean(), pct_eod=(tr.exit_reason == "eod").mean(),
        ann=(1 + ret) ** (1 / years) - 1 if years > 0 else np.nan,
        sharpe=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan,
        mdd=((cv.equity - cv.equity.cummax()) / cv.equity.cummax()).min(),
    )


FMT = {"win": "{:.1%}".format, "avg_R": "{:+.3f}".format, "r_bps_med": "{:.0f}".format,
       "pct_stop": "{:.0%}".format, "pct_eod": "{:.0%}".format, "ann": "{:+.1%}".format,
       "sharpe": "{:.2f}".format, "mdd": "{:.1%}".format}


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    df = load_1m(symbol)
    ins = df[df.index <= IS_END]
    oos = df[df.index > IS_END]
    pd.set_option("display.width", 220)

    print(f"IN-SAMPLE  {ins.index.min().date()} -> {ins.index.max().date()}   "
          f"(friccion observada, stop armado al instante, 5x, sin filtro)\n")
    grid = {}
    for n in [5, 10, 15, 30, 60]:
        for tgt, label in [(10.0, "10R"), (None, "EoD")]:
            tr, cv = run(ins, range_min=n, target_R=tgt)
            grid[f"{n:2d}min  {label}"] = summarize(tr, cv)
    g = pd.DataFrame(grid).T
    print(g.to_string(formatters=FMT))

    best = g.sort_values("sharpe", ascending=False).head(3)
    print(f"\ntop-3 in-sample por Sharpe: {list(best.index)}")

    print(f"\nOUT-OF-SAMPLE  {oos.index.min().date()} -> {oos.index.max().date()}  (los mismos, sin tocar nada)\n")
    rows = {}
    for name in list(best.index) + [" 5min  10R"]:
        n = int(name.split("min")[0]); tgt = 10.0 if "10R" in name else None
        tr, cv = run(oos, range_min=n, target_R=tgt)
        rows[name + ("  (baseline)" if name.strip().startswith("5min") and name not in best.index else "")] = summarize(tr, cv)
    print(pd.DataFrame(rows).T.to_string(formatters=FMT))
