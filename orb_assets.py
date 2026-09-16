"""
Cross-asset test of the validated base config with ZERO changes:
60-minute opening range, stop at the range extreme, 1.5R target, EoD at
15:59 ET, observed QQQ friction (1.50 / 0.35 / 0.56 bps), 5x, no filters.

The only per-asset choice is the range anchor: 09:30 ET (cash open) for
everything, plus 09:00 ET (NYMEX pit open) as a market-structure variant
for crude. Same IS/OOS split as QQQ (2021-05..2023-12 / 2024-01..2026-09).

Friction caveat: 1.50 bps entry was MEASURED on QQQ. Other assets are
run with the same number; the `spread_bps` column (median quoted spread
from the bars) shows how far each is from QQQ so the reader can judge.
"""
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from orb_engine import RISK_PCT
from orb_backtest import ACCOUNT_START
from orb_friction import OBSERVED
from orb_limit import load_1m, walk_exit
from orb_range import summarize, IS_END

ENTRY_BPS, STOP_BPS, COMM_BPS = OBSERVED["entry_bps"], OBSERVED["stop_bps"], OBSERVED["comm_bps"]
RANGE_MIN, TARGET_R, EOD = 60, 1.5, "15:59"


def simulate(sess, capital, anchor="09:30", bar_min=1, entry_bps=ENTRY_BPS, stop_bps=STOP_BPS,
             comm_bps=COMM_BPS, leverage=5.0):
    a = datetime.strptime(anchor, "%H:%M")
    rng_end = (a + timedelta(minutes=RANGE_MIN - bar_min)).time()
    rng = sess.between_time(anchor, rng_end)
    if len(rng) < (RANGE_MIN // bar_min) // 2:
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
    shares = int(min((capital * RISK_PCT) / r, (leverage * capital) / ref))
    if shares <= 0:
        return None

    entry_adv, stop_slip, comm = (ref * x / 1e4 for x in (entry_bps, stop_bps, comm_bps))
    fill = ref + sign * entry_adv
    target = ref + sign * TARGET_R * r
    exit_price, reason = walk_exit(after.iloc[1:], long, fill, stop, target, stop_slip, entry_adv)
    pnl = (exit_price - fill) * shares * sign - comm * shares
    return dict(long=long, ref=ref, r=r, r_bps=r / ref * 1e4, shares=shares,
                exit_reason=reason, pnl=pnl, pnl_R=pnl / (r * shares))


def run(df, anchor="09:30", bar_min=1, **kw):
    capital = ACCOUNT_START
    trades, curve = [], []
    for day, g in df.groupby("date"):
        sess = g.between_time(anchor, EOD)
        if len(sess) < 90 // bar_min:
            continue
        t = simulate(sess, capital, anchor=anchor, bar_min=bar_min, **kw)
        if t is not None:
            capital += t["pnl"]
            t["date"] = day
            trades.append(t)
        curve.append((sess.index[-1], capital))
    return pd.DataFrame(trades), pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")


def load_5m(symbol):
    df = pd.read_csv(f"data_{symbol}.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index = df.index.tz_convert("America/New_York")
    df["date"] = df.index.date
    return df


def spread_bps(df, anchor):
    rth = df.between_time(anchor, EOD)
    px = rth.close
    dec = rth.close.astype(str).str.split(".").str[1].str.len().median()
    point = 10 ** (-dec) if pd.notna(dec) else 0.01
    return float((rth.spread * point / px * 1e4).median())


FMT = {"win": "{:.1%}".format, "avg_R": "{:+.3f}".format, "r_bps_med": "{:.0f}".format,
       "pct_stop": "{:.0%}".format, "pct_eod": "{:.0%}".format, "ann": "{:+.1%}".format,
       "sharpe": "{:.2f}".format, "mdd": "{:.1%}".format, "spread_bps": "{:.1f}".format}

if __name__ == "__main__":
    assets = [("QQQ", "09:30"), ("AAPL", "09:30"), ("NDX", "09:30"), ("SP500", "09:30"),
              ("XTIUSD", "09:30"), ("XTIUSD", "09:00")]
    if len(sys.argv) > 1:
        assets = [(s, "09:30") for s in sys.argv[1:]]
    pd.set_option("display.width", 220)
    for label, sel in [("IN-SAMPLE 2021-05..2023-12", lambda d: d[d.index <= IS_END]),
                       ("OUT-OF-SAMPLE 2024-01..2026-09", lambda d: d[d.index > IS_END])]:
        rows = {}
        for sym, anchor in assets:
            try:
                df = load_1m(sym)
            except FileNotFoundError:
                continue
            part = sel(df)
            if part.empty:
                continue
            s = summarize(*run(part, anchor=anchor))
            s["spread_bps"] = spread_bps(part, anchor)
            rows[f"{sym:7s}@{anchor}"] = s
        print(f"\n=== {label}  (60min / stop rango / 1.5R / EoD 15:59 / friccion QQQ) ===")
        print(pd.DataFrame(rows).T[["trades", "win", "avg_R", "r_bps_med", "pct_stop", "pct_eod",
                                    "ann", "sharpe", "mdd", "spread_bps"]].to_string(formatters=FMT))
