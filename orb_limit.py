"""
Limit-order entry vs market entry for the ORB rule, simulated on 1-minute
bars (data1m_<SYMBOL>.csv from download_data_1m.py).

Why: the live bot's market order fills ~1.5 bps worse than the candle
reference the rule anchors stop/target to, and orb_friction.py showed
that friction is most of the edge. A limit at the reference price pays
zero adverse fill on the days it fills -- but it may not fill precisely
on the momentum days the strategy is built to catch (adverse selection).
This measures that trade-off with real bars instead of guessing.

Entry variants, all with the MAX_R_BPS filter applied:
  M      market at 09:35 with entry_bps adverse fill (= current live bot)
  L      limit at ref (+offset_bps in our favour... i.e. worse for us),
         live for `timeout_min` minutes; if unfilled -> SKIP the day
  Lm     same limit; if unfilled -> market at the close of the timeout
         minute, paying entry_bps

Fill model on 1m bars: a long limit at price P fills in minute m if
bar[m].low <= P (short: high >= P), at P. The fill minute itself is not
scanned for stop/target, matching orb_engine's convention of not scanning
the entry bar; exits are scanned from the next minute.
"""
import sys
import numpy as np
import pandas as pd

from orb_engine import RISK_PCT, TARGET_R, MAX_R_BPS
from orb_backtest import ACCOUNT_START
from orb_friction import OBSERVED, PESSIMISTIC

ENTRY_BPS = OBSERVED["entry_bps"]
STOP_BPS = OBSERVED["stop_bps"]
COMM_BPS = OBSERVED["comm_bps"]


def load_1m(symbol):
    df = pd.read_csv(f"data1m_{symbol}.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index = df.index.tz_convert("America/New_York")
    df["date"] = df.index.date
    return df


def signal_from_1m(sess):
    """sess: one day's 1m bars 09:30..15:59. First 5 minutes = opening range."""
    first5 = sess.between_time("09:30", "09:34")
    if len(first5) < 3 or len(sess) < 8:
        return None
    o1, h1, l1, c1 = first5.iloc[0].open, first5.high.max(), first5.low.min(), first5.iloc[-1].close
    if o1 == c1:
        return None
    after = sess.between_time("09:35", "15:59")
    if after.empty:
        return None
    ref = after.iloc[0].open
    long = c1 > o1
    stop = l1 if long else h1
    r = abs(ref - stop)
    if r <= 0 or r / ref * 1e4 > MAX_R_BPS:
        return None
    return dict(long=long, sign=1 if long else -1, ref=ref, stop=stop, r=r,
                target=ref + (1 if long else -1) * TARGET_R * r, after=after)


def walk_exit(bars, long, fill, stop, target, stop_slip, entry_adv):
    """Scan bars for stop/target; EoD at last close (market, pays entry_adv)."""
    sign = 1 if long else -1
    for _, b in bars.iterrows():
        if long:
            if b.low <= stop:
                return stop - stop_slip, "stop"
            if b.high >= target:
                return target, "target"
        else:
            if b.high >= stop:
                return stop + stop_slip, "stop"
            if b.low <= target:
                return target, "target"
    return bars.iloc[-1].close - sign * entry_adv, "eod"


def simulate(sess, capital, mode, entry_bps=ENTRY_BPS, stop_bps=STOP_BPS,
             comm_bps=COMM_BPS, leverage=5.0, offset_bps=0.0, timeout_min=1,
             grace_min=0):
    """grace_min: extra minutes after the fill minute that are NOT scanned for
    stop/target. 0 = live behaviour (broker stop is armed immediately).
    4 with mode M reproduces the 5-minute backtest's convention of not
    scanning the 09:35-09:40 entry bar at all."""
    sig = signal_from_1m(sess)
    if sig is None:
        return None
    ref, stop, r, target, after = sig["ref"], sig["stop"], sig["r"], sig["target"], sig["after"]
    sign, long = sig["sign"], sig["long"]

    shares = int(min((capital * RISK_PCT) / r, (leverage * capital) / ref))
    if shares <= 0:
        return None

    entry_adv = ref * entry_bps / 1e4
    stop_slip = ref * stop_bps / 1e4
    comm = ref * comm_bps / 1e4

    if mode == "M":
        fill, fill_idx, how = ref + sign * entry_adv, 0, "market"
    else:
        limit = ref + sign * ref * offset_bps / 1e4  # offset>0 = willing to pay a bit
        fill, fill_idx, how = None, None, None
        for i in range(min(timeout_min, len(after))):
            b = after.iloc[i]
            if i == 0:
                # our order lands ~3s into this minute; its low/high may predate
                # the order. Only a close through the limit is certain evidence
                # price was at/through it after we were live.
                hit = (long and b.close <= limit) or (not long and b.close >= limit)
            else:
                hit = (long and b.low <= limit) or (not long and b.high >= limit)
            if hit:
                fill, fill_idx, how = limit, i, "limit"
                break
        if fill is None:
            if mode == "L":
                return dict(long=long, ref=ref, r=r, shares=shares, filled=False,
                            how="unfilled", pnl=0.0, pnl_R=0.0, exit_reason="skip")
            # Lm: market at the close of the timeout minute
            i = min(timeout_min, len(after)) - 1
            fill, fill_idx, how = after.iloc[i].close + sign * entry_adv, i, "market_after_timeout"

    bars_after = after.iloc[fill_idx + 1 + grace_min:]
    if bars_after.empty:
        return None
    exit_price, reason = walk_exit(bars_after, long, fill, stop, target, stop_slip, entry_adv)
    pnl = (exit_price - fill) * shares * sign - comm * shares
    return dict(long=long, ref=ref, fill=fill, r=r, shares=shares, filled=True, how=how,
                exit=exit_price, exit_reason=reason, pnl=pnl, pnl_R=pnl / (r * shares))


def run(df, **kw):
    capital = ACCOUNT_START
    trades, curve = [], []
    for day, g in df.groupby("date"):
        sess = g.between_time("09:30", "15:59")
        if len(sess) < 8:
            continue
        t = simulate(sess, capital, **kw)
        if t is not None:
            capital += t["pnl"]
            t["date"] = day
            trades.append(t)
        curve.append((sess.index[-1], capital))
    return pd.DataFrame(trades), pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")


def summarize(tr, cv, years):
    if tr.empty:
        return {}
    taken = tr[tr.filled]
    ret = cv.equity.iloc[-1] / ACCOUNT_START - 1
    d = cv.equity.pct_change().dropna()
    return dict(
        signals=len(tr), filled=len(taken), fill_rate=len(taken) / len(tr),
        win_rate=(taken.pnl > 0).mean() if len(taken) else np.nan,
        avg_R_filled=taken.pnl_R.mean() if len(taken) else np.nan,
        avg_R_all=tr.pnl_R.mean(),
        ann_return=(1 + ret) ** (1 / years) - 1,
        sharpe=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan,
        mdd=((cv.equity - cv.equity.cummax()) / cv.equity.cummax()).min(),
    )


FMT = {"fill_rate": "{:.0%}".format, "win_rate": "{:.1%}".format, "avg_R_filled": "{:+.3f}".format,
       "avg_R_all": "{:+.3f}".format, "ann_return": "{:+.1%}".format, "sharpe": "{:.2f}".format,
       "mdd": "{:.1%}".format}

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    df = load_1m(symbol)
    years = (df.index.max() - df.index.min()).days / 365.25
    print(f"{symbol} 1m  {df.index.min().date()} -> {df.index.max().date()}  ({years:.1f} anos)\n")

    scen = {
        "M4  mercado, gracia 5min (=backtest 5m)": dict(mode="M", grace_min=4),
        "M   mercado (bot actual)":                dict(mode="M"),
        "L   limite@ref, 1min, skip":             dict(mode="L", timeout_min=1),
        "L   limite@ref, 2min, skip":             dict(mode="L", timeout_min=2),
        "L   limite@ref, 5min, skip":             dict(mode="L", timeout_min=5),
        "Lm  limite@ref, 1min, luego mercado":    dict(mode="Lm", timeout_min=1),
        "Lm  limite@ref, 2min, luego mercado":    dict(mode="Lm", timeout_min=2),
        "L   limite@ref+0.5bps, 1min, skip":      dict(mode="L", timeout_min=1, offset_bps=0.5),
        "Lm  limite@ref+0.5bps, 1min, mercado":   dict(mode="Lm", timeout_min=1, offset_bps=0.5),
    }
    rows, keep = {}, {}
    for name, kw in scen.items():
        tr, cv = run(df, **kw)
        rows[name] = summarize(tr, cv, years)
        keep[name] = tr
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).T.to_string(formatters=FMT))

    # adverse selection: on the days the 1-min limit did NOT fill, what did
    # the market-entry variant make?
    m = keep["M   mercado (bot actual)"].set_index("date")
    l1 = keep["L   limite@ref, 1min, skip"].set_index("date")
    unf = l1[~l1.filled].index
    fil = l1[l1.filled].index
    print(f"\n=== seleccion adversa (limite 1 min) ===")
    print(f"dias con senal: {len(l1)}   llenaron: {len(fil)} ({len(fil)/len(l1):.0%})   NO llenaron: {len(unf)}")
    print(f"con entrada a MERCADO, esos mismos dias hubieran dado:")
    print(f"   dias que la limite SI llena : avg R {m.loc[fil].pnl_R.mean():+.3f}   win {(m.loc[fil].pnl>0).mean():.1%}")
    print(f"   dias que la limite NO llena : avg R {m.loc[unf].pnl_R.mean():+.3f}   win {(m.loc[unf].pnl>0).mean():.1%}")
