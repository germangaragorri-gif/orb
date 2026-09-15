"""
Friction-aware ORB backtest. Same rule as orb_engine.simulate_day, plus the
execution costs measured on the first 15 live trades (2026-08-25..09-15):

  - entry_bps: the market order fills this much worse than the candle open
    the rule uses as reference (mean 1.50 bps = $0.107 at $712, p75 2.67).
    Stop and target stay anchored to the candle, so this both raises the
    real $ at risk and shrinks the distance to target.
  - stop_bps: a stop-loss fills this much beyond its level (mean 0.35 bps).
  - comm_bps: round-trip commission (0.56 bps = $0.04/unit at $712).
  - leverage: the account's REAL per-symbol margin is ~5x for QQQ, not the
    20x the original backtest assumed, so live sizing is margin-capped on
    most days.

With every friction at 0 and leverage=20 this reproduces orb_engine's
numbers exactly (checked in validate()).
"""
import sys
import numpy as np
import pandas as pd

from orb_backtest import load_data, ACCOUNT_START
from orb_engine import RISK_PCT, TARGET_R, simulate_day

# All frictions in basis points of price, so they scale correctly across
# 13 years in which QQQ went from ~$70 to ~$700. Measured on 15 live trades
# at a mean reference price of $712.69.
OBSERVED = dict(entry_bps=1.50, stop_bps=0.35, comm_bps=0.56)
PESSIMISTIC = dict(entry_bps=2.67, stop_bps=0.70, comm_bps=0.56)


def simulate_day_friction(g, capital, entry_bps=0.0, stop_bps=0.0,
                          comm_bps=0.0, leverage=20.0, min_r=0.0, max_r_bps=None):
    if len(g) < 2:
        return None
    first = g.iloc[0]
    o1, h1, l1, c1 = first.open, first.high, first.low, first.close
    if o1 == c1:
        return None

    ref = g.iloc[1].open
    long = c1 > o1
    sign = 1 if long else -1
    stop = l1 if long else h1
    r_ref = abs(ref - stop)
    if r_ref <= 0 or r_ref < min_r:
        return None
    if max_r_bps is not None and r_ref / ref * 1e4 > max_r_bps:
        return None

    # sizing is done from the reference R, exactly as the live bot does
    risk_shares = (capital * RISK_PCT) / r_ref
    max_shares = (leverage * capital) / ref
    shares = int(min(risk_shares, max_shares))
    if shares <= 0:
        return None

    target = ref + sign * TARGET_R * r_ref
    entry_adverse = ref * entry_bps / 1e4
    stop_slip = ref * stop_bps / 1e4
    comm_per_unit = ref * comm_bps / 1e4
    fill = ref + sign * entry_adverse

    exit_price, reason = None, None
    for _, bar in g.iloc[2:].iterrows():
        if long:
            if bar.low <= stop:
                exit_price, reason = stop - stop_slip, "stop"; break
            if bar.high >= target:
                exit_price, reason = target, "target"; break
        else:
            if bar.high >= stop:
                exit_price, reason = stop + stop_slip, "stop"; break
            if bar.low <= target:
                exit_price, reason = target, "target"; break
    if exit_price is None:
        # EoD liquidation is a market order too; charge the same adverse fill
        exit_price, reason = g.iloc[-1].close - sign * entry_adverse, "eod"

    gross = (exit_price - fill) * shares * sign
    pnl = gross - comm_per_unit * shares
    return {
        "long": long, "ref": ref, "fill": fill, "exit": exit_price, "exit_reason": reason,
        "stop": stop, "shares": shares, "R": r_ref, "pnl": pnl,
        "pnl_R": pnl / (r_ref * shares),
    }


def run(df, **kw):
    capital = ACCOUNT_START
    trades, curve = [], []
    for day, g in df.groupby("date"):
        g = g.between_time("09:30", "16:00")
        if len(g) < 2:
            continue
        t = simulate_day_friction(g, capital, **kw)
        if t is not None:
            capital += t["pnl"]
            t["date"] = day
            trades.append(t)
        curve.append((g.index[-1], capital))
    return pd.DataFrame(trades), pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")


def summarize(trades, curve, years):
    if trades.empty:
        return {}
    ret = curve.equity.iloc[-1] / ACCOUNT_START - 1
    daily = curve.equity.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    mdd = ((curve.equity - curve.equity.cummax()) / curve.equity.cummax()).min()
    stops = trades[trades.exit_reason == "stop"]
    return dict(
        trades=len(trades), win_rate=(trades.pnl > 0).mean(), avg_R=trades.pnl_R.mean(),
        avg_R_on_stop=stops.pnl_R.mean() if len(stops) else np.nan,
        ann_return=(1 + ret) ** (1 / years) - 1, sharpe=sharpe, mdd=mdd,
    )


def validate(df):
    """Friction=0 must match orb_engine.simulate_day trade-for-trade."""
    capital = ACCOUNT_START
    for day, g in df.groupby("date"):
        g = g.between_time("09:30", "16:00")
        if len(g) < 2:
            continue
        # orb_engine deducts the candle's quoted spread as its cost model; the
        # friction module models cost explicitly instead, so compare gross.
        a = simulate_day(g, capital)
        b = simulate_day_friction(g, capital)
        if (a is None) != (b is None):
            raise AssertionError(f"{day}: trade/no-trade mismatch")
        if a is not None:
            gross_a = a["pnl"] + g.iloc[1].spread_price * a["shares"]
            if abs(gross_a - b["pnl"]) > 1e-6 or a["shares"] != b["shares"]:
                raise AssertionError(f"{day}: pnl {gross_a} vs {b['pnl']}")
            capital += a["pnl"]
    print("validate: friction=0 reproduce orb_engine.simulate_day trade por trade  OK")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    df = load_data(symbol)
    years = (df.index.max() - df.index.min()).days / 365.25
    validate(df)

    scenarios = {
        "A  sin friccion, 20x (backtest original)": dict(leverage=20),
        "B  sin friccion, 5x (margen real)":         dict(leverage=5),
        "C  friccion OBSERVADA, 5x":                 dict(leverage=5, **OBSERVED),
        "D  friccion PESIMISTA (p75), 5x":           dict(leverage=5, **PESSIMISTIC),
    }
    rows = {}
    for name, kw in scenarios.items():
        tr, cv = run(df, **kw)
        rows[name] = summarize(tr, cv, years)
        if name.startswith("C"):
            tr_c, cv_c = tr, cv
    out = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(f"\n{symbol}  {df.index.min().date()} -> {df.index.max().date()}  ({years:.1f} anos)\n")
    print(out.to_string(formatters={
        "win_rate": "{:.1%}".format, "avg_R": "{:+.3f}".format, "avg_R_on_stop": "{:+.3f}".format,
        "ann_return": "{:+.1%}".format, "sharpe": "{:.2f}".format, "mdd": "{:.1%}".format,
    }))

    print("\n=== escenario C (friccion observada) por ano ===")
    tr_c["year"] = pd.to_datetime(tr_c.date).dt.year
    cv_c["year"] = cv_c.index.year
    yrs = []
    for y, g in tr_c.groupby("year"):
        eq = cv_c[cv_c.year == y].equity
        d = eq.pct_change().dropna()
        yrs.append(dict(year=y, trades=len(g), win_rate=(g.pnl > 0).mean(), avg_R=g.pnl_R.mean(),
                        ret=eq.iloc[-1] / eq.iloc[0] - 1,
                        sharpe=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan))
    print(pd.DataFrame(yrs).set_index("year").to_string(formatters={
        "win_rate": "{:.1%}".format, "avg_R": "{:+.3f}".format, "ret": "{:+.1%}".format, "sharpe": "{:.2f}".format}))
    print(f"\nanos positivos: {sum(r['ret'] > 0 for r in yrs)}/{len(yrs)}")
