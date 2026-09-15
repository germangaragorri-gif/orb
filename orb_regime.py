"""
Regime diagnostic on the validated base (60-min range, stop at range
extreme, 1.5R target): where does the edge live by side x daily trend?

Trend from Wilder ADX(14) + DI+/DI- on daily RTH bars built from the 1m
data, using the PRIOR day's close (no lookahead):
  bull : ADX > ADX_MIN and DI+ > DI-
  bear : ADX > ADX_MIN and DI- > DI+
  none : ADX <= ADX_MIN

Then three variants, IS and OOS:
  base       long and short always
  short_bear long always, short only in bear      (user's hypothesis)
  symmetric  long only in bull, short only in bear

Pre-committed rule: a side/regime filter is only supported if the same
cell is clearly better in BOTH periods.
"""
import sys
import numpy as np
import pandas as pd

from orb_combo import run, load_1m, daily_atr, summarize, IS_END

ADX_MIN = 20
BASE = dict(range_min=60, stop_mode="range", target_R=1.5)


def wilder_adx(df, period=14):
    rth = df.between_time("09:30", "15:59")
    d = rth.groupby("date").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    up, dn = d.high.diff(), -d.low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=alpha, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    out = pd.DataFrame({"adx": adx, "pdi": pdi, "mdi": mdi}).shift(1)  # known at prior close
    out["regime"] = np.select(
        [(out.adx > ADX_MIN) & (out.pdi > out.mdi), (out.adx > ADX_MIN) & (out.mdi > out.pdi)],
        ["bull", "bear"], default="none")
    return out


def with_regime(tr, reg):
    tr = tr.copy()
    tr["regime"] = tr.date.map(reg.regime)
    tr["side"] = np.where(tr.long, "long", "short")
    return tr


def cell_table(tr):
    g = tr.groupby(["side", "regime"]).agg(n=("pnl_R", "size"), win=("pnl", lambda x: (x > 0).mean()),
                                          avg_R=("pnl_R", "mean"))
    return g


def apply_filter(tr, cv_dates, mode):
    """Rebuild the equity curve keeping only the trades the filter allows.
    Position sizing stays as simulated (1% of running equity in the base
    run), so this is a slight approximation of a true re-run; fine for a
    diagnostic."""
    if mode == "base":
        keep = tr
    elif mode == "short_bear":
        keep = tr[(tr.side == "long") | (tr.regime == "bear")]
    elif mode == "symmetric":
        keep = tr[((tr.side == "long") & (tr.regime == "bull")) | ((tr.side == "short") & (tr.regime == "bear"))]
    return keep


def stats(keep, years):
    from orb_backtest import ACCOUNT_START
    if keep.empty:
        return {}
    daily = keep.groupby("date").pnl.sum()
    eq = ACCOUNT_START + daily.cumsum()
    r = eq.pct_change().dropna()
    return dict(trades=len(keep), win=(keep.pnl > 0).mean(), avg_R=keep.pnl_R.mean(),
                ann=(eq.iloc[-1] / ACCOUNT_START) ** (1 / years) - 1,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                mdd=((eq - eq.cummax()) / eq.cummax()).min())


FMT = {"win": "{:.1%}".format, "avg_R": "{:+.3f}".format, "ann": "{:+.1%}".format,
       "sharpe": "{:.2f}".format, "mdd": "{:.1%}".format}

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    df = load_1m(symbol)
    atr = daily_atr(df)
    reg = wilder_adx(df)
    pd.set_option("display.width", 200)

    for label, part in [("IN-SAMPLE", df[df.index <= IS_END]), ("OUT-OF-SAMPLE", df[df.index > IS_END])]:
        years = (part.index.max() - part.index.min()).days / 365.25
        tr, cv = run(part, atr, **BASE)
        tr = with_regime(tr, reg)
        print(f"\n{'='*70}\n{label}  {part.index.min().date()} -> {part.index.max().date()}   "
              f"regimen: {tr.regime.value_counts(normalize=True).round(2).to_dict()}")
        print("\n--- edge por lado x regimen (base 60min/rango/1.5R) ---")
        print(cell_table(tr).to_string(formatters={"win": "{:.1%}".format, "avg_R": "{:+.3f}".format}))
        print("\n--- variantes ---")
        rows = {m: stats(apply_filter(tr, None, m), years) for m in ["base", "short_bear", "symmetric"]}
        print(pd.DataFrame(rows).T.to_string(formatters=FMT))
