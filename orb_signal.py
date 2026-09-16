"""
The purest test of the ORB premise on QQQ: does the first hour's direction
predict the rest of the day? No stop, no target -- enter at 10:30 in the
direction of the 09:30-10:29 candle, exit at 15:59, net of observed
friction. Rest-of-day return in bps by year, 2013-2026 (5m bars).

Result (see commit): 2013-2021 gross -2.4 bps/day, t=-3.5, 1/9 years
positive. 2022-2026 gross +7.6, t=+1.5, fading each year. The premise is
a 2022+ regime, not a persistent property of QQQ. Every ORB variant
tested inherits that.

Also disproves an earlier inference: breakout beating fade in every year
was a stop-geometry artefact, not directional information.
"""
import numpy as np
import pandas as pd
import orb_assets as A

d5 = A.load_5m("QQQ")
rows = []
for day, g in d5.groupby("date"):
    sess = g.between_time("09:30", "15:59")
    rng = sess.between_time("09:30", "10:25")
    if len(rng) < 6:
        continue
    after = sess[sess.index > rng.index[-1]]
    if len(after) < 5:
        continue
    o1, c1 = rng.iloc[0].open, rng.iloc[-1].close
    if o1 == c1:
        continue
    sign = 1 if c1 > o1 else -1
    ref, eod = after.iloc[0].open, after.iloc[-1].close
    gross = sign * (eod - ref) / ref * 1e4
    rows.append(dict(year=day.year, gross=gross, net=gross - 2 * A.ENTRY_BPS - A.COMM_BPS))
t = pd.DataFrame(rows)
yr = t.groupby("year").agg(n=("net", "size"), gross=("gross", "mean"), net=("net", "mean"),
                           win=("net", lambda x: (x > 0).mean()), sd=("net", "std"))
yr["t"] = yr.net / (yr.sd / np.sqrt(yr.n))
print(yr.to_string(formatters={"gross": "{:+.1f}".format, "net": "{:+.1f}".format,
                               "win": "{:.1%}".format, "sd": "{:.0f}".format, "t": "{:+.2f}".format}))
for lo, hi in [(2013, 2021), (2022, 2026)]:
    s = t[(t.year >= lo) & (t.year <= hi)]
    print(f"{lo}-{hi}: gross {s.gross.mean():+.2f}  net {s.net.mean():+.2f} bps/day  "
          f"t={s.net.mean() / (s.net.std() / np.sqrt(len(s))):+.1f}  win {(s.net > 0).mean():.1%}")
