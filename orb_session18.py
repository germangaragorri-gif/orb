"""
Last ORB variant tested (2026-09-15): anchor the range to the index CFD's
real session open, 18:00 ET the previous day, instead of the 09:30 cash
open. NDX and SP500, 5m bars 2022-12..2026-08 (inside the favourable
2022+ regime), observed-QQQ friction, 1.5R, stop at extreme.

  A  range 18:00-18:59, enter 19:00, exit 15:59 next day
  B  range = whole overnight 18:00-09:25, enter 09:30, exit 15:59

            NDX                      SP500
  A   -0.241R  Sharpe -1.37     -0.308R  Sharpe -2.01
  B   -0.126R  Sharpe -1.84     -0.112R  Sharpe -1.68

Negative every year in all four. Spreads in the 18:00 window are 0.4-0.5
bps, same as daytime, so it is not a cost artefact: the overnight
direction carries no information about the day. Closes the ORB family.
(Runner kept inline in the session transcript; this file documents it.)
"""
