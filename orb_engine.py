"""
Shared ORB decision logic (single source of truth), used identically by the
backtest and the shadow paper-trading script so both can never drift apart.

Rules (Zarattini & Aziz): first 5-min candle of the 9:30 ET session sets the
range; enter at the open of the second candle in the direction of the first
candle; stop at the extreme of the first candle; target 10R, or liquidate at
end of day if not reached first.
"""
RISK_PCT = 0.01
LEVERAGE = 20.0
TARGET_R = 10.0


def simulate_day(g, capital):
    """g: one day's 5m bars (already sliced to the 09:30-16:00 session),
    with columns open/high/low/close/spread_price. Returns a trade dict, or
    None if no trade was taken (doji, zero shares, etc.)."""
    if len(g) < 2:
        return None

    first = g.iloc[0]
    o1, h1, l1, c1 = first.open, first.high, first.low, first.close
    if o1 == c1:
        return None

    entry_bar = g.iloc[1]
    entry_price = entry_bar.open
    long = c1 > o1
    stop_price = l1 if long else h1
    r_dollars = abs(entry_price - stop_price)
    if r_dollars <= 0:
        return None

    risk_shares = (capital * RISK_PCT) / r_dollars
    max_shares = (LEVERAGE * capital) / entry_price
    shares = int(min(risk_shares, max_shares))
    if shares <= 0:
        return None

    target_price = entry_price + TARGET_R * r_dollars if long else entry_price - TARGET_R * r_dollars

    rest = g.iloc[2:]
    exit_price = None
    exit_time = None
    exit_reason = None
    for ts, bar in rest.iterrows():
        if long:
            if bar.low <= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break
            if bar.high >= target_price:
                exit_price, exit_reason = target_price, "target"
                break
        else:
            if bar.high >= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break
            if bar.low <= target_price:
                exit_price, exit_reason = target_price, "target"
                break
        exit_time = ts
    if exit_price is None:
        exit_price = g.iloc[-1].close
        exit_reason = "eod"
        exit_time = g.index[-1]

    gross_pnl = (exit_price - entry_price) * shares if long else (entry_price - exit_price) * shares
    cost = entry_bar.spread_price * shares
    pnl = gross_pnl - cost

    return {
        "long": long, "entry": entry_price, "entry_time": entry_bar.name if hasattr(entry_bar, "name") else None,
        "exit": exit_price, "exit_time": exit_time, "exit_reason": exit_reason,
        "stop": stop_price, "shares": shares, "R": r_dollars,
        "pnl": pnl, "pnl_R": pnl / (r_dollars * shares) if shares else float("nan"),
    }
