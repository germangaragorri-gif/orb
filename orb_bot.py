"""
ORB live execution bot (demo account "GermanG88", isolated from swing-trader).

Meant to run via cron every 5 minutes, Mon-Fri, across a UTC window wide
enough to cover both EDT/EST offsets for the 09:30-16:00 America/New_York
session. It is idempotent: a state file tracks what's already been done
today, and an flock guarantees only one instance runs at a time, so extra or
late cron ticks are harmless no-ops.

Three actions, decided purely from the current New York time:
  - ENTRY_PREWARM (~09:30 ET): the slow work — opening the MetaApi websocket,
    waiting for terminal sync, reading the balance — happens BEFORE the
    opening range closes, while the market data we need doesn't exist yet.
    The process then waits until 09:35:01, reads the just-closed first candle,
    and fires the order within ~1-2s. This matters: the stop is derived from
    the candle (per the ORB rule), but the fill happens at whatever the market
    is doing when the order lands, so every second of setup latency eats real
    risk budget. On 2026-09-10 a 15s delay consumed 49% of that day's R before
    the trade even started, and it stopped out 2 seconds later.
  - ENTRY fallback (~09:36-09:55 ET): if the prewarm tick never ran or died,
    take the trade the slow way rather than skip the day entirely.
  - EOD_CLOSE (~15:45-15:57 ET): if today's position is still open (i.e.
    neither stop nor target was hit during the day), close it at market,
    matching the backtest's "liquidate at EoD" rule. QQQ's actual broker
    trade session closes at 15:58:59 ET (confirmed via get_symbol_specification),
    not 16:00 - the window must fire comfortably before that cutoff or the
    close order would be rejected as outside trading hours.

The strategy rule: first 5-min candle sets the range, enter at the open of
the second candle in its direction, stop at the extreme of the first candle,
target 10R, liquidate at EoD. Plus one filter added 2026-09-15 and shared
with the backtest via orb_engine.MAX_R_BPS: skip the day if the opening
range is wider than 30 bps of price (those days lose even before costs;
validated out of sample 2020-2026, see orb_friction.py).

Every action is logged to trades_log.csv for comparison against the backtest.
"""
import asyncio
import fcntl
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

from orb_engine import RISK_PCT, TARGET_R, MAX_R_BPS

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]
SYMBOL = os.environ.get("ORB_SYMBOL", "QQQ")

NY = ZoneInfo("America/New_York")
STATE_FILE = Path(f"/root/orb-bot/state_{SYMBOL}.json")
LOG_FILE = Path(f"/root/orb-bot/trades_log_{SYMBOL}.csv")
LOCK_FILE = Path(f"/root/orb-bot/.lock_{SYMBOL}")

# Fraction of free margin usable for one position (leaves a cushion so the
# broker never rejects the order for insufficient margin).
MARGIN_BUDGET = 0.9


def acquire_lock():
    """Returns an open, locked file handle, or None if another instance holds
    it. The prewarm run stays alive from ~09:30 to ~09:35, straddling the
    09:35 cron tick — without this they could both enter."""
    handle = open(LOCK_FILE, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"date": None, "entered": False, "closed": False, "position_id": None}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_event(row):
    df = pd.DataFrame([row])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)


async def get_first_candle(account, day_et):
    """Read-only REST call: first two 5m candles of today's session."""
    start = datetime(day_et.year, day_et.month, day_et.day, 13, 45, tzinfo=ZoneInfo("UTC"))
    candles = await account.get_historical_candles(SYMBOL, "5m", start_time=start, limit=20)
    df = pd.DataFrame(candles)
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(NY)
    df = df.set_index("time").sort_index()
    sess = df.between_time("09:30", "16:00")
    sess = sess[sess.index.date == day_et.date()]
    if len(sess) < 2:
        return None
    sample = sess["close"].astype(str)
    decimals = sample.str.split(".").str[1].str.len().median()
    point = 10 ** (-decimals) if pd.notna(decimals) else 0.01
    sess = sess.copy()
    sess["spread_price"] = sess["spread"] * point
    return sess


def compute_signal(sess):
    """The ORB rule, unchanged. Returns a dict describing the trade, or a
    no-trade dict with a reason."""
    first = sess.iloc[0]
    o1, h1, l1, c1 = first.open, first.high, first.low, first.close
    if o1 == c1:
        return {"trade": False, "reason": "doji"}

    entry_ref = sess.iloc[1].open
    long = c1 > o1
    stop_price = l1 if long else h1
    r_dollars = abs(entry_ref - stop_price)
    if r_dollars <= 0:
        return {"trade": False, "reason": "zero_r"}

    r_bps = r_dollars / entry_ref * 1e4
    if r_bps > MAX_R_BPS:
        return {"trade": False, "reason": "range_too_wide", "r_bps": r_bps}

    target_price = (entry_ref + TARGET_R * r_dollars) if long else (entry_ref - TARGET_R * r_dollars)
    return {
        "trade": True, "long": long, "entry_ref": entry_ref, "stop": stop_price,
        "r_dollars": r_dollars, "r_bps": r_bps, "target": target_price,
    }


def mark_no_trade(state, now_et, sig):
    reason = sig["reason"]
    detail = f" (R={sig['r_bps']:.1f} bps > {MAX_R_BPS:.0f})" if "r_bps" in sig else ""
    print(f"{SYMBOL}: {reason}{detail}, no trade today.")
    state.update(date=str(now_et.date()), entered=True, closed=True, position_id=None)
    save_state(state)
    log_event({"date": now_et.date(), "event": f"no_trade_{reason}",
               "r_bps": round(sig.get("r_bps", float("nan")), 2), "time": str(now_et)})


async def submit_entry(connection, sig, balance, free_margin, state, now_et, path):
    """Size from live balance and fire the market order with stop/target
    attached, so the broker manages both exits server-side."""
    order_type = "ORDER_TYPE_BUY" if sig["long"] else "ORDER_TYPE_SELL"
    margin_check = await asyncio.wait_for(connection.calculate_margin({
        "symbol": SYMBOL, "type": order_type, "volume": 1, "openPrice": sig["entry_ref"],
    }), timeout=20)
    margin_per_unit = margin_check["margin"]

    risk_shares = (balance * RISK_PCT) / sig["r_dollars"]
    margin_budget_shares = (free_margin * MARGIN_BUDGET) / margin_per_unit if margin_per_unit > 0 else 0
    volume = int(min(risk_shares, margin_budget_shares))
    if volume <= 0:
        print(f"{SYMBOL}: computed volume<=0 (balance={balance}, freeMargin={free_margin}, "
              f"R={sig['r_dollars']}, margin_per_unit={margin_per_unit}), skipping.")
        return

    if sig["long"]:
        result = await connection.create_market_buy_order(
            SYMBOL, volume, stop_loss=sig["stop"], take_profit=sig["target"])
    else:
        result = await connection.create_market_sell_order(
            SYMBOL, volume, stop_loss=sig["stop"], take_profit=sig["target"])

    submitted_at = datetime.now(NY)
    range_close = submitted_at.replace(hour=9, minute=35, second=0, microsecond=0)
    latency_s = (submitted_at - range_close).total_seconds()

    position_id = result.get("positionId") or result.get("orderId")
    print(f"{SYMBOL}: ENTERED {'LONG' if sig['long'] else 'SHORT'} vol={volume} "
          f"ref={sig['entry_ref']:.2f} stop={sig['stop']:.2f} target={sig['target']:.2f} "
          f"latency={latency_s:.1f}s position_id={position_id}")

    state.update(date=str(now_et.date()), entered=True, closed=False, position_id=position_id)
    save_state(state)
    log_event({
        "date": now_et.date(), "event": "entry", "long": sig["long"], "volume": volume,
        "margin_per_unit": margin_per_unit, "free_margin": free_margin,
        "entry_signal": sig["entry_ref"], "stop": sig["stop"], "target": sig["target"],
        "r_bps": round(sig["r_bps"], 2),
        "balance_before": balance, "position_id": position_id,
        "path": path, "latency_s": round(latency_s, 2), "time": str(submitted_at),
    })


async def wait_for_range_close():
    """Sleep until just after the opening range candle closes at 09:35:00 ET."""
    target = datetime.now(NY).replace(hour=9, minute=35, second=1, microsecond=0)
    while True:
        remaining = (target - datetime.now(NY)).total_seconds()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 5.0))


async def poll_first_candle(account, attempts=20, delay=0.5):
    """The second candle appears the moment 09:35 ticks over, but retry
    briefly in case the feed lags."""
    for _ in range(attempts):
        sess = await get_first_candle(account, datetime.now(NY))
        if sess is not None and len(sess) >= 2:
            return sess
        await asyncio.sleep(delay)
    return None


async def do_entry_prewarm(state, now_et):
    """Do all the slow setup before 09:35, then fire as soon as the range closes."""
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    connection = account.get_rpc_connection()
    try:
        t0 = time.monotonic()
        await asyncio.wait_for(connection.connect(), timeout=60)
        await asyncio.wait_for(connection.wait_synchronized(), timeout=120)
        info = await asyncio.wait_for(connection.get_account_information(), timeout=30)
        balance = info["balance"]
        free_margin = info.get("freeMargin", balance)
        print(f"{SYMBOL}: prewarm ready in {time.monotonic() - t0:.1f}s "
              f"(balance={balance}, freeMargin={free_margin}); waiting for 09:35 range close.")

        await wait_for_range_close()

        sess = await poll_first_candle(account)
        if sess is None:
            print(f"{SYMBOL}: range closed but no candle data arrived; leaving it to the fallback tick.")
            return

        sig = compute_signal(sess)
        if not sig["trade"]:
            mark_no_trade(state, datetime.now(NY), sig)
            return

        await submit_entry(connection, sig, balance, free_margin, state, datetime.now(NY), "prewarm")
    finally:
        await connection.close()


async def do_entry(state, now_et):
    """Fallback path: same rule, but pays the full connect/sync latency."""
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    sess = await get_first_candle(account, now_et)
    if sess is None:
        print(f"{SYMBOL}: no session data yet, skipping entry check.")
        return

    sig = compute_signal(sess)
    if not sig["trade"]:
        mark_no_trade(state, now_et, sig)
        return

    connection = account.get_rpc_connection()
    try:
        await asyncio.wait_for(connection.connect(), timeout=30)
        await asyncio.wait_for(connection.wait_synchronized(), timeout=60)
        info = await asyncio.wait_for(connection.get_account_information(), timeout=30)
        balance = info["balance"]
        free_margin = info.get("freeMargin", balance)
        await submit_entry(connection, sig, balance, free_margin, state, now_et, "fallback")
    finally:
        await connection.close()


async def do_eod_close(state, now_et):
    if state.get("closed") or not state.get("entered"):
        return

    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    connection = account.get_rpc_connection()
    try:
        await asyncio.wait_for(connection.connect(), timeout=30)
        await asyncio.wait_for(connection.wait_synchronized(), timeout=60)
        positions = await connection.get_positions()
        mine = [p for p in positions if p["symbol"] == SYMBOL]

        if not mine:
            print(f"{SYMBOL}: no open position at EOD check (stop or target already hit).")
            state["closed"] = True
            save_state(state)
            log_event({"date": now_et.date(), "event": "already_closed_intraday", "time": str(now_et)})
            return

        for pos in mine:
            await connection.close_position(pos["id"])
            print(f"{SYMBOL}: EOD-closed position {pos['id']} "
                  f"(volume={pos['volume']}, profit={pos.get('profit')})")
            log_event({
                "date": now_et.date(), "event": "eod_close", "position_id": pos["id"],
                "volume": pos["volume"], "profit": pos.get("profit"), "time": str(now_et),
            })

        state["closed"] = True
        save_state(state)
    finally:
        await connection.close()


async def main():
    now_et = datetime.now(NY)
    if now_et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return

    state = load_state()
    if state.get("date") != str(now_et.date()):
        state = {"date": str(now_et.date()), "entered": False, "closed": False, "position_id": None}

    hm = (now_et.hour, now_et.minute)
    if not state["entered"] and (9, 28) <= hm <= (9, 33):
        await do_entry_prewarm(state, now_et)
    elif not state["entered"] and (9, 36) <= hm <= (9, 55):
        await do_entry(state, now_et)
    elif state["entered"] and not state["closed"] and (15, 45) <= hm <= (15, 57):
        await do_eod_close(state, now_et)
    else:
        print(f"{now_et}: outside action windows or already handled today "
              f"(entered={state.get('entered')}, closed={state.get('closed')}).")


if __name__ == "__main__":
    lock = acquire_lock()
    if lock is None:
        print("Another instance is running (prewarm in progress); exiting.")
    else:
        try:
            asyncio.run(main())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
