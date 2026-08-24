"""
ORB live execution bot (demo account "GermanG88", isolated from swing-trader).

Meant to run via cron every 5 minutes, Mon-Fri, across a UTC window wide
enough to cover both EDT/EST offsets for the 09:30-16:00 America/New_York
session. It is idempotent: a state file tracks what's already been done
today, so extra/late cron ticks are harmless no-ops.

Two actions, decided purely from the current New York time:
  - ENTRY  (~09:35-09:50 ET): read the first 5m candle of the session (REST,
    read-only), decide direction/entry/stop per the ORB rule (orb_engine),
    size the position from the account's live balance, and place a real
    market order on the demo account with stop_loss/take_profit attached —
    the broker manages the exit automatically from there.
  - EOD_CLOSE (~15:45-15:57 ET): if today's position is still open (i.e.
    neither stop nor target was hit during the day), close it at market,
    matching the backtest's "liquidate at EoD" rule. QQQ's actual broker
    trade session closes at 15:58:59 ET (confirmed via get_symbol_specification),
    not 16:00 - the window must fire comfortably before that cutoff or the
    close order would be rejected as outside trading hours.

Every action is logged to trades_log.csv for comparison against the backtest.
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

from orb_engine import RISK_PCT, TARGET_R

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]
SYMBOL = os.environ.get("ORB_SYMBOL", "QQQ")
LEVERAGE = float(os.environ.get("ORB_LEVERAGE", "200"))  # confirmed live on GermanG88

NY = ZoneInfo("America/New_York")
STATE_FILE = Path(f"/root/orb-bot/state_{SYMBOL}.json")
LOG_FILE = Path(f"/root/orb-bot/trades_log_{SYMBOL}.csv")


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


async def do_entry(state, now_et):
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    sess = await get_first_candle(account, now_et)
    if sess is None:
        print(f"{SYMBOL}: no session data yet, skipping entry check.")
        return

    first = sess.iloc[0]
    o1, h1, l1, c1 = first.open, first.high, first.low, first.close
    if o1 == c1:
        print(f"{SYMBOL}: doji first candle, no trade today.")
        state.update(date=str(now_et.date()), entered=True, closed=True, position_id=None)
        save_state(state)
        log_event({"date": now_et.date(), "event": "no_trade_doji", "time": str(now_et)})
        return

    entry_bar = sess.iloc[1]
    entry_price = entry_bar.open
    long = c1 > o1
    stop_price = l1 if long else h1
    r_dollars = abs(entry_price - stop_price)
    if r_dollars <= 0:
        print(f"{SYMBOL}: zero-R first candle, no trade today.")
        state.update(date=str(now_et.date()), entered=True, closed=True, position_id=None)
        save_state(state)
        return

    target_price = entry_price + TARGET_R * r_dollars if long else entry_price - TARGET_R * r_dollars

    connection = account.get_rpc_connection()
    try:
        await asyncio.wait_for(connection.connect(), timeout=30)
        await asyncio.wait_for(connection.wait_synchronized(), timeout=60)
        info = await asyncio.wait_for(connection.get_account_information(), timeout=30)
        balance = info["balance"]
        free_margin = info.get("freeMargin", balance)

        # Real per-unit margin for this symbol (broker-specific, can differ a
        # lot from the account's nominal "leverage" figure — e.g. QQQ here
        # requires ~5x-equivalent margin even though the account shows 200x).
        order_type = "ORDER_TYPE_SELL" if not long else "ORDER_TYPE_BUY"
        margin_check = await asyncio.wait_for(connection.calculate_margin({
            "symbol": SYMBOL, "type": order_type, "volume": 1, "openPrice": entry_price,
        }), timeout=20)
        margin_per_unit = margin_check["margin"]

        risk_shares = (balance * RISK_PCT) / r_dollars
        margin_budget_shares = (free_margin * 0.9) / margin_per_unit if margin_per_unit > 0 else 0
        volume = int(min(risk_shares, margin_budget_shares))
        if volume <= 0:
            print(f"{SYMBOL}: computed volume<=0 (balance={balance}, freeMargin={free_margin}, "
                  f"R={r_dollars}, margin_per_unit={margin_per_unit}), skipping.")
            return

        if long:
            result = await connection.create_market_buy_order(
                SYMBOL, volume, stop_loss=stop_price, take_profit=target_price)
        else:
            result = await connection.create_market_sell_order(
                SYMBOL, volume, stop_loss=stop_price, take_profit=target_price)

        position_id = result.get("positionId") or result.get("orderId")
        print(f"{SYMBOL}: ENTERED {'LONG' if long else 'SHORT'} vol={volume} "
              f"entry~{entry_price:.2f} stop={stop_price:.2f} target={target_price:.2f} "
              f"position_id={position_id}")

        state.update(date=str(now_et.date()), entered=True, closed=False, position_id=position_id)
        save_state(state)
        log_event({
            "date": now_et.date(), "event": "entry", "long": long, "volume": volume,
            "margin_per_unit": margin_per_unit, "free_margin": free_margin,
            "entry_signal": entry_price, "stop": stop_price, "target": target_price,
            "balance_before": balance, "position_id": position_id, "time": str(now_et),
        })
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

    t = now_et.time()
    if not state["entered"] and (9, 35) <= (t.hour, t.minute) <= (9, 55):
        await do_entry(state, now_et)
    elif state["entered"] and not state["closed"] and (15, 45) <= (t.hour, t.minute) <= (15, 57):
        await do_eod_close(state, now_et)
    else:
        print(f"{now_et}: outside action windows or already handled today "
              f"(entered={state.get('entered')}, closed={state.get('closed')}).")


if __name__ == "__main__":
    asyncio.run(main())
