"""
Shadow paper trading for the ORB strategy: read-only, never places real
orders. Meant to run once per day (after the 16:00 ET close) via a scheduled
routine. Each run:
  1. Pulls the latest 5m candles for SYMBOL via MetaApi's REST
     historical-candles endpoint (no live/streaming connection, same as the
     backtest downloader).
  2. Finds any complete trading day(s) not yet processed.
  3. Applies the exact same ORB rule as the backtest (orb_engine.simulate_day)
     to each one, walking forward through that day's own bars only (no
     lookahead beyond what would have been known live).
  4. Appends the result to paper_trades_log.csv and updates paper_state.json
     (running "paper capital", starting at PAPER_START).

This never touches a broker order book — it is a parallel bookkeeping
exercise meant to confirm, with real out-of-sample data going forward, that
live spreads/behavior match what the backtest assumed.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

from orb_engine import simulate_day

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
PAPER_START = 1_000.0
STATE_FILE = Path(f"paper_state_{SYMBOL}.json")
LOG_FILE = Path(f"paper_trades_log_{SYMBOL}.csv")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_processed_date": None, "capital": PAPER_START}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def fetch_recent(symbol, lookback_days=10):
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    cursor = datetime.now(timezone.utc)
    all_candles = []
    for _ in range(lookback_days * 300 // 1000 + 3):  # ~300 5m bars/day incl. off-hours cushion
        candles = await account.get_historical_candles(symbol, "5m", start_time=cursor, limit=1000)
        if not candles:
            break
        all_candles = candles + all_candles
        cursor = candles[0]["time"]
        if cursor < datetime.now(timezone.utc) - timedelta(days=lookback_days):
            break
    return all_candles


def to_df(candles):
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index = df.index.tz_convert("America/New_York")
    df["date"] = df.index.date
    sample = df["close"].iloc[:200].astype(str)
    decimals = sample.str.split(".").str[1].str.len().median()
    point = 10 ** (-decimals) if pd.notna(decimals) else 0.01
    df["spread_price"] = df["spread"] * point
    return df


def main():
    state = load_state()
    capital = state["capital"]
    last_date = state["last_processed_date"]

    candles = asyncio.run(fetch_recent(SYMBOL))
    if not candles:
        print(f"No candles returned for {SYMBOL}; nothing to do.")
        return
    df = to_df(candles)

    now_et = pd.Timestamp.now(tz="America/New_York")
    trading_days = sorted(set(df["date"]))

    log_rows = []
    processed_any = False
    for day in trading_days:
        if last_date is not None and str(day) <= last_date:
            continue
        # only process a day once its session has fully closed
        session_close = pd.Timestamp(f"{day} 16:00:00", tz="America/New_York")
        if now_et < session_close:
            continue

        g = df[df["date"] == day].between_time("09:30", "16:00")
        trade = simulate_day(g, capital)
        if trade is None:
            log_rows.append({"date": day, "symbol": SYMBOL, "traded": False, "capital": capital})
        else:
            capital += trade["pnl"]
            row = {"date": day, "symbol": SYMBOL, "traded": True, "capital": capital, **trade}
            log_rows.append(row)
        last_date = str(day)
        processed_any = True

    if not processed_any:
        print(f"{SYMBOL}: no new complete trading days to process "
              f"(last processed: {last_date}, now: {now_et}).")
        return

    log_df = pd.DataFrame(log_rows)
    header = not LOG_FILE.exists()
    log_df.to_csv(LOG_FILE, mode="a", header=header, index=False)

    state["last_processed_date"] = last_date
    state["capital"] = capital
    save_state(state)

    n_trades = log_df["traded"].sum()
    print(f"{SYMBOL}: processed {len(log_df)} day(s), {n_trades} trade(s). "
          f"Capital: ${capital:,.2f} (start ${PAPER_START:,.0f})")
    for _, r in log_df.iterrows():
        if r["traded"]:
            print(f"  {r['date']} {'LONG' if r['long'] else 'SHORT'} "
                  f"entry={r['entry']:.2f} exit={r['exit']:.2f} ({r['exit_reason']}) "
                  f"pnl=${r['pnl']:.2f} ({r['pnl_R']:+.2f}R)")
        else:
            print(f"  {r['date']} no trade (doji/zero-R)")


if __name__ == "__main__":
    main()
