"""
Read-only REST downloader: paginates backwards through MetaApi's
historical-candles endpoint and saves 5m OHLCV data per symbol to CSV.
No live/streaming connection is opened.
"""
import asyncio
import os
import sys
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]

MAX_PAGES = 3000
STOP_BEFORE = datetime(2021, 5, 1, tzinfo=timezone.utc)  # align with QQQ 1m window


async def download(account, symbol):
    cursor = datetime.now(timezone.utc)
    all_candles = []
    pages = 0
    oldest = None
    while pages < MAX_PAGES:
        candles = await account.get_historical_candles(symbol, "1m", start_time=cursor, limit=1000)
        if not candles:
            break
        new_oldest = candles[0]["time"]
        if oldest is not None and new_oldest >= oldest:
            break
        all_candles = candles + all_candles
        oldest = new_oldest
        cursor = oldest
        pages += 1
        if oldest < STOP_BEFORE:
            break
        if pages % 20 == 0:
            print(f"  {symbol}: page {pages}, back to {oldest}", flush=True)
    return all_candles


async def main():
    symbols = sys.argv[1:] or ["NDX", "QQQ", "SP500", "SPY"]
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    for sym in symbols:
        print(f"Downloading {sym}...", flush=True)
        candles = await download(account, sym)
        df = pd.DataFrame(candles)
        df.to_csv(f"data1m_{sym}.csv", index=False)
        print(f"{sym}: {len(df)} candles saved -> data1m_{sym}.csv "
              f"({df.time.min()} -> {df.time.max()})", flush=True)


asyncio.run(main())
