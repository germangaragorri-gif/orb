"""Read-only REST check: how far back does 5m history actually go per symbol."""
import asyncio
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]
SYMBOLS = ["NDX", "QQQ", "SP500", "SPY"]


async def main():
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    for sym in SYMBOLS:
        cursor = datetime.now(timezone.utc)
        oldest = None
        pages = 0
        while pages < 200:
            candles = await account.get_historical_candles(sym, "5m", start_time=cursor, limit=1000)
            if not candles:
                break
            new_oldest = candles[0]["time"]
            if oldest is not None and new_oldest >= oldest:
                break  # no progress, hit the real edge
            oldest = new_oldest
            cursor = oldest
            pages += 1
            if pages % 20 == 0:
                print(f"  {sym}: page {pages}, back to {oldest}", flush=True)
        print(f"{sym}: reached back to {oldest} after {pages} pages ({pages*1000} candles max)", flush=True)


asyncio.run(main())
