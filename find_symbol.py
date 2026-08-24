"""
Read-only REST call only (no live streaming/RPC connection to the terminal,
so this cannot interact with or interfere with the running bot's connection).
Tries candidate Nasdaq-100 CFD ticker names via the historical-candles REST
endpoint to find which one this broker uses.
"""
import asyncio
import os
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]

CANDIDATES = [
    # Nasdaq-100 index
    "NDX", "NDX.pro", "NDXm", "NDX100", "USNDX", "US.NDX",
    # QQQ ETF (as traded literally in the paper)
    "QQQ", "QQQ.pro", "QQQm", "US.QQQ",
    # S&P 500 index / SPY ETF
    "SPX500", "SPX500.pro", "SPX500m", "US500", "US500.pro", "US500m",
    "SP500", "SPX", "SPY", "SPY.pro", "SPYm", "US.SPY",
]


async def main():
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    print("Account state:", account.state)

    for sym in CANDIDATES:
        try:
            candles = await account.get_historical_candles(sym, "5m", limit=2)
            if candles:
                print(f"FOUND: {sym!r} -> {len(candles)} candles, last: {candles[-1]}")
            else:
                print(f"{sym!r}: no data returned")
        except Exception as e:
            print(f"{sym!r}: {type(e).__name__}: {str(e)[:120]}")


asyncio.run(main())
