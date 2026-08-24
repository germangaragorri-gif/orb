"""Read-only: list available symbols on the MetaApi account to find the Nasdaq CFD ticker."""
import asyncio
import os
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()
TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]


async def main():
    api = MetaApi(TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    print("Account state:", account.state, "| connection:", account.connection_status)

    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    symbols = await connection.get_symbols()
    nasdaq_like = [s for s in symbols if any(k in s.upper() for k in
                   ["NAS", "US100", "USTEC", "100", "NDX", "TECH"])]
    print(f"\nTotal symbols: {len(symbols)}")
    print("Nasdaq-like candidates:", nasdaq_like)

    await connection.close()


asyncio.run(main())
