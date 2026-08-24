"""Read-only: fetch account information (balance/equity/leverage) via a single RPC call."""
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
    connection = account.get_rpc_connection()
    try:
        await asyncio.wait_for(connection.connect(), timeout=30)
        info = await asyncio.wait_for(connection.get_account_information(), timeout=30)
        print("Account info:", info)
    finally:
        await connection.close()


asyncio.run(main())
