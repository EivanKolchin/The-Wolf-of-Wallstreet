
import sys, os
import asyncio
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
from api.routes import get_portfolio
async def run():
    try:
        res = await get_portfolio()
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
asyncio.run(run())
