import asyncio
from backend.memory.redis_client import get_redis
async def main():
    r = await get_redis()
    print(await r.get('agent_frontend_status'))
asyncio.run(main())
