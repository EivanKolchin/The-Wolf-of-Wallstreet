import asyncio
from backend.memory.redis_client import get_redis
async def main():
    r=await get_redis()
    v=await r.get('agent_visual_predictions')
    print('VISUAL:', v)
asyncio.run(main())
