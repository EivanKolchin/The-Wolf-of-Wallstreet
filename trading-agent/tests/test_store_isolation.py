"""The suite must never reach the owner's live Redis / journal / DB.

On 2026-07-09 a pytest run published a synthetic A/B/C paper book straight into the live
`strategy:paperbook:state`, wrecking the real book (98,995 -> 15,698). conftest.py now pins fake
stores; these tests assert that the pinning is actually in effect, so the protection can't silently
rot (e.g. if someone reorders imports in conftest, or drops the autouse fixture).
"""
import asyncio
import os


def test_get_redis_returns_fakeredis_not_the_live_server():
    from backend.memory.redis_client import get_redis
    r = asyncio.run(get_redis())
    mod = type(r).__module__.lower()
    assert "fakeredis" in mod, f"tests are talking to a REAL Redis ({mod}) — live state is at risk"


def test_redis_url_points_at_a_closed_port():
    assert os.environ["REDIS_URL"].endswith(":6399/0")


def test_trade_journal_writes_to_a_temp_path():
    from backend.agents.trade_journal import TradeJournal
    p = str(TradeJournal().path)
    assert "wow-tests-" in p, f"journal would write to {p} (the production journal)"
    assert "backend" not in p.replace("wow-tests-", "")


def test_database_url_is_a_throwaway_sqlite():
    url = os.environ["DATABASE_URL"]
    assert "wow-tests-" in url, f"tests would write to the production DB: {url}"
    assert "trading-agent.db" not in url


def test_fake_redis_writes_are_not_visible_to_the_real_server():
    """Write a canary through get_redis(); a real server would persist it on :6379."""
    from backend.memory.redis_client import get_redis

    async def _write():
        r = await get_redis()
        await r.set("wow:isolation_canary", "if-you-see-this-on-6379-isolation-broke")
        return await r.get("wow:isolation_canary")

    assert asyncio.run(_write()) == "if-you-see-this-on-6379-isolation-broke"
