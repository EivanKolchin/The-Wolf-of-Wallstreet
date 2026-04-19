import asyncio
import hashlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Coroutine, List
from urllib.parse import urlparse

import feedparser
from structlog import get_logger

log = get_logger("data.news_feed")

DEFAULT_RSS_FEEDS = [
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://finance.yahoo.com/news/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.skynews.com/feeds/rss/business.xml",
    "https://www.theguardian.com/business/rss"
]

KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "rate", "inflation", "wallstreet",
    "cftc", "exchange", "regulation", "stablecoin", "defi", "interest rate",
    "market", "stock", "nasdaq", "sp500", "recession", "tariff", "sanctions", "war", "stock", "pandemic"
]

@dataclass
class NewsArticle:
    headline: str
    body: str
    source_domain: str
    url: str
    published_at: datetime
    article_hash: str


class NewsIngestionPipeline:
    def __init__(
        self,
        rss_urls: List[str] = DEFAULT_RSS_FEEDS,
        poll_interval_seconds: int = 60,
        on_article: Callable[[NewsArticle], Coroutine] = None
    ):
        self.rss_urls = rss_urls
        self.poll_interval_seconds = poll_interval_seconds
        
        async def noop(article): pass
        self.on_article = on_article or noop
        
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Deduplication cache
        self.seen_hashes_queue = deque(maxlen=10000)
        self.seen_hashes_set = set()
        self.cache_lock = asyncio.Lock()

    def filter_relevant(self, article: NewsArticle) -> bool:
        headline_lower = article.headline.lower()
        for kw in KEYWORDS:
            if kw in headline_lower:
                return True
        return False

    async def _add_to_seen(self, h: str) -> bool:
        async with self.cache_lock:
            if h in self.seen_hashes_set:
                return False
                
            if len(self.seen_hashes_queue) == self.seen_hashes_queue.maxlen:
                oldest = self.seen_hashes_queue.popleft()
                self.seen_hashes_set.discard(oldest)
                
            self.seen_hashes_queue.append(h)
            self.seen_hashes_set.add(h)
            return True

    def _parse_published_date(self, entry) -> datetime:
        # feedparser standardises to 'published_parsed' struct_time usually
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            from time import mktime
            return datetime.utcfromtimestamp(mktime(entry.published_parsed))
        return datetime.utcnow()

    async def poll_feed(self, url: str):
        backoff = 30.0
        max_backoff = 600.0
        domain = urlparse(url).netloc.replace('www.', '')
        
        polls = 0

        while self.running:
            try:
                # Wrap feedparser in an async thread to prevent blocking the event loop
                feed = await asyncio.to_thread(feedparser.parse, url)
                
                if feed.bozo:  # feedparser flag for malformed XML or connection error
                    raise Exception(getattr(feed, 'bozo_exception', 'Unknown feed error'))
                
                # Reset backoff on success
                backoff = 30.0
                
                for entry in feed.entries:
                    headline = entry.get('title', '')
                    body = entry.get('summary', '') or entry.get('description', '')
                    item_url = entry.get('link', '')
                    
                    if not headline:
                        continue
                        
                    raw_str = headline + domain
                    art_hash = hashlib.sha256(raw_str.encode('utf-8')).hexdigest()
                    
                    # Deduplicate
                    if await self._add_to_seen(art_hash):
                        
                        published_at = self._parse_published_date(entry)
                        
                        article = NewsArticle(
                            headline=headline,
                            body=body,
                            source_domain=domain,
                            url=item_url,
                            published_at=published_at,
                            article_hash=art_hash
                        )
                        
                        if self.filter_relevant(article):
                            try:
                                await self.on_article(article)
                            except Exception as cb_err:
                                log.error("Error in on_article callback", error=str(cb_err))
                                
                polls += 1
                if polls % 10 == 0:
                    log.info("feed alive", feed_domain=domain, url=url)
                    
                await asyncio.sleep(self.poll_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Feed poll failed", url=url, domain=domain, error=str(e), next_retry=backoff)
                if not self.running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)

    async def start(self) -> None:
        self.running = True
        for url in self.rss_urls:
            task = asyncio.create_task(self.poll_feed(url))
            self.tasks.append(task)
        log.info("News ingestion pipeline started", feeds_count=len(self.rss_urls))

    async def stop(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()
            
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()
            
        log.info("News ingestion pipeline stopped")