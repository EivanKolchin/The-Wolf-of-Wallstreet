"""Cross-sectional anomaly scanner — the "scan wide, flag weird" idea, adapted honestly.

The external system that inspired this scans ~300k prediction markets for alpha. In OUR domain
(liquid price markets) breadth is correlation-capped and wide scanning adds ~no alpha (measured:
crypto XS over 20 names Sharpe −0.39, DSR 0.002). So this scanner's job is deliberately NOT
alpha — it is a RISK/ATTENTION input:

  * **wide_spread / illiquid** flags → a tighten-only de-gear multiplier on the TS sleeve
    (a spread blowout means real execution cost + liquidity risk right now);
  * **abnormal_move / volume_spike** flags → attention/observability (published to Redis for
    the dashboard, cross-checkable against the entity graph + news: a flagged move with no
    narrative is a microstructure event, not information).

Cost: TWO public REST calls for the ENTIRE perp universe (~300 symbols):
  /fapi/v1/ticker/24hr (per-symbol 24h stats) + /fapi/v1/ticker/bookTicker (best bid/ask).
Pure classification is unit-tested; the fetch is best-effort and the de-gear FAILS OPEN to 1.0
(no data → no scaling → the validated baseline book).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger("anomaly_scanner")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("anomaly_scanner")


@dataclass
class ScannerConfig:
    spread_warn_bps: float = 10.0     # (ask-bid)/mid above this → wide_spread (severity ramps)
    spread_max_bps: float = 50.0      # at/above this the de-gear hits the floor
    min_quote_volume: float = 5e6     # 24h quote volume below this → illiquid
    move_z: float = 3.0               # cross-sectional |z| of 24h % change → abnormal_move
    volume_z: float = 3.0             # cross-sectional z of log quote volume → volume_spike
    degear_floor: float = 0.5         # liquidity flags never scale below this (de-risk, not exit)


@dataclass
class Flag:
    symbol: str
    kind: str          # wide_spread | illiquid | abnormal_move | volume_spike
    value: float       # the measured quantity (bps, z, volume)
    severity: float    # 0..1


def _zscores(vals: List[float]) -> List[float]:
    n = len(vals)
    if n < 8:
        return [0.0] * n
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(n - 1, 1))
    if sd <= 1e-12:
        return [0.0] * n
    return [(v - mu) / sd for v in vals]


def classify_anomalies(stats: List[dict], books: Dict[str, dict],
                       cfg: Optional[ScannerConfig] = None) -> Dict[str, List[Flag]]:
    """Pure classification. ``stats`` rows need {symbol, priceChangePercent, quoteVolume};
    ``books`` maps symbol → {bid, ask}. Returns {symbol: [flags]} for flagged symbols only."""
    cfg = cfg or ScannerConfig()
    out: Dict[str, List[Flag]] = {}

    def add(sym: str, f: Flag):
        out.setdefault(sym, []).append(f)

    rows = [r for r in stats if r.get("symbol")]
    moves = [float(r.get("priceChangePercent", 0.0) or 0.0) for r in rows]
    vols = [math.log1p(max(0.0, float(r.get("quoteVolume", 0.0) or 0.0))) for r in rows]
    mz, vz = _zscores(moves), _zscores(vols)

    for r, m_z, v_z in zip(rows, mz, vz):
        sym = str(r["symbol"]).upper()
        qv = float(r.get("quoteVolume", 0.0) or 0.0)
        if abs(m_z) >= cfg.move_z:
            add(sym, Flag(sym, "abnormal_move", round(m_z, 2),
                          min(1.0, (abs(m_z) - cfg.move_z) / cfg.move_z + 0.34)))
        if v_z >= cfg.volume_z:
            add(sym, Flag(sym, "volume_spike", round(v_z, 2),
                          min(1.0, (v_z - cfg.volume_z) / cfg.volume_z + 0.34)))
        if qv < cfg.min_quote_volume:
            add(sym, Flag(sym, "illiquid", qv, 1.0))

    for sym, b in (books or {}).items():
        try:
            bid, ask = float(b.get("bid", 0.0)), float(b.get("ask", 0.0))
        except (TypeError, ValueError):
            continue
        if bid <= 0 or ask <= bid:
            continue
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 1e4
        if spread_bps >= cfg.spread_warn_bps:
            sev = min(1.0, (spread_bps - cfg.spread_warn_bps)
                      / max(cfg.spread_max_bps - cfg.spread_warn_bps, 1e-9))
            add(sym.upper(), Flag(sym.upper(), "wide_spread", round(spread_bps, 1), sev))
    return out


def degear_from_flags(flags: List[Flag], cfg: Optional[ScannerConfig] = None) -> float:
    """Tighten-only multiplier in [floor, 1] from a symbol's LIQUIDITY flags. Attention flags
    (abnormal_move / volume_spike) do NOT de-gear — momentum sleeves *want* moves; only real
    execution-cost risk (spread) and illiquidity scale exposure down."""
    cfg = cfg or ScannerConfig()
    scale = 1.0
    for f in flags or []:
        if f.kind == "wide_spread":
            scale = min(scale, 1.0 - (1.0 - cfg.degear_floor) * float(f.severity))
        elif f.kind == "illiquid":
            scale = min(scale, cfg.degear_floor)
    return max(cfg.degear_floor, scale)


class AnomalyScanner:
    """Live wrapper: fetch → classify → cache, with a per-symbol de-gear lookup for the agent
    and a Redis publish for observability. All failure paths return neutral (no scaling)."""

    def __init__(self, cfg: Optional[ScannerConfig] = None, *, ttl_seconds: float = 900.0,
                 testnet: Optional[bool] = None, stock_cfg: Optional[ScannerConfig] = None):
        self.cfg = cfg or ScannerConfig()
        # Equities need their OWN thresholds. Alpaca's free feed is IEX-only, so (a) the quoted
        # spread is a single small venue's book — much wider than the consolidated NBBO — and
        # (b) dailyBar volume is IEX's slice (~a few % of consolidated), not the real tape. Reusing
        # the crypto-perp thresholds flagged 12/15 megacaps as `wide_spread` and RGTI as `illiquid`.
        # move_z/volume_z are LOWER than crypto's 3.0: the stock cross-section is ~15 names, and a
        # z-score over n samples is bounded by (n-1)/√n ≈ 3.6 — a 3.0 threshold is nearly
        # unreachable there, so it would never flag. 2.5 is attainable yet still a genuine outlier.
        self.stock_cfg = stock_cfg or ScannerConfig(
            spread_warn_bps=25.0, spread_max_bps=100.0,
            min_quote_volume=5e5,     # IEX-slice dollar volume; ~$500k IEX ≈ genuinely thin
            move_z=2.5, volume_z=2.5, degear_floor=self.cfg.degear_floor,
        )
        self.ttl = ttl_seconds
        # Mainnet only: the scanner needs REAL cross-sectional moves / spreads / volumes; testnet
        # tickers are synthetic and would fabricate anomaly flags (or hide real ones).
        self.base = "https://fapi.binance.com"
        self._flags: Dict[str, List[Flag]] = {}
        self._at: float = 0.0

    def _fetch(self, path: str):
        import requests
        r = requests.get(f"{self.base}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None

    @staticmethod
    def _alpaca_creds() -> Optional[tuple]:
        """(key, secret) for Alpaca market data, or None when not configured. Fail-open:
        no creds → the stock leg is silently skipped and only crypto flags are returned."""
        try:
            from backend.core.config import settings
            key = (getattr(settings, "ALPACA_API_KEY", "") or "").strip()
            secret = (getattr(settings, "ALPACA_SECRET_KEY", "") or
                      getattr(settings, "ALPACA_SECRET", "") or "").strip()
            if key and secret and "your_" not in key.lower():
                return key, secret
        except Exception:
            pass
        return None

    def _fetch_stock_snapshots(self):
        """Alpaca snapshots for the STOCK universe, shaped into the (stats, books) contract
        classify_anomalies expects. One REST call for all names. Returns ([], {}) on any
        failure / missing creds so the stock leg never breaks the crypto scan."""
        creds = self._alpaca_creds()
        if not creds:
            return [], {}
        try:
            from backend.core import universe as _universe
            syms = list(_universe.STOCK_UNDERLYINGS)
        except Exception:
            return [], {}
        if not syms:
            return [], {}
        try:
            from backend.core.config import settings
            feed = (getattr(settings, "ALPACA_DATA_FEED", "iex") or "iex").strip().lower()
        except Exception:
            feed = "iex"
        try:
            import requests
            key, secret = creds
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                params={"symbols": ",".join(syms), "feed": feed},
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                timeout=10)
            if r.status_code != 200:
                logger.warning("anomaly_stock_snapshot_http", status=r.status_code)
                return [], {}
            data = r.json() or {}
        except Exception as e:
            logger.warning("anomaly_stock_snapshot_failed", error=str(e)[:100])
            return [], {}

        # Emit spread flags ONLY when the quote is a meaningful execution-cost signal:
        #   * a CONSOLIDATED feed (sip) — IEX's single-venue book is routinely far off the NBBO
        #     (measured: SNDK 245bps / TSLA 98bps on IEX vs ~1bps for NVDA), so an IEX-derived
        #     "wide_spread" says more about IEX than about the stock; and
        #   * during the REGULAR session — outside it quotes are stale/arbitrarily wide.
        # Otherwise we return no books, so stocks are flagged on move/volume/liquidity only.
        try:
            from backend.core.market_hours import us_session_state
            rth = us_session_state() == "regular"
        except Exception:
            rth = False
        spread_ok = rth and feed not in ("iex",)

        stats, books = [], {}
        for sym, snap in data.items():
            if not isinstance(snap, dict):
                continue
            daily = snap.get("dailyBar") or {}
            prev = snap.get("prevDailyBar") or {}
            quote = snap.get("latestQuote") or {}
            try:
                c = float(daily.get("c", 0.0) or 0.0)
                pc = float(prev.get("c", 0.0) or 0.0)
                vol = float(daily.get("v", 0.0) or 0.0)
                vw = float(daily.get("vw", 0.0) or c or 0.0)   # VWAP ≈ avg price for $-volume
            except (TypeError, ValueError):
                continue
            change_pct = ((c / pc - 1.0) * 100.0) if pc > 0 else 0.0
            stats.append({
                "symbol": str(sym).upper(),
                "priceChangePercent": change_pct,
                "quoteVolume": vol * (vw if vw > 0 else c),      # dollar volume (IEX slice)
            })
            if not spread_ok:
                continue
            try:
                bid = float(quote.get("bp", 0.0) or 0.0)
                ask = float(quote.get("ap", 0.0) or 0.0)
                if bid > 0 and ask > 0:
                    books[str(sym).upper()] = {"bid": bid, "ask": ask}
            except (TypeError, ValueError):
                pass
        return stats, books

    def scan(self) -> Dict[str, List[Flag]]:
        """Refresh if stale. Best-effort. Crypto leg = 2 Binance-futures REST calls over the
        whole perp universe; stock leg = 1 Alpaca snapshots call over STOCK_UNDERLYINGS. The
        two asset classes are classified in SEPARATE cross-sections (z-scores are only
        meaningful within a class) then merged."""
        now = time.time()
        if now - self._at < self.ttl and self._flags:
            return self._flags
        try:
            stats = self._fetch("/fapi/v1/ticker/24hr") or []
            book_rows = self._fetch("/fapi/v1/ticker/bookTicker") or []
            books = {str(b.get("symbol", "")).upper(): {"bid": b.get("bidPrice"),
                                                        "ask": b.get("askPrice")}
                     for b in book_rows if b.get("symbol")}
            crypto_flags = classify_anomalies(stats, books, self.cfg)

            # Stock leg — its OWN cross-section AND its own thresholds (equity return/volume/spread
            # scales differ from crypto perps, and the IEX feed reports a single venue's slice).
            s_stats, s_books = self._fetch_stock_snapshots()
            stock_flags = classify_anomalies(s_stats, s_books, self.stock_cfg) if s_stats else {}

            self._flags = {**crypto_flags, **stock_flags}
            self._at = now
            logger.info("anomaly_scan_done", flagged=len(self._flags),
                        crypto_flagged=len(crypto_flags), stock_flagged=len(stock_flags),
                        crypto_universe=len(stats), stock_universe=len(s_stats))
        except Exception as e:
            logger.warning("anomaly_scan_failed", error=str(e)[:100])
        return self._flags

    def degear_scale(self, symbol: str) -> float:
        """[floor, 1] multiplier for one symbol; 1.0 when unflagged/no data (fail open)."""
        flags = self.scan().get(str(symbol).upper())
        return degear_from_flags(flags, self.cfg) if flags else 1.0

    @staticmethod
    def _publishable(symbol: str) -> bool:
        """Only surface flags for assets in OUR universe. The crypto leg classifies the whole ~700
        perp cross-section (breadth is what makes the z-scores meaningful), but publishing all ~490
        flagged altcoins we never trade drowns the dashboard — and made the panel read as
        'crypto-only'. De-gear lookups still consult the full, unfiltered flag set."""
        try:
            from backend.core import universe as _u
            return symbol.upper() in set(_u.CRYPTO_SYMBOLS) | set(_u.STOCK_UNDERLYINGS)
        except Exception:
            return True

    async def publish(self, redis) -> None:
        """Best-effort Redis publish (strategy:anomalies) for the dashboard."""
        try:
            import json as _json
            payload = {s: [{"kind": f.kind, "value": f.value, "severity": round(f.severity, 2)}
                           for f in fl]
                       for s, fl in self.scan().items() if self._publishable(s)}
            await redis.set("strategy:anomalies", _json.dumps(
                {"flags": payload, "updated_at": time.time()}))
        except Exception as e:  # pragma: no cover
            logger.warning("anomaly_publish_failed", error=str(e)[:80])
