"""Semantic news-text embedding for the NN feature vector (Phase 3).

Turns a piece of news text (a headline, an LLM rationale, an Alpaca/Benzinga
article summary) into a fixed ``NEWS_EMBED_DIM``-wide vector that occupies the
``feature_spec.NEWS_EMBED`` slots [70:86] of the model input. This is what lets
the policy learn from *what the news says*, not just the 4 hand-rolled scalars
(direction/magnitude/confidence/age) at slots 49-52.

Backend: ``sentence-transformers`` (all-MiniLM-L6-v2, 384-d → 16-d via fixed
random projection) when installed — the high-quality semantic path. If
sentence-transformers is NOT available (e.g. a lean offline/Colab env), it
degrades gracefully to a DETERMINISTIC token-hashing embedding rather than
crashing, so training, tests and the live agent keep running with no heavy dep.
``effective_backend()`` reports which path is actually in use (recorded in the
checkpoint so offline and live stay in lock-step).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import numpy as np

from backend.signals import feature_spec as fs

try:
    from backend.core.config import settings
except Exception:
    settings = None

DIM = fs.NEWS_EMBED_DIM  # 16
_PROJECTION_SEED = 20240517

_MODEL_NAME = "all-MiniLM-L6-v2"


def _setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


def text_for_news_impact(impact: Any) -> str:
    """Compose the embeddable text for a live ``NewsImpact``."""
    if impact is None:
        return ""
    parts = [
        str(getattr(impact, "asset", "") or ""),
        str(getattr(impact, "direction", "") or ""),
        str(getattr(impact, "severity", "") or ""),
        str(getattr(impact, "rationale", "") or ""),
    ]
    mk = getattr(impact, "matched_keywords", None)
    if isinstance(mk, dict):
        for kws in mk.values():
            if isinstance(kws, (list, tuple)):
                parts.extend(str(k) for k in kws)
    return " ".join(p for p in parts if p).strip()


class NewsEmbedder:
    """Stateless-ish embedder with an optional Redis cache.

    Requires ``sentence-transformers`` (hard dependency). Construct once per
    process; the model + projection matrix are loaded lazily on first use.
    """

    def __init__(self, redis_client: Any = None, backend: str = None):
        self.redis = redis_client
        # "transformer" (default, falls back to hashing if the dep is missing) | "hashing"
        self.backend = (backend or str(_setting("NN_NEWS_EMBED_BACKEND", "transformer"))).lower()
        self._st_model = None
        self._projection: Optional[np.ndarray] = None
        self._st_failed = False

    def effective_backend(self) -> str:
        """Which path is ACTUALLY used: 'transformer' if available (and not forced to
        hashing), else 'hashing'. Recorded in the checkpoint so offline == live."""
        if self.backend == "hashing":
            return "hashing"
        return "transformer" if self._ensure_transformer() else "hashing"

    def _ensure_transformer(self) -> bool:
        if self._st_model is not None:
            return True
        if self._st_failed or self.backend == "hashing":
            return False
        try:
            from sentence_transformers import SentenceTransformer
            model_name = str(_setting("NN_NEWS_EMBED_MODEL", _MODEL_NAME))
            self._st_model = SentenceTransformer(model_name)
            src_dim = int(self._st_model.get_sentence_embedding_dimension())
            rng = np.random.default_rng(_PROJECTION_SEED)
            self._projection = (rng.standard_normal((src_dim, DIM)) / np.sqrt(DIM)).astype(np.float32)
            return True
        except Exception:
            # No sentence-transformers (or model load failed) → degrade to the deterministic
            # hashing embedding instead of crashing the whole pipeline.
            self._st_failed = True
            return False

    def _hashing_embed(self, text: str) -> np.ndarray:
        """Deterministic, dependency-free token-hashing embedding: each token hashes to a
        signed bucket; the accumulated vector is L2-normalised. Same text → same vector
        across processes (so offline alignment and live inference never drift)."""
        v = np.zeros(DIM, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % DIM] += 1.0 if ((h >> 8) & 1) else -1.0
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 0 else v

    def _transformer_embed(self, text: str) -> np.ndarray:
        emb = self._st_model.encode([text], normalize_embeddings=True)
        proj = np.asarray(emb, dtype=np.float32) @ self._projection
        out = proj[0]
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        return out.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(DIM, dtype=np.float32)
        if self._ensure_transformer():
            return self._transformer_embed(text)
        return self._hashing_embed(text)         # graceful fallback (no sentence-transformers)

    async def embed_text_cached(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(DIM, dtype=np.float32)
        cache_key = f"news_embed:transformer:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        if self.redis is not None:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    arr = np.asarray(json.loads(cached), dtype=np.float32)
                    if arr.shape[0] == DIM:
                        return arr
            except Exception:
                pass
        vec = self.embed_text(text)
        if self.redis is not None:
            try:
                await self.redis.setex(cache_key, 86400, json.dumps(vec.tolist()))
            except Exception:
                pass
        return vec

    def embed_news_impact(self, impact: Any) -> np.ndarray:
        return self.embed_text(text_for_news_impact(impact))

    async def embed_news_impact_cached(self, impact: Any) -> np.ndarray:
        return await self.embed_text_cached(text_for_news_impact(impact))


_DEFAULT_EMBEDDER: Optional[NewsEmbedder] = None


def get_embedder(redis_client: Any = None) -> NewsEmbedder:
    global _DEFAULT_EMBEDDER
    if redis_client is not None:
        return NewsEmbedder(redis_client=redis_client)
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = NewsEmbedder()
    return _DEFAULT_EMBEDDER
