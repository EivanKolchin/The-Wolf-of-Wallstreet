"""Semantic news-text embedding for the NN feature vector (Phase 3).

Turns a piece of news text (a headline, an LLM rationale, an Alpaca/Benzinga
article summary) into a fixed ``NEWS_EMBED_DIM``-wide vector that occupies the
``feature_spec.NEWS_EMBED`` slots [70:86] of the model input. This is what lets
the policy learn from *what the news says*, not just the 4 hand-rolled scalars
(direction/magnitude/confidence/age) at slots 49-52.

Two interchangeable backends, selected by ``settings.NN_NEWS_EMBED_BACKEND``:

* ``"hashing"`` (default) — a deterministic signed feature-hashing embedding.
  Zero heavy dependencies, no model download, byte-for-byte reproducible across
  processes (uses hashlib, NOT Python's salted ``hash()``), so the offline
  pretraining alignment and the live agent always produce the *same* vector for
  the same text. Captures lexical/entity signal.

* ``"transformer"`` — ``sentence-transformers`` MiniLM (384-d) projected down to
  ``NEWS_EMBED_DIM`` via a FIXED seeded random projection (Johnson-Lindenstrauss),
  again fully deterministic. Richer semantics; requires
  ``pip install sentence-transformers`` (optional — see requirements.txt). Falls
  back to hashing automatically if the import fails.

The CRITICAL invariant is determinism + dimension-stability: the same text must
map to the same vector everywhere, regardless of backend, so offline-trained
weights stay valid against live vectors.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

import numpy as np

from backend.signals import feature_spec as fs

try:
    from backend.core.config import settings
except Exception:  # pragma: no cover - keep importable in bare scripts
    settings = None  # type: ignore

DIM = fs.NEWS_EMBED_DIM  # 16
_PROJECTION_SEED = 20240517  # fixed — must never change or offline/live drift
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


def text_for_news_impact(impact: Any) -> str:
    """Compose the embeddable text for a live ``NewsImpact``.

    NewsImpact doesn't carry the original headline, but its ``rationale`` is the
    LLM's semantic summary of the story; we enrich it with the asset, direction
    and severity so the embedding reflects both content and stance.
    """
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


def _tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    # add adjacent bigrams for a little word-order signal
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams


def _stable_hash(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")


def _hashing_embed(text: str) -> np.ndarray:
    """Deterministic signed feature-hashing → L2-normalized DIM vector."""
    vec = np.zeros(DIM, dtype=np.float32)
    toks = _tokenize(text)
    if not toks:
        return vec
    for tok in toks:
        h = _stable_hash(tok)
        bucket = h % DIM
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class NewsEmbedder:
    """Stateless-ish embedder with an optional Redis cache.

    Construct once and reuse; the transformer model and projection matrix are
    loaded lazily on first use so importing this module stays cheap.
    """

    def __init__(self, redis_client: Any = None, backend: Optional[str] = None):
        self.redis = redis_client
        self.backend = (backend or _setting("NN_NEWS_EMBED_BACKEND", "hashing") or "hashing").lower()
        self._st_model = None
        self._projection: Optional[np.ndarray] = None
        self._transformer_failed = False

    # ---- transformer backend (optional) -------------------------------------
    def _ensure_transformer(self) -> bool:
        if self._transformer_failed:
            return False
        if self._st_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_name = str(_setting("NN_NEWS_EMBED_MODEL", "all-MiniLM-L6-v2"))
            self._st_model = SentenceTransformer(model_name)
            src_dim = int(self._st_model.get_sentence_embedding_dimension())
            rng = np.random.default_rng(_PROJECTION_SEED)
            # JL random projection (src_dim -> DIM), columns unit-scaled
            self._projection = (rng.standard_normal((src_dim, DIM)) / np.sqrt(DIM)).astype(np.float32)
            return True
        except Exception:
            self._transformer_failed = True
            self._st_model = None
            return False

    def _transformer_embed(self, text: str) -> np.ndarray:
        emb = self._st_model.encode([text], normalize_embeddings=True)  # type: ignore
        proj = np.asarray(emb, dtype=np.float32) @ self._projection  # (1, DIM)
        out = proj[0]
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        return out.astype(np.float32)

    def effective_backend(self) -> str:
        """The backend ACTUALLY in effect: ``'transformer'`` only if it's selected
        *and* the model loaded successfully, otherwise ``'hashing'``. Use this (not
        ``self.backend``) for offline↔live consistency checks — ``'transformer'``
        silently falls back to ``'hashing'`` when ``sentence-transformers`` is
        missing, and a mismatch would feed the model meaningless news features."""
        if self.backend == "transformer" and self._ensure_transformer():
            return "transformer"
        return "hashing"

    # ---- public API ----------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(DIM, dtype=np.float32)
        if self.backend == "transformer" and self._ensure_transformer():
            try:
                return self._transformer_embed(text)
            except Exception:
                pass  # fall through to hashing
        return _hashing_embed(text)

    async def embed_text_cached(self, text: str) -> np.ndarray:
        """Async variant that memoizes embeddings in Redis (key by text hash).

        Safe to call without Redis — degrades to a direct compute."""
        text = (text or "").strip()
        if not text:
            return np.zeros(DIM, dtype=np.float32)
        cache_key = f"news_embed:{self.backend}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
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


# Module-level singleton for the live agent (no Redis cache by default; the
# agent passes its Redis client when constructing its own instance).
_DEFAULT_EMBEDDER: Optional[NewsEmbedder] = None


def get_embedder(redis_client: Any = None) -> NewsEmbedder:
    global _DEFAULT_EMBEDDER
    if redis_client is not None:
        return NewsEmbedder(redis_client=redis_client)
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = NewsEmbedder()
    return _DEFAULT_EMBEDDER
