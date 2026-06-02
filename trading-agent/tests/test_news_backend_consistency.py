"""Cycle 3: semantic news backend + offline↔live consistency guard.

The 16 NEWS_EMBED dims only mean the same thing live as in training if produced by
the SAME backend. These cover: the effective-backend probe (which the guard uses),
deterministic embeddings, and that the checkpoint records the backend so the live
loader can compare.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.signals.news_embedding import NewsEmbedder  # noqa: E402

_spec = importlib.util.spec_from_file_location("pretrain_news_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def test_effective_backend_hashing():
    assert NewsEmbedder(backend="hashing").effective_backend() == "hashing"


def test_effective_backend_transformer_falls_back_safely():
    # 'transformer' resolves to itself only if sentence-transformers is installed;
    # otherwise it must report 'hashing' (what it actually produces) — never lie.
    eb = NewsEmbedder(backend="transformer").effective_backend()
    assert eb in ("transformer", "hashing")


def test_embed_text_is_deterministic_and_dim16():
    e = NewsEmbedder(backend="hashing")
    txt = "Fed signals rate cut; chipmakers rally on AI demand"
    a, b = e.embed_text(txt), e.embed_text(txt)
    assert a.shape == (16,) and np.allclose(a, b)
    assert not np.allclose(a, e.embed_text("unrelated weather report"))


def test_checkpoint_records_news_backend(tmp_path):
    model = pt.ImprovedTradingLSTM(
        input_size=pt.INPUT_SIZE, hidden_size=8, num_layers=1, dropout=0.0,
        num_symbols=len(pt.SYMBOLS), symbol_embed_dim=4,
        num_horizons=len(pt.HORIZONS), num_classes=pt.NUM_CLASSES,
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    p = tmp_path / "ck.pt"
    pt.save_checkpoint(model, opt, epoch=1, val_loss=0.5, path=p, label="test")
    ck = torch.load(p, weights_only=False)
    assert "news_backend" in ck
    assert ck["news_backend"] in ("disabled", "hashing", "transformer")
