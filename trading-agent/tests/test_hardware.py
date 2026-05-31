"""Workstream B tests: hardware-aware compute budget."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.core import hardware as hw  # noqa: E402
from backend.core.config import settings  # noqa: E402


def test_detect_has_sane_fields():
    info = hw.detect()
    assert info.logical_cores >= 1
    assert info.physical_cores >= 1
    assert info.physical_cores <= info.logical_cores


def test_auto_tune_off_pins_one_thread(monkeypatch):
    monkeypatch.setattr(settings, "HW_AUTO_TUNE", False, raising=False)
    assert hw.recommended_threads() == 1


def test_threads_within_bounds(monkeypatch):
    monkeypatch.setattr(settings, "HW_AUTO_TUNE", True, raising=False)
    monkeypatch.setattr(settings, "HW_THREAD_CAP", 8, raising=False)
    monkeypatch.setattr(settings, "HW_RESERVED_CORES", 2, raising=False)
    n = hw.recommended_threads()
    assert 2 <= n <= 8


def test_explicit_omp_override_wins(monkeypatch):
    monkeypatch.setattr(settings, "HW_AUTO_TUNE", True, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    b = hw.apply_startup_threads()
    assert b["threads_applied"] == 3
    assert os.environ["MKL_NUM_THREADS"] == "3"


def test_budget_has_required_keys():
    b = hw.get_budget()
    for k in ("physical_cores", "logical_cores", "threads", "mc_samples_cap",
              "batch_cap", "use_igpu", "auto_tune"):
        assert k in b
    assert b["use_igpu"] is False  # opt-in only; default off
