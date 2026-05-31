"""Hardware-aware compute budget (Workstream B).

The runtime used to hard-pin every BLAS/torch pool to a single thread (a
conservative choice for a "small CPU"). On a multi-core box that leaves cores
idle. This module *measures* the machine once and sets a sensible thread budget
for the heavy process (the NN trading agent), so the model can use the cores it
has — while staying RAM-friendly (threads share the model weights; we never add
processes here).

Design goals:
- **Safe by default**: leave headroom for the OS/I/O; cap threads so the
  i5-12500H's slower E-cores don't cause cache thrash; never oversubscribe.
- **Overridable**: explicit env (`OMP_NUM_THREADS`) or `HW_AUTO_TUNE=false`
  always wins.
- **No hard deps**: psutil is optional; falls back to `os.cpu_count()`.
- **iGPU is opt-in only**: Intel Iris Xe shares system RAM (the bottleneck), so
  we detect it but default it OFF and recommend CPU threading instead.

Call `apply_startup_threads()` once, as early as possible (before torch's first
tensor op), in each process.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class HardwareInfo:
    logical_cores: int
    physical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    cpu_brand: str
    has_intel_gpu: bool


def _safe_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def detect() -> HardwareInfo:
    logical = os.cpu_count() or 1
    physical = logical
    total_gb = 0.0
    avail_gb = 0.0
    brand = ""
    try:
        import psutil  # type: ignore
        physical = psutil.cpu_count(logical=False) or logical
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / 1e9, 1)
        avail_gb = round(vm.available / 1e9, 1)
    except Exception:
        physical = max(1, logical // 2)  # assume 2 threads/core if psutil absent

    try:
        import platform
        brand = platform.processor() or platform.machine() or ""
    except Exception:
        pass

    return HardwareInfo(
        logical_cores=int(logical),
        physical_cores=int(physical),
        total_ram_gb=float(total_gb),
        available_ram_gb=float(avail_gb),
        cpu_brand=str(brand),
        has_intel_gpu=_detect_intel_gpu(),
    )


def _detect_intel_gpu() -> bool:
    """Best-effort, informational only. Never fatal.

    Deliberately does NOT import torch — this runs at process startup before the
    thread env is finalized, and importing torch here would be premature. Only
    probes torch.xpu if torch is *already* loaded; otherwise returns False (the
    iGPU path is opt-in via HW_USE_IGPU anyway)."""
    import sys
    torch = sys.modules.get("torch")
    try:
        if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    return False


def _setting(name: str, default):
    try:
        from backend.core.config import settings
        return getattr(settings, name, default)
    except Exception:
        return default


def recommended_threads(hw: Optional[HardwareInfo] = None) -> int:
    """Intra-op thread count for the heavy (NN-agent) process.

    Targets physical cores minus reserved headroom, capped so the i5-12500H's
    E-cores don't dominate. Other processes (news agent, FastAPI) inherit this
    but do little torch work, so a generous cap is harmless in practice."""
    hw = hw or detect()
    if not bool(_setting("HW_AUTO_TUNE", True)):
        return 1
    reserved = _safe_int(_setting("HW_RESERVED_CORES", 2), 2)
    cap = _safe_int(_setting("HW_THREAD_CAP", 8), 8)
    usable = max(1, hw.physical_cores - reserved)
    # ~60% of usable physical cores for compute, bounded by [2, cap].
    n = max(2, min(cap, round(usable * 0.6)))
    return int(n)


def get_budget() -> dict:
    """Full budget dict for logging + downstream consumers (e.g. A3 batch caps)."""
    hw = detect()
    threads = recommended_threads(hw)
    # RAM-bounded caps (consumed by the cross-symbol batching + MC code). When
    # RAM is unknown (psutil absent → 0.0) stay on a moderate middle tier rather
    # than assuming a big box.
    avail = hw.available_ram_gb or hw.total_ram_gb
    if avail <= 0.0:
        mc_cap, batch_cap = 16, 128          # unknown RAM → moderate defaults
    elif avail < 2.0:
        mc_cap, batch_cap = 8, 32
    elif avail < 4.0:
        mc_cap, batch_cap = 16, 128
    else:
        mc_cap, batch_cap = 32, 256
    use_igpu = bool(_setting("HW_USE_IGPU", False)) and hw.has_intel_gpu
    return {
        **asdict(hw),
        "threads": threads,
        "mc_samples_cap": mc_cap,
        "batch_cap": batch_cap,
        "use_igpu": use_igpu,
        "auto_tune": bool(_setting("HW_AUTO_TUNE", True)),
    }


def apply_startup_threads() -> dict:
    """Set the BLAS/OMP thread env (and torch threads if already imported) from
    the budget. Idempotent and safe to call in every process before torch's
    first op. Explicit env or HW_AUTO_TUNE=false keeps the legacy single thread.

    Returns the budget dict (for logging by the caller)."""
    budget = get_budget()
    n = int(budget["threads"]) if budget.get("auto_tune", True) else 1

    # Respect an explicit operator override of OMP_NUM_THREADS.
    if "OMP_NUM_THREADS" in os.environ:
        try:
            n = int(os.environ["OMP_NUM_THREADS"])
        except Exception:
            pass

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)

    # If torch is already imported, set its intra-op pool now (best-effort).
    try:
        import torch  # type: ignore
        torch.set_num_threads(max(1, n))
    except Exception:
        pass

    budget["threads_applied"] = n
    return budget


def summary_line(budget: Optional[dict] = None) -> str:
    b = budget or get_budget()
    return (
        f"hardware budget: cores={b.get('physical_cores')}p/{b.get('logical_cores')}l "
        f"ram={b.get('available_ram_gb')}/{b.get('total_ram_gb')}GB "
        f"threads={b.get('threads_applied', b.get('threads'))} "
        f"mc_cap={b.get('mc_samples_cap')} batch_cap={b.get('batch_cap')} "
        f"igpu={'on' if b.get('use_igpu') else 'off'} auto_tune={b.get('auto_tune')}"
    )
