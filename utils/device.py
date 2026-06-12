"""
Device detection
================
One place to pick the best available compute backend, so every script and
notebook runs unchanged on:

    - NVIDIA GPU         -> CUDA
    - Apple Silicon Mac  -> MPS (Metal)
    - anything else      -> CPU

Usage
-----
    from utils.device import get_device, device_report
    device = get_device()          # torch.device, best available
    print(device_report())         # human-readable summary
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """
    Return the best available torch.device.

    Priority: CUDA > MPS > CPU.
    Pass prefer="cpu"/"cuda"/"mps" to force a specific backend (falls back to
    CPU with a warning if the requested backend isn't available).
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
        print(f"[device] requested '{prefer}' is unavailable; falling back to auto-detect.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report() -> str:
    """A short multi-line description of the selected device and what's available."""
    dev = get_device()
    lines = [f"Selected device : {dev.type.upper()}"]

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"CUDA GPU        : {name} ({mem:.1f} GB)")
    else:
        lines.append("CUDA GPU        : not available")

    mps = torch.backends.mps.is_available()
    lines.append(f"Apple MPS (Metal): {'available' if mps else 'not available'}")
    lines.append(f"CPU threads     : {torch.get_num_threads()}")
    return "\n".join(lines)


# A note on MPS: a few PyG sparse ops still fall back to CPU on Metal. This is
# fine for the small datasets in this repo. If you ever hit an MPS-unsupported
# op, set the env var PYTORCH_ENABLE_MPS_FALLBACK=1 before running, or call
# get_device(prefer="cpu").


if __name__ == "__main__":
    print(device_report())
