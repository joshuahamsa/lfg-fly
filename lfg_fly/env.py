"""CPU discipline for a box whose cores belong to an XRPL validator.

Imported first by lfg_fly/__init__.py, so the thread caps are in the
environment before numpy, torch or pyarrow is imported anywhere in the package.
"""

from __future__ import annotations

import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def apply_thread_caps() -> None:
    for var in THREAD_VARS:
        os.environ[var] = "1"


def configure_libraries() -> None:
    """Cap the libraries' own pools. Call once, before heavy work."""
    import pyarrow
    import torch

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # torch allows this once per process; a second call is harmless to skip


apply_thread_caps()
