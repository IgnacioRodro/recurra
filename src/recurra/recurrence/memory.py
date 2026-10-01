"""Memory planning.

Design note (DD-34) -- the library refuses rather than thrashes.

Principle P7 says the memory budget is declared, not discovered when the
machine starts swapping. ``plan`` reports what each storage strategy would
cost and picks one that fits; when nothing fits it raises with the specific
numbers, so the user knows whether to window, subsample or move to a
streaming backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get_config
from ..exceptions import MemoryBudgetError

GIB = 1024.0 ** 3
DTYPE_BYTES = {"bool": 1, "uint8": 1, "float32": 4, "float64": 8}


@dataclass(frozen=True)
class MemoryPlan:
    n_rows: int
    n_cols: int
    dim: int
    dtype: str
    tile_size: int
    backend: str
    store: str
    matrix_gb: float
    tile_gb: float
    budget_gb: float
    estimated_rr: float | None = None
    sparse_gb: float | None = None
    note: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "n_rows": self.n_rows, "n_cols": self.n_cols, "dim": self.dim,
            "dtype": self.dtype, "backend": self.backend, "store": self.store,
            "tile_size": self.tile_size,
            "dense_matrix_gb": round(self.matrix_gb, 4),
            "working_tile_gb": round(self.tile_gb, 5),
            "sparse_gb": None if self.sparse_gb is None else round(self.sparse_gb, 4),
            "budget_gb": self.budget_gb, "note": self.note,
        }])

    def summary(self) -> str:
        return (
            f"MemoryPlan: {self.n_rows} x {self.n_cols}, dim {self.dim}\n"
            f"  dense matrix ({self.dtype}) : {self.matrix_gb:8.3f} GiB\n"
            f"  working tile  ({self.tile_size}) : {self.tile_gb:8.4f} GiB\n"
            + (f"  sparse estimate            : {self.sparse_gb:8.3f} GiB\n"
               if self.sparse_gb is not None else "")
            + f"  budget                     : {self.budget_gb:8.3f} GiB\n"
            f"  chosen                     : backend={self.backend}, store={self.store}"
            + (f"\n  note: {self.note}" if self.note else "")
        )


def plan(n_rows: int, n_cols: int, dim: int, *, dtype: str = "uint8",
         budget_gb: float | None = None, tile_size: int | None = None,
         backend: str = "auto", store: str = "auto",
         estimated_rr: float | None = None) -> MemoryPlan:
    """Choose a backend and tile size that fit the declared budget."""
    from .core import canonical_store

    store = canonical_store(store)
    budget_gb = get_config().memory_budget_gb if budget_gb is None else budget_gb
    item = DTYPE_BYTES.get(dtype, 1)
    matrix_gb = n_rows * n_cols * item / GIB

    if tile_size is None:
        # A tile costs one float64 distance block plus the output block.
        per_cell = 8 + item
        target = max(0.25 * budget_gb * GIB, 8 * 1024**2)
        tile_size = int(np.clip(np.sqrt(target / per_cell), 128, 8192))
    tile_gb = tile_size * tile_size * (8 + item) / GIB

    sparse_gb = None
    if estimated_rr is not None:
        # CSR: one int32 column index plus a fraction of the row pointers.
        sparse_gb = n_rows * n_cols * estimated_rr * 4 / GIB + n_rows * 8 / GIB

    note = ""
    if store == "auto":
        if matrix_gb <= 0.5 * budget_gb:
            store, backend_auto = "memory", "dense"
        elif sparse_gb is not None and sparse_gb <= 0.5 * budget_gb:
            store, backend_auto = "sparse", "tiled"
            note = "dense matrix exceeds half the budget; storing sparse"
        else:
            store, backend_auto = "stream", "tiled"
            note = ("dense matrix does not fit; streaming without materialising. "
                    "Metrics remain exact; the full matrix is never available.")
    else:
        backend_auto = "dense" if store == "memory" else "tiled"

    if backend == "auto":
        backend = backend_auto

    if store == "memory" and matrix_gb > budget_gb:
        raise MemoryBudgetError(
            f"a dense {n_rows}x{n_cols} {dtype} matrix needs {matrix_gb:.2f} GiB but the "
            f"budget is {budget_gb:.2f} GiB. Options: store='stream' (exact "
            f"metrics), store='sparse', window the trajectory, subsample it, or raise "
            f"the budget with recurra.set_config(memory_budget_gb=...)."
        )
    if tile_gb > budget_gb:
        raise MemoryBudgetError(
            f"tile_size={tile_size} needs {tile_gb:.2f} GiB per tile, over the "
            f"{budget_gb:.2f} GiB budget. Lower tile_size."
        )

    return MemoryPlan(n_rows, n_cols, dim, dtype, int(tile_size), backend, store,
                      matrix_gb, tile_gb, budget_gb, estimated_rr, sparse_gb, note)
