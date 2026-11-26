"""Compute engines: where recurrence plots and their line histograms are computed.

Two engines give the same answer [DD-125]:

* ``numpy`` -- the reference, on the CPU, always available;
* ``torch`` -- PyTorch, on a CUDA device (or on the CPU, for testing), from
  the ``gpu`` extra.

``engine="auto"`` takes the best one available for the job and says which it
took. Asking for ``torch`` where it cannot run is an error, never a silent
fall back to the CPU.
"""
from .select import (
    AUTO_MIN_POINTS,
    ENGINES,
    EngineChoice,
    cuda_devices,
    physical_cores,
    resolve_engine,
    torch_available,
)

__all__ = ["AUTO_MIN_POINTS", "ENGINES", "EngineChoice", "cuda_devices",
           "physical_cores", "resolve_engine", "torch_available"]
