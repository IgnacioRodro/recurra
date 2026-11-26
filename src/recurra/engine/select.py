"""Choosing the engine, and saying what was chosen.

Design note (DD-125) -- the library picks the fastest engine, and never in
silence.

recurra is meant to run on laptops without a GPU and on servers with two of
them, so the default is ``engine="auto"``: the torch engine on the first CUDA
device when PyTorch and a device are there and the job is large enough to
gain from it, the numpy engine otherwise. Two rules keep that from becoming a
silent simplification (P3):

* the choice is announced once per process, with a ``CaveatWarning`` of fixed
  text per choice, and recorded in the provenance of the result;
* an explicit ``engine="torch"`` that cannot run raises, with the reason and
  what to install or pass instead. It never falls back to the CPU.

Small jobs stay on the CPU under ``auto``: below ``AUTO_MIN_POINTS`` points
per window, moving data to the device costs more than it saves. The value is
provisional until measured on the target hardware.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import warnings
from dataclasses import dataclass, field

from ..exceptions import BackendUnavailable, CaveatWarning, ParameterError

ENGINES = ("auto", "numpy", "torch")

#: Under ``engine="auto"``, windows shorter than this stay on the CPU.
AUTO_MIN_POINTS = 512


@dataclass(frozen=True)
class EngineChoice:
    """Which engine runs, on which device, and why."""

    engine: str                 # "numpy" or "torch"
    device: str                 # "cpu" or "cuda:N"
    reason: str
    details: dict = field(default_factory=dict)

    def provenance(self) -> dict:
        """What a result should record about where it was computed."""
        out = {"engine": self.engine, "device": self.device,
               "engine_reason": self.reason}
        out.update(self.details)
        return out


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@functools.lru_cache(maxsize=1)
def cuda_devices() -> tuple[str, ...]:
    """The CUDA devices PyTorch can see (after ``CUDA_VISIBLE_DEVICES``)."""
    if not torch_available():
        return ()
    import torch

    try:
        if not torch.cuda.is_available():
            return ()
        return tuple(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    except Exception:                                   # pragma: no cover
        return ()


def physical_cores() -> int:
    """Physical cores, not hardware threads: what ``n_jobs`` should be.

    On a 20-core server with two threads per core, 38 workers did no better
    than 20 [DD-82]. Uses joblib's count when it is installed.
    """
    try:
        from joblib import cpu_count

        return max(1, int(cpu_count(only_physical_cores=True)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _torch_details(device: str) -> dict:
    import torch

    d = {"torch": torch.__version__, "cuda": torch.version.cuda,
         "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
         "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if device.startswith("cuda"):
        d["device_name"] = torch.cuda.get_device_name(torch.device(device))
    return d


_ANNOUNCED: set[str] = set()


def _announce(choice: EngineChoice) -> None:
    key = f"{choice.engine}:{choice.device}"
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    name = choice.details.get("device_name", choice.device)
    warnings.warn(
        f"engine='auto' chose the {choice.engine} engine on {name}: {choice.reason}. "
        "Pass engine='numpy' or engine='torch' to choose yourself [DD-125].",
        CaveatWarning, stacklevel=4)


def resolve_engine(engine: str = "auto", device: str | None = None, *,
                   n_points: int | None = None, announce: bool = True) -> EngineChoice:
    """Decide which engine runs a job [DD-125].

    Parameters
    ----------
    engine
        ``"auto"``, ``"numpy"`` or ``"torch"``.
    device
        ``None`` (the engine's default), ``"cpu"``, ``"cuda"`` or ``"cuda:N"``.
        With ``engine="torch"``, ``"cpu"`` runs the torch engine on the CPU,
        which is how its logic is tested without a GPU.
    n_points
        Points per window, for the ``auto`` rule on small jobs.
    """
    if engine not in ENGINES:
        raise ParameterError(f"engine must be one of {ENGINES}, got {engine!r}")
    if device is not None and not (device == "cpu" or device == "cuda"
                                   or device.startswith("cuda:")):
        raise ParameterError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")

    if engine == "numpy":
        if device not in (None, "cpu"):
            raise ParameterError(
                f"the numpy engine runs on the CPU; device={device!r} needs engine='torch'")
        return EngineChoice("numpy", "cpu", "asked for")

    if engine == "torch":
        if not torch_available():
            raise BackendUnavailable("torch", "gpu")
        if device == "cpu":
            return EngineChoice("torch", "cpu", "asked for", _torch_details("cpu"))
        visible = cuda_devices()
        if not visible:
            raise ParameterError(
                "engine='torch' was asked for, but PyTorch sees no CUDA device. It does "
                "not fall back to the CPU on its own: pass device='cpu' to run the torch "
                "engine on the CPU, engine='numpy' for the reference engine, or "
                "engine='auto' to let recurra choose [DD-125].")
        dev = visible[0] if device in (None, "cuda") else device
        if dev not in visible:
            raise ParameterError(f"device {dev!r} is not visible; visible: {visible}")
        return EngineChoice("torch", dev, "asked for", _torch_details(dev))

    # engine == "auto"
    if device == "cpu":
        return EngineChoice("numpy", "cpu", "device='cpu' was asked for")
    visible = cuda_devices()
    if not visible:
        reason = ("PyTorch is not installed" if not torch_available()
                  else "PyTorch sees no CUDA device")
        choice = EngineChoice("numpy", "cpu", reason)
    elif n_points is not None and n_points < AUTO_MIN_POINTS:
        choice = EngineChoice(
            "numpy", "cpu",
            f"windows of fewer than {AUTO_MIN_POINTS} points are faster on the CPU")
    else:
        dev = visible[0] if device in (None, "cuda") else device
        if dev not in visible:
            raise ParameterError(f"device {dev!r} is not visible; visible: {visible}")
        choice = EngineChoice("torch", dev, "a CUDA device is available",
                              _torch_details(dev))
    # Announced when a device exists, whichever way the choice went; a machine
    # with no GPU gets the reference engine, recorded but not announced.
    if announce and visible:
        _announce(choice)
    return choice
