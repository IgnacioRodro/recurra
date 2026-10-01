"""File readers. Each returns (channels, fs, roles, meta) for ingest()."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import IngestError

_EXT_FORMAT = {
    ".csv": "csv", ".tsv": "csv", ".txt": "csv",
    ".npy": "npy", ".npz": "npz",
    ".h5": "hdf5", ".hdf5": "hdf5",
    ".edf": "edf", ".bdf": "edf",
    ".parquet": "parquet",
    ".mat": "mat",
}


def sniff_format(path: str) -> str:
    ext = os.path.splitext(str(path))[1].lower()
    fmt = _EXT_FORMAT.get(ext)
    if fmt is None:
        raise IngestError(
            f"cannot infer format from extension {ext!r}. "
            f"Pass kind= explicitly. Known: {sorted(set(_EXT_FORMAT.values()))}"
        )
    return fmt


def read_csv(path, fs=None, time_column=None, sep=None, **kw):
    """Read a CSV/TSV where rows are samples and columns are channels.

    If ``time_column`` is given (or a column named 'time'/'t' is found) it is
    used to infer ``fs`` and is not treated as a channel.
    """
    if sep is None:
        sep = "\t" if str(path).lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, **kw)

    if time_column is None:
        for cand in ("time", "t", "Time", "timestamp"):
            if cand in df.columns:
                time_column = cand
                break

    meta: dict[str, Any] = {"source": str(path), "format": "csv"}
    if time_column is not None and time_column in df.columns:
        t = df[time_column].to_numpy(dtype=float)
        df = df.drop(columns=[time_column])
        dt = np.diff(t)
        if dt.size and np.all(dt > 0):
            inferred = 1.0 / float(np.median(dt))
            jitter = float(np.std(dt) / np.mean(dt)) if np.mean(dt) else 0.0
            meta["fs_inferred_from"] = time_column
            meta["sampling_jitter"] = jitter
            if fs is None:
                fs = inferred
            meta["t0"] = float(t[0])

    channels = {str(c): df[c].to_numpy(dtype=float) for c in df.columns}
    return channels, fs, {}, meta


def read_npy(path, fs=None, labels=None, **kw):
    arr = np.load(path, allow_pickle=False)
    arr = np.atleast_2d(arr)
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 512:
        # heuristic: more columns than rows and few rows -> channels-first
        arr = arr.T
    labels = labels or [f"ch{i:02d}" for i in range(arr.shape[1])]
    channels = {str(labels[i]): np.asarray(arr[:, i], dtype=float) for i in range(arr.shape[1])}
    return channels, fs, {}, {"source": str(path), "format": "npy"}


def read_npz(path, fs=None, **kw):
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.keys())
        if "fs" in keys and fs is None:
            fs = float(z["fs"])
            keys.remove("fs")
        channels = {k: np.asarray(z[k], dtype=float).ravel() for k in keys}
    return channels, fs, {}, {"source": str(path), "format": "npz"}


def read_hdf5(path, fs=None, group="/", **kw):
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        from ..exceptions import BackendUnavailable

        raise BackendUnavailable("hdf5", "io") from e

    channels: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    meta: dict[str, Any] = {"source": str(path), "format": "hdf5"}
    with h5py.File(path, "r") as f:
        g = f[group]
        if fs is None:
            for key in ("fs", "sfreq", "sampling_rate"):
                if key in g.attrs:
                    fs = float(g.attrs[key])
                    break
        for k, v in g.attrs.items():
            meta[f"attr_{k}"] = v.item() if hasattr(v, "item") else v
        for name, ds in g.items():
            if not hasattr(ds, "shape"):
                continue
            a = np.asarray(ds[()], dtype=float)
            if a.ndim == 1:
                channels[name] = a
            elif a.ndim == 2:
                for j in range(a.shape[1]):
                    channels[f"{name}_{j:02d}"] = a[:, j]
            if "role" in ds.attrs:
                roles[name] = str(ds.attrs["role"])
    return channels, fs, roles, meta


def read_edf(path, fs=None, channels=None, exclude=None,
             keep_annotations: bool = True, **kw):
    """Read an EDF or BDF recording.

    Design note (DD-88) -- a real recording carries more than its samples.

    The first version returned the arrays and the sampling rate and dropped
    everything else: the physical units, who was recorded and when, the
    equipment, the annotations. All of that is needed to interpret a result and
    none of it can be recovered later, so it is read and attached.

    Mixed sampling rates are the normal case, not an error: an EEG montage at
    500 Hz commonly sits beside an ECG at 250 and an annotation channel. The
    reader therefore names the offending channels and their rates instead of
    refusing with a bare list of numbers, and ``channels`` selects a
    homogeneous subset.
    """
    try:
        import pyedflib
    except ImportError as e:
        from ..exceptions import BackendUnavailable

        raise BackendUnavailable("edf", "io") from e

    with pyedflib.EdfReader(str(path)) as f:
        n_signals = f.signals_in_file
        labels = [str(v).strip() for v in f.getSignalLabels()]
        rates = [float(f.getSampleFrequency(i)) for i in range(n_signals)]
        units = [str(f.getPhysicalDimension(i)).strip() for i in range(n_signals)]

        wanted = list(range(n_signals))
        if channels is not None:
            missing = [c for c in channels if c not in labels]
            if missing:
                raise IngestError(
                    f"channel(s) {missing} not in {path}. It has: {labels}")
            wanted = [labels.index(c) for c in channels]
        if exclude:
            wanted = [i for i in wanted if labels[i] not in set(exclude)]
        if not wanted:
            raise IngestError(f"no channel left to read from {path}")

        chosen_rates = {labels[i]: rates[i] for i in wanted}
        if len(set(chosen_rates.values())) > 1:
            by_rate: dict[float, list[str]] = {}
            for name, rate in chosen_rates.items():
                by_rate.setdefault(rate, []).append(name)
            detail = "; ".join(f"{r:g} Hz: {sorted(v)}" for r, v in
                               sorted(by_rate.items()))
            raise IngestError(
                f"{path} mixes sampling rates ({detail}). Recurra needs one rate "
                "per Recording. Select a homogeneous set with channels=[...], "
                "drop the others with exclude=[...], or read the groups "
                "separately and resample."
            )
        rate = fs or float(next(iter(chosen_rates.values())))
        data = {labels[i]: np.asarray(f.readSignal(i), dtype=float) for i in wanted}
        chosen_units = {labels[i]: units[i] for i in wanted if units[i]}

        meta = {"source": str(path), "format": "edf",
                "n_signals_in_file": n_signals,
                "channels_in_file": labels}
        for key, getter in (("patient", "getPatientCode"),
                            ("patient_name", "getPatientName"),
                            ("recording", "getRecordingAdditional"),
                            ("equipment", "getEquipment"),
                            ("technician", "getTechnician"),
                            ("admincode", "getAdmincode")):
            try:
                value = str(getattr(f, getter)()).strip()
                if value:
                    meta[key] = value
            except Exception:
                pass
        try:
            meta["start_datetime"] = f.getStartdatetime().isoformat()
        except Exception:
            pass
        try:
            meta["file_duration_s"] = float(f.getFileDuration())
        except Exception:
            pass
        if keep_annotations:
            try:
                onsets, durations, texts = f.readAnnotations()
                if len(onsets):
                    meta["annotations"] = [
                        {"onset_s": float(o), "duration_s": float(d),
                         "label": str(t)}
                        for o, d, t in zip(onsets, durations, texts, strict=False)]
            except Exception:
                pass

    if meta.get("patient"):
        meta.setdefault("subject", meta["patient"])
    return data, rate, {}, {**meta, "units": chosen_units}


READERS = {
    "csv": read_csv, "npy": read_npy, "npz": read_npz,
    "hdf5": read_hdf5, "edf": read_edf,
}


def read_file(path, fmt=None, fs=None, **kw):
    fmt = fmt or sniff_format(path)
    if fmt not in READERS:
        raise IngestError(f"no reader for format {fmt!r}; have {sorted(READERS)}")
    return READERS[fmt](path, fs=fs, **kw)
