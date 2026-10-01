"""Constructors: every route from data to a state space.

Design note (DD-25) -- three routes, one downstream.

The project specification states the two ways of building the phase space: directly from
observable state variables, or by delay reconstruction (Takens). recurra
supports both plus their combination, and everything downstream is identical:

* ``observable`` -- coordinates are physical quantities. This is where the
  CFC state spaces live, e.g. (cos phi_low, sin phi_low, A_high) for PAC.
* ``delay``      -- Takens reconstruction from one or several channels.
* ``hybrid``     -- observable coordinates *plus* delays applied to a subset.
  The canonical case: keep the phase circle intact (its geometry is already
  right) but embed the envelope in m dimensions, because the envelope is a
  scalar observable of a subsystem with its own dynamics.

Nothing forces the coupling reading. ``channels()`` builds a plain
multivariate state space from raw columns, and ``custom()`` takes any
callable, so signals with no coupling at all are first-class.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..capabilities import Capability, require
from ..exceptions import InferenceWarning, ParameterError
from ..provenance import Step
from .core import CoordGroup, StateSpace
from .embedding import _embed, estimate_embedding

#: How a spec entry turns into coordinates.
COORD_MODES = ("phase_circle", "amplitude", "raw", "inst_freq", "delay", "phase_raw")


def _warn_choice(note: str) -> None:
    from ..config import get_config

    if get_config().warn_on_caveat:
        import warnings

        warnings.warn(f"statespace: {note}", InferenceWarning, stacklevel=3)


def _resolve(rec, name: str | None, role: str, what: str) -> str:
    """Find a channel by name, or the unique one carrying a role."""
    if name is not None:
        if name not in rec.channels:
            raise ParameterError(f"no channel {name!r}; available: {rec.names}")
        return name
    cands = rec.by_role(role)
    if not cands:
        raise ParameterError(
            f"{what}: no channel with role {role!r}. Run analytic() first, or name "
            f"the channel explicitly. Available: "
            + ", ".join(f"{n}({rec.roles[n]})" for n in rec.names)
        )
    if len(cands) > 1:
        raise ParameterError(
            f"{what}: several channels have role {role!r} ({cands}); name one explicitly"
        )
    return cands[0]


def _band_edges(rec, channel: str) -> tuple[float, float] | None:
    """(low, high) edges of the band a derived channel came from, if known.

    The registry lives in ``rec.meta["bands"]``, filled by :func:`bandpass`
    or passed to :func:`ingest` when the decomposition was done elsewhere. A
    channel is looked up by its own name and then by the base name behind a
    ``phase_`` / ``amp_`` / ``ifreq_`` prefix.
    """
    bands = rec.meta.get("bands", {}) or {}
    if channel in bands:
        lo, hi = bands[channel]
        return float(lo), float(hi)
    for prefix in ("phase_", "amp_", "ifreq_"):
        if channel.startswith(prefix):
            base = channel[len(prefix):]
            if base in bands:
                lo, hi = bands[base]
                return float(lo), float(hi)
    return None


def _band_center(rec, channel: str) -> float | None:
    """Centre frequency of the band a derived channel came from, if known."""
    edges = _band_edges(rec, channel)
    return None if edges is None else 0.5 * (edges[0] + edges[1])


def _pick_by_frequency(rec, role: str, which: str, what: str) -> tuple[str, str]:
    """Pick the slowest or fastest channel of a role, using recorded bands.

    Returns (name, note). Falls back to a hard error when the bands are
    unknown, because guessing which rhythm is the modulator is not something
    a library should do silently.
    """
    cands = rec.by_role(role)
    if not cands:
        raise ParameterError(f"{what}: no channel with role {role!r}")
    if len(cands) == 1:
        return cands[0], ""
    freqs = {c: _band_center(rec, c) for c in cands}
    known = {c: f for c, f in freqs.items() if f is not None}
    if len(known) < len(cands):
        raise ParameterError(
            f"{what}: several channels have role {role!r} ({cands}) and their source "
            "bands are not all recorded, so the slow/fast one cannot be identified. "
            "Name the channel explicitly."
        )
    pick = (min(known, key=known.get) if which == "slowest" else max(known, key=known.get))
    note = f"{role} channel {pick!r} chosen as {which} ({known[pick]:g} Hz) among {cands}"
    return pick, note


SMOOTH_METHODS = ("gaussian", "butterworth")


def _smooth(X: np.ndarray, cutoff_hz: float, fs: float,
            method: str = "gaussian") -> np.ndarray:
    """Zero-phase low-pass of an envelope [DD-29, DD-117].

    The envelope of a band centred at f_high carries meaningful variation only
    up to roughly half the band's width; faster fluctuation is estimation
    noise that fills the state space and obscures the modulation geometry.
    Smoothing is standard practice in PAC work, and is recorded as a caveat
    because it does discard information.

    Both methods have a gain of exactly 1/2 at ``cutoff_hz``, so the number
    means the same thing whichever is used. ``"gaussian"`` (default) convolves
    with a non-negative kernel, so a non-negative envelope stays non-negative.
    ``"butterworth"`` is the order-4 filter applied forward and backward that
    was the only method before 0.25: flatter below the cut-off and sharper
    above it, but its impulse response has negative lobes, and on a strongly
    modulated envelope it rang below zero -- 4.7% of samples at alpha 0.9,
    down to -54% of the mean [DD-117].
    """
    nyq = fs / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ParameterError(f"smooth={cutoff_hz} Hz invalid for fs={fs} (Nyquist {nyq})")
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        # H(f) = exp(-2 pi^2 sigma^2 f^2) equals 1/2 at the cut-off.
        sigma = fs * np.sqrt(np.log(2.0) / 2.0) / (np.pi * cutoff_hz)
        return np.column_stack([gaussian_filter1d(X[:, k], sigma, mode="reflect",
                                                  truncate=4.0)
                                for k in range(X.shape[1])])
    if method == "butterworth":
        from scipy import signal as sps

        sos = sps.butter(4, cutoff_hz, btype="lowpass", fs=fs, output="sos")
        return np.column_stack([sps.sosfiltfilt(sos, X[:, k]) for k in range(X.shape[1])])
    raise ParameterError(f"smooth_method must be one of {SMOOTH_METHODS}, got {method!r}")


def _circle(phase: np.ndarray) -> np.ndarray:
    p = np.asarray(phase, dtype=float)
    return np.column_stack([np.cos(p), np.sin(p)])


def _align(blocks: list[np.ndarray]) -> list[np.ndarray]:
    """Trim every block to the shortest, since delays shorten trajectories."""
    n = min(b.shape[0] for b in blocks)
    return [b[-n:] if b.shape[0] != n else b for b in blocks]


# ===================================================================== core
def build(rec, spec: Sequence[Mapping[str, Any]], *, route: str = "observable",
          scaling: str = "rms_balanced", lambda_: float | None = None,
          weights: Sequence[float] | None = None, builder: str = "build",
          rng=None) -> StateSpace:
    """Declarative constructor. Every named builder funnels through here.

    Parameters
    ----------
    spec
        List of entries. Each has ``source`` (channel name) and ``as`` (one of
        :data:`COORD_MODES`), plus mode-specific keys:

        * ``phase_circle`` -- maps phase to (cos, sin). No extra keys.
        * ``amplitude`` / ``raw`` / ``inst_freq`` -- one column as-is.
        * ``delay`` -- ``m`` and ``tau``; either may be ``"auto"``.
        * ``phase_raw`` -- the wrapped phase as one column. Discouraged: it
          has a metric discontinuity at 2*pi, which is exactly what the
          circle embedding exists to remove. Kept for comparison studies.

        Optional per-entry: ``label``, ``weight``, ``scale``
        (``zscore`` / ``robust`` / ``minmax`` / ``rank`` / ``none``).
    """
    if not spec:
        raise ParameterError("spec is empty: a state space needs at least one block")

    blocks: list[np.ndarray] = []
    metas: list[dict] = []
    emb_used: dict[str, Any] = {}
    smoothed: list[tuple[str, float]] = []

    for i, entry in enumerate(spec):
        mode = entry.get("as", "raw")
        if mode not in COORD_MODES:
            raise ParameterError(f"spec[{i}]: unknown 'as' {mode!r}; use one of {COORD_MODES}")
        src = entry.get("source")
        label = entry.get("label")

        if mode == "phase_circle":
            name = _resolve(rec, src, "phase", f"spec[{i}] phase_circle")
            X = _circle(rec.get(name))
            kind, dim = "phase_circle", 2
        elif mode == "phase_raw":
            name = _resolve(rec, src, "phase", f"spec[{i}] phase_raw")
            X = np.asarray(rec.get(name), dtype=float).reshape(-1, 1)
            kind, dim = "raw", 1
        elif mode in ("amplitude", "inst_freq", "raw"):
            role = {"amplitude": "amplitude", "inst_freq": "inst_freq", "raw": "raw"}[mode]
            name = _resolve(rec, src, role, f"spec[{i}] {mode}")
            X = np.asarray(rec.get(name), dtype=float).reshape(-1, 1)
            if entry.get("smooth"):
                X = _smooth(X, float(entry["smooth"]), rec.fs,
                            method=entry.get("smooth_method", "gaussian"))
                smoothed.append((name, float(entry["smooth"])))
            kind, dim = ("amplitude" if mode == "amplitude"
                         else "inst_freq" if mode == "inst_freq" else "raw"), 1
        else:  # delay
            if src is None:
                raise ParameterError(f"spec[{i}]: 'delay' needs an explicit source")
            if src not in rec.channels:
                raise ParameterError(f"spec[{i}]: no channel {src!r}; have {rec.names}")
            x = np.asarray(rec.get(src), dtype=float)
            m, tau = entry.get("m", "auto"), entry.get("tau", "auto")
            if m == "auto" or tau == "auto":
                est = estimate_embedding(
                    x, tau_method=entry.get("tau_method", "ami"),
                    m_method=entry.get("m_method", "fnn"),
                    tau_range=entry.get("tau_range", (1, min(200, x.size // 4))),
                    m_range=entry.get("m_range", (1, 10)), rng=rng,
                )
                tau = est.tau if tau == "auto" else int(tau)
                m = est.m if m == "auto" else int(m)
                emb_used[src] = {"tau": tau, "m": m,
                                 "tau_method": est.tau_method, "m_method": est.m_method}
            m, tau = int(m), int(tau)
            X = _embed(x, m, tau)
            kind, dim = "delay", m

        blocks.append(np.atleast_2d(X))
        metas.append({"label": label or (name if mode != "delay" else src),
                      "kind": kind, "dim": dim, "source": name if mode != "delay" else src,
                      "mode": mode,
                      "params": {**({"m": m, "tau": tau} if mode == "delay" else {}),
                                 **({"smooth_hz": float(entry["smooth"]),
                                     "smooth_method": entry.get("smooth_method", "gaussian")}
                                    if entry.get("smooth") and mode != "delay" else {}),
                                 **({"scale": entry["scale"]}
                                    if entry.get("scale") and entry["scale"] != "none"
                                    else {})},
                      "weight": entry.get("weight"), "scale": entry.get("scale")})

    # Per-block pre-scaling, then align lengths (delays shorten the record)
    from ..preprocess.scaling import scale_array

    for k, meta in enumerate(metas):
        if meta["scale"] and meta["scale"] != "none":
            blocks[k] = scale_array(blocks[k], policy=meta["scale"])

    n_before = [b.shape[0] for b in blocks]
    blocks = _align(blocks)
    n_after = blocks[0].shape[0]

    coords = np.column_stack(blocks)
    groups, pos = [], 0
    for meta in metas:
        groups.append(CoordGroup(
            label=meta["label"], start=pos, stop=pos + meta["dim"], kind=meta["kind"],
            source=meta["source"], params=meta["params"],
        ))
        pos += meta["dim"]

    caveats = []
    if len(set(n_before)) > 1:
        caveats.append(f"blocks had lengths {n_before}; aligned to the shortest "
                       f"({n_after}) by trimming from the start")
    for nm, hz in smoothed:
        caveats.append(f"envelope {nm!r} low-pass filtered at {hz:g} Hz; "
                       "faster envelope fluctuations are removed from the geometry")
    caveats = tuple(caveats)

    step = Step(
        "statespace",
        {"route": route, "builder": builder, "dim": coords.shape[1],
         "n_points": n_after,
         "blocks": [f"{m['label']}:{m['mode']}({m['dim']})" for m in metas],
         **({"embedding_auto": emb_used} if emb_used else {}),
         **({"smoothed_hz": dict(smoothed)} if smoothed else {})},
        inputs=tuple(m["source"] for m in metas),
        caveats=caveats,
    )

    t_offset = (rec.n_samples - n_after) / rec.fs
    ss = StateSpace(
        coords=coords, groups=tuple(groups), fs=rec.fs, t0=rec.t0 + t_offset,
        route=route, provenance=rec.provenance + (step,),
        meta={"subject": rec.subject, "builder": builder, **({"embedding": emb_used} if emb_used else {})},
    )

    explicit = [m["weight"] for m in metas]
    if any(w is not None for w in explicit):
        return ss.rescale("custom", weights=[1.0 if w is None else float(w) for w in explicit])
    if scaling and scaling != "none":
        return ss.rescale(scaling, lambda_=lambda_, weights=weights, rng=rng)
    return ss


# ======================================================= observable route
def phase_circle(rec, phase: str | None = None, **kw) -> StateSpace:
    """(cos phi, sin phi). The 2-D circular embedding of one phase."""
    require("phase_circle", rec.caps, Capability.PHASE)
    return build(rec, [{"source": phase, "as": "phase_circle"}],
                 builder="phase_circle", **kw)


def pac_space(rec, phase: str | None = None, amplitude: str | None = None,
              *, embed_amplitude: int | str | None = None, tau: int | str = "auto",
              smooth: float | None = None, smooth_method: str = "gaussian",
              scale_amplitude: str | None = None,
              **kw) -> StateSpace:
    """PAC: s(t) = (cos phi_low, sin phi_low, A_high).

    With ``embed_amplitude=m`` the envelope is delay-embedded in m dimensions
    while the phase circle stays intact -- the hybrid route.

    ``smooth`` low-passes the envelope [DD-29]: the cut-off in Hz, where the
    gain is 1/2. It must not cut below the upper edge of the phase band, or the
    modulation is removed before it is measured; when the band edges are
    recorded in ``rec.meta["bands"]`` a cut-off below them raises a
    :class:`GeometryWarning` [DD-112].

    ``smooth_method`` is ``"gaussian"`` (default) or ``"butterworth"``
    [DD-117]. The Gaussian kernel is non-negative, so the envelope never
    goes below zero, but its roll-off is gentle: with the cut-off at the upper
    edge of a 4-8 Hz phase band it keeps 68% of a 6 Hz modulation, against 91%
    for the Butterworth. Place the cut-off at about 1.5 times the band's upper
    edge (12 Hz for theta) to keep three quarters of the modulation at the
    edge itself. ``"butterworth"`` reproduces results from before 0.25.
    """
    require("pac_space", rec.caps, Capability.PHASE | Capability.AMPLITUDE,
            hint="run analytic() on a slow and a fast band first.")
    notes = []
    if phase is None:
        ph, n1 = _pick_by_frequency(rec, "phase", "slowest", "pac_space")
        notes.append(n1)
    else:
        ph = _resolve(rec, phase, "phase", "pac_space")
    if amplitude is None:
        am, n2 = _pick_by_frequency(rec, "amplitude", "fastest", "pac_space")
        notes.append(n2)
    else:
        am = _resolve(rec, amplitude, "amplitude", "pac_space")
    for n in notes:
        if n:
            _warn_choice(n)
    if embed_amplitude:
        spec = [{"source": ph, "as": "phase_circle"},
                {"source": am, "as": "delay", "m": embed_amplitude, "tau": tau}]
        return build(rec, spec, route="hybrid", builder="pac_space(hybrid)", **kw)

    # Design note (DD-112) -- the smoothing has to pass the phase band.
    #
    # The envelope is low-passed so that estimation noise faster than the
    # modulation does not fill the space [DD-29]. But the modulation the
    # space exists to show runs *at the phase band's frequency*: a cut-off
    # below the band's upper edge removes it by construction, and every
    # downstream metric then describes an envelope with the coupling
    # filtered out. 8 Hz is right for theta and blind to alpha or beta.
    # The edges are known when the band was recorded, and then this is a
    # mistake the library can see.
    edges = _band_edges(rec, ph)
    # Design note (DD-113) -- the amplitude band has to hold the sidebands.
    #
    # An envelope modulated at f Hz around a carrier at f_c lives at f_c +- f.
    # A band narrower than 2 f cannot pass both sidebands, and the modulation
    # is removed by the band-pass before any envelope exists to measure.
    # Measured on a synthetic 40 Hz response with a theta modulation of depth
    # 0.5 at 6 Hz: a 35-45 Hz band gives MVL 0.014, a 30-50 Hz band 0.203.
    amp_edges = _band_edges(rec, am)
    if edges is not None and amp_edges is not None:
        width = amp_edges[1] - amp_edges[0]
        if width < 2.0 * edges[1]:
            import warnings

            from ..exceptions import GeometryWarning

            warnings.warn(
                f"pac_space: the amplitude band behind {am!r} is {width:g} Hz wide "
                f"and the phase band behind {ph!r} reaches {edges[1]:g} Hz. A "
                f"modulation at f Hz needs sidebands at +-f, so the band must be "
                f"at least {2 * edges[1]:g} Hz wide to carry it; as built, the "
                "band-pass removes the coupling before the envelope is taken "
                "[DD-113].", GeometryWarning, stacklevel=2)
    if smooth and edges is not None and float(smooth) < edges[1]:
        import warnings

        from ..exceptions import GeometryWarning

        warnings.warn(
            f"pac_space: smooth={float(smooth):g} Hz is below the upper edge "
            f"({edges[1]:g} Hz) of the phase band behind {ph!r}. The envelope "
            "low-pass removes the modulation at the phase frequency, so the "
            f"space cannot show coupling with this band. Use smooth >= "
            f"{edges[1]:g} or smooth=None [DD-112].",
            GeometryWarning, stacklevel=2)
    return build(rec, [{"source": ph, "as": "phase_circle"},
                       {"source": am, "as": "amplitude", "smooth": smooth,
                        "smooth_method": smooth_method,
                        "scale": scale_amplitude}],
                 route="observable", builder="pac_space", **kw)


def ppc_space(rec, phase1: str | None = None, phase2: str | None = None,
              **kw) -> StateSpace:
    """PPC: two phase circles in one space. For CRP, use ``.split()`` after."""
    require("ppc_space", rec.caps, Capability.PHASE)
    ps = rec.by_role("phase")
    p1 = phase1 or (ps[0] if len(ps) >= 2 else _resolve(rec, phase1, "phase", "ppc_space"))
    p2 = phase2 or (ps[1] if len(ps) >= 2 else None)
    if p2 is None:
        raise ParameterError("ppc_space needs two phase channels")
    return build(rec, [{"source": p1, "as": "phase_circle", "label": p1},
                       {"source": p2, "as": "phase_circle", "label": p2}],
                 route="observable", builder="ppc_space", **kw)


def envelope_space(rec, amplitudes: Sequence[str] | None = None, *,
                   embed: int | None = None, tau: int | str = "auto", **kw) -> StateSpace:
    """AAC: the envelopes as coordinates. For JRP, build one space per envelope."""
    require("envelope_space", rec.caps, Capability.AMPLITUDE)
    names = list(amplitudes) if amplitudes else rec.by_role("amplitude")
    if not names:
        raise ParameterError("envelope_space: no amplitude channels found")
    if embed:
        spec = [{"source": n, "as": "delay", "m": embed, "tau": tau, "label": n}
                for n in names]
        return build(rec, spec, route="hybrid", builder="envelope_space(hybrid)", **kw)
    return build(rec, [{"source": n, "as": "amplitude", "label": n} for n in names],
                 route="observable", builder="envelope_space", **kw)


def ppa_space(rec, phase1: str, phase2: str, amplitude: str, **kw) -> StateSpace:
    """PPA: (cos p1, sin p1, cos p2, sin p2, A). Trivariate coupling, 5-D."""
    require("ppa_space", rec.caps, Capability.PHASE | Capability.AMPLITUDE)
    return build(rec, [{"source": phase1, "as": "phase_circle", "label": phase1},
                       {"source": phase2, "as": "phase_circle", "label": phase2},
                       {"source": amplitude, "as": "amplitude", "label": amplitude}],
                 route="observable", builder="ppa_space", **kw)


def nested_space(rec, hierarchy: Sequence[Mapping[str, str]], **kw) -> StateSpace:
    """Nested hierarchies, e.g. phi_delta -> A_theta -> A_gamma.

    ``hierarchy`` is a list of ``{"source": name, "as": mode}`` entries in
    order from slowest to fastest.
    """
    return build(rec, list(hierarchy), route="observable", builder="nested_space", **kw)


def freq_amp_space(rec, inst_freq: str | None = None, amplitude: str | None = None,
                   **kw) -> StateSpace:
    """FFC: (instantaneous frequency, amplitude)."""
    require("freq_amp_space", rec.caps, Capability.INST_FREQ | Capability.AMPLITUDE)
    f = _resolve(rec, inst_freq, "inst_freq", "freq_amp_space")
    a = _resolve(rec, amplitude, "amplitude", "freq_amp_space")
    return build(rec, [{"source": f, "as": "inst_freq"}, {"source": a, "as": "amplitude"}],
                 route="observable", builder="freq_amp_space", **kw)


def channels(rec, names: Sequence[str] | None = None, *, scaling: str = "rms_balanced",
             **kw) -> StateSpace:
    """Plain multivariate state space from raw columns. No coupling assumed."""
    if names is None:
        names = rec.by_role("state") or rec.by_role("raw") or rec.names
    spec = [{"source": n, "as": ("raw" if rec.roles.get(n) in ("raw", "state", "band")
                                 else rec.roles.get(n)), "label": n} for n in names]
    for e, n in zip(spec, names, strict=False):
        if rec.roles.get(n) in ("state", "band"):
            e["as"] = "raw"
    return build(rec, spec, route="observable", builder="channels", scaling=scaling, **kw)


def custom(rec, fn: Callable[[Any], np.ndarray], *, labels: Sequence[str] | None = None,
           kind: str = "derived", scaling: str = "none", **kw) -> StateSpace:
    """Escape hatch: any callable Recording -> (N, K) array."""
    X = np.atleast_2d(np.asarray(fn(rec), dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    groups = (CoordGroup(labels[0] if labels else "custom", 0, X.shape[1], kind=kind),)
    step = Step("statespace", {"route": "external", "builder": "custom",
                               "dim": X.shape[1], "n_points": X.shape[0]},
                notes=getattr(fn, "__name__", "callable"))
    ss = StateSpace(coords=X, groups=groups, fs=rec.fs, t0=rec.t0, route="external",
                    provenance=rec.provenance + (step,),
                    meta={"subject": rec.subject, "builder": "custom"})
    return ss.rescale(scaling) if scaling != "none" else ss


# ============================================================ delay route
def takens(rec, source: str | None = None, *, m: int | str = "auto",
           tau: int | str = "auto", scaling: str = "none", **kw) -> StateSpace:
    """Univariate Takens reconstruction."""
    src = source or (rec.by_role("raw") or rec.names)[0]
    return build(rec, [{"source": src, "as": "delay", "m": m, "tau": tau, "label": src}],
                 route="delay", builder="takens", scaling=scaling, **kw)


def takens_multivariate(rec, sources: Sequence[str] | None = None, *,
                        m: int | str = "auto", tau: int | str = "auto",
                        scaling: str = "rms_balanced", **kw) -> StateSpace:
    """Delay embedding of several channels, concatenated."""
    names = list(sources) if sources else (rec.by_role("raw") or rec.names)
    spec = [{"source": n, "as": "delay", "m": m, "tau": tau, "label": n} for n in names]
    return build(rec, spec, route="delay", builder="takens_multivariate",
                 scaling=scaling, **kw)


# ============================================================ declarative
def from_bands(rec, spec: Sequence[Mapping[str, Any]], *, scaling: str = "rms_balanced",
               filter_kwargs: Mapping[str, Any] | None = None,
               edge_policy: str = "trim", **kw) -> StateSpace:
    """From a raw signal straight to a state space: filter, Hilbert, build.

    Each entry: ``{"band": (lo, hi), "as": "phase"|"amplitude"|"inst_freq"|"delay"}``.
    """
    from ..preprocess import analytic, filterbank

    bands, modes = {}, []
    for i, e in enumerate(spec):
        if "band" not in e:
            raise ParameterError(f"from_bands spec[{i}] needs a 'band'")
        lo, hi = e["band"]
        label = e.get("label", f"b{lo:g}_{hi:g}")
        bands[label] = (lo, hi)
        modes.append({**e, "label": label})

    rec2 = filterbank(rec, bands, **(filter_kwargs or {}))
    rec2 = analytic(rec2, sources=list(bands), edge_policy=edge_policy)

    inner = []
    for e in modes:
        mode = e.get("as", "amplitude")
        label = e["label"]
        if mode == "phase":
            inner.append({"source": f"phase_{label}", "as": "phase_circle", "label": label,
                          "weight": e.get("weight"), "scale": e.get("scale")})
        elif mode == "amplitude":
            inner.append({"source": f"amp_{label}", "as": "amplitude", "label": label,
                          "weight": e.get("weight"), "scale": e.get("scale")})
        elif mode == "inst_freq":
            inner.append({"source": f"ifreq_{label}", "as": "inst_freq", "label": label})
        elif mode == "delay":
            inner.append({"source": f"amp_{label}", "as": "delay", "label": label,
                          "m": e.get("m", "auto"), "tau": e.get("tau", "auto")})
        else:
            raise ParameterError(f"from_bands: unknown 'as' {mode!r}")

    route = "hybrid" if any(e.get("as") == "delay" for e in modes) else "observable"
    return build(rec2, inner, route=route, builder="from_bands", scaling=scaling, **kw)
