"""StateSpace: the point cloud on which every recurrence structure is built.

Design note (DD-21) -- weights fold into the coordinates.

The combined distance across heterogeneous blocks is

    d^2(x, y) = sum_g w_g * ||x_g - y_g||^2

Rather than teach every downstream engine about blocks and weights, note
that this is exactly the plain Euclidean distance on rescaled coordinates:

    d^2 = || sqrt(w_g) * (x_g - y_g) ||^2

So ``StateSpace.weighted_coords`` returns the array on which ordinary
Euclidean distance already *is* the grouped weighted distance. The whole
scaling apparatus [DD-02] collapses into one array transform, and the
recurrence engine stays ignorant of phases, amplitudes and blocks.

(This identity holds for Euclidean block metrics. Non-Euclidean block
metrics -- Chebyshev per block, angular -- need the explicit grouped path,
which the recurrence layer will provide in F5.)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ..config import SeedLike
from ..exceptions import GeometryWarning, ParameterError
from ..preprocess.scaling import Scaler, ScaleReport
from ..provenance import Step, provenance_frame

#: What a block of coordinates represents. Drives default metric and scaling.
COORD_KINDS = ("phase_circle", "amplitude", "delay", "raw", "inst_freq", "derived")

#: Routes to a state space, per the project specification: observable variables vs Takens.
ROUTES = ("observable", "delay", "hybrid", "external")


@dataclass(frozen=True)
class CoordGroup:
    """A contiguous block of coordinates with its own geometry."""

    label: str
    start: int
    stop: int
    kind: str = "raw"
    metric: str = "euclidean"
    weight: float = 1.0
    source: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def dim(self) -> int:
        return self.stop - self.start

    def __post_init__(self) -> None:
        if self.kind not in COORD_KINDS:
            raise ParameterError(f"unknown coord kind {self.kind!r}; use one of {COORD_KINDS}")
        if self.stop <= self.start:
            raise ParameterError(f"empty coord group {self.label!r}")


@dataclass(frozen=True)
class StateSpace:
    """A trajectory in state space, plus how it was built.

    Attributes
    ----------
    coords : (N, K) array
        Raw (unweighted) coordinates.
    groups : tuple of CoordGroup
        Contiguous blocks partitioning the K columns.
    """

    coords: np.ndarray
    groups: tuple[CoordGroup, ...]
    fs: float
    t0: float = 0.0
    route: str = "observable"
    scaling: str = "none"
    scale_report: ScaleReport | None = None
    provenance: tuple[Step, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        C = np.asarray(self.coords, dtype=float)
        if C.ndim != 2:
            raise ParameterError(f"coords must be 2-D (N, K), got shape {C.shape}")
        if C.shape[0] < 2:
            raise ParameterError(f"state space needs at least 2 points, got {C.shape[0]}")
        covered = sum(g.dim for g in self.groups)
        if covered != C.shape[1]:
            raise ParameterError(
                f"groups cover {covered} columns but coords has {C.shape[1]}"
            )
        if C.flags.writeable:
            C = C.view()
            C.flags.writeable = False
        object.__setattr__(self, "coords", C)
        if self.route not in ROUTES:
            raise ParameterError(f"unknown route {self.route!r}; use one of {ROUTES}")

    # ------------------------------------------------------------ shape
    @property
    def n_points(self) -> int:
        return self.coords.shape[0]

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    @property
    def labels(self) -> list[str]:
        out = []
        for g in self.groups:
            if g.dim == 1:
                out.append(g.label)
            elif g.kind == "phase_circle":
                out += [f"cos({g.label})", f"sin({g.label})"]
            else:
                out += [f"{g.label}[{i}]" for i in range(g.dim)]
        return out

    @property
    def block_labels(self) -> list[str]:
        return [g.label for g in self.groups]

    @property
    def duration(self) -> float:
        return self.n_points / self.fs

    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.n_points) / self.fs

    # ------------------------------------------------------------ access
    def block(self, label: str) -> np.ndarray:
        for g in self.groups:
            if g.label == label:
                return self.coords[:, g.slice]
        raise ParameterError(f"no block {label!r}; have {self.block_labels}")

    def group(self, label: str) -> CoordGroup:
        for g in self.groups:
            if g.label == label:
                return g
        raise ParameterError(f"no block {label!r}; have {self.block_labels}")

    @property
    def weights(self) -> np.ndarray:
        """Per-column weight vector."""
        w = np.ones(self.dim)
        for g in self.groups:
            w[g.slice] = g.weight
        return w

    @property
    def weighted_coords(self) -> np.ndarray:
        """Coordinates on which plain Euclidean distance is the grouped one."""
        return self.coords * np.sqrt(self.weights)

    def blocks(self) -> list[tuple[str, np.ndarray]]:
        return [(g.label, self.coords[:, g.slice]) for g in self.groups]

    # --------------------------------------------------------- transform
    def rescale(self, policy: str = "rms_balanced", *, lambda_: float | None = None,
                weights: Sequence[float] | None = None, rng: SeedLike = None,
                estimator: str = "theoretical") -> StateSpace:
        """Recompute block weights under a scaling policy [DD-02]."""
        scaler = Scaler(policy, lambda_=lambda_, weights=weights, rng=rng,
                        estimator=estimator)
        report = scaler.fit(self.blocks())
        new_groups = tuple(
            replace(g, weight=float(w)) for g, w in zip(self.groups, report.weights, strict=False)
        )
        cav = tuple(c for c in (report.warning, report.shape_warning) if c)
        step = Step(
            "rescale",
            {"policy": policy, "lambda": lambda_,
             "weights": [round(float(w), 6) for w in report.weights],
             "variance_share": [round(float(s), 6) for s in report.variance_share],
             "tail_ratio": [round(float(t), 3) for t in report.tail_ratio]},
            caveats=cav,
        )
        out = replace(self, groups=new_groups, scaling=policy, scale_report=report,
                      provenance=self.provenance + (step,))
        _maybe_warn(report)
        return out

    def reweight(self, lambda_: float) -> StateSpace:
        """Shorthand for the two-block lambda policy."""
        return self.rescale("weighted", lambda_=lambda_)

    def subsample(self, factor: int | None = None, *, indices=None,
                  max_points: int | None = None, rng: SeedLike = None
                  ) -> StateSpace:
        """Decimate the trajectory. Recorded, never silent [P3].

        Design note (DD-47) -- decimation changes the sampling rate, and the
        object must say so.

        Keeping every k-th point means consecutive states are k original
        samples apart, so the effective rate is fs/k. An earlier version left
        ``fs`` untouched, and the space then reported a duration k times
        shorter than the stretch of signal it actually covered -- which fed
        wrong times into figure axes, window boundaries and density profiles.
        The rate is now divided by the realised decimation factor.

        **Decimation changes the metrics, not only the cost** [DD-80]. Measured
        at factor 2: RR, DET and LAM are unchanged; L, TT and Vmax halve, being
        counts of samples; Lmax collapses from 9001 to 157 and should not be
        reported from decimated data. Factor 5 already moves DET from 0.9996 to
        0.9512, and factor 10 to 0.7753.

        **No anti-alias filter is applied**: every k-th point is kept and the
        rest dropped. That is exact for coordinates whose content lies below
        the new Nyquist rate, which holds for a phase circle or an envelope
        of a band ending at 80 Hz down to factor 2 at 500 Hz, and fails past
        it -- part of what DD-80 measured at factor 5 is the gamma circle
        folding back, not the metrics reacting to sparser sampling. Resample
        the recording first when a larger factor is wanted.

        With irregular explicit ``indices`` there is no single rate to report.
        The rate is left as it was, ``t0`` becomes meaningless, and a warning
        says so: nothing downstream that is expressed in seconds can be
        trusted for such a space.
        """
        n = self.n_points
        caveats: list[str] = []
        irregular = False

        if indices is not None:
            idx = np.asarray(indices, dtype=int)
            mode = "explicit"
            gaps = np.unique(np.diff(np.sort(idx))) if idx.size > 1 else np.array([1])
            irregular = gaps.size > 1
            step = float(gaps[0]) if not irregular else float(np.mean(np.diff(np.sort(idx))))
        elif factor is not None:
            step = float(int(factor))
            idx = np.arange(0, n, int(factor))
            mode = f"stride:{factor}"
        elif max_points is not None and n > max_points:
            # A uniform stride rather than linspace: linspace leaves unequal
            # gaps, which destroys the notion of a sampling rate for the sake
            # of hitting the point count exactly. Returning slightly fewer
            # points with a well-defined rate is the better trade.
            step = float(int(np.ceil(n / int(max_points))))
            idx = np.arange(0, n, int(step))
            mode = f"max_points:{max_points}"
        else:
            return self

        if irregular:
            fs_new = self.fs
            caveats.append(
                "decimation is not uniform, so there is no single effective "
                "sampling rate; every quantity expressed in seconds (durations, "
                "figure time axes, window boundaries) is meaningless for this "
                "space. Use a uniform factor, or resample the recording before "
                "building the space."
            )
        else:
            fs_new = self.fs / step
            caveats.append(
                f"trajectory decimated {n} -> {idx.size} points (factor "
                f"{step:.4g}); the effective sampling rate drops from "
                f"{self.fs:g} Hz to {fs_new:g} Hz, and every rate-dependent "
                "quantity follows. Points are dropped, not filtered: any "
                f"coordinate with content above {fs_new / 2:g} Hz folds back"
            )

        step_rec = Step("subsample",
                        {"mode": mode, "n_before": n, "n_after": int(idx.size),
                         "factor": round(float(step), 6),
                         "fs_before": self.fs, "fs_after": round(float(fs_new), 6),
                         "uniform": not irregular},
                        caveats=tuple(caveats))
        out = replace(self, coords=self.coords[idx], fs=float(fs_new),
                      provenance=self.provenance + (step_rec,))
        _warn_caveats(caveats)
        return out

    def window(self, start: int, stop: int) -> StateSpace:
        return replace(
            self, coords=self.coords[start:stop], t0=self.t0 + start / self.fs,
            provenance=self.provenance + (Step("window", {"start": start, "stop": stop}),),
        )

    def reduce(self, method: str = "pca", n_components: int = 3) -> StateSpace:
        """Dimensionality reduction (contingency for high-N interpretability)."""
        X = self.weighted_coords
        if method == "pca":
            Xc = X - X.mean(axis=0)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(n_components, Vt.shape[0])
            Y = Xc @ Vt[:k].T
            explained = (S[:k] ** 2 / np.sum(S**2)).tolist()
        elif method == "umap":  # pragma: no cover - optional
            try:
                import umap
            except ImportError as e:
                from ..exceptions import BackendUnavailable

                raise BackendUnavailable("umap", "viz") from e
            Y = umap.UMAP(n_components=n_components).fit_transform(X)
            explained = []
        else:
            raise ParameterError(f"unknown reduction {method!r}; use 'pca' or 'umap'")

        groups = (CoordGroup("reduced", 0, Y.shape[1], kind="derived",
                             params={"method": method}),)
        step = Step("reduce", {"method": method, "n_components": Y.shape[1],
                               "explained_variance": [round(e, 4) for e in explained]},
                    caveats=("dimensionality reduced; block structure and physical "
                             "interpretation of coordinates are lost",))
        return replace(self, coords=Y, groups=groups, scaling="none",
                       scale_report=None, provenance=self.provenance + (step,))

    def split(self, labels: Sequence[str]) -> StateSpace:
        """Sub-space made of selected blocks (for CRP / JRP operands)."""
        gs = [self.group(lbl) for lbl in labels]
        X = np.column_stack([self.coords[:, g.slice] for g in gs])
        new, pos = [], 0
        for g in gs:
            new.append(replace(g, start=pos, stop=pos + g.dim))
            pos += g.dim
        return replace(self, coords=X, groups=tuple(new),
                       provenance=self.provenance + (Step("split", {"blocks": list(labels)}),))

    # --------------------------------------------------------- reporting
    def scale_frame(self) -> pd.DataFrame:
        if self.scale_report is not None:
            df = self.scale_report.to_frame()
        else:
            # No ScaleReport, because no policy was applied. The diagnostic is
            # wanted here more than anywhere: an unscaled space is the one most
            # likely to have a dominant or heavy-tailed block [DD-28]. So the
            # shape statistics are computed rather than left out.
            from ..preprocess.scaling import shape_stats, theoretical_rms_pairwise

            rms = [theoretical_rms_pairwise(X) for _, X in self.blocks()]
            shapes = [shape_stats(X) for _, X in self.blocks()]
            contrib = np.array([g.weight * r**2 for g, r in zip(self.groups, rms, strict=False)])
            tot = contrib.sum() or 1.0
            df = pd.DataFrame({"block": self.block_labels, "rms_pairwise": rms,
                               "weight": [g.weight for g in self.groups],
                               "variance_share": contrib / tot,
                               "policy": self.scaling,
                               "tail_ratio": [sh["tail_ratio"] for sh in shapes],
                               "skewness": [sh["skew"] for sh in shapes]})
        df.insert(0, "route", self.route)
        df["kind"] = [g.kind for g in self.groups]
        df["dim"] = [g.dim for g in self.groups]
        return df

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(self.coords), columns=self.labels,
                            index=self.times())

    def provenance_frame(self) -> pd.DataFrame:
        return provenance_frame(self.provenance)

    def describe(self) -> pd.DataFrame:
        """One row: the compact record of this state space, for CSV export."""
        sh = self.scale_frame()["variance_share"].to_numpy()
        return pd.DataFrame([{
            "route": self.route, "dim": self.dim, "n_points": self.n_points,
            "fs_hz": self.fs, "duration_s": round(self.duration, 3),
            "n_blocks": len(self.groups),
            "blocks": "+".join(f"{g.label}({g.dim})" for g in self.groups),
            "kinds": "+".join(g.kind for g in self.groups),
            "scaling": self.scaling,
            "weights": ";".join(f"{g.weight:.4g}" for g in self.groups),
            "variance_share": ";".join(f"{s:.3f}" for s in sh),
            "max_share": float(sh.max()) if sh.size else np.nan,
            "builder": self.meta.get("builder", ""),
        }])

    def summary(self) -> str:
        lines = [
            f"StateSpace(route={self.route!r}, {self.n_points} points, dim={self.dim}, "
            f"{self.duration:.2f} s @ {self.fs:g} Hz)",
            f"  scaling : {self.scaling}",
            "  blocks  :",
        ]
        sh = self.scale_frame()["variance_share"].to_numpy()
        for g, s in zip(self.groups, sh, strict=False):
            lines.append(
                f"    {g.label:<18s} dim={g.dim} kind={g.kind:<12s} "
                f"w={g.weight:9.4g}  share={s:6.1%}"
            )
        if self.scale_report:
            for msg in (self.scale_report.warning, self.scale_report.shape_warning):
                if msg:
                    lines.append(f"  WARNING : {msg}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<StateSpace {self.route} N={self.n_points} K={self.dim}>"


def _warn_caveats(messages) -> None:
    from ..config import get_config

    if not get_config().warn_on_caveat:
        return
    import warnings

    from ..exceptions import CaveatWarning

    for msg in messages:
        warnings.warn(f"statespace: {msg}", CaveatWarning, stacklevel=3)


def _maybe_warn(report: ScaleReport) -> None:
    from ..config import get_config

    if not get_config().warn_on_caveat:
        return
    import warnings

    for msg in (report.warning, report.shape_warning):
        if msg:
            warnings.warn(f"statespace: {msg}", GeometryWarning, stacklevel=3)


def make_groups(specs: list[tuple[str, int, str, str, str]]) -> tuple[CoordGroup, ...]:
    """Build contiguous groups from (label, dim, kind, metric, source) tuples."""
    groups, pos = [], 0
    for label, dim, kind, metric, source in specs:
        groups.append(CoordGroup(label, pos, pos + dim, kind=kind, metric=metric,
                                 source=source))
        pos += dim
    return tuple(groups)
