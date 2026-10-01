"""Exception hierarchy for recurra.

Design note (DD-09): every failure a user can act on gets its own exception
type carrying enough context to fix it. Bare ValueError is reserved for
genuine programming errors.
"""
from __future__ import annotations


class RecurraError(Exception):
    """Base class for all recurra errors."""


class RecurraWarning(UserWarning):
    """Base class for every warning the library raises.

    Design note (DD-46). The library warns about things it did without being
    asked -- edge trimming, envelope smoothing, a channel chosen by frequency,
    a block dominating the geometry, records that are not comparable. Those
    are valuable during analysis and noise during a test run, and a user who
    wants to quieten them should not have to silence every warning in the
    process, including the ones numpy and scipy raise for good reason.

    So they carry their own category:

        warnings.simplefilter("ignore", recurra.RecurraWarning)

    and finer control is available through the subclasses.
    """


class CaveatWarning(RecurraWarning):
    """Something was done that the caller did not explicitly ask for.

    Always also recorded in the provenance trail, so silencing the warning
    loses the notification, not the record.
    """


class InferenceWarning(RecurraWarning):
    """A choice was made on the caller's behalf, such as which channel to use."""


class GeometryWarning(RecurraWarning):
    """The geometry of the state space may not measure what was intended."""


class ParameterWarning(RecurraWarning):
    """A parameter combination is valid but unlikely to do what was intended."""


class ComparabilityWarning(RecurraWarning):
    """Analyses that are about to be compared were not built the same way."""


class CapabilityError(RecurraError):
    """A stage was asked to run without the data it requires.

    Raised at *plan* time, not halfway through a computation.
    """

    def __init__(self, stage: str, required, available, hint: str = ""):
        self.stage = stage
        self.required = required
        self.available = available
        missing = required & ~available
        msg = (
            f"stage {stage!r} requires {required!s} but the recording only "
            f"provides {available!s} (missing: {missing!s})."
        )
        if hint:
            msg += f" {hint}"
        super().__init__(msg)


class IngestError(RecurraError):
    """The input could not be interpreted as a Recording."""


class QualityError(RecurraError):
    """Input data failed a hard validation check."""


class ParameterError(RecurraError):
    """A parameter is outside its valid domain, or a combination is invalid."""


class BackendUnavailable(RecurraError):
    """An optional backend was requested but is not installed."""

    def __init__(self, backend: str, extra: str):
        super().__init__(
            f"backend {backend!r} is not available. Install it with: "
            f'pip install "recurra[{extra}]"'
        )


class MemoryBudgetError(RecurraError):
    """The requested computation does not fit the declared memory budget."""
