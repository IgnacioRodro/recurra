"""Capability flags: what a Recording currently holds.

Design note (DD-03). Each processing stage declares what it *requires* and
what it *provides*; a Recording carries what it *has*. This makes three
things possible:

1. The user enters the pipeline at any point (raw signal, already-Hilbert
   components, a prebuilt state space) without pretending to have data they
   do not have.
2. Stages already covered get skipped instead of recomputed.
3. Impossible requests fail when the plan is built, naming the missing
   capability, rather than three stages later with a shape error.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Flag, auto


class Capability(Flag):
    """What information a Recording provides."""

    NONE = 0
    RAW = auto()          #: raw time-domain samples
    BANDS = auto()        #: band-limited components
    PHASE = auto()        #: instantaneous phase
    AMPLITUDE = auto()    #: instantaneous amplitude (envelope)
    INST_FREQ = auto()    #: instantaneous frequency
    STATESPACE = auto()   #: an explicit point cloud in state space
    THRESHOLD = auto()    #: a recurrence threshold has been fixed
    RECURRENCE = auto()   #: a recurrence structure is available

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self is Capability.NONE:
            return "NONE"
        return "|".join(
            c.name for c in Capability if c is not Capability.NONE and c in self
        )


#: Which capability a channel role grants.
ROLE_TO_CAPABILITY: dict[str, Capability] = {
    "raw": Capability.RAW,
    "band": Capability.BANDS,
    "phase": Capability.PHASE,
    "amplitude": Capability.AMPLITUDE,
    "inst_freq": Capability.INST_FREQ,
    "state": Capability.STATESPACE,
}

VALID_ROLES = frozenset(ROLE_TO_CAPABILITY)


def capabilities_from_roles(roles: Mapping[str, str]) -> Capability:
    """Derive the capability set implied by a mapping of channel -> role."""
    caps = Capability.NONE
    for role in roles.values():
        caps |= ROLE_TO_CAPABILITY.get(role, Capability.NONE)
    return caps


def require(stage: str, available: Capability, required: Capability, hint: str = "") -> None:
    """Raise CapabilityError unless ``available`` covers ``required``."""
    from .exceptions import CapabilityError

    if (required & ~available) != Capability.NONE:
        raise CapabilityError(stage, required, available, hint)
