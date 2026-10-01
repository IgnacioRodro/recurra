import numpy as np
import pytest

import recurra as rc
from recurra.capabilities import Capability, capabilities_from_roles, require


def test_flags_compose():
    c = capabilities_from_roles({"a": "phase", "b": "amplitude"})
    assert Capability.PHASE in c and Capability.AMPLITUDE in c
    assert Capability.RAW not in c


def test_require_raises_with_named_missing():
    with pytest.raises(rc.CapabilityError) as e:
        require("demo", Capability.RAW, Capability.RAW | Capability.PHASE)
    assert "PHASE" in str(e.value)


def test_stage_refuses_without_capability():
    r = rc.ingest(np.random.default_rng(2100).standard_normal((500, 3)), fs=100.0, kind="statespace")
    with pytest.raises(rc.CapabilityError):
        rc.bandpass(r, (4, 8))
