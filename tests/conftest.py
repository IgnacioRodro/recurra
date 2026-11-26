import importlib.util

import numpy as np
import pytest

import recurra as rc


@pytest.fixture(autouse=True)
def _one_core_per_test_worker():
    """Batch functions take every physical core by default [DD-126]. Under
    ``pytest -n auto`` each test worker already holds a core, so a test that
    did the same would start one process per core from each of them. Tests run
    on one core unless they ask; ``test_parallel.py`` exercises the default.
    """
    before = rc.get_config().n_jobs
    rc.set_config(n_jobs=1)
    yield
    rc.set_config(n_jobs=before)


@pytest.fixture
def pac_signal():
    return rc.generate_cfc(modality="pac", duration=12.0, fs=500.0, alpha=0.8,
                           preferred_phase=np.pi / 2, snr_db=20.0, seed=42)


@pytest.fixture
def rec(pac_signal):
    return pac_signal.to_recording("test01")


def pytest_runtest_setup(item):
    """Skip a test whose optional backend is not installed [DD-17, P6].

    Explicit, per test: a blanket rule turning ImportError into a skip would
    also hide the bug this suite most needs to catch, an optional backend
    imported on a core code path. With only the core installed, every test
    without this marker must pass.
    """
    for marker in item.iter_markers("requires"):
        missing = [m for m in marker.args if importlib.util.find_spec(m) is None]
        if missing:
            pytest.skip(f"optional backend not installed: {', '.join(missing)}")
    if item.get_closest_marker("cuda") is not None:
        # The GPU tests run only where PyTorch sees a CUDA device; continuous
        # integration has none, so there they are skipped [DD-125].
        from recurra.engine import cuda_devices

        if not cuda_devices():
            pytest.skip("no CUDA device visible to PyTorch")
