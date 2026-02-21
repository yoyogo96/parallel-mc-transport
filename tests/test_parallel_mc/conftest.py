"""
Shared pytest fixtures for parallel_mc test suite.
"""
import numpy as np
import pytest

from parallel_mc.geometry import MCFRGeometry
from parallel_mc.materials import build_fuel_salt, build_reflector_beo
from parallel_mc.particle import ParticleBank, FissionBank
from parallel_mc.backends.cpu import CPUBackend


@pytest.fixture
def rng():
    """Numpy Generator with fixed seed for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def geometry():
    """Default MCFRGeometry instance."""
    return MCFRGeometry()


@pytest.fixture
def fuel():
    """Fuel salt material."""
    return build_fuel_salt()


@pytest.fixture
def reflector_beo():
    """BeO reflector material."""
    return build_reflector_beo()


@pytest.fixture
def materials(fuel, reflector_beo):
    """Materials dict keyed by region ID (0=core, 1=reflector)."""
    return {0: fuel, 1: reflector_beo}


@pytest.fixture
def source_bank(geometry, fuel, rng):
    """1000-particle source bank distributed uniformly in core."""
    return ParticleBank.create_source(1000, geometry, fuel.chi, rng)


@pytest.fixture
def cpu_backend():
    """CPUBackend with 2 workers (no Numba to keep tests fast)."""
    return CPUBackend(n_workers=2, use_numba=False)
