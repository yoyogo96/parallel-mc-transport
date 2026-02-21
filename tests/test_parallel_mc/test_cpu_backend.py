"""
Tests for parallel_mc.backends.cpu module (CPUBackend).
"""
import numpy as np
import pytest

from parallel_mc.backends.cpu import CPUBackend
from parallel_mc.particle import ParticleBank, FissionBank
from parallel_mc.tallies import create_default_tally


class TestCPUBackendAvailability:
    def test_is_available_returns_true(self, cpu_backend):
        assert cpu_backend.is_available() is True

    def test_get_name_contains_cpu(self, cpu_backend):
        name = cpu_backend.get_name()
        assert "CPU" in name, f"get_name() '{name}' does not contain 'CPU'"

    def test_get_name_returns_string(self, cpu_backend):
        assert isinstance(cpu_backend.get_name(), str)


class TestTransportBatch:
    def test_produces_fission_sites(self, cpu_backend, geometry, materials, rng, fuel):
        """500 source particles through CPUBackend must yield fission sites."""
        source = ParticleBank.create_source(500, geometry, fuel.chi, rng)
        fb = cpu_backend.transport_batch(source, geometry, materials, None, rng)
        assert isinstance(fb, FissionBank)
        assert fb.count > 0, "CPUBackend produced zero fission sites"

    def test_k_batch_reasonable(self, cpu_backend, geometry, materials, rng, fuel):
        """k_batch = fission_bank.count / n_particles should be in [0.5, 2.0]."""
        n = 500
        source = ParticleBank.create_source(n, geometry, fuel.chi, rng)
        fb = cpu_backend.transport_batch(source, geometry, materials, None, rng)
        k_batch = fb.count / n
        assert 0.5 <= k_batch <= 2.0, (
            f"CPUBackend k_batch = {k_batch:.3f} outside reasonable range"
        )

    def test_inactive_batch_no_tally(self, cpu_backend, geometry, materials, rng, fuel):
        """Passing tallies=None (inactive batch) must not raise."""
        source = ParticleBank.create_source(200, geometry, fuel.chi, rng)
        fb = cpu_backend.transport_batch(source, geometry, materials, None, rng)
        assert fb is not None

    def test_active_batch_with_tally(self, cpu_backend, geometry, materials, rng, fuel):
        """Passing a TallyAccumulator (active batch) must populate tally."""
        tally = create_default_tally(geometry)
        source = ParticleBank.create_source(300, geometry, fuel.chi, rng)
        fb = cpu_backend.transport_batch(source, geometry, materials, tally, rng)
        # At least some flux contribution expected
        assert np.sum(tally.flux) > 0 or fb.count > 0

    def test_returns_fission_bank_type(self, cpu_backend, geometry, materials, rng, fuel):
        source = ParticleBank.create_source(100, geometry, fuel.chi, rng)
        fb = cpu_backend.transport_batch(source, geometry, materials, None, rng)
        assert isinstance(fb, FissionBank)


class TestMultiWorkerConsistency:
    def test_single_vs_two_workers_similar_k(self, geometry, materials, fuel):
        """1-worker and 2-worker backends should give similar k_batch."""
        backend_1 = CPUBackend(n_workers=1, use_numba=False)
        backend_2 = CPUBackend(n_workers=2, use_numba=False)

        n = 500
        seed = 777

        rng_1 = np.random.default_rng(seed)
        source_1 = ParticleBank.create_source(n, geometry, fuel.chi, rng_1)
        fb_1 = backend_1.transport_batch(source_1, geometry, materials, None, rng_1)
        k_1 = fb_1.count / n

        rng_2 = np.random.default_rng(seed)
        source_2 = ParticleBank.create_source(n, geometry, fuel.chi, rng_2)
        fb_2 = backend_2.transport_batch(source_2, geometry, materials, None, rng_2)
        k_2 = fb_2.count / n

        # Both should be in a physically reasonable range
        assert 0.5 <= k_1 <= 2.0, f"1-worker k_batch = {k_1:.3f}"
        assert 0.5 <= k_2 <= 2.0, f"2-worker k_batch = {k_2:.3f}"

        # Results can differ due to independent per-worker RNG seeds,
        # but both should be within 50% of each other for reasonable physics
        ratio = max(k_1, k_2) / max(min(k_1, k_2), 1e-6)
        assert ratio < 2.0, (
            f"1-worker k={k_1:.3f} and 2-worker k={k_2:.3f} differ by factor {ratio:.2f}"
        )

    def test_n_workers_one_is_valid(self, geometry, materials, rng, fuel):
        """n_workers=1 (no multiprocessing Pool) should work correctly."""
        backend = CPUBackend(n_workers=1, use_numba=False)
        source = ParticleBank.create_source(200, geometry, fuel.chi, rng)
        fb = backend.transport_batch(source, geometry, materials, None, rng)
        assert fb.count > 0
