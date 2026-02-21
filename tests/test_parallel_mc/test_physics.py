"""
Tests for parallel_mc.physics module.
"""
import numpy as np
import pytest

from parallel_mc.physics import transport_particle, transport_batch_sequential
from parallel_mc.particle import ParticleBank, FissionBank
from parallel_mc.tallies import create_default_tally


class TestTransportParticle:
    def test_produces_fission_sites_from_core(self, geometry, materials, rng):
        """A particle starting in the fissile core should produce fission sites."""
        # Run many single-particle transports to get at least one fission event
        fission_bank = FissionBank(max_sites=1000)
        n_trials = 200

        for _ in range(n_trials):
            x, y, z = geometry.sample_in_core(rng, 1)
            ux, uy, uz = 1.0, 0.0, 0.0
            transport_particle(
                float(x[0]), float(y[0]), float(z[0]),
                ux, uy, uz,
                group=2,
                weight=1.0,
                geometry=geometry,
                materials=materials,
                fission_bank=fission_bank,
                tally_accumulator=None,
                rng=rng,
            )

        assert fission_bank.count > 0, (
            f"No fission sites produced after {n_trials} particles in core"
        )

    def test_fission_sites_inside_core(self, geometry, materials, rng):
        """All fission sites must be within the core volume."""
        fission_bank = FissionBank(max_sites=5000)

        for _ in range(300):
            x, y, z = geometry.sample_in_core(rng, 1)
            transport_particle(
                float(x[0]), float(y[0]), float(z[0]),
                1.0, 0.0, 0.0,
                group=1,
                weight=1.0,
                geometry=geometry,
                materials=materials,
                fission_bank=fission_bank,
                tally_accumulator=None,
                rng=rng,
            )

        n = fission_bank.count
        assert n > 0

        fx = fission_bank.x[:n]
        fy = fission_bank.y[:n]
        fz = fission_bank.z[:n]
        r = np.sqrt(fx**2 + fy**2)

        assert np.all(r <= geometry.core_radius + 1e-6), (
            "Fission sites found outside core radially"
        )
        assert np.all(np.abs(fz) <= geometry.core_half_height + 1e-6), (
            "Fission sites found outside core axially"
        )

    def test_with_tally_accumulator(self, geometry, materials, rng):
        """transport_particle must not crash when a tally is provided."""
        tally = create_default_tally(geometry)
        fission_bank = FissionBank(max_sites=500)

        for _ in range(50):
            x, y, z = geometry.sample_in_core(rng, 1)
            transport_particle(
                float(x[0]), float(y[0]), float(z[0]),
                0.0, 0.0, 1.0,
                group=0,
                weight=1.0,
                geometry=geometry,
                materials=materials,
                fission_bank=fission_bank,
                tally_accumulator=tally,
                rng=rng,
            )

        # Tally flux should have at least some scored contributions
        assert np.sum(tally.flux) > 0


class TestTransportBatchSequential:
    def test_produces_non_zero_fission_bank(self, geometry, materials, rng, fuel):
        """A batch of 100 core particles must produce fission sites."""
        source = ParticleBank.create_source(100, geometry, fuel.chi, rng)
        fission_bank = transport_batch_sequential(source, geometry, materials, None, rng)
        assert fission_bank.count > 0

    def test_k_batch_reasonable(self, geometry, materials, rng, fuel):
        """k_batch = fission_bank.count / n must be in [0.5, 2.0]."""
        n = 200
        source = ParticleBank.create_source(n, geometry, fuel.chi, rng)
        fission_bank = transport_batch_sequential(source, geometry, materials, None, rng)
        k_batch = fission_bank.count / n
        assert 0.5 <= k_batch <= 2.0, (
            f"k_batch = {k_batch:.3f} outside reasonable range [0.5, 2.0]"
        )

    def test_fission_sites_in_core(self, geometry, materials, rng, fuel):
        """All fission sites produced by sequential batch must be in the core."""
        source = ParticleBank.create_source(150, geometry, fuel.chi, rng)
        fb = transport_batch_sequential(source, geometry, materials, None, rng)
        n = fb.count
        assert n > 0

        r = np.sqrt(fb.x[:n]**2 + fb.y[:n]**2)
        assert np.all(r <= geometry.core_radius + 1e-6)
        assert np.all(np.abs(fb.z[:n]) <= geometry.core_half_height + 1e-6)

    def test_with_tally_no_crash(self, geometry, materials, rng, fuel):
        """transport_batch_sequential with tallies must complete without error."""
        tally = create_default_tally(geometry)
        source = ParticleBank.create_source(100, geometry, fuel.chi, rng)
        fb = transport_batch_sequential(source, geometry, materials, tally, rng)
        # At least some flux should have been scored
        assert np.sum(tally.flux) > 0 or fb.count > 0  # either counts as success
