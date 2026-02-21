"""
Tests for parallel_mc.particle module.
"""
import numpy as np
import pytest

from parallel_mc.particle import (
    ParticleBank,
    FissionBank,
    sample_isotropic,
    sample_from_cdf,
)
from parallel_mc.constants import N_GROUPS


class TestParticleBankCreateEmpty:
    def test_all_dead(self):
        bank = ParticleBank.create_empty(50)
        assert np.all(~bank.alive), "create_empty should produce all dead particles"

    def test_correct_count(self):
        bank = ParticleBank.create_empty(50)
        assert bank.n_particles == 50
        assert bank.n_alive == 0

    def test_array_shapes(self):
        n = 50
        bank = ParticleBank.create_empty(n)
        for attr in ("x", "y", "z", "ux", "uy", "uz", "weight"):
            assert getattr(bank, attr).shape == (n,), f"Wrong shape for {attr}"
        assert bank.group.shape == (n,)
        assert bank.alive.shape == (n,)


class TestParticleBankCreateSource:
    def test_all_alive(self, source_bank):
        assert np.all(source_bank.alive), "create_source should produce all alive particles"

    def test_correct_count(self, source_bank):
        assert source_bank.n_particles == 1000
        assert source_bank.n_alive == 1000

    def test_positions_in_core(self, source_bank, geometry):
        r = np.sqrt(source_bank.x**2 + source_bank.y**2)
        assert np.all(r < geometry.core_radius), "Source particles outside core radially"
        assert np.all(np.abs(source_bank.z) < geometry.core_half_height), (
            "Source particles outside core axially"
        )

    def test_weights_are_one(self, source_bank):
        np.testing.assert_array_equal(source_bank.weight, 1.0)

    def test_groups_in_range(self, source_bank):
        assert np.all(source_bank.group >= 0)
        assert np.all(source_bank.group < N_GROUPS)

    def test_directions_are_unit_vectors(self, source_bank):
        mag = np.sqrt(source_bank.ux**2 + source_bank.uy**2 + source_bank.uz**2)
        np.testing.assert_allclose(mag, 1.0, atol=1e-12)


class TestSampleIsotropic:
    def test_produces_unit_vectors(self, rng):
        ux, uy, uz = sample_isotropic(rng, 1000)
        mag = np.sqrt(ux**2 + uy**2 + uz**2)
        np.testing.assert_allclose(mag, 1.0, atol=1e-12,
                                   err_msg="sample_isotropic produced non-unit vectors")

    def test_mean_near_zero(self, rng):
        """Isotropic distribution has zero mean component."""
        ux, uy, uz = sample_isotropic(rng, 5000)
        assert abs(np.mean(ux)) < 0.05, "ux mean too far from 0"
        assert abs(np.mean(uy)) < 0.05, "uy mean too far from 0"
        assert abs(np.mean(uz)) < 0.05, "uz mean too far from 0"

    def test_correct_shape(self, rng):
        n = 100
        ux, uy, uz = sample_isotropic(rng, n)
        assert ux.shape == (n,)
        assert uy.shape == (n,)
        assert uz.shape == (n,)


class TestSampleFromCdf:
    def test_uniform_distribution(self, rng):
        """Uniform probabilities -> each bin gets ~1/N samples."""
        n_bins = 4
        probs = np.ones(n_bins) / n_bins
        n_samples = 10000
        samples = sample_from_cdf(probs, rng, n_samples)
        for g in range(n_bins):
            fraction = np.sum(samples == g) / n_samples
            assert abs(fraction - 0.25) < 0.03, (
                f"Bin {g} fraction {fraction:.3f} far from expected 0.25"
            )

    def test_concentrated_distribution(self, rng):
        """All probability in one bin -> all samples in that bin."""
        probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        samples = sample_from_cdf(probs, rng, 200)
        assert np.all(samples == 2)

    def test_samples_in_valid_range(self, rng):
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0])
        samples = sample_from_cdf(probs, rng, 1000)
        assert np.all(samples >= 0)
        assert np.all(samples < N_GROUPS)


class TestFissionBank:
    def test_initial_count_zero(self):
        fb = FissionBank()
        assert fb.count == 0

    def test_add_increments_count(self):
        fb = FissionBank()
        fb.add(0.1, 0.2, 0.3, 1)
        assert fb.count == 1

    def test_add_stores_values(self):
        fb = FissionBank()
        fb.add(1.0, 2.0, 3.0, 5)
        assert fb.x[0] == pytest.approx(1.0)
        assert fb.y[0] == pytest.approx(2.0)
        assert fb.z[0] == pytest.approx(3.0)
        assert fb.group[0] == 5

    def test_clear_resets_count(self):
        fb = FissionBank()
        fb.add(0.0, 0.0, 0.0, 0)
        fb.add(1.0, 1.0, 1.0, 1)
        fb.clear()
        assert fb.count == 0

    def test_to_particle_bank_correct_count(self, rng, fuel):
        fb = FissionBank()
        for i in range(50):
            fb.add(float(i) * 0.01, 0.0, 0.0, 0)

        n_target = 100
        bank = fb.to_particle_bank(n_target, fuel.chi, rng)
        assert bank.n_particles == n_target
        assert bank.n_alive == n_target

    def test_to_particle_bank_all_alive(self, rng, fuel):
        fb = FissionBank()
        for _ in range(20):
            fb.add(0.0, 0.0, 0.0, 0)
        bank = fb.to_particle_bank(50, fuel.chi, rng)
        assert np.all(bank.alive)


class TestParticleBankSplit:
    def test_total_count_preserved(self, source_bank):
        chunks = source_bank.split(3)
        total = sum(c.n_particles for c in chunks)
        assert total == source_bank.n_particles

    def test_correct_number_of_chunks(self, source_bank):
        n_chunks = 4
        chunks = source_bank.split(n_chunks)
        assert len(chunks) == n_chunks

    def test_chunks_are_particle_banks(self, source_bank):
        chunks = source_bank.split(2)
        for chunk in chunks:
            assert isinstance(chunk, ParticleBank)

    def test_split_into_one(self, source_bank):
        chunks = source_bank.split(1)
        assert len(chunks) == 1
        assert chunks[0].n_particles == source_bank.n_particles
