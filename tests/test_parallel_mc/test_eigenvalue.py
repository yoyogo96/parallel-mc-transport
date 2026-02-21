"""
Tests for parallel_mc.eigenvalue module.
"""
import json
import pytest
import numpy as np

from parallel_mc.eigenvalue import quick_run, EigenvalueResult, PowerIteration
from parallel_mc.backends.cpu import CPUBackend


# Use small numbers to keep tests fast
N_PARTICLES = 200
N_BATCHES = 15
N_INACTIVE = 5


@pytest.fixture
def quick_result(cpu_backend):
    """Run a minimal eigenvalue calculation and return the result."""
    return quick_run(
        backend=cpu_backend,
        n_particles=N_PARTICLES,
        n_batches=N_BATCHES,
        n_inactive=N_INACTIVE,
        seed=123,
    )


class TestQuickRunNoError:
    def test_runs_without_exception(self, cpu_backend):
        result = quick_run(
            backend=cpu_backend,
            n_particles=N_PARTICLES,
            n_batches=N_BATCHES,
            n_inactive=N_INACTIVE,
            seed=99,
        )
        assert result is not None


class TestEigenvalueResultStructure:
    def test_is_eigenvalue_result(self, quick_result):
        assert isinstance(quick_result, EigenvalueResult)

    def test_keff_in_valid_range(self, quick_result):
        assert 0 < quick_result.keff < 3, (
            f"keff = {quick_result.keff:.4f} outside (0, 3)"
        )

    def test_keff_std_non_negative(self, quick_result):
        assert quick_result.keff_std >= 0

    def test_keff_history_correct_length(self, quick_result):
        assert len(quick_result.keff_history) == N_BATCHES

    def test_keff_history_all_positive(self, quick_result):
        assert all(k > 0 for k in quick_result.keff_history), (
            "Some k_batch values are <= 0"
        )

    def test_entropy_history_correct_length(self, quick_result):
        # Entropy is computed every batch
        assert len(quick_result.entropy_history) == N_BATCHES

    def test_entropy_history_all_non_negative(self, quick_result):
        assert all(h >= 0 for h in quick_result.entropy_history)

    def test_n_particles_recorded(self, quick_result):
        assert quick_result.n_particles == N_PARTICLES

    def test_n_batches_recorded(self, quick_result):
        assert quick_result.n_batches == N_BATCHES

    def test_n_inactive_recorded(self, quick_result):
        assert quick_result.n_inactive == N_INACTIVE

    def test_n_active_recorded(self, quick_result):
        assert quick_result.n_active == N_BATCHES - N_INACTIVE

    def test_total_time_positive(self, quick_result):
        assert quick_result.total_time > 0

    def test_leakage_fraction_in_range(self, quick_result):
        assert 0.0 <= quick_result.leakage_fraction <= 1.0

    def test_flux_mean_shape(self, quick_result):
        """Flux mean should be (n_r, n_z, N_GROUPS)."""
        assert quick_result.flux_mean.ndim == 3

    def test_r_edges_and_z_edges_present(self, quick_result):
        assert len(quick_result.r_edges) > 1
        assert len(quick_result.z_edges) > 1


class TestToDict:
    def test_to_dict_returns_dict(self, quick_result):
        d = quick_result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_json_serializable(self, quick_result):
        d = quick_result.to_dict()
        json_str = json.dumps(d)  # should not raise
        parsed = json.loads(json_str)
        assert parsed['n_particles'] == N_PARTICLES

    def test_to_dict_contains_expected_keys(self, quick_result):
        d = quick_result.to_dict()
        expected_keys = {
            'keff', 'keff_std', 'keff_history', 'entropy_history',
            'leakage_fraction', 'total_time', 'backend_name',
            'n_particles', 'n_batches', 'n_inactive', 'n_active',
            'r_edges', 'z_edges',
        }
        for key in expected_keys:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_to_dict_keff_matches(self, quick_result):
        d = quick_result.to_dict()
        assert d['keff'] == pytest.approx(quick_result.keff)
