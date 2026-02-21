"""
Tests for parallel_mc.materials module.
"""
import numpy as np
import pytest

from parallel_mc.materials import (
    Material,
    build_fuel_salt,
    build_reflector_beo,
    build_reflector_ss316h,
)
from parallel_mc.constants import N_GROUPS


class TestFuelSalt:
    def test_returns_material(self, fuel):
        assert isinstance(fuel, Material)

    def test_is_fissile(self, fuel):
        assert fuel.is_fissile is True

    def test_sigma_t_consistency(self, fuel):
        """sigma_t == sigma_a + row_sum(sigma_s) for every group."""
        sigma_t_check = fuel.sigma_a + np.sum(fuel.sigma_s, axis=1)
        np.testing.assert_allclose(
            fuel.sigma_t, sigma_t_check, rtol=1e-10,
            err_msg="Fuel: sigma_t != sigma_a + sum(sigma_s, axis=1)",
        )

    def test_all_macroscopic_xs_positive(self, fuel):
        assert np.all(fuel.sigma_t > 0), "Fuel: some sigma_t <= 0"
        assert np.all(fuel.sigma_a >= 0), "Fuel: some sigma_a < 0"
        assert np.all(fuel.sigma_s >= 0), "Fuel: some sigma_s < 0"
        assert np.all(fuel.sigma_f > 0), "Fuel: some sigma_f <= 0 (fissile material)"

    def test_sigma_t_reasonable_range(self, fuel):
        """Fast-spectrum fuel sigma_t should be 5-30 1/m per group."""
        assert np.all(fuel.sigma_t >= 5.0), (
            f"Fuel: sigma_t too low: {fuel.sigma_t}"
        )
        assert np.all(fuel.sigma_t <= 30.0), (
            f"Fuel: sigma_t too high: {fuel.sigma_t}"
        )

    def test_chi_sums_to_one(self, fuel):
        assert np.sum(fuel.chi) == pytest.approx(1.0, abs=1e-9)

    def test_chi_non_negative(self, fuel):
        assert np.all(fuel.chi >= 0)

    def test_correct_number_of_groups(self, fuel):
        assert fuel.sigma_t.shape == (N_GROUPS,)
        assert fuel.sigma_s.shape == (N_GROUPS, N_GROUPS)
        assert fuel.chi.shape == (N_GROUPS,)


class TestBeOReflector:
    def test_returns_material(self, reflector_beo):
        assert isinstance(reflector_beo, Material)

    def test_is_not_fissile(self, reflector_beo):
        assert reflector_beo.is_fissile is False

    def test_sigma_t_consistency(self, reflector_beo):
        sigma_t_check = reflector_beo.sigma_a + np.sum(reflector_beo.sigma_s, axis=1)
        np.testing.assert_allclose(
            reflector_beo.sigma_t, sigma_t_check, rtol=1e-10,
        )

    def test_all_macroscopic_xs_non_negative(self, reflector_beo):
        assert np.all(reflector_beo.sigma_t >= 0)
        assert np.all(reflector_beo.sigma_a >= 0)
        assert np.all(reflector_beo.sigma_s >= 0)

    def test_sigma_f_zero(self, reflector_beo):
        """BeO reflector has no fission cross-section."""
        np.testing.assert_array_equal(reflector_beo.sigma_f, 0.0)

    def test_correct_number_of_groups(self, reflector_beo):
        assert reflector_beo.sigma_t.shape == (N_GROUPS,)
        assert reflector_beo.sigma_s.shape == (N_GROUPS, N_GROUPS)


class TestSS316HReflector:
    @pytest.fixture
    def reflector_ss(self):
        return build_reflector_ss316h()

    def test_returns_material(self, reflector_ss):
        assert isinstance(reflector_ss, Material)

    def test_is_not_fissile(self, reflector_ss):
        assert reflector_ss.is_fissile is False

    def test_sigma_t_consistency(self, reflector_ss):
        sigma_t_check = reflector_ss.sigma_a + np.sum(reflector_ss.sigma_s, axis=1)
        np.testing.assert_allclose(
            reflector_ss.sigma_t, sigma_t_check, rtol=1e-10,
        )

    def test_all_macroscopic_xs_non_negative(self, reflector_ss):
        assert np.all(reflector_ss.sigma_t >= 0)
        assert np.all(reflector_ss.sigma_a >= 0)
        assert np.all(reflector_ss.sigma_s >= 0)

    def test_sigma_f_zero(self, reflector_ss):
        np.testing.assert_array_equal(reflector_ss.sigma_f, 0.0)

    def test_correct_number_of_groups(self, reflector_ss):
        assert reflector_ss.sigma_t.shape == (N_GROUPS,)
        assert reflector_ss.sigma_s.shape == (N_GROUPS, N_GROUPS)
