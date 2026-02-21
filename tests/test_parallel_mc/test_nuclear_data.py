"""
Tests for parallel_mc.nuclear_data module.
"""
import numpy as np
import pytest

from parallel_mc.nuclear_data import NUCLIDE_LIBRARY, NuclideData
from parallel_mc.constants import N_GROUPS

# Nuclides that can fission
FISSILE_NUCLIDES = {"U235", "U238"}
# Nuclides that should have zero fission XS
NON_FISSILE_NUCLIDES = {"Na23", "K39", "Cl35", "Cl37"}


class TestLibraryContents:
    def test_nuclide_count(self):
        """Library must contain exactly 11 nuclides."""
        assert len(NUCLIDE_LIBRARY) == 11

    def test_expected_keys_present(self):
        expected = {
            "U235", "U238",
            "Na23", "K39", "Cl35", "Cl37",
            "Fe56", "Cr52", "Ni58",
            "Be9", "O16",
        }
        assert set(NUCLIDE_LIBRARY.keys()) == expected

    def test_all_values_are_nuclide_data(self):
        for name, nuc in NUCLIDE_LIBRARY.items():
            assert isinstance(nuc, NuclideData), f"{name} is not a NuclideData"


class TestCrossSectionConsistency:
    """sigma_total = sigma_absorption + sum(scatter_matrix, axis=1)."""

    @pytest.mark.parametrize("name", list(NUCLIDE_LIBRARY.keys()))
    def test_sigma_total_identity(self, name):
        nuc = NUCLIDE_LIBRARY[name]
        sigma_t_from_parts = nuc.sigma_absorption + np.sum(nuc.scatter_matrix, axis=1)
        np.testing.assert_allclose(
            nuc.sigma_total, sigma_t_from_parts,
            rtol=1e-10,
            err_msg=f"{name}: sigma_total != sigma_a + row_sums(scatter_matrix)",
        )

    @pytest.mark.parametrize("name", list(NUCLIDE_LIBRARY.keys()))
    def test_all_cross_sections_non_negative(self, name):
        nuc = NUCLIDE_LIBRARY[name]
        assert np.all(nuc.sigma_total >= 0), f"{name}: negative sigma_total"
        assert np.all(nuc.sigma_absorption >= 0), f"{name}: negative sigma_absorption"
        assert np.all(nuc.sigma_capture >= 0), f"{name}: negative sigma_capture"
        assert np.all(nuc.sigma_fission >= 0), f"{name}: negative sigma_fission"
        assert np.all(nuc.nu >= 0), f"{name}: negative nu"

    @pytest.mark.parametrize("name", list(NUCLIDE_LIBRARY.keys()))
    def test_scatter_matrix_non_negative(self, name):
        smat = NUCLIDE_LIBRARY[name].scatter_matrix
        assert np.all(smat >= 0), f"{name}: negative scatter_matrix entry"

    @pytest.mark.parametrize("name", list(NUCLIDE_LIBRARY.keys()))
    def test_scatter_matrix_shape(self, name):
        smat = NUCLIDE_LIBRARY[name].scatter_matrix
        assert smat.shape == (N_GROUPS, N_GROUPS), (
            f"{name}: scatter_matrix shape {smat.shape} != ({N_GROUPS}, {N_GROUPS})"
        )


class TestChiNormalization:
    """Fission spectrum must sum to 1 for fissile nuclides."""

    @pytest.mark.parametrize("name", sorted(FISSILE_NUCLIDES))
    def test_chi_sums_to_one(self, name):
        chi = NUCLIDE_LIBRARY[name].chi
        assert np.sum(chi) == pytest.approx(1.0, abs=1e-9), (
            f"{name}: chi sums to {np.sum(chi):.6f}, expected 1.0"
        )

    @pytest.mark.parametrize("name", sorted(FISSILE_NUCLIDES))
    def test_chi_non_negative(self, name):
        chi = NUCLIDE_LIBRARY[name].chi
        assert np.all(chi >= 0), f"{name}: negative chi entry"


class TestFissileProperties:
    @pytest.mark.parametrize("name", sorted(FISSILE_NUCLIDES))
    def test_fissile_has_positive_sigma_fission(self, name):
        nuc = NUCLIDE_LIBRARY[name]
        assert np.any(nuc.sigma_fission > 0), (
            f"{name}: expected sigma_fission > 0 in at least one group"
        )

    @pytest.mark.parametrize("name", sorted(FISSILE_NUCLIDES))
    def test_fissile_has_positive_nu_in_fission_groups(self, name):
        nuc = NUCLIDE_LIBRARY[name]
        fission_groups = nuc.sigma_fission > 0
        assert np.any(nuc.nu[fission_groups] > 0), (
            f"{name}: nu == 0 in all groups where sigma_fission > 0"
        )


class TestNonFissileProperties:
    @pytest.mark.parametrize("name", sorted(NON_FISSILE_NUCLIDES))
    def test_non_fissile_sigma_fission_zero(self, name):
        nuc = NUCLIDE_LIBRARY[name]
        np.testing.assert_array_equal(
            nuc.sigma_fission, 0.0,
            err_msg=f"{name}: expected sigma_fission == 0 for all groups",
        )

    def test_u235_fission_in_fast_groups(self):
        """U-235 has non-zero fission XS in the first two fast groups."""
        u235 = NUCLIDE_LIBRARY["U235"]
        assert u235.sigma_fission[0] > 0, "U-235: no fission in group 0 (6-20 MeV)"
        assert u235.sigma_fission[1] > 0, "U-235: no fission in group 1 (2-6 MeV)"

    def test_u238_no_fission_in_thermal_groups(self):
        """U-238 fast fission stops below ~1.2 MeV (groups 3-7 = 0)."""
        u238 = NUCLIDE_LIBRARY["U238"]
        for g in range(3, N_GROUPS):
            assert u238.sigma_fission[g] == 0.0, (
                f"U-238: unexpected fission in group {g}"
            )
