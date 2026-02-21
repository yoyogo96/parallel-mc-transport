"""
Tests for parallel_mc.constants module.
"""
import numpy as np
import pytest

from parallel_mc.constants import (
    N_GROUPS,
    GROUP_BOUNDARIES_MEV,
    GROUP_CENTERS_MEV,
)


class TestGroupBoundaries:
    def test_boundary_count(self):
        """8 groups require 9 boundaries."""
        assert len(GROUP_BOUNDARIES_MEV) == 9

    def test_monotonically_decreasing(self):
        """Boundaries must decrease from high to low energy."""
        diffs = np.diff(GROUP_BOUNDARIES_MEV)
        assert np.all(diffs < 0), "Boundaries are not monotonically decreasing"

    def test_first_boundary_is_20_mev(self):
        assert GROUP_BOUNDARIES_MEV[0] == pytest.approx(20.0)

    def test_last_boundary_is_1e5_ev(self):
        """Last boundary is 1e-5 MeV = 10 eV."""
        assert GROUP_BOUNDARIES_MEV[-1] == pytest.approx(1e-5)

    def test_all_boundaries_positive(self):
        assert np.all(GROUP_BOUNDARIES_MEV > 0)


class TestNGroups:
    def test_n_groups_equals_8(self):
        assert N_GROUPS == 8


class TestGroupCenters:
    def test_center_count(self):
        """One geometric-mean center per group."""
        assert len(GROUP_CENTERS_MEV) == 8

    def test_centers_within_boundaries(self):
        """Each center must lie strictly between its group boundaries."""
        for g in range(N_GROUPS):
            lo = GROUP_BOUNDARIES_MEV[g + 1]
            hi = GROUP_BOUNDARIES_MEV[g]
            center = GROUP_CENTERS_MEV[g]
            assert lo < center < hi, (
                f"Group {g} center {center:.4e} not in ({lo:.4e}, {hi:.4e})"
            )

    def test_centers_positive(self):
        assert np.all(GROUP_CENTERS_MEV > 0)
