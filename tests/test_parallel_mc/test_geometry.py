"""
Tests for parallel_mc.geometry module.
"""
import numpy as np
import pytest

from parallel_mc.geometry import MCFRGeometry, CORE, REFLECTOR, VACUUM


class TestRegionLookupScalar:
    def test_origin_is_core(self, geometry):
        assert geometry.region(0.0, 0.0, 0.0) == CORE

    def test_inside_core_cylinder(self, geometry):
        """Point at r=0.5, z=0 is inside the core."""
        assert geometry.region(0.5, 0.0, 0.0) == CORE

    def test_radial_reflector(self, geometry):
        """Point just outside core radius is in reflector."""
        r = geometry.core_radius + 0.05   # 5 cm into radial reflector
        assert geometry.region(r, 0.0, 0.0) == REFLECTOR

    def test_axial_reflector_above(self, geometry):
        """Point above core top is in axial reflector."""
        z = geometry.core_half_height + 0.05   # 5 cm above top
        assert geometry.region(0.0, 0.0, z) == REFLECTOR

    def test_axial_reflector_below(self, geometry):
        """Point below core bottom is in axial reflector."""
        z = -(geometry.core_half_height + 0.05)
        assert geometry.region(0.0, 0.0, z) == REFLECTOR

    def test_vacuum_radially_outside(self, geometry):
        """Point beyond outer radius is vacuum."""
        r = geometry.outer_radius + 0.05
        assert geometry.region(r, 0.0, 0.0) == VACUUM

    def test_vacuum_axially_outside(self, geometry):
        """Point beyond outer half-height is vacuum."""
        z = geometry.outer_half_height + 0.05
        assert geometry.region(0.0, 0.0, z) == VACUUM

    def test_test_spec_reflector_point(self, geometry):
        """Spec point: region(0.9, 0, 0) == REFLECTOR."""
        # core_radius ~ 0.8008, reflector_radial = 0.20 -> outer_radius ~ 1.0008
        # 0.9 is inside outer but outside core -> REFLECTOR
        assert geometry.region(0.9, 0.0, 0.0) == REFLECTOR

    def test_test_spec_vacuum_point(self, geometry):
        """Spec point: region(2.0, 0, 0) == VACUUM."""
        assert geometry.region(2.0, 0.0, 0.0) == VACUUM

    def test_test_spec_axial_reflector(self, geometry):
        """Spec point: region(0, 0, 0.85) == REFLECTOR."""
        # core_half_height ~ 0.8008, 0.85 > 0.8008 -> above core
        # outer_half_height = 0.8008 + 0.15 = 0.9508 -> 0.85 < 0.9508 -> REFLECTOR
        assert geometry.region(0.0, 0.0, 0.85) == REFLECTOR


class TestRegionLookupVectorized:
    def test_vectorized_matches_scalar(self, geometry, rng):
        """Vectorized region() must agree with per-point scalar calls."""
        n = 200
        x = rng.uniform(-1.5, 1.5, n)
        y = rng.uniform(-1.5, 1.5, n)
        z = rng.uniform(-1.2, 1.2, n)

        vec_result = geometry.region(x, y, z)

        for i in range(n):
            scalar = geometry.region(x[i], y[i], z[i])
            assert vec_result[i] == scalar, (
                f"Mismatch at point {i}: ({x[i]:.3f}, {y[i]:.3f}, {z[i]:.3f}) "
                f"vec={vec_result[i]} scalar={scalar}"
            )

    def test_vectorized_returns_array(self, geometry):
        x = np.array([0.0, 0.9, 2.0])
        y = np.zeros(3)
        z = np.zeros(3)
        result = geometry.region(x, y, z)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == CORE
        assert result[1] == REFLECTOR
        assert result[2] == VACUUM


class TestDistanceToBoundary:
    def test_from_center_radial(self, geometry):
        """From origin moving radially outward, distance == core_radius."""
        d, next_reg = geometry.distance_to_boundary(
            0.0, 0.0, 0.0,   # position
            1.0, 0.0, 0.0,   # direction (purely radial)
            CORE,
        )
        assert d == pytest.approx(geometry.core_radius, rel=1e-6)
        assert next_reg == REFLECTOR

    def test_from_center_axial(self, geometry):
        """From origin moving axially, distance == core_half_height."""
        d, next_reg = geometry.distance_to_boundary(
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,   # axially upward
            CORE,
        )
        assert d == pytest.approx(geometry.core_half_height, rel=1e-6)
        assert next_reg == REFLECTOR

    def test_distance_positive(self, geometry):
        """Distance must always be positive."""
        d, _ = geometry.distance_to_boundary(
            0.0, 0.0, 0.0,
            0.5, 0.5, 1.0 / np.sqrt(3),
            CORE,
        )
        assert d > 0

    def test_reflector_to_core(self, geometry):
        """From reflector directed toward core, next region should be CORE."""
        r_mid = geometry.core_radius + 0.05   # midpoint of radial reflector
        # Position just outside core, pointing inward
        d, next_reg = geometry.distance_to_boundary(
            r_mid, 0.0, 0.0,
            -1.0, 0.0, 0.0,   # pointing inward
            REFLECTOR,
        )
        assert d == pytest.approx(0.05, rel=1e-4)
        assert next_reg == CORE

    def test_reflector_to_vacuum(self, geometry):
        """From reflector directed outward, next region should be VACUUM."""
        r_mid = geometry.core_radius + 0.05
        d, next_reg = geometry.distance_to_boundary(
            r_mid, 0.0, 0.0,
            1.0, 0.0, 0.0,   # pointing outward
            REFLECTOR,
        )
        expected = geometry.outer_radius - r_mid
        assert d == pytest.approx(expected, rel=1e-4)
        assert next_reg == VACUUM


class TestSampleInCore:
    def test_all_points_inside_core(self, geometry, rng):
        n = 500
        x, y, z = geometry.sample_in_core(rng, n)
        r = np.sqrt(x**2 + y**2)
        assert np.all(r < geometry.core_radius), "Some sampled points outside core radially"
        assert np.all(np.abs(z) < geometry.core_half_height), (
            "Some sampled points outside core axially"
        )

    def test_returns_correct_count(self, geometry, rng):
        n = 100
        x, y, z = geometry.sample_in_core(rng, n)
        assert len(x) == n
        assert len(y) == n
        assert len(z) == n

    def test_uniform_radial_distribution(self, geometry, rng):
        """r^2 should be roughly uniform (cylindrical sampling)."""
        n = 5000
        x, y, z = geometry.sample_in_core(rng, n)
        r2 = x**2 + y**2
        # r^2 should be uniform in [0, R^2]: mean ~ R^2/2
        expected_mean = geometry.core_radius**2 / 2.0
        actual_mean = np.mean(r2)
        # Allow 5% tolerance (statistical)
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05
