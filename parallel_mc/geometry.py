"""
MCFR Cylindrical Geometry - Pool-Type Core with Reflector

Two regions:
  0 (CORE): Fuel salt cylinder, R=core_radius, |z| < core_half_height
  1 (REFLECTOR): Annular/axial reflector surrounding core
 -1 (VACUUM): Outside everything

Centered at origin. Z-axis is cylinder axis.

The geometry supports both vectorized (numpy) and scalar operations
for CPU and GPU backends respectively.
"""
import numpy as np
from dataclasses import dataclass

# Region IDs
CORE = 0
REFLECTOR = 1
VACUUM = -1


@dataclass
class MCFRGeometry:
    """Pool-type MCFR geometry.

    Default dimensions from config.py:
    - Core: R~0.80m, H~1.60m (H/D=1.0)
    - Reflector: 20cm radial (BeO), 15cm axial
    """
    core_radius: float = 0.8008        # m (from compute_derived)
    core_half_height: float = 0.8008   # m (H/2, H/D=1.0)
    reflector_radial: float = 0.20     # m (BeO radial thickness)
    reflector_axial: float = 0.15      # m (BeO axial thickness)

    @property
    def outer_radius(self):
        return self.core_radius + self.reflector_radial

    @property
    def outer_half_height(self):
        return self.core_half_height + self.reflector_axial

    @property
    def core_volume(self):
        return np.pi * self.core_radius**2 * (2 * self.core_half_height)

    @property
    def reflector_volume(self):
        outer_vol = np.pi * self.outer_radius**2 * (2 * self.outer_half_height)
        return outer_vol - self.core_volume

    def region(self, x, y, z):
        """Determine region for particle position(s).

        Works with scalars or numpy arrays.
        Returns: int or ndarray of ints (CORE=0, REFLECTOR=1, VACUUM=-1)
        """
        r2 = x**2 + y**2
        # Check core first
        in_core = (r2 < self.core_radius**2) & (np.abs(z) < self.core_half_height)
        # Check reflector
        in_outer = (r2 < self.outer_radius**2) & (np.abs(z) < self.outer_half_height)
        in_reflector = in_outer & ~in_core

        if np.ndim(x) == 0:
            # Scalar
            if in_core:
                return CORE
            elif in_reflector:
                return REFLECTOR
            else:
                return VACUUM
        else:
            result = np.full_like(x, VACUUM, dtype=np.int32)
            result[in_reflector] = REFLECTOR
            result[in_core] = CORE  # core overrides reflector where both true
            return result

    def distance_to_boundary(self, x, y, z, ux, uy, uz, current_region):
        """Distance to nearest boundary and next region.

        For scalar inputs only (called per-particle in transport loop).

        Returns: (distance, next_region)

        Algorithm:
        - In CORE: check distance to core cylinder (radial + axial)
        - In REFLECTOR: check distance to core inner surface AND outer surface
        - Result is minimum positive distance, with corresponding next region
        """
        r2 = x**2 + y**2

        if current_region == CORE:
            # Distance to core cylinder wall (radial)
            d_radial = self._dist_to_cylinder(x, y, ux, uy, self.core_radius, outward=True)
            # Distance to core top/bottom planes
            d_axial_top = self._dist_to_plane_z(z, uz, self.core_half_height)
            d_axial_bot = self._dist_to_plane_z(z, uz, -self.core_half_height)

            d_min = min(d_radial, d_axial_top, d_axial_bot)
            next_reg = REFLECTOR  # core always transitions to reflector
            return d_min, next_reg

        elif current_region == REFLECTOR:
            distances = []
            next_regs = []

            # Distance to core inner surface (going inward)
            d_core_r = self._dist_to_cylinder(x, y, ux, uy, self.core_radius, outward=False)
            if d_core_r < 1e30:
                # Check if z at intersection is within core height
                z_at = z + uz * d_core_r
                if abs(z_at) < self.core_half_height:
                    distances.append(d_core_r)
                    next_regs.append(CORE)

            # Distance to core axial planes (going inward)
            d_core_top = self._dist_to_plane_z(z, uz, self.core_half_height)
            if d_core_top < 1e30:
                x_at = x + ux * d_core_top
                y_at = y + uy * d_core_top
                if x_at**2 + y_at**2 < self.core_radius**2:
                    distances.append(d_core_top)
                    next_regs.append(CORE)

            d_core_bot = self._dist_to_plane_z(z, uz, -self.core_half_height)
            if d_core_bot < 1e30:
                x_at = x + ux * d_core_bot
                y_at = y + uy * d_core_bot
                if x_at**2 + y_at**2 < self.core_radius**2:
                    distances.append(d_core_bot)
                    next_regs.append(CORE)

            # Distance to outer surface (going outward -> vacuum)
            d_outer_r = self._dist_to_cylinder(x, y, ux, uy, self.outer_radius, outward=True)
            distances.append(d_outer_r)
            next_regs.append(VACUUM)

            d_outer_top = self._dist_to_plane_z(z, uz, self.outer_half_height)
            distances.append(d_outer_top)
            next_regs.append(VACUUM)

            d_outer_bot = self._dist_to_plane_z(z, uz, -self.outer_half_height)
            distances.append(d_outer_bot)
            next_regs.append(VACUUM)

            if not distances:
                return 1e30, VACUUM

            idx = np.argmin(distances)
            return distances[idx], next_regs[idx]

        else:
            return 1e30, VACUUM

    def _dist_to_cylinder(self, x, y, ux, uy, R, outward=True):
        """Distance to cylindrical surface at radius R.

        Solves: (x + ux*t)^2 + (y + uy*t)^2 = R^2
        -> a*t^2 + 2*b*t + c = 0
        where a = ux^2 + uy^2, b = x*ux + y*uy, c = x^2 + y^2 - R^2
        """
        a = ux**2 + uy**2
        if a < 1e-20:
            return 1e30  # parallel to cylinder axis

        b = x * ux + y * uy
        c = x**2 + y**2 - R**2
        disc = b**2 - a * c

        if disc < 0:
            return 1e30  # no intersection

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / a
        t2 = (-b + sqrt_disc) / a

        eps = 1e-10
        if outward:
            # Want the exit intersection (farther one for inside, any positive for outside)
            if c < 0:
                # Inside cylinder -> take the positive root
                return t2 if t2 > eps else 1e30
            else:
                # Outside cylinder -> shouldn't happen for outward from inside
                if t1 > eps:
                    return t1
                elif t2 > eps:
                    return t2
                return 1e30
        else:
            # Going inward to cylinder (for reflector checking core boundary)
            # Want the entry intersection
            if c > 0:
                # Outside the inner cylinder
                if t1 > eps:
                    return t1
                return 1e30
            else:
                # Inside (shouldn't happen for inward check)
                return 1e30

    def _dist_to_plane_z(self, z, uz, z_plane):
        """Distance to a horizontal plane at z=z_plane."""
        if abs(uz) < 1e-20:
            return 1e30
        t = (z_plane - z) / uz
        return t if t > 1e-10 else 1e30

    def sample_in_core(self, rng, n):
        """Sample n uniform random positions within the core cylinder.

        Returns: (x, y, z) each shape [n]
        """
        # Sample uniformly in cylinder: r^2 uniform in [0, R^2], theta in [0, 2pi], z uniform
        r = self.core_radius * np.sqrt(rng.random(n))
        theta = 2.0 * np.pi * rng.random(n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = self.core_half_height * (2.0 * rng.random(n) - 1.0)
        return x, y, z
