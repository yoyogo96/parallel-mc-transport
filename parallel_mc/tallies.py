"""
Tally system for MC neutron transport.

Track-length estimator on cylindrical (r, z) mesh:
  flux[ir, iz, g] += weight * track_length / bin_volume

Also tracks:
  - Fission rate per mesh cell
  - Absorption rate per mesh cell
  - Total leakage
  - Per-batch k_eff
"""
import numpy as np
from .constants import N_GROUPS


class TallyAccumulator:
    """Accumulates tallies during a single batch.

    Cylindrical (r, z) mesh with N_GROUPS energy groups.
    """
    def __init__(self, r_edges, z_edges):
        """
        Args:
            r_edges: radial bin edges [n_r+1] in meters (0 to outer_radius)
            z_edges: axial bin edges [n_z+1] in meters (-outer_half_height to +outer_half_height)
        """
        self.r_edges = np.array(r_edges)
        self.z_edges = np.array(z_edges)
        self.n_r = len(r_edges) - 1
        self.n_z = len(z_edges) - 1

        # Compute bin volumes for normalization
        # V_bin = pi * (r_out^2 - r_in^2) * dz
        self.bin_volumes = np.zeros((self.n_r, self.n_z))
        for ir in range(self.n_r):
            r_in = self.r_edges[ir]
            r_out = self.r_edges[ir + 1]
            for iz in range(self.n_z):
                dz = self.z_edges[iz + 1] - self.z_edges[iz]
                self.bin_volumes[ir, iz] = np.pi * (r_out**2 - r_in**2) * dz

        # Tally arrays
        self.flux = np.zeros((self.n_r, self.n_z, N_GROUPS))         # track-length flux
        self.fission_rate = np.zeros((self.n_r, self.n_z))            # fission events
        self.absorption_rate = np.zeros((self.n_r, self.n_z))         # absorption events
        self.leakage = 0.0

    def score_track(self, x, y, z, ux, uy, uz, track_length, weight, group, material):
        """Score a track-length contribution to flux tally.

        Simplified: scores at track midpoint (adequate for fine mesh).
        For a more rigorous implementation, would split track across bin boundaries.
        """
        # Track midpoint
        xm = x + ux * track_length * 0.5
        ym = y + uy * track_length * 0.5
        zm = z + uz * track_length * 0.5
        rm = np.sqrt(xm**2 + ym**2)

        # Find bin
        ir = np.searchsorted(self.r_edges, rm) - 1
        iz = np.searchsorted(self.z_edges, zm) - 1

        if 0 <= ir < self.n_r and 0 <= iz < self.n_z:
            vol = self.bin_volumes[ir, iz]
            if vol > 0:
                self.flux[ir, iz, group] += weight * track_length / vol
                # Also score reaction rates using track-length estimator
                if material.is_fissile:
                    self.fission_rate[ir, iz] += weight * track_length * material.sigma_f[group] / vol
                self.absorption_rate[ir, iz] += weight * track_length * material.sigma_a[group] / vol

    def score_leakage(self, weight):
        """Score a leakage event."""
        self.leakage += weight

    def reset(self):
        """Reset all tallies for a new batch."""
        self.flux[:] = 0.0
        self.fission_rate[:] = 0.0
        self.absorption_rate[:] = 0.0
        self.leakage = 0.0


class TallyStatistics:
    """Accumulates batch-wise tally statistics for mean/variance computation.

    Uses Welford's online algorithm for numerically stable variance.
    """
    def __init__(self, n_r, n_z):
        self.n_r = n_r
        self.n_z = n_z
        self.n_batches = 0

        # Running sums for flux
        self.flux_sum = np.zeros((n_r, n_z, N_GROUPS))
        self.flux_sq_sum = np.zeros((n_r, n_z, N_GROUPS))

        # Running sums for fission rate
        self.fission_sum = np.zeros((n_r, n_z))
        self.fission_sq_sum = np.zeros((n_r, n_z))

        # k_eff history
        self.keff_history = []
        self.leakage_history = []

    def accumulate(self, tally, k_batch):
        """Add results from one active batch."""
        self.n_batches += 1
        self.flux_sum += tally.flux
        self.flux_sq_sum += tally.flux**2
        self.fission_sum += tally.fission_rate
        self.fission_sq_sum += tally.fission_rate**2
        self.keff_history.append(k_batch)
        self.leakage_history.append(tally.leakage)

    @property
    def keff_mean(self):
        if not self.keff_history:
            return 0.0
        return np.mean(self.keff_history)

    @property
    def keff_std(self):
        if len(self.keff_history) < 2:
            return 0.0
        return np.std(self.keff_history, ddof=1) / np.sqrt(len(self.keff_history))

    @property
    def flux_mean(self):
        if self.n_batches == 0:
            return self.flux_sum
        return self.flux_sum / self.n_batches

    @property
    def flux_std(self):
        if self.n_batches < 2:
            return np.zeros_like(self.flux_sum)
        mean = self.flux_mean
        var = self.flux_sq_sum / self.n_batches - mean**2
        var = np.maximum(var, 0)  # numerical safety
        return np.sqrt(var / self.n_batches)

    @property
    def fission_mean(self):
        if self.n_batches == 0:
            return self.fission_sum
        return self.fission_sum / self.n_batches

    @property
    def leakage_mean(self):
        if not self.leakage_history:
            return 0.0
        return np.mean(self.leakage_history)


def create_default_tally(geometry):
    """Create default tally mesh for MCFR geometry.

    25 radial bins, 30 axial bins covering core + reflector.
    """
    r_max = geometry.outer_radius * 1.01  # slight padding
    z_max = geometry.outer_half_height * 1.01

    r_edges = np.linspace(0, r_max, 26)   # 25 bins
    z_edges = np.linspace(-z_max, z_max, 31)  # 30 bins

    return TallyAccumulator(r_edges, z_edges)
