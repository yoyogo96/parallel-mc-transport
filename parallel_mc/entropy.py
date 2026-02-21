"""
Shannon entropy of fission source distribution.

Used to monitor convergence of the fission source during inactive batches.
The entropy should plateau when the source is converged.

H = -sum_i (p_i * ln(p_i))

where p_i = (sites in bin i) / (total sites), computed on a coarse (r,z) mesh.
"""
import numpy as np
from .constants import ENTROPY_NR, ENTROPY_NZ


class EntropyMonitor:
    """Shannon entropy of fission source on coarse mesh."""

    def __init__(self, geometry, n_r=ENTROPY_NR, n_z=ENTROPY_NZ):
        self.n_r = n_r
        self.n_z = n_z
        self.r_max = geometry.outer_radius
        self.z_min = -geometry.outer_half_height
        self.z_max = geometry.outer_half_height

        self.r_edges = np.linspace(0, self.r_max, n_r + 1)
        self.z_edges = np.linspace(self.z_min, self.z_max, n_z + 1)

        self.history = []
        self.max_entropy = np.log(n_r * n_z)  # maximum possible entropy

    def compute(self, fission_bank):
        """Compute Shannon entropy of the fission source.

        Args:
            fission_bank: FissionBank with fission sites

        Returns:
            H: Shannon entropy value
        """
        if fission_bank.count == 0:
            return 0.0

        # Get positions of fission sites
        x = fission_bank.x[:fission_bank.count]
        y = fission_bank.y[:fission_bank.count]
        z = fission_bank.z[:fission_bank.count]
        r = np.sqrt(x**2 + y**2)

        # Bin the fission sites
        ir = np.searchsorted(self.r_edges, r) - 1
        iz = np.searchsorted(self.z_edges, z) - 1

        # Clip to valid range
        ir = np.clip(ir, 0, self.n_r - 1)
        iz = np.clip(iz, 0, self.n_z - 1)

        # Count sites per bin
        counts = np.zeros((self.n_r, self.n_z))
        for i in range(len(ir)):
            counts[ir[i], iz[i]] += 1

        # Compute entropy
        total = np.sum(counts)
        if total == 0:
            return 0.0

        probs = counts.flatten() / total
        probs = probs[probs > 0]  # remove zeros (log(0) undefined)
        H = -np.sum(probs * np.log(probs))

        self.history.append(H)
        return H

    @property
    def is_converged(self):
        """Check if entropy has converged (simple heuristic).

        Converged if last 10 values have std < 5% of mean.
        """
        if len(self.history) < 20:
            return False
        last_10 = self.history[-10:]
        mean = np.mean(last_10)
        std = np.std(last_10)
        return std < 0.05 * mean if mean > 0 else False
