"""
Particle state and bank data structures.

SoA (Structure of Arrays) layout for vectorization and GPU compatibility.
Each field is a contiguous numpy array of length N (batch size).
"""
import numpy as np
from dataclasses import dataclass, field
from .constants import N_GROUPS


@dataclass
class ParticleBank:
    """Structure-of-Arrays storage for N particles.

    All coordinate arrays use float64 for geometric precision.
    Group indices use int32 for GPU compatibility.
    """
    x: np.ndarray       # float64[N] position
    y: np.ndarray
    z: np.ndarray
    ux: np.ndarray      # float64[N] direction (unit vector)
    uy: np.ndarray
    uz: np.ndarray
    group: np.ndarray   # int32[N] energy group index [0..7]
    weight: np.ndarray  # float64[N] statistical weight
    alive: np.ndarray   # bool[N]

    @property
    def n_particles(self):
        return len(self.x)

    @property
    def n_alive(self):
        return int(np.sum(self.alive))

    @classmethod
    def create_empty(cls, n):
        """Create bank with n particles, all dead."""
        return cls(
            x=np.zeros(n),
            y=np.zeros(n),
            z=np.zeros(n),
            ux=np.zeros(n),
            uy=np.zeros(n),
            uz=np.ones(n),
            group=np.zeros(n, dtype=np.int32),
            weight=np.ones(n),
            alive=np.zeros(n, dtype=bool),
        )

    @classmethod
    def create_source(cls, n, geometry, chi, rng):
        """Create initial fission source in core.

        Args:
            n: number of particles
            geometry: MCFRGeometry
            chi: fission spectrum array [N_GROUPS]
            rng: numpy random Generator

        Returns:
            ParticleBank with n particles uniformly distributed in core
        """
        x, y, z = geometry.sample_in_core(rng, n)

        # Isotropic directions
        ux, uy, uz = sample_isotropic(rng, n)

        # Sample energy groups from fission spectrum chi
        group = sample_from_cdf(chi, rng, n)

        return cls(
            x=x, y=y, z=z,
            ux=ux, uy=uy, uz=uz,
            group=group.astype(np.int32),
            weight=np.ones(n),
            alive=np.ones(n, dtype=bool),
        )

    def slice(self, mask):
        """Return a new ParticleBank with only particles where mask is True."""
        return ParticleBank(
            x=self.x[mask].copy(),
            y=self.y[mask].copy(),
            z=self.z[mask].copy(),
            ux=self.ux[mask].copy(),
            uy=self.uy[mask].copy(),
            uz=self.uz[mask].copy(),
            group=self.group[mask].copy(),
            weight=self.weight[mask].copy(),
            alive=self.alive[mask].copy(),
        )

    def split(self, n_chunks):
        """Split bank into n_chunks roughly equal parts for multiprocessing."""
        indices = np.array_split(np.arange(self.n_particles), n_chunks)
        chunks = []
        for idx in indices:
            chunks.append(ParticleBank(
                x=self.x[idx].copy(),
                y=self.y[idx].copy(),
                z=self.z[idx].copy(),
                ux=self.ux[idx].copy(),
                uy=self.uy[idx].copy(),
                uz=self.uz[idx].copy(),
                group=self.group[idx].copy(),
                weight=self.weight[idx].copy(),
                alive=self.alive[idx].copy(),
            ))
        return chunks


@dataclass
class FissionSite:
    """Single fission site for the fission bank."""
    x: float
    y: float
    z: float
    group: int  # energy group of born neutron


class FissionBank:
    """Dynamic collection of fission sites produced during a batch.

    Pre-allocated for efficiency, with an atomic-style counter.
    """
    def __init__(self, max_sites=200000):
        self.max_sites = max_sites
        self.x = np.zeros(max_sites)
        self.y = np.zeros(max_sites)
        self.z = np.zeros(max_sites)
        self.group = np.zeros(max_sites, dtype=np.int32)
        self.count = 0

    def add(self, x, y, z, group):
        """Add a fission site."""
        if self.count < self.max_sites:
            self.x[self.count] = x
            self.y[self.count] = y
            self.z[self.count] = z
            self.group[self.count] = group
            self.count += 1

    def add_batch(self, x_arr, y_arr, z_arr, group_arr):
        """Add multiple fission sites."""
        n = len(x_arr)
        end = min(self.count + n, self.max_sites)
        actual = end - self.count
        self.x[self.count:end] = x_arr[:actual]
        self.y[self.count:end] = y_arr[:actual]
        self.z[self.count:end] = z_arr[:actual]
        self.group[self.count:end] = group_arr[:actual]
        self.count = end

    def to_particle_bank(self, n_target, chi, rng):
        """Resample fission bank to exactly n_target source particles.

        Standard MC power iteration normalization.
        """
        if self.count == 0:
            # No fission sites - create uniform source (subcritical fallback)
            from .geometry import MCFRGeometry
            geom = MCFRGeometry()
            return ParticleBank.create_source(n_target, geom, chi, rng)

        # Resample with replacement
        indices = rng.integers(0, self.count, size=n_target)
        ux, uy, uz = sample_isotropic(rng, n_target)
        # Resample groups from chi (fission-born neutrons)
        group = sample_from_cdf(chi, rng, n_target)

        return ParticleBank(
            x=self.x[indices].copy(),
            y=self.y[indices].copy(),
            z=self.z[indices].copy(),
            ux=ux, uy=uy, uz=uz,
            group=group.astype(np.int32),
            weight=np.ones(n_target),
            alive=np.ones(n_target, dtype=bool),
        )

    def clear(self):
        self.count = 0


def sample_isotropic(rng, n):
    """Sample n isotropic unit direction vectors."""
    cos_theta = 2.0 * rng.random(n) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * rng.random(n)
    ux = sin_theta * np.cos(phi)
    uy = sin_theta * np.sin(phi)
    uz = cos_theta
    return ux, uy, uz


def sample_from_cdf(probabilities, rng, n):
    """Sample n indices from discrete probability distribution."""
    cdf = np.cumsum(probabilities)
    cdf /= cdf[-1]  # normalize
    xi = rng.random(n)
    return np.searchsorted(cdf, xi).astype(np.int32)
