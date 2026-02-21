"""
k-Eigenvalue Power Iteration Solver

Standard Monte Carlo power iteration:
1. Initialize N source neutrons in fuel
2. For each batch:
   a. Transport all source neutrons -> fission bank
   b. k_batch = |fission_bank| / N
   c. Compute Shannon entropy
   d. If active batch: accumulate tallies
   e. Resample fission bank -> next source
3. Statistics from active batches
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .constants import N_GROUPS
from .geometry import MCFRGeometry
from .materials import Material, build_fuel_salt, build_reflector_beo
from .particle import ParticleBank, FissionBank
from .tallies import TallyAccumulator, TallyStatistics, create_default_tally
from .entropy import EntropyMonitor


@dataclass
class EigenvalueResult:
    """Complete results from a k-eigenvalue calculation."""
    keff: float
    keff_std: float
    keff_history: List[float]
    entropy_history: List[float]
    flux_mean: np.ndarray           # [n_r, n_z, N_GROUPS]
    flux_std: np.ndarray
    fission_rate_mean: np.ndarray   # [n_r, n_z]
    leakage_fraction: float
    total_time: float               # seconds
    backend_name: str
    n_particles: int
    n_batches: int
    n_inactive: int
    n_active: int
    # Mesh info
    r_edges: np.ndarray
    z_edges: np.ndarray

    def summary(self):
        """Print human-readable summary."""
        print("=" * 60)
        print(f"  k-Eigenvalue Result ({self.backend_name})")
        print("=" * 60)
        print(f"  k_eff = {self.keff:.5f} +/- {self.keff_std:.5f}")
        print(f"  Batches: {self.n_batches} ({self.n_inactive} inactive + {self.n_active} active)")
        print(f"  Particles/batch: {self.n_particles:,}")
        print(f"  Total histories: {self.n_particles * self.n_batches:,}")
        print(f"  Leakage fraction: {self.leakage_fraction:.4f}")
        print(f"  Wall time: {self.total_time:.1f} s")
        rate = self.n_particles * self.n_batches / self.total_time
        print(f"  Rate: {rate:,.0f} particles/s")
        print("=" * 60)

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            'keff': float(self.keff),
            'keff_std': float(self.keff_std),
            'keff_history': [float(k) for k in self.keff_history],
            'entropy_history': [float(h) for h in self.entropy_history],
            'leakage_fraction': float(self.leakage_fraction),
            'total_time': float(self.total_time),
            'backend_name': self.backend_name,
            'n_particles': self.n_particles,
            'n_batches': self.n_batches,
            'n_inactive': self.n_inactive,
            'n_active': self.n_active,
            'r_edges': self.r_edges.tolist(),
            'z_edges': self.z_edges.tolist(),
        }


class PowerIteration:
    """k-eigenvalue power iteration driver.

    Uses any MCBackend for the transport computation.
    """

    def __init__(
        self,
        backend,                          # MCBackend instance
        geometry: MCFRGeometry = None,
        materials: Dict[int, Material] = None,
        n_particles: int = 50000,
        n_batches: int = 300,
        n_inactive: int = 50,
        seed: int = 42,
    ):
        self.backend = backend
        self.geometry = geometry or MCFRGeometry()

        if materials is None:
            fuel = build_fuel_salt()
            refl = build_reflector_beo()
            self.materials = {0: fuel, 1: refl}
        else:
            self.materials = materials

        self.n_particles = n_particles
        self.n_batches = n_batches
        self.n_inactive = n_inactive
        self.n_active = n_batches - n_inactive
        self.seed = seed

    def solve(self, verbose=True) -> EigenvalueResult:
        """Run the full k-eigenvalue calculation.

        Returns:
            EigenvalueResult with all statistics
        """
        rng = np.random.default_rng(self.seed)

        # Get chi from fuel material
        fuel = self.materials[0]
        chi = fuel.chi

        # Create tally system
        tally = create_default_tally(self.geometry)
        stats = TallyStatistics(tally.n_r, tally.n_z)
        entropy_monitor = EntropyMonitor(self.geometry)

        # Initialize source
        source_bank = ParticleBank.create_source(
            self.n_particles, self.geometry, chi, rng
        )

        if verbose:
            print(f"Starting k-eigenvalue calculation")
            print(f"  Backend: {self.backend.get_name()}")
            print(f"  Particles/batch: {self.n_particles:,}")
            print(f"  Batches: {self.n_batches} ({self.n_inactive} inactive + {self.n_active} active)")
            print(f"  Geometry: R={self.geometry.core_radius:.3f}m, H={2*self.geometry.core_half_height:.3f}m")
            print()

        t_start = time.time()
        all_keff = []

        for batch in range(1, self.n_batches + 1):
            t_batch = time.time()
            is_active = batch > self.n_inactive

            # Reset tally for this batch
            tally.reset()

            # Transport all particles
            fission_bank = self.backend.transport_batch(
                source_bank, self.geometry, self.materials,
                tally if is_active else None,
                rng,
            )

            # Batch k_eff
            k_batch = fission_bank.count / self.n_particles
            all_keff.append(k_batch)

            # Shannon entropy
            entropy = entropy_monitor.compute(fission_bank)

            # Accumulate statistics for active batches
            if is_active:
                stats.accumulate(tally, k_batch)

            # Resample fission bank for next generation
            source_bank = fission_bank.to_particle_bank(self.n_particles, chi, rng)

            # Progress output
            if verbose and (batch % 10 == 0 or batch <= 5):
                elapsed = time.time() - t_batch
                status = "active" if is_active else "inactive"
                cum_keff = np.mean(all_keff[self.n_inactive:]) if is_active else np.mean(all_keff)
                cum_std = (
                    np.std(all_keff[self.n_inactive:], ddof=1) / np.sqrt(max(1, batch - self.n_inactive))
                    if is_active and batch > self.n_inactive
                    else 0
                )
                print(f"  Batch {batch:4d}/{self.n_batches} [{status:8s}] "
                      f"k={k_batch:.5f}  "
                      f"cumul={cum_keff:.5f}+/-{cum_std:.5f}  "
                      f"H={entropy:.3f}  "
                      f"fiss={fission_bank.count:6d}  "
                      f"dt={elapsed:.2f}s")

        total_time = time.time() - t_start

        # Build result
        result = EigenvalueResult(
            keff=stats.keff_mean,
            keff_std=stats.keff_std,
            keff_history=all_keff,
            entropy_history=entropy_monitor.history,
            flux_mean=stats.flux_mean,
            flux_std=stats.flux_std,
            fission_rate_mean=stats.fission_mean,
            leakage_fraction=stats.leakage_mean / self.n_particles if self.n_particles > 0 else 0,
            total_time=total_time,
            backend_name=self.backend.get_name(),
            n_particles=self.n_particles,
            n_batches=self.n_batches,
            n_inactive=self.n_inactive,
            n_active=self.n_active,
            r_edges=tally.r_edges,
            z_edges=tally.z_edges,
        )

        if verbose:
            print()
            result.summary()

        return result


def quick_run(backend=None, n_particles=10000, n_batches=100, n_inactive=20, seed=42):
    """Quick test run with default parameters."""
    if backend is None:
        from .backends import auto_select_backend
        backend = auto_select_backend()

    solver = PowerIteration(
        backend=backend,
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        seed=seed,
    )
    return solver.solve()


def production_run(backend=None, n_particles=50000, n_batches=300, n_inactive=50, seed=42):
    """Production run with higher statistics."""
    if backend is None:
        from .backends import auto_select_backend
        backend = auto_select_backend()

    solver = PowerIteration(
        backend=backend,
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        seed=seed,
    )
    return solver.solve()
