"""
Compare parallel MC results against OpenMC reference.

OpenMC reference data is stored in:
  results/openmc_results.json

Key reference values:
  k_eff = 0.9327 +/- 0.00035
  200 active batches, 20000 particles/batch = 4M active histories
"""
import json
import os
import numpy as np


def load_openmc_reference():
    """Load OpenMC reference results."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ref_path = os.path.join(base_dir, 'results', 'openmc_results.json')

    if not os.path.exists(ref_path):
        print(f"WARNING: OpenMC reference file not found at {ref_path}")
        return None

    with open(ref_path) as f:
        data = json.load(f)
    return data


def run_validation(backend_name='auto', n_particles=20000, n_batches=150, seed=42):
    """Run validation comparison against OpenMC.

    Parameters
    ----------
    backend_name : str
        Backend to use: 'auto', 'cpu', 'cuda', or 'metal'.
    n_particles : int
        Particles per batch for the parallel MC run.
    n_batches : int
        Total number of batches (including inactive).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    int
        Exit status: 0 for pass (within 10%), 1 for fail.
    """
    from ..backends import auto_select_backend, get_backend
    from ..eigenvalue import PowerIteration
    from ..materials import build_fuel_salt, build_reflector_beo
    from ..geometry import MCFRGeometry

    print("=" * 70)
    print("  Parallel MC vs OpenMC Validation")
    print("=" * 70)

    # Load OpenMC reference
    ref = load_openmc_reference()
    if ref:
        ref_keff = ref.get('keff', 0.9327)
        ref_keff_std = ref.get('keff_std', 0.00035)
        print(f"\n  OpenMC Reference: k_eff = {ref_keff:.5f} +/- {ref_keff_std:.5f}")
        print(f"  OpenMC Histories: {ref.get('active_batches', 200) * ref.get('particles_per_batch', 20000):,}")
    else:
        ref_keff = 0.9327
        ref_keff_std = 0.00035
        print(f"\n  OpenMC Reference (hardcoded): k_eff = {ref_keff:.5f} +/- {ref_keff_std:.5f}")

    # Select backend
    if backend_name == 'auto':
        backend = auto_select_backend()
    else:
        backend = get_backend(backend_name)

    # Run parallel MC
    fuel = build_fuel_salt()
    refl = build_reflector_beo()  # BeO to match OpenMC reference

    print(f"\n  Running parallel MC ({backend.get_name()})...")
    print(f"  Particles/batch: {n_particles:,}, Batches: {n_batches}")
    print()

    solver = PowerIteration(
        backend=backend,
        materials={0: fuel, 1: refl},
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=min(30, n_batches // 5),
        seed=seed,
    )
    result = solver.solve(verbose=True)

    # Compare
    delta_k = result.keff - ref_keff
    delta_k_pcm = delta_k * 1e5
    combined_std = np.sqrt(result.keff_std**2 + ref_keff_std**2)
    n_sigma = abs(delta_k) / combined_std if combined_std > 0 else 0

    print()
    print("=" * 70)
    print("  VALIDATION COMPARISON")
    print("=" * 70)
    print(f"  OpenMC k_eff:          {ref_keff:.5f} +/- {ref_keff_std:.5f}")
    print(f"  Parallel MC k_eff:     {result.keff:.5f} +/- {result.keff_std:.5f}")
    print(f"  Delta-k:               {delta_k:+.5f} ({delta_k_pcm:+.0f} pcm)")
    print(f"  Combined sigma:        {combined_std:.5f}")
    print(f"  Deviation:             {n_sigma:.1f} sigma")
    print()

    # Note about expected discrepancy
    pct_diff = abs(delta_k / ref_keff) * 100
    print(f"  Relative difference:   {pct_diff:.1f}%")
    print()

    if pct_diff < 2.0:
        print("  RESULT: EXCELLENT - Within 2% of OpenMC (expected for 8-group)")
        status = 0
    elif pct_diff < 5.0:
        print("  RESULT: GOOD - Within 5% of OpenMC (acceptable for multi-group)")
        status = 0
    elif pct_diff < 10.0:
        print("  RESULT: FAIR - Within 10% (cross-section tuning may help)")
        status = 0
    else:
        print("  RESULT: POOR - >10% difference (check physics and cross-sections)")
        status = 1

    print()
    print("  NOTE: 8-group approximation inherently differs from continuous-energy.")
    print("  Dominant error sources:")
    print("    - Group-averaged cross-sections (spectrum-weighting uncertainty)")
    print("    - U-238 resonance self-shielding (collapsed in 8 groups)")
    print("    - Scattering transfer matrix (P0 approximation)")
    print("    - Cl-35 nuclear data uncertainty (ENDF/B-VIII.0 Cl-35(n,p) issue)")
    print("=" * 70)

    # Save validation report
    report = {
        'openmc_keff': ref_keff,
        'openmc_keff_std': ref_keff_std,
        'parallel_mc_keff': result.keff,
        'parallel_mc_keff_std': result.keff_std,
        'delta_k': delta_k,
        'delta_k_pcm': delta_k_pcm,
        'relative_pct': pct_diff,
        'n_sigma': n_sigma,
        'backend': result.backend_name,
        'n_particles': n_particles,
        'n_batches': n_batches,
    }

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    report_path = os.path.join(base_dir, 'results', 'parallel_mc_validation.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Validation report saved to {report_path}")

    return status
