"""
CPU Backend: multiprocessing + Numba JIT

Parallelization strategy:
- Split source particles into n_workers chunks
- Each worker runs JIT-compiled transport on its chunk
- Collect and merge fission banks
- Aggregate tallies

Numba JIT compiles the transport kernel to native code,
providing ~50-100x speedup over pure Python per core.
Combined with multiprocessing, this gives ~400-800x total speedup
on an 8-core machine.
"""
import os
import math
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple

from ..constants import N_GROUPS, MAX_COLLISIONS, EPSILON, WEIGHT_CUTOFF, SURVIVAL_WEIGHT
from ..geometry import MCFRGeometry, CORE, REFLECTOR, VACUUM
from ..materials import Material
from ..particle import ParticleBank, FissionBank
from ..tallies import TallyAccumulator
from .base import MCBackend

# ---------------------------------------------------------------------------
# Try to import numba; set flag for fallback path
# ---------------------------------------------------------------------------
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ===================================================================
# Numba JIT transport kernels (module-level, compiled once)
# ===================================================================

if HAS_NUMBA:
    # ---------------------------------------------------------------
    # Region lookup
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _region_jit(x, y, z, core_r, core_hh, outer_r, outer_hh):
        """Determine which region (x, y, z) belongs to.

        Returns: CORE (0), REFLECTOR (1), or VACUUM (-1).
        """
        r2 = x * x + y * y
        abs_z = abs(z)
        if r2 < core_r * core_r and abs_z < core_hh:
            return 0   # CORE
        if r2 < outer_r * outer_r and abs_z < outer_hh:
            return 1   # REFLECTOR
        return -1      # VACUUM

    # ---------------------------------------------------------------
    # Distance to cylindrical surface
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _dist_to_cylinder_jit(x, y, ux, uy, R, outward):
        """Distance from (x,y) along (ux,uy) to cylinder of radius R.

        outward=True  -> particle inside cylinder, wanting exit distance
        outward=False -> particle outside cylinder, wanting entry distance
        Returns large sentinel (1e30) when no valid intersection.
        """
        a = ux * ux + uy * uy
        if a < 1.0e-20:
            return 1.0e30

        b = x * ux + y * uy
        c = x * x + y * y - R * R
        disc = b * b - a * c
        if disc < 0.0:
            return 1.0e30

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / a
        t2 = (-b + sqrt_disc) / a
        eps = 1.0e-10

        if outward:
            if c < 0.0:
                # Inside cylinder -> positive root is exit
                if t2 > eps:
                    return t2
                return 1.0e30
            else:
                # Outside cylinder (shouldn't happen for outward-from-inside)
                if t1 > eps:
                    return t1
                if t2 > eps:
                    return t2
                return 1.0e30
        else:
            # Inward: want entry point
            if c > 0.0:
                if t1 > eps:
                    return t1
                return 1.0e30
            else:
                return 1.0e30

    # ---------------------------------------------------------------
    # Distance to horizontal plane
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _dist_to_plane_jit(z, uz, z_plane):
        """Distance along uz to the plane at z = z_plane."""
        if abs(uz) < 1.0e-20:
            return 1.0e30
        t = (z_plane - z) / uz
        if t > 1.0e-10:
            return t
        return 1.0e30

    # ---------------------------------------------------------------
    # Distance to nearest boundary + next region
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _dist_to_boundary_jit(x, y, z, ux, uy, uz, reg,
                              core_r, core_hh, outer_r, outer_hh):
        """Compute (distance, next_region) for a particle in *reg*.

        Mirrors MCFRGeometry.distance_to_boundary exactly.
        """
        if reg == 0:
            # CORE -> nearest of radial wall, top, bottom  -> all go to REFLECTOR
            d_rad = _dist_to_cylinder_jit(x, y, ux, uy, core_r, True)
            d_top = _dist_to_plane_jit(z, uz, core_hh)
            d_bot = _dist_to_plane_jit(z, uz, -core_hh)
            d_min = d_rad
            if d_top < d_min:
                d_min = d_top
            if d_bot < d_min:
                d_min = d_bot
            return d_min, 1  # next = REFLECTOR

        elif reg == 1:
            # REFLECTOR -> check core inner surfaces AND outer surfaces
            best_d = 1.0e30
            best_reg = -1  # default: VACUUM

            # --- core radial surface (inward) ---
            d_core_r = _dist_to_cylinder_jit(x, y, ux, uy, core_r, False)
            if d_core_r < 1.0e30:
                z_at = z + uz * d_core_r
                if abs(z_at) < core_hh:
                    if d_core_r < best_d:
                        best_d = d_core_r
                        best_reg = 0  # CORE

            # --- core top plane (inward) ---
            d_core_top = _dist_to_plane_jit(z, uz, core_hh)
            if d_core_top < 1.0e30:
                x_at = x + ux * d_core_top
                y_at = y + uy * d_core_top
                if x_at * x_at + y_at * y_at < core_r * core_r:
                    if d_core_top < best_d:
                        best_d = d_core_top
                        best_reg = 0

            # --- core bottom plane (inward) ---
            d_core_bot = _dist_to_plane_jit(z, uz, -core_hh)
            if d_core_bot < 1.0e30:
                x_at = x + ux * d_core_bot
                y_at = y + uy * d_core_bot
                if x_at * x_at + y_at * y_at < core_r * core_r:
                    if d_core_bot < best_d:
                        best_d = d_core_bot
                        best_reg = 0

            # --- outer radial surface (outward -> VACUUM) ---
            d_outer_r = _dist_to_cylinder_jit(x, y, ux, uy, outer_r, True)
            if d_outer_r < best_d:
                best_d = d_outer_r
                best_reg = -1

            # --- outer top plane (-> VACUUM) ---
            d_outer_top = _dist_to_plane_jit(z, uz, outer_hh)
            if d_outer_top < best_d:
                best_d = d_outer_top
                best_reg = -1

            # --- outer bottom plane (-> VACUUM) ---
            d_outer_bot = _dist_to_plane_jit(z, uz, -outer_hh)
            if d_outer_bot < best_d:
                best_d = d_outer_bot
                best_reg = -1

            return best_d, best_reg

        # Outside everything
        return 1.0e30, -1

    # ---------------------------------------------------------------
    # Sample isotropic direction (scalar, for a single neutron)
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _sample_direction(rng_state):
        """Return (ux, uy, uz) isotropic unit vector.

        Uses two uniform randoms from the provided Numba-compatible
        random state (we simply use numpy random inside njit).
        """
        cos_theta = 2.0 * np.random.random() - 1.0
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * math.pi * np.random.random()
        ux = sin_theta * math.cos(phi)
        uy = sin_theta * math.sin(phi)
        uz = cos_theta
        return ux, uy, uz

    # ---------------------------------------------------------------
    # Sample discrete CDF (single sample)
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _sample_cdf_jit(probs, n_groups):
        """Sample one index from a discrete probability array."""
        # Build CDF
        xi = np.random.random()
        cumulative = 0.0
        for g in range(n_groups):
            cumulative += probs[g]
            if xi < cumulative:
                return g
        return n_groups - 1

    # ---------------------------------------------------------------
    # Transport a single particle (full physics)
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _transport_particle_jit(
        x, y, z, ux, uy, uz, group, weight,
        core_r, core_hh, outer_r, outer_hh,
        # Material arrays for region 0 (fuel/core):
        sigma_t_0, sigma_s_0, sigma_a_0, sigma_f_0,
        nu_sigma_f_0, chi_0, is_fissile_0,
        # Material arrays for region 1 (reflector):
        sigma_t_1, sigma_s_1, sigma_a_1, sigma_f_1,
        nu_sigma_f_1, chi_1, is_fissile_1,
        # Tally mesh edges and volumes
        r_edges, z_edges, n_r, n_z,
        do_tally,
        # Output arrays (pre-allocated fission sites for this particle)
        fission_x, fission_y, fission_z, fission_g, fission_offset,
        # Output tally arrays (thread-local)
        tally_flux, tally_fission, tally_absorption,
    ):
        """Transport one particle to completion.

        Returns:
            (fission_count, leakage_weight)
        """
        alive = True
        n_collisions = 0
        fission_count = 0
        leakage_wt = 0.0

        while alive and n_collisions < MAX_COLLISIONS:
            reg = _region_jit(x, y, z, core_r, core_hh, outer_r, outer_hh)
            if reg < 0:
                alive = False
                break

            # Select material arrays for this region
            if reg == 0:
                sig_t = sigma_t_0
                sig_s = sigma_s_0
                sig_a = sigma_a_0
                sig_f = sigma_f_0
                nu_sig_f = nu_sigma_f_0
                chi_mat = chi_0
                fissile = is_fissile_0
            else:
                sig_t = sigma_t_1
                sig_s = sigma_s_1
                sig_a = sigma_a_1
                sig_f = sigma_f_1
                nu_sig_f = nu_sigma_f_1
                chi_mat = chi_1
                fissile = is_fissile_1

            sigma_t_g = sig_t[group]

            if sigma_t_g < 1.0e-20:
                # Transparent medium -- stream to boundary
                d_bnd, next_reg = _dist_to_boundary_jit(
                    x, y, z, ux, uy, uz, reg,
                    core_r, core_hh, outer_r, outer_hh)
                if do_tally:
                    _score_track_jit(x, y, z, ux, uy, uz, d_bnd, weight,
                                     group, sig_f, sig_a, fissile,
                                     r_edges, z_edges, n_r, n_z,
                                     tally_flux, tally_fission, tally_absorption)
                x += ux * (d_bnd + EPSILON)
                y += uy * (d_bnd + EPSILON)
                z += uz * (d_bnd + EPSILON)
                continue

            # Sample free-flight distance
            s_collision = -math.log(np.random.random()) / sigma_t_g

            # Distance to boundary
            d_bnd, next_reg = _dist_to_boundary_jit(
                x, y, z, ux, uy, uz, reg,
                core_r, core_hh, outer_r, outer_hh)

            if s_collision < d_bnd:
                # ---- COLLISION ----
                if do_tally:
                    _score_track_jit(x, y, z, ux, uy, uz, s_collision, weight,
                                     group, sig_f, sig_a, fissile,
                                     r_edges, z_edges, n_r, n_z,
                                     tally_flux, tally_fission, tally_absorption)
                x += ux * s_collision
                y += uy * s_collision
                z += uz * s_collision
                n_collisions += 1

                # Reaction type
                xi = np.random.random()
                sigma_s_total = 0.0
                for gg in range(N_GROUPS):
                    sigma_s_total += sig_s[group, gg]

                if xi < sigma_s_total / sigma_t_g:
                    # ---- SCATTERING ----
                    # Normalize scattering probabilities and sample outgoing group
                    scatter_sum = sigma_s_total
                    if scatter_sum > 0.0:
                        # Build normalized probabilities
                        xi2 = np.random.random() * scatter_sum
                        cumul = 0.0
                        new_group = N_GROUPS - 1
                        for gg in range(N_GROUPS):
                            cumul += sig_s[group, gg]
                            if xi2 < cumul:
                                new_group = gg
                                break
                        group = new_group

                    # Isotropic scattering (P0)
                    cos_theta = 2.0 * np.random.random() - 1.0
                    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
                    phi = 2.0 * math.pi * np.random.random()
                    ux = sin_theta * math.cos(phi)
                    uy = sin_theta * math.sin(phi)
                    uz = cos_theta

                else:
                    # ---- ABSORPTION ----
                    if fissile and sig_f[group] > 0.0:
                        xi2 = np.random.random()
                        if xi2 < sig_f[group] / sig_a[group]:
                            # ---- FISSION ----
                            nu = nu_sig_f[group] / sig_f[group]
                            n_new = int(nu)
                            if np.random.random() < (nu - n_new):
                                n_new += 1
                            for _ in range(n_new):
                                fg = _sample_cdf_jit(chi_mat, N_GROUPS)
                                idx = fission_offset + fission_count
                                if idx < fission_x.shape[0]:
                                    fission_x[idx] = x
                                    fission_y[idx] = y
                                    fission_z[idx] = z
                                    fission_g[idx] = fg
                                fission_count += 1

                    alive = False

            else:
                # ---- BOUNDARY CROSSING ----
                if do_tally:
                    _score_track_jit(x, y, z, ux, uy, uz, d_bnd, weight,
                                     group, sig_f, sig_a, fissile,
                                     r_edges, z_edges, n_r, n_z,
                                     tally_flux, tally_fission, tally_absorption)
                x += ux * (d_bnd + EPSILON)
                y += uy * (d_bnd + EPSILON)
                z += uz * (d_bnd + EPSILON)

                if next_reg < 0:
                    if do_tally:
                        leakage_wt += weight
                    alive = False

            # Russian roulette
            if alive and weight < WEIGHT_CUTOFF:
                if np.random.random() < 0.5:
                    weight = SURVIVAL_WEIGHT
                else:
                    alive = False

        return fission_count, leakage_wt

    # ---------------------------------------------------------------
    # Track-length tally scoring (JIT helper)
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _score_track_jit(x, y, z, ux, uy, uz, track_length, weight, group,
                         sigma_f, sigma_a, is_fissile,
                         r_edges, z_edges, n_r, n_z,
                         tally_flux, tally_fission, tally_absorption):
        """Score track-length contribution at midpoint (simplified)."""
        xm = x + ux * track_length * 0.5
        ym = y + uy * track_length * 0.5
        zm = z + uz * track_length * 0.5
        rm = math.sqrt(xm * xm + ym * ym)

        # Binary search for r bin
        ir = -1
        for i in range(n_r):
            if rm >= r_edges[i] and rm < r_edges[i + 1]:
                ir = i
                break

        # Binary search for z bin
        iz = -1
        for j in range(n_z):
            if zm >= z_edges[j] and zm < z_edges[j + 1]:
                iz = j
                break

        if ir >= 0 and iz >= 0:
            # Compute bin volume: pi * (r_out^2 - r_in^2) * dz
            r_in = r_edges[ir]
            r_out = r_edges[ir + 1]
            dz = z_edges[iz + 1] - z_edges[iz]
            vol = math.pi * (r_out * r_out - r_in * r_in) * dz
            if vol > 0.0:
                contrib = weight * track_length / vol
                tally_flux[ir, iz, group] += contrib
                if is_fissile:
                    tally_fission[ir, iz] += contrib * sigma_f[group]
                tally_absorption[ir, iz] += contrib * sigma_a[group]

    # ---------------------------------------------------------------
    # Transport an entire chunk of particles (called by each worker)
    # ---------------------------------------------------------------
    @njit(cache=True)
    def _transport_chunk_jit(
        # Particle arrays
        px, py, pz, pux, puy, puz, pgroup, pweight, palive,
        # Geometry scalars
        core_r, core_hh, outer_r, outer_hh,
        # Material 0 (core/fuel)
        sigma_t_0, sigma_s_0, sigma_a_0, sigma_f_0,
        nu_sigma_f_0, chi_0, is_fissile_0,
        # Material 1 (reflector)
        sigma_t_1, sigma_s_1, sigma_a_1, sigma_f_1,
        nu_sigma_f_1, chi_1, is_fissile_1,
        # Tally mesh
        r_edges, z_edges, n_r, n_z, do_tally,
        # Max fission sites for pre-allocation
        max_fission_sites,
        # Random seed for this chunk
        seed,
    ):
        """Transport all particles in a chunk.

        Returns:
            (fission_x, fission_y, fission_z, fission_g, total_fission_count,
             tally_flux, tally_fission, tally_absorption, total_leakage)
        """
        # Seed Numba's random state for this chunk
        np.random.seed(seed)

        n_particles = px.shape[0]

        # Pre-allocate fission site storage
        fission_x = np.zeros(max_fission_sites)
        fission_y = np.zeros(max_fission_sites)
        fission_z = np.zeros(max_fission_sites)
        fission_g = np.zeros(max_fission_sites, dtype=numba.int32)
        total_fission_count = 0

        # Allocate tally arrays
        if do_tally:
            tally_flux = np.zeros((n_r, n_z, N_GROUPS))
            tally_fission = np.zeros((n_r, n_z))
            tally_absorption = np.zeros((n_r, n_z))
        else:
            tally_flux = np.zeros((1, 1, 1))
            tally_fission = np.zeros((1, 1))
            tally_absorption = np.zeros((1, 1))

        total_leakage = 0.0

        for i in range(n_particles):
            if not palive[i]:
                continue

            fc, lk = _transport_particle_jit(
                px[i], py[i], pz[i],
                pux[i], puy[i], puz[i],
                pgroup[i], pweight[i],
                core_r, core_hh, outer_r, outer_hh,
                sigma_t_0, sigma_s_0, sigma_a_0, sigma_f_0,
                nu_sigma_f_0, chi_0, is_fissile_0,
                sigma_t_1, sigma_s_1, sigma_a_1, sigma_f_1,
                nu_sigma_f_1, chi_1, is_fissile_1,
                r_edges, z_edges, n_r, n_z, do_tally,
                fission_x, fission_y, fission_z, fission_g,
                total_fission_count,
                tally_flux, tally_fission, tally_absorption,
            )
            total_fission_count += fc
            total_leakage += lk

        return (fission_x, fission_y, fission_z, fission_g,
                total_fission_count,
                tally_flux, tally_fission, tally_absorption,
                total_leakage)


# ===================================================================
# Helper: pack materials and geometry into flat arrays / scalars
# ===================================================================

def _pack_geometry(geometry: MCFRGeometry):
    """Extract geometry scalars for JIT kernels."""
    return (
        geometry.core_radius,
        geometry.core_half_height,
        geometry.outer_radius,
        geometry.outer_half_height,
    )


def _pack_material(mat: Material):
    """Pack one Material into flat numpy arrays for JIT kernels.

    Returns:
        (sigma_t, sigma_s, sigma_a, sigma_f, nu_sigma_f, chi, is_fissile)
        All arrays are contiguous float64 except is_fissile (bool -> int).
    """
    return (
        np.ascontiguousarray(mat.sigma_t, dtype=np.float64),
        np.ascontiguousarray(mat.sigma_s, dtype=np.float64),
        np.ascontiguousarray(mat.sigma_a, dtype=np.float64),
        np.ascontiguousarray(mat.sigma_f, dtype=np.float64),
        np.ascontiguousarray(mat.nu_sigma_f, dtype=np.float64),
        np.ascontiguousarray(mat.chi, dtype=np.float64),
        1 if mat.is_fissile else 0,
    )


def _pack_tally(tallies: Optional[TallyAccumulator]):
    """Pack tally mesh into arrays for JIT.

    Returns:
        (r_edges, z_edges, n_r, n_z, do_tally)
    """
    if tallies is None:
        # Dummy mesh (won't be used because do_tally=False)
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            1, 1, False,
        )
    return (
        np.ascontiguousarray(tallies.r_edges, dtype=np.float64),
        np.ascontiguousarray(tallies.z_edges, dtype=np.float64),
        tallies.n_r,
        tallies.n_z,
        True,
    )


# ===================================================================
# Multiprocessing worker function (top-level for pickle)
# ===================================================================

def _worker_transport(args):
    """Worker function executed in each Pool process.

    Receives serialisable numpy arrays / scalars (no class instances).
    Returns fission sites and partial tally arrays.
    """
    (
        px, py, pz, pux, puy, puz, pgroup, pweight, palive,
        core_r, core_hh, outer_r, outer_hh,
        sigma_t_0, sigma_s_0, sigma_a_0, sigma_f_0,
        nu_sigma_f_0, chi_0, is_fissile_0,
        sigma_t_1, sigma_s_1, sigma_a_1, sigma_f_1,
        nu_sigma_f_1, chi_1, is_fissile_1,
        r_edges, z_edges, n_r, n_z, do_tally,
        max_fission_sites, seed,
    ) = args

    result = _transport_chunk_jit(
        px, py, pz, pux, puy, puz, pgroup, pweight, palive,
        core_r, core_hh, outer_r, outer_hh,
        sigma_t_0, sigma_s_0, sigma_a_0, sigma_f_0,
        nu_sigma_f_0, chi_0, is_fissile_0,
        sigma_t_1, sigma_s_1, sigma_a_1, sigma_f_1,
        nu_sigma_f_1, chi_1, is_fissile_1,
        r_edges, z_edges, n_r, n_z, do_tally,
        max_fission_sites, seed,
    )

    (fx, fy, fz, fg, fc,
     t_flux, t_fission, t_absorption, t_leakage) = result

    # Trim fission arrays to actual count
    fc = int(fc)
    fx = fx[:fc].copy()
    fy = fy[:fc].copy()
    fz = fz[:fc].copy()
    fg = fg[:fc].copy()

    return (fx, fy, fz, fg, fc,
            t_flux, t_fission, t_absorption, float(t_leakage))


# ===================================================================
# Fallback worker (pure Python, no Numba)
# ===================================================================

def _worker_transport_fallback(args):
    """Fallback worker using the pure-Python transport from physics.py.

    Each worker process creates its own RNG from the provided seed,
    constructs temporary geometry/material/tally objects, and calls
    the reference transport_batch_sequential.
    """
    (
        px, py, pz, pux, puy, puz, pgroup, pweight, palive,
        core_r, core_hh, refl_radial, refl_axial,
        mat0_dict, mat1_dict,
        r_edges, z_edges, n_r, n_z, do_tally,
        seed,
    ) = args

    from ..physics import transport_particle as _tp_ref
    from ..particle import FissionBank as _FB, sample_from_cdf as _sample_cdf

    rng = np.random.default_rng(seed)

    # Reconstruct geometry
    geom = MCFRGeometry(
        core_radius=core_r,
        core_half_height=core_hh,
        reflector_radial=refl_radial,
        reflector_axial=refl_axial,
    )

    # Reconstruct materials
    mat0 = Material(**mat0_dict)
    mat1 = Material(**mat1_dict)
    mats = {0: mat0, 1: mat1}

    # Reconstruct tallies
    if do_tally:
        tallies = TallyAccumulator(r_edges, z_edges)
    else:
        tallies = None

    n_particles = len(px)
    fb = FissionBank(max_sites=4 * n_particles)

    for i in range(n_particles):
        if not palive[i]:
            continue
        _tp_ref(
            px[i], py[i], pz[i],
            pux[i], puy[i], puz[i],
            int(pgroup[i]), float(pweight[i]),
            geom, mats, fb, tallies, rng,
        )

    fc = fb.count
    fx = fb.x[:fc].copy()
    fy = fb.y[:fc].copy()
    fz = fb.z[:fc].copy()
    fg = fb.group[:fc].copy()

    if tallies is not None:
        return (fx, fy, fz, fg, fc,
                tallies.flux.copy(), tallies.fission_rate.copy(),
                tallies.absorption_rate.copy(), float(tallies.leakage))
    else:
        return (fx, fy, fz, fg, fc,
                np.zeros((1, 1, 1)), np.zeros((1, 1)),
                np.zeros((1, 1)), 0.0)


def _material_to_dict(mat: Material) -> dict:
    """Serialize a Material dataclass into a pickle-safe dict."""
    return dict(
        name=mat.name,
        mat_id=mat.mat_id,
        density=mat.density,
        temperature=mat.temperature,
        sigma_t=mat.sigma_t.copy(),
        sigma_s=mat.sigma_s.copy(),
        sigma_a=mat.sigma_a.copy(),
        sigma_f=mat.sigma_f.copy(),
        nu_sigma_f=mat.nu_sigma_f.copy(),
        chi=mat.chi.copy(),
        is_fissile=mat.is_fissile,
    )


# ===================================================================
# CPUBackend class
# ===================================================================

class CPUBackend(MCBackend):
    """CPU-parallel Monte Carlo transport backend.

    Uses multiprocessing.Pool for inter-core parallelism and (optionally)
    Numba JIT for intra-core speedup of the transport kernel.

    Parameters
    ----------
    n_workers : int or None
        Number of worker processes.  ``None`` -> ``os.cpu_count()``.
    use_numba : bool
        If True (default) and Numba is installed, use JIT-compiled kernels.
        Falls back to pure-Python if Numba is unavailable regardless of
        this flag.
    """

    def __init__(self, n_workers: Optional[int] = None, use_numba: bool = True):
        if n_workers is None:
            n_workers = cpu_count() or 1
        self._n_workers = max(1, n_workers)
        self._use_numba = use_numba and HAS_NUMBA

    # ------------------------------------------------------------------
    # MCBackend interface
    # ------------------------------------------------------------------

    def transport_batch(
        self,
        source_bank: ParticleBank,
        geometry: MCFRGeometry,
        materials: Dict[int, Material],
        tallies: Optional[TallyAccumulator],
        rng: np.random.Generator,
    ) -> FissionBank:
        """Transport all particles in *source_bank* through one generation.

        Splits the source bank into ``n_workers`` chunks, dispatches each
        to a worker process, then merges fission banks and aggregates
        tally contributions.
        """
        n = source_bank.n_particles
        if n == 0:
            return FissionBank(max_sites=1)

        # Generate independent seeds for each worker
        seeds = rng.integers(0, 2**31, size=self._n_workers).tolist()

        # Split particle arrays into chunks
        chunks = source_bank.split(self._n_workers)

        if self._use_numba:
            result = self._dispatch_numba(chunks, geometry, materials,
                                          tallies, seeds, n)
        else:
            result = self._dispatch_fallback(chunks, geometry, materials,
                                             tallies, seeds, n)
        return result

    def get_name(self) -> str:
        n = self._n_workers
        mode = "Numba JIT" if self._use_numba else "pure-Python"
        return f"CPU ({n} core{'s' if n > 1 else ''}, {mode})"

    def is_available(self) -> bool:
        return True  # CPU is always available

    # ------------------------------------------------------------------
    # Internal: Numba path
    # ------------------------------------------------------------------

    def _dispatch_numba(self, chunks, geometry, materials, tallies, seeds, n_total):
        """Build worker args, dispatch via Pool, merge results (Numba path)."""
        core_r, core_hh, outer_r, outer_hh = _pack_geometry(geometry)
        (st0, ss0, sa0, sf0, nsf0, chi0, fis0) = _pack_material(materials[CORE])
        (st1, ss1, sa1, sf1, nsf1, chi1, fis1) = _pack_material(materials[REFLECTOR])
        r_edges, z_edges, n_r, n_z, do_tally = _pack_tally(tallies)

        max_fission_per_chunk = 4 * n_total  # generous upper bound

        worker_args = []
        for i, chunk in enumerate(chunks):
            worker_args.append((
                np.ascontiguousarray(chunk.x),
                np.ascontiguousarray(chunk.y),
                np.ascontiguousarray(chunk.z),
                np.ascontiguousarray(chunk.ux),
                np.ascontiguousarray(chunk.uy),
                np.ascontiguousarray(chunk.uz),
                np.ascontiguousarray(chunk.group, dtype=np.int64),
                np.ascontiguousarray(chunk.weight),
                np.ascontiguousarray(chunk.alive),
                core_r, core_hh, outer_r, outer_hh,
                st0, ss0, sa0, sf0, nsf0, chi0, fis0,
                st1, ss1, sa1, sf1, nsf1, chi1, fis1,
                r_edges, z_edges, n_r, n_z, do_tally,
                max_fission_per_chunk, seeds[i],
            ))

        # Dispatch
        if self._n_workers == 1:
            results = [_worker_transport(worker_args[0])]
        else:
            with Pool(processes=self._n_workers) as pool:
                results = pool.map(_worker_transport, worker_args)

        return self._merge_results(results, tallies, n_total)

    # ------------------------------------------------------------------
    # Internal: fallback path (no Numba)
    # ------------------------------------------------------------------

    def _dispatch_fallback(self, chunks, geometry, materials, tallies, seeds, n_total):
        """Dispatch using pure-Python reference transport (fallback)."""
        r_edges, z_edges, n_r, n_z, do_tally = _pack_tally(tallies)

        mat0_dict = _material_to_dict(materials[CORE])
        mat1_dict = _material_to_dict(materials[REFLECTOR])

        refl_radial = geometry.reflector_radial
        refl_axial = geometry.reflector_axial
        core_r = geometry.core_radius
        core_hh = geometry.core_half_height

        worker_args = []
        for i, chunk in enumerate(chunks):
            worker_args.append((
                chunk.x.copy(), chunk.y.copy(), chunk.z.copy(),
                chunk.ux.copy(), chunk.uy.copy(), chunk.uz.copy(),
                chunk.group.copy(), chunk.weight.copy(), chunk.alive.copy(),
                core_r, core_hh, refl_radial, refl_axial,
                mat0_dict, mat1_dict,
                r_edges, z_edges, n_r, n_z, do_tally,
                seeds[i],
            ))

        if self._n_workers == 1:
            results = [_worker_transport_fallback(worker_args[0])]
        else:
            with Pool(processes=self._n_workers) as pool:
                results = pool.map(_worker_transport_fallback, worker_args)

        return self._merge_results(results, tallies, n_total)

    # ------------------------------------------------------------------
    # Merge worker results
    # ------------------------------------------------------------------

    def _merge_results(self, results, tallies, n_total):
        """Merge fission banks and aggregate tallies from all workers."""
        total_fission = sum(r[4] for r in results)
        fission_bank = FissionBank(max_sites=max(total_fission + 1, 1))

        for (fx, fy, fz, fg, fc, t_flux, t_fiss, t_abs, t_leak) in results:
            if fc > 0:
                fission_bank.add_batch(fx, fy, fz, fg)

            if tallies is not None:
                # Aggregate partial tallies
                if t_flux.shape == tallies.flux.shape:
                    tallies.flux += t_flux
                if t_fiss.shape == tallies.fission_rate.shape:
                    tallies.fission_rate += t_fiss
                if t_abs.shape == tallies.absorption_rate.shape:
                    tallies.absorption_rate += t_abs
                tallies.leakage += t_leak

        return fission_bank
