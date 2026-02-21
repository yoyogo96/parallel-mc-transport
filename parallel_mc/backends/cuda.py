"""
CUDA GPU Backend using Numba CUDA kernels.

Each GPU thread transports one particle through its complete history.
Uses Xoroshiro128+ RNG for per-thread random numbers.
Fission bank stored in pre-allocated device buffer with atomic counter.

Tally scoring on GPU is skipped for simplicity: the kernel only accumulates
fission sites. For active batches that require flux/fission-rate tallies,
callers should fall back to the CPU backend or implement a separate reduction
pass after the kernel returns. This matches the common MC pattern where
inactive (source-convergence) batches dominate, and tallies are only needed
for a handful of active batches.
"""

import math

import numpy as np
from typing import Dict, Optional

from ..constants import N_GROUPS, MAX_COLLISIONS, EPSILON, WEIGHT_CUTOFF, SURVIVAL_WEIGHT
from ..geometry import MCFRGeometry, CORE, REFLECTOR, VACUUM
from ..materials import Material
from ..particle import ParticleBank, FissionBank
from ..tallies import TallyAccumulator
from .base import MCBackend

# ---------------------------------------------------------------------------
# Optional CUDA imports -- failure leaves HAS_CUDA = False
# ---------------------------------------------------------------------------
try:
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float64,
    )
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# ---------------------------------------------------------------------------
# Constants used inside the kernel (must be Python-level literals for Numba)
# ---------------------------------------------------------------------------
_N_GROUPS = 8          # == N_GROUPS, repeated as a literal for Numba
_N_MATS = 2            # 0 = fuel core, 1 = reflector
_MAX_COLLISIONS = 500  # == MAX_COLLISIONS
_PI = 3.141592653589793

# ---------------------------------------------------------------------------
# CUDA transport kernel
# ---------------------------------------------------------------------------

if HAS_CUDA:

    @cuda.jit
    def _transport_kernel(
        # ---- Particle SoA (device, read-only) ----
        x, y, z,
        ux, uy, uz,
        group, weight, alive,
        # ---- Fission bank output (pre-allocated device arrays) ----
        fiss_x, fiss_y, fiss_z, fiss_g,
        fiss_count,           # int32[1] atomic counter
        # ---- Flattened material tables (device, read-only) ----
        # Layout: sigma_t_flat[mat_id * 8 + g]
        sigma_t_flat,
        # Layout: sigma_s_flat[mat_id * 64 + g_in * 8 + g_out]
        sigma_s_flat,
        sigma_a_flat,
        sigma_f_flat,
        nu_sigma_f_flat,
        # Layout: chi_flat[mat_id * 8 + g]
        chi_flat,
        # Layout: is_fissile_flat[mat_id]  (1.0 = fissile, 0.0 = not)
        is_fissile_flat,
        # ---- Geometry scalars ----
        core_r,      # core radius   (m)
        core_hh,     # core half-height (m)
        outer_r,     # outer (core+reflector) radius (m)
        outer_hh,    # outer half-height (m)
        # ---- RNG states ----
        rng_states,
        # ---- Capacity limit ----
        max_fiss,
    ):
        """
        One CUDA thread transports one particle for its full history.

        Physical model
        --------------
        - Woodcock / delta-tracking is NOT used; explicit surface tracking.
        - Only two regions: CORE (reg=0) and REFLECTOR (reg=1).
        - Collision types: scatter (isotropic), absorption (fission or capture).
        - Russian roulette below WEIGHT_CUTOFF; no splitting.
        - Tallies are NOT scored here (see module docstring).

        Geometry
        --------
        Finite cylinder centred at origin, Z-axis is the cylinder axis.
        CORE    : r < core_r  AND  |z| < core_hh
        REFLECTOR: r < outer_r AND |z| < outer_hh  (excluding CORE)
        VACUUM  : everything else (particle terminates immediately)
        """
        tid = cuda.grid(1)
        n_particles = x.shape[0]
        if tid >= n_particles:
            return
        if not alive[tid]:
            return

        # ------------------------------------------------------------------ #
        # Load particle state into local registers
        # ------------------------------------------------------------------ #
        px = x[tid]
        py = y[tid]
        pz = z[tid]
        pux = ux[tid]
        puy = uy[tid]
        puz = uz[tid]
        pg = group[tid]
        pw = weight[tid]

        # Pre-compute squares for reuse
        core_r2  = core_r  * core_r
        outer_r2 = outer_r * outer_r

        # ------------------------------------------------------------------ #
        # Main transport loop
        # ------------------------------------------------------------------ #
        collision_count = 0
        while collision_count < _MAX_COLLISIONS:
            collision_count += 1

            # -------------------------------------------------------------- #
            # 1. Determine current region
            # -------------------------------------------------------------- #
            r2 = px * px + py * py
            az = pz if pz >= 0.0 else -pz  # abs(pz)

            if r2 < core_r2 and az < core_hh:
                reg = 0  # CORE
            elif r2 < outer_r2 and az < outer_hh:
                reg = 1  # REFLECTOR
            else:
                # Particle has escaped to vacuum
                break

            # -------------------------------------------------------------- #
            # 2. Retrieve total cross-section for current group & region
            # -------------------------------------------------------------- #
            sig_t = sigma_t_flat[reg * _N_GROUPS + pg]
            if sig_t < 1.0e-20:
                # Transparent material -- stream to boundary
                sig_t = 1.0e-20

            # -------------------------------------------------------------- #
            # 3. Sample free-flight distance to collision
            # -------------------------------------------------------------- #
            xi = xoroshiro128p_uniform_float64(rng_states, tid)
            if xi < 1.0e-30:
                xi = 1.0e-30
            s_coll = -math.log(xi) / sig_t

            # -------------------------------------------------------------- #
            # 4. Compute distance to the nearest surface (d_bnd, next_reg)
            # -------------------------------------------------------------- #
            d_bnd   = 1.0e30
            next_reg = -1  # vacuum by default

            a_r = pux * pux + puy * puy   # radial direction^2
            b_r = px * pux + py * puy     # used for cylinder intersections

            if reg == 0:
                # ---- CORE: only boundary is the core cylinder / planes -> REFLECTOR ----

                # Radial exit from core cylinder
                if a_r > 1.0e-20:
                    c_r  = r2 - core_r2
                    disc = b_r * b_r - a_r * c_r
                    if disc >= 0.0:
                        sqrt_d = math.sqrt(disc)
                        t = (-b_r + sqrt_d) / a_r   # exit root (larger)
                        if t > 1.0e-10 and t < d_bnd:
                            d_bnd    = t
                            next_reg = 1

                # Axial exit from core: top plane (puz > 0) or bottom (puz < 0)
                if puz > 1.0e-20:
                    t_top = (core_hh - pz) / puz
                    if t_top > 1.0e-10 and t_top < d_bnd:
                        d_bnd    = t_top
                        next_reg = 1
                elif puz < -1.0e-20:
                    t_bot = (-core_hh - pz) / puz
                    if t_bot > 1.0e-10 and t_bot < d_bnd:
                        d_bnd    = t_bot
                        next_reg = 1

            else:
                # ---- REFLECTOR: can exit to VACUUM (outer surfaces)
                #                 or re-enter CORE (inner surfaces)        ----

                # --- Outer radial boundary -> VACUUM ---
                if a_r > 1.0e-20:
                    c_out  = r2 - outer_r2
                    disc   = b_r * b_r - a_r * c_out
                    if disc >= 0.0:
                        sqrt_d = math.sqrt(disc)
                        t = (-b_r + sqrt_d) / a_r
                        if t > 1.0e-10 and t < d_bnd:
                            d_bnd    = t
                            next_reg = -1  # VACUUM

                # --- Outer axial boundaries -> VACUUM ---
                if puz > 1.0e-20:
                    t_top = (outer_hh - pz) / puz
                    if t_top > 1.0e-10 and t_top < d_bnd:
                        d_bnd    = t_top
                        next_reg = -1
                elif puz < -1.0e-20:
                    t_bot = (-outer_hh - pz) / puz
                    if t_bot > 1.0e-10 and t_bot < d_bnd:
                        d_bnd    = t_bot
                        next_reg = -1

                # --- Inner radial boundary -> CORE ---
                # Particle is outside core_r2 (in reflector), heading inward.
                # Entry root is the smaller t (t1 = (-b - sqrt)/a).
                if a_r > 1.0e-20:
                    c_in   = r2 - core_r2   # > 0 since we are outside core
                    disc   = b_r * b_r - a_r * c_in
                    if disc >= 0.0 and c_in > 0.0:
                        sqrt_d = math.sqrt(disc)
                        t1 = (-b_r - sqrt_d) / a_r
                        if t1 > 1.0e-10 and t1 < d_bnd:
                            # Confirm the z-coordinate is inside the core height
                            z_at = pz + puz * t1
                            az_at = z_at if z_at >= 0.0 else -z_at
                            if az_at < core_hh:
                                d_bnd    = t1
                                next_reg = 0  # CORE

                # --- Inner axial planes -> CORE ---
                if puz > 1.0e-20:
                    t_ctop = (core_hh - pz) / puz
                    if t_ctop > 1.0e-10 and t_ctop < d_bnd:
                        x_at = px + pux * t_ctop
                        y_at = py + puy * t_ctop
                        if x_at * x_at + y_at * y_at < core_r2:
                            d_bnd    = t_ctop
                            next_reg = 0
                elif puz < -1.0e-20:
                    t_cbot = (-core_hh - pz) / puz
                    if t_cbot > 1.0e-10 and t_cbot < d_bnd:
                        x_at = px + pux * t_cbot
                        y_at = py + puy * t_cbot
                        if x_at * x_at + y_at * y_at < core_r2:
                            d_bnd    = t_cbot
                            next_reg = 0

            # -------------------------------------------------------------- #
            # 5. Collision vs. boundary crossing
            # -------------------------------------------------------------- #
            if s_coll < d_bnd:
                # ---- COLLISION ----
                px += pux * s_coll
                py += puy * s_coll
                pz += puz * s_coll

                # Total outscatter XS (summed over outgoing groups)
                sig_s_total = 0.0
                s_base = reg * 64 + pg * _N_GROUPS  # index into sigma_s_flat
                for g2 in range(_N_GROUPS):
                    sig_s_total += sigma_s_flat[s_base + g2]

                xi2 = xoroshiro128p_uniform_float64(rng_states, tid)

                if xi2 * sig_t < sig_s_total:
                    # ---- SCATTER ----
                    # Sample outgoing group proportional to sigma_s[pg, g_out]
                    xi3 = xoroshiro128p_uniform_float64(rng_states, tid) * sig_s_total
                    cum = 0.0
                    new_g = pg
                    for g2 in range(_N_GROUPS):
                        cum += sigma_s_flat[s_base + g2]
                        if xi3 <= cum:
                            new_g = g2
                            break
                    pg = new_g

                    # Sample new isotropic direction
                    cos_th  = 2.0 * xoroshiro128p_uniform_float64(rng_states, tid) - 1.0
                    sin_th  = math.sqrt(max(0.0, 1.0 - cos_th * cos_th))
                    phi     = 2.0 * _PI * xoroshiro128p_uniform_float64(rng_states, tid)
                    pux = sin_th * math.cos(phi)
                    puy = sin_th * math.sin(phi)
                    puz = cos_th

                else:
                    # ---- ABSORPTION ----
                    fissile = is_fissile_flat[reg]
                    sig_f   = sigma_f_flat[reg * _N_GROUPS + pg]
                    sig_a   = sigma_a_flat[reg * _N_GROUPS + pg]

                    if fissile > 0.5 and sig_f > 1.0e-20 and sig_a > 1.0e-20:
                        xi4 = xoroshiro128p_uniform_float64(rng_states, tid)
                        if xi4 * sig_a < sig_f:
                            # ---- FISSION ----
                            # Expected number of neutrons produced
                            nu_sf = nu_sigma_f_flat[reg * _N_GROUPS + pg]
                            nu    = nu_sf / sig_f if sig_f > 1.0e-20 else 0.0

                            # Integer sampling of n_new (stochastic rounding)
                            n_new = int(nu)
                            if xoroshiro128p_uniform_float64(rng_states, tid) < (nu - n_new):
                                n_new += 1

                            # Emit each fission neutron
                            chi_base = reg * _N_GROUPS
                            for _ in range(n_new):
                                # Sample outgoing group from chi spectrum
                                xi5    = xoroshiro128p_uniform_float64(rng_states, tid)
                                cum_c  = 0.0
                                fg     = 0
                                for g2 in range(_N_GROUPS):
                                    cum_c += chi_flat[chi_base + g2]
                                    if xi5 <= cum_c:
                                        fg = g2
                                        break

                                # Atomic increment of fission counter
                                idx = cuda.atomic.add(fiss_count, 0, 1)
                                if idx < max_fiss:
                                    fiss_x[idx] = px
                                    fiss_y[idx] = py
                                    fiss_z[idx] = pz
                                    fiss_g[idx] = fg

                    # Particle absorbed (fission or capture) -- end history
                    break

            else:
                # ---- BOUNDARY CROSSING ----
                # Advance particle just past the surface (EPSILON nudge)
                px += pux * (d_bnd + EPSILON)
                py += puy * (d_bnd + EPSILON)
                pz += puz * (d_bnd + EPSILON)

                if next_reg < 0:
                    # Escaped to vacuum -- end history
                    break
                # Otherwise continue in next_reg (loop re-checks region)

            # -------------------------------------------------------------- #
            # 6. Russian roulette on low-weight particles
            # -------------------------------------------------------------- #
            if pw < WEIGHT_CUTOFF:
                if xoroshiro128p_uniform_float64(rng_states, tid) < 0.5:
                    pw = SURVIVAL_WEIGHT   # survive with doubled weight
                else:
                    break                   # killed

        # Particle history complete -- write-back not required (SoA is input-only)


# ---------------------------------------------------------------------------
# CUDABackend class
# ---------------------------------------------------------------------------

class CUDABackend(MCBackend):
    """
    CUDA GPU backend for MC neutron transport.

    Requirements
    ------------
    - NVIDIA GPU with CUDA compute capability >= 6.0
    - numba >= 0.57  (``pip install numba``)
    - numpy >= 1.20

    cupy is NOT required; all device memory is managed via numba.cuda.

    Usage
    -----
    >>> from parallel_mc.backends.cuda import CUDABackend
    >>> backend = CUDABackend()
    >>> if backend.is_available():
    ...     fission_bank = backend.transport_batch(source, geom, mats, None, rng)
    """

    def __init__(self):
        self._available = False
        self._device_name = "N/A"

        if not HAS_CUDA:
            return

        try:
            # Query current device to confirm CUDA is functional
            device = cuda.get_current_device()
            raw_name = device.name
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode("utf-8", errors="replace")
            self._device_name = raw_name
            self._available = True
        except Exception:
            self._available = False

    # ------------------------------------------------------------------
    # MCBackend interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def get_name(self) -> str:
        if self._available:
            return f"CUDA ({self._device_name})"
        if not HAS_CUDA:
            return "CUDA (not available -- install numba: pip install numba)"
        return "CUDA (GPU not detected)"

    def transport_batch(
        self,
        source_bank: ParticleBank,
        geometry: MCFRGeometry,
        materials: Dict[int, Material],
        tallies: Optional[TallyAccumulator],
        rng: np.random.Generator,
    ) -> FissionBank:
        """
        Transport all particles in source_bank on the GPU.

        Steps
        -----
        1. Pack material cross-sections into flat float64 device arrays.
        2. Transfer particle SoA to device.
        3. Allocate fission bank device buffer (4 * n_particles slots).
        4. Create per-thread Xoroshiro128+ RNG states.
        5. Launch _transport_kernel with 256 threads/block.
        6. Synchronise and copy fission bank back to host.
        7. Return populated FissionBank.

        Note: tallies argument is accepted for API compatibility but is
        NOT scored on the GPU. Pass tallies=None for inactive batches.
        For active batches requiring tally data, use the CPU backend.

        Parameters
        ----------
        source_bank : ParticleBank
        geometry    : MCFRGeometry
        materials   : dict  {0: fuel_material, 1: reflector_material}
        tallies     : TallyAccumulator | None   (ignored on GPU)
        rng         : numpy.random.Generator

        Returns
        -------
        FissionBank
        """
        if not self._available:
            raise RuntimeError(
                "CUDA backend is not available. "
                "Check GPU presence and numba installation."
            )

        n = source_bank.n_particles
        if n == 0:
            return FissionBank(max_sites=1)

        # ------------------------------------------------------------------ #
        # Build flattened material tables
        # ------------------------------------------------------------------ #
        n_mats = _N_MATS   # 0 = core fuel, 1 = reflector

        sigma_t_flat    = np.zeros(n_mats * _N_GROUPS,              dtype=np.float64)
        sigma_s_flat    = np.zeros(n_mats * _N_GROUPS * _N_GROUPS,  dtype=np.float64)
        sigma_a_flat    = np.zeros(n_mats * _N_GROUPS,              dtype=np.float64)
        sigma_f_flat    = np.zeros(n_mats * _N_GROUPS,              dtype=np.float64)
        nu_sigma_f_flat = np.zeros(n_mats * _N_GROUPS,              dtype=np.float64)
        chi_flat        = np.zeros(n_mats * _N_GROUPS,              dtype=np.float64)
        is_fissile_flat = np.zeros(n_mats,                          dtype=np.float64)

        for mat_id, mat in materials.items():
            if mat_id < 0 or mat_id >= n_mats:
                continue
            t_off = mat_id * _N_GROUPS
            s_off = mat_id * _N_GROUPS * _N_GROUPS
            sigma_t_flat[t_off : t_off + _N_GROUPS]         = mat.sigma_t
            sigma_a_flat[t_off : t_off + _N_GROUPS]         = mat.sigma_a
            sigma_f_flat[t_off : t_off + _N_GROUPS]         = mat.sigma_f
            nu_sigma_f_flat[t_off : t_off + _N_GROUPS]      = mat.nu_sigma_f
            chi_flat[t_off : t_off + _N_GROUPS]              = mat.chi
            is_fissile_flat[mat_id]                          = 1.0 if mat.is_fissile else 0.0
            # sigma_s is stored row-major [g_in, g_out] -> flatten directly
            sigma_s_flat[s_off : s_off + _N_GROUPS * _N_GROUPS] = mat.sigma_s.ravel()

        # Normalise chi so each material's spectrum sums to 1
        for mat_id in range(n_mats):
            t_off = mat_id * _N_GROUPS
            chi_sum = chi_flat[t_off : t_off + _N_GROUPS].sum()
            if chi_sum > 0.0:
                chi_flat[t_off : t_off + _N_GROUPS] /= chi_sum

        # ------------------------------------------------------------------ #
        # Transfer particle SoA to device
        # ------------------------------------------------------------------ #
        d_x     = cuda.to_device(np.ascontiguousarray(source_bank.x,     dtype=np.float64))
        d_y     = cuda.to_device(np.ascontiguousarray(source_bank.y,     dtype=np.float64))
        d_z     = cuda.to_device(np.ascontiguousarray(source_bank.z,     dtype=np.float64))
        d_ux    = cuda.to_device(np.ascontiguousarray(source_bank.ux,    dtype=np.float64))
        d_uy    = cuda.to_device(np.ascontiguousarray(source_bank.uy,    dtype=np.float64))
        d_uz    = cuda.to_device(np.ascontiguousarray(source_bank.uz,    dtype=np.float64))
        d_group = cuda.to_device(np.ascontiguousarray(source_bank.group, dtype=np.int32))
        d_weight= cuda.to_device(np.ascontiguousarray(source_bank.weight,dtype=np.float64))
        d_alive = cuda.to_device(np.ascontiguousarray(source_bank.alive, dtype=np.bool_))

        # ------------------------------------------------------------------ #
        # Allocate fission bank on device
        # ------------------------------------------------------------------ #
        max_fiss = max(4 * n, 1024)   # generous upper bound; atomic counter guards it
        d_fiss_x     = cuda.device_array(max_fiss, dtype=np.float64)
        d_fiss_y     = cuda.device_array(max_fiss, dtype=np.float64)
        d_fiss_z     = cuda.device_array(max_fiss, dtype=np.float64)
        d_fiss_g     = cuda.device_array(max_fiss, dtype=np.int32)
        d_fiss_count = cuda.to_device(np.zeros(1, dtype=np.int32))

        # ------------------------------------------------------------------ #
        # Transfer material tables to device
        # ------------------------------------------------------------------ #
        d_sigma_t    = cuda.to_device(sigma_t_flat)
        d_sigma_s    = cuda.to_device(sigma_s_flat)
        d_sigma_a    = cuda.to_device(sigma_a_flat)
        d_sigma_f    = cuda.to_device(sigma_f_flat)
        d_nu_sf      = cuda.to_device(nu_sigma_f_flat)
        d_chi        = cuda.to_device(chi_flat)
        d_fissile    = cuda.to_device(is_fissile_flat)

        # ------------------------------------------------------------------ #
        # Initialise per-thread RNG states
        # ------------------------------------------------------------------ #
        seed = int(rng.integers(0, 2**31 - 1))
        rng_states = create_xoroshiro128p_states(n, seed=seed)

        # ------------------------------------------------------------------ #
        # Kernel launch configuration: 256 threads/block
        # ------------------------------------------------------------------ #
        threads_per_block = 256
        blocks = (n + threads_per_block - 1) // threads_per_block

        _transport_kernel[blocks, threads_per_block](
            # Particle SoA
            d_x, d_y, d_z,
            d_ux, d_uy, d_uz,
            d_group, d_weight, d_alive,
            # Fission bank
            d_fiss_x, d_fiss_y, d_fiss_z, d_fiss_g,
            d_fiss_count,
            # Material tables
            d_sigma_t, d_sigma_s, d_sigma_a, d_sigma_f, d_nu_sf, d_chi, d_fissile,
            # Geometry
            float(geometry.core_radius),
            float(geometry.core_half_height),
            float(geometry.outer_radius),
            float(geometry.outer_half_height),
            # RNG
            rng_states,
            # Capacity
            max_fiss,
        )

        cuda.synchronize()

        # ------------------------------------------------------------------ #
        # Copy fission bank back to host and build FissionBank
        # ------------------------------------------------------------------ #
        fiss_count = int(d_fiss_count.copy_to_host()[0])
        fiss_count = min(fiss_count, max_fiss)  # guard against overflow

        fission_bank = FissionBank(max_sites=max(fiss_count, 1))
        if fiss_count > 0:
            fission_bank.x[:fiss_count]     = d_fiss_x.copy_to_host()[:fiss_count]
            fission_bank.y[:fiss_count]     = d_fiss_y.copy_to_host()[:fiss_count]
            fission_bank.z[:fiss_count]     = d_fiss_z.copy_to_host()[:fiss_count]
            fission_bank.group[:fiss_count] = d_fiss_g.copy_to_host()[:fiss_count]
            fission_bank.count              = fiss_count

        return fission_bank
