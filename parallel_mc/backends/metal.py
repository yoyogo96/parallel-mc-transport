"""
Apple Metal GPU Backend using metalcompute.

Particle-level parallelism on Apple Silicon GPU.
Transport kernel written in Metal Shading Language (MSL).
Uses Philox counter-based PRNG for reproducible random numbers.

Requirements:
    pip install metalcompute
    macOS with Apple Silicon (M1/M2/M3/M4)

metalcompute API (v0.2.9):
    device = mc.Device([index])
    kernel = device.kernel(msl_source_string)
    fn = kernel.function(function_name)
    fn(n_threads, buf0, buf1, ...)   # dispatch
    out_buf = device.buffer(n_bytes) # GPU-backed buffer
    result = np.frombuffer(out_buf, dtype=np.float32)  # readback

    Input buffers: pass numpy arrays or array.array directly.
    Output buffers: create with device.buffer(n_bytes), read via np.frombuffer().
"""
import numpy as np
import platform
from array import array as pyarray
from typing import Dict, Optional

from ..constants import N_GROUPS, MAX_COLLISIONS, EPSILON
from ..geometry import MCFRGeometry
from ..materials import Material
from ..particle import ParticleBank, FissionBank
from ..tallies import TallyAccumulator
from .base import MCBackend

try:
    import metalcompute as mc
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


# ---------------------------------------------------------------------------
# Metal Shading Language kernel
# ---------------------------------------------------------------------------
# Full neutron transport physics in MSL:
#   - Philox 2x32 counter-based PRNG (stateless, no bank conflicts)
#   - Region lookup: core cylinder (reg=0), reflector annulus (reg=1), vacuum
#   - Free-flight distance sampling
#   - Ray-cylinder and ray-plane intersection with exact analytic formulas
#   - Scattering: sample outgoing group from 8x8 cross-section matrix
#   - Absorption: capture vs fission branching via sigma_f/sigma_a
#   - Fission neutron production: stochastic floor/ceil from nu
#   - Russian roulette at weight < WEIGHT_CUTOFF
#   - Atomic increment of fission bank counter (no race condition)
#
# Buffer layout (21 buffers, [[buffer(N)]] indices):
#   0-7:  particle SoA (x,y,z,ux,uy,uz,group,weight) - float32 except group=int32
#   8-11: fission bank output (fiss_x, fiss_y, fiss_z, fiss_g)
#   12:   fission bank counter (uint32, atomic)
#   13-19: material data (sigma_t, sigma_s, sigma_a, sigma_f, nu_sf, chi, is_fissile)
#   20:   geometry + run params (float32[7])
#
# Material data layout (2 materials x 8 groups):
#   sigma_t[mat*8 + g]            total cross-section (1/m)
#   sigma_s[mat*64 + from_g*8+g]  scatter matrix row = from_group
#   sigma_a[mat*8 + g]            absorption
#   sigma_f[mat*8 + g]            fission
#   nu_sf[mat*8 + g]              nu*sigma_f (production)
#   chi[mat*8 + g]                fission spectrum
#   is_fissile[mat]               1.0 if fissile, 0.0 otherwise
#
# params[7]:
#   [0] core_radius, [1] core_half_height, [2] outer_radius, [3] outer_half_height
#   [4] n_particles, [5] max_fiss_sites, [6] seed (float cast of uint32)

METAL_SHADER = r'''
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Philox 2x32-10 counter-based PRNG
// Stateless: each particle thread gets independent, high-quality random stream.
// No shared state, no bank conflicts, fully parallelisable.
// ---------------------------------------------------------------------------
struct PhiloxState {
    uint2 counter;   // 64-bit counter split into two 32-bit halves
    uint2 key;       // 64-bit key (set from thread ID + run seed)
};

// One round of Philox-2x32: multiply-high + XOR
static uint2 philox2x32_round(uint2 ctr, uint2 key) {
    // Multiply lo word by the Philox multiplier; get hi and lo halves
    ulong prod = (ulong)ctr.x * 0xD2511F53UL;
    uint prod_hi = (uint)(prod >> 32);
    uint prod_lo = (uint)(prod & 0xFFFFFFFFUL);
    return uint2(ctr.y ^ key.x ^ prod_hi, prod_lo);
}

// Apply 10 rounds of Philox-2x32 and return a uniform float in [0,1)
static float philox_next(thread PhiloxState &s) {
    s.counter.x += 1u;
    if (s.counter.x == 0u) s.counter.y += 1u;  // carry

    uint2 ctr = s.counter;
    uint2 key = s.key;

    // 10 rounds (security parameter; 7 is enough but 10 is the standard)
    for (int i = 0; i < 10; i++) {
        ctr = philox2x32_round(ctr, key);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    // Map to [0, 1) via division; avoid exact 1.0
    return (float)(ctr.x) * (1.0f / 4294967296.0f);
}

// ---------------------------------------------------------------------------
// Transport kernel
// One GPU thread per source particle.
// Each thread transports its particle through one full history.
// ---------------------------------------------------------------------------
kernel void transport_kernel(
    // --- Particle SoA (input, read-only for position/direction) ---
    device const float *px_in    [[ buffer(0) ]],
    device const float *py_in    [[ buffer(1) ]],
    device const float *pz_in    [[ buffer(2) ]],
    device const float *pux_in   [[ buffer(3) ]],
    device const float *puy_in   [[ buffer(4) ]],
    device const float *puz_in   [[ buffer(5) ]],
    device const int   *pg_in    [[ buffer(6) ]],
    device const float *pw_in    [[ buffer(7) ]],
    // --- Fission bank output ---
    device float       *fiss_x   [[ buffer(8) ]],
    device float       *fiss_y   [[ buffer(9) ]],
    device float       *fiss_z   [[ buffer(10) ]],
    device int         *fiss_g   [[ buffer(11) ]],
    device atomic_uint *fiss_cnt [[ buffer(12) ]],
    // --- Material data (2 materials, 8 groups) ---
    device const float *sigma_t  [[ buffer(13) ]],   // [mat*8 + g]
    device const float *sigma_s  [[ buffer(14) ]],   // [mat*64 + from_g*8 + to_g]
    device const float *sigma_a  [[ buffer(15) ]],   // [mat*8 + g]
    device const float *sigma_f  [[ buffer(16) ]],   // [mat*8 + g]
    device const float *nu_sf    [[ buffer(17) ]],   // [mat*8 + g]
    device const float *chi      [[ buffer(18) ]],   // [mat*8 + g]
    device const float *is_fiss  [[ buffer(19) ]],   // [mat]
    // --- Geometry + run params ---
    device const float *params   [[ buffer(20) ]],
    // params[0]=core_r, params[1]=core_hh, params[2]=outer_r, params[3]=outer_hh
    // params[4]=n_particles, params[5]=max_fiss_sites, params[6]=seed

    uint tid [[ thread_position_in_grid ]]
) {
    const int n_particles = (int)params[4];
    if ((int)tid >= n_particles) return;

    const float core_r   = params[0];
    const float core_hh  = params[1];
    const float outer_r  = params[2];
    const float outer_hh = params[3];
    const int   max_fiss = (int)params[5];
    const uint  seed     = as_type<uint>(params[6]);

    // -----------------------------------------------------------------------
    // Init Philox PRNG state: each thread gets a unique stream
    // -----------------------------------------------------------------------
    PhiloxState rng;
    rng.counter = uint2(0u, (uint)tid);
    rng.key     = uint2(seed ^ (uint)tid * 0x9E3779B9u,
                        0xBB67AE85u + (uint)tid);

    // Load particle state (float precision adequate for ~1m scale geometry)
    float px  = px_in[tid];
    float py  = py_in[tid];
    float pz  = pz_in[tid];
    float pux = pux_in[tid];
    float puy = puy_in[tid];
    float puz = puz_in[tid];
    int   pg  = pg_in[tid];
    float pw  = pw_in[tid];

    const float WCUTOFF = 1e-4f;  // Russian roulette weight threshold
    const float EPS     = 1e-7f;  // boundary nudge (float-safe for ~1m scale)
    const float INF     = 1e30f;

    // -----------------------------------------------------------------------
    // History loop: up to MAX_COLLISIONS events per particle
    // -----------------------------------------------------------------------
    for (int col = 0; col < 500; col++) {

        // --- Region lookup ---
        float r2 = px*px + py*py;
        float az = fabs(pz);
        int reg;
        if (r2 < core_r*core_r && az < core_hh) {
            reg = 0;   // CORE
        } else if (r2 < outer_r*outer_r && az < outer_hh) {
            reg = 1;   // REFLECTOR
        } else {
            break;     // VACUUM: particle escaped
        }

        // --- Total cross-section for current region + group ---
        float sig_t = sigma_t[reg * 8 + pg];
        if (sig_t < 1e-20f) break;  // transparent material (shouldn't happen)

        // --- Sample free-flight distance ---
        float xi = philox_next(rng);
        if (xi < 1e-38f) xi = 1e-38f;  // guard against log(0)
        float s_coll = -log(xi) / sig_t;

        // -------------------------------------------------------------------
        // Distance to geometry boundary
        // Returns: (distance, next_region)  where next_region=-1 means vacuum
        // -------------------------------------------------------------------
        float d_bnd    = INF;
        int   next_reg = -1;

        float a_r = pux*pux + puy*puy;   // radial direction component squared
        float b_r = px*pux + py*puy;     // dot(pos_xy, dir_xy)

        if (reg == 0) {
            // ---------------------------------------------------------------
            // CORE region: find exit to reflector
            // Boundaries: outer lateral cylinder at core_r, axial planes at ±core_hh
            // ---------------------------------------------------------------

            // Radial: (px+pux*t)^2+(py+puy*t)^2 = core_r^2 -> quadratic
            if (a_r > 1e-20f) {
                float c_r  = r2 - core_r*core_r;   // negative (inside cylinder)
                float disc = b_r*b_r - a_r*c_r;
                if (disc >= 0.0f) {
                    float t = (-b_r + sqrt(disc)) / a_r;  // positive root (exit)
                    if (t > EPS && t < d_bnd) {
                        d_bnd    = t;
                        next_reg = 1;  // exits into reflector
                    }
                }
            }
            // Axial top/bottom planes
            if (fabs(puz) > 1e-20f) {
                float t_top = (core_hh - pz) / puz;
                float t_bot = (-core_hh - pz) / puz;
                if (t_top > EPS && t_top < d_bnd) { d_bnd = t_top; next_reg = 1; }
                if (t_bot > EPS && t_bot < d_bnd) { d_bnd = t_bot; next_reg = 1; }
            }

        } else {
            // ---------------------------------------------------------------
            // REFLECTOR region: find exit to vacuum OR re-entry to core
            // Outer boundaries: cylinder at outer_r, planes at ±outer_hh -> vacuum
            // Inner boundary: cylinder at core_r (only if coming from outside),
            //                 planes at ±core_hh intersected with core cylinder
            // ---------------------------------------------------------------

            // --- Outer cylinder -> VACUUM ---
            if (a_r > 1e-20f) {
                float c_r  = r2 - outer_r*outer_r;  // positive (outside -> no, inside)
                float disc = b_r*b_r - a_r*c_r;
                if (disc >= 0.0f) {
                    float t = (-b_r + sqrt(disc)) / a_r;  // exit from inside
                    if (t > EPS && t < d_bnd) { d_bnd = t; next_reg = -1; }
                }
            }
            // Outer axial planes -> VACUUM
            if (fabs(puz) > 1e-20f) {
                float t_top = (outer_hh - pz) / puz;
                float t_bot = (-outer_hh - pz) / puz;
                if (t_top > EPS && t_top < d_bnd) { d_bnd = t_top; next_reg = -1; }
                if (t_bot > EPS && t_bot < d_bnd) { d_bnd = t_bot; next_reg = -1; }
            }

            // --- Inner core cylinder (re-entry to CORE) ---
            // Only intersects if particle is outside core cylinder (r2 > core_r^2)
            // and heading inward (b_r < 0 roughly, but quadratic handles it exactly)
            if (a_r > 1e-20f) {
                float c_r  = r2 - core_r*core_r;  // positive (particle is in reflector)
                float disc = b_r*b_r - a_r*c_r;
                if (disc >= 0.0f) {
                    float t = (-b_r - sqrt(disc)) / a_r;  // near-entry root
                    if (t > EPS && t < d_bnd) {
                        // Check z at intersection is within core height
                        float z_at = pz + puz * t;
                        if (fabs(z_at) < core_hh) { d_bnd = t; next_reg = 0; }
                    }
                }
            }
            // Core axial caps: plane at ±core_hh, only if footprint inside core cylinder
            if (fabs(puz) > 1e-20f) {
                float t_top = (core_hh - pz) / puz;
                if (t_top > EPS && t_top < d_bnd) {
                    float xa = px + pux * t_top;
                    float ya = py + puy * t_top;
                    if (xa*xa + ya*ya < core_r*core_r) { d_bnd = t_top; next_reg = 0; }
                }
                float t_bot = (-core_hh - pz) / puz;
                if (t_bot > EPS && t_bot < d_bnd) {
                    float xa = px + pux * t_bot;
                    float ya = py + puy * t_bot;
                    if (xa*xa + ya*ya < core_r*core_r) { d_bnd = t_bot; next_reg = 0; }
                }
            }
        }

        // -------------------------------------------------------------------
        // Decide: collision or boundary crossing?
        // -------------------------------------------------------------------
        if (s_coll < d_bnd) {
            // ---------------------------------------------------------------
            // COLLISION at (px + pux*s_coll, ...)
            // ---------------------------------------------------------------
            px += pux * s_coll;
            py += puy * s_coll;
            pz += puz * s_coll;

            // --- Compute total scattering cross-section for this group ---
            float sig_s_total = 0.0f;
            int s_base = reg * 64 + pg * 8;
            for (int g2 = 0; g2 < 8; g2++) {
                sig_s_total += sigma_s[s_base + g2];
            }

            float xi_rxn = philox_next(rng);
            if (xi_rxn * sig_t < sig_s_total) {
                // -----------------------------------------------------------
                // SCATTER: sample outgoing energy group from scatter matrix
                // -----------------------------------------------------------
                float xi_g = philox_next(rng) * sig_s_total;
                float cum  = 0.0f;
                int new_g  = pg;  // fallback to current group
                for (int g2 = 0; g2 < 8; g2++) {
                    cum += sigma_s[s_base + g2];
                    if (xi_g <= cum) { new_g = g2; break; }
                }
                pg = new_g;

                // Sample isotropic direction (transport in fast reactor is ~isotropic)
                float cos_th = 2.0f * philox_next(rng) - 1.0f;
                float sin_th = sqrt(max(0.0f, 1.0f - cos_th*cos_th));
                float phi    = 6.2831853f * philox_next(rng);
                pux = sin_th * cos(phi);
                puy = sin_th * sin(phi);
                puz = cos_th;

            } else {
                // -----------------------------------------------------------
                // ABSORPTION: check for fission vs radiative capture
                // -----------------------------------------------------------
                float sf  = sigma_f[reg * 8 + pg];
                float sa  = sigma_a[reg * 8 + pg];

                if (is_fiss[reg] > 0.5f && sf > 0.0f && sa > 0.0f) {
                    float xi_fiss = philox_next(rng);
                    if (xi_fiss < sf / sa) {
                        // ---------------------------------------------------
                        // FISSION: sample number of secondaries from nu
                        // ---------------------------------------------------
                        float nu_val = nu_sf[reg * 8 + pg] / sf;  // nu = nu*sf / sf
                        int n_nu = (int)nu_val;
                        if (philox_next(rng) < (nu_val - (float)n_nu)) n_nu++;

                        // Store each secondary in fission bank (atomically)
                        int chi_base = reg * 8;
                        for (int fn = 0; fn < n_nu; fn++) {
                            // Sample fission-born energy group from chi
                            float xi_chi = philox_next(rng);
                            float cum_c  = 0.0f;
                            int   fg     = 0;
                            for (int g2 = 0; g2 < 8; g2++) {
                                cum_c += chi[chi_base + g2];
                                if (xi_chi <= cum_c) { fg = g2; break; }
                            }

                            // Atomic increment to claim a fission bank slot
                            uint idx = atomic_fetch_add_explicit(fiss_cnt, 1u,
                                                                 memory_order_relaxed);
                            if ((int)idx < max_fiss) {
                                fiss_x[idx] = px;
                                fiss_y[idx] = py;
                                fiss_z[idx] = pz;
                                fiss_g[idx] = fg;
                            }
                        }
                    }
                }
                break;  // Absorbed: particle history ends
            }

        } else {
            // ---------------------------------------------------------------
            // BOUNDARY CROSSING: advance to surface + epsilon nudge
            // ---------------------------------------------------------------
            px += pux * (d_bnd + EPS);
            py += puy * (d_bnd + EPS);
            pz += puz * (d_bnd + EPS);

            if (next_reg < 0) break;  // Crossed into vacuum: particle escapes
            // Otherwise continue in next_reg (will be detected next iteration)
        }

        // -------------------------------------------------------------------
        // Russian roulette: kill low-weight particles stochastically
        // -------------------------------------------------------------------
        if (pw < WCUTOFF) {
            if (philox_next(rng) > 0.5f) {
                pw = 2.0f * WCUTOFF;  // survived: restore weight
            } else {
                break;  // killed
            }
        }

    }  // end history loop
}
'''


class MetalBackend(MCBackend):
    """Apple Metal GPU backend for Monte Carlo neutron transport.

    Uses the metalcompute package to compile and run an MSL compute shader
    that transports an entire particle batch in parallel on Apple Silicon.

    One GPU thread per source particle. The shader implements the full
    transport loop including: region tracking, free-flight sampling,
    scattering, absorption, fission, and Russian roulette.

    Limitations vs CPU backend:
    - Tallies (flux, fission rate) are NOT scored on GPU; tally argument
      is accepted but ignored. Use CPU backend for tally accumulation.
    - float32 precision (vs float64 on CPU); adequate for ~1m scale geometry.
    - Maximum 2 material regions (core=0, reflector=1) hardcoded in shader.
    """

    def __init__(self):
        self._available = False
        self._device_name = "Unknown"
        self._device = None

        if not HAS_METAL:
            return
        if platform.system() != 'Darwin':
            return

        try:
            devices = mc.get_devices()
            if not devices:
                return
            self._device = mc.Device()
            self._device_name = devices[0].deviceName
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
            return f"Metal ({self._device_name})"
        return "Metal (not available - requires macOS + metalcompute)"

    def transport_batch(
        self,
        source_bank: ParticleBank,
        geometry: MCFRGeometry,
        materials: Dict[int, Material],
        tallies: Optional[TallyAccumulator],
        rng: np.random.Generator,
    ) -> FissionBank:
        """Transport all particles on the Metal GPU.

        Args:
            source_bank: Source particles for this generation.
            geometry:    MCFR cylindrical geometry (core + reflector).
            materials:   Dict mapping region_id -> Material (keys 0 and 1).
            tallies:     Tally accumulator (accepted but NOT scored on GPU).
            rng:         NumPy Generator used only to draw the per-batch seed.

        Returns:
            FissionBank with fission sites produced during this batch.
        """
        if not self._available:
            raise RuntimeError(
                "Metal backend not available. "
                "Install metalcompute on macOS with Apple Silicon."
            )

        n = source_bank.n_particles
        if n == 0:
            return FissionBank(max_sites=1)

        # -------------------------------------------------------------------
        # Cap fission bank at 4x source size (generous; typical nu~2.5)
        # -------------------------------------------------------------------
        max_fiss = max(4 * n, 1000)

        # -------------------------------------------------------------------
        # Pack material data into flat float32 arrays
        # Layout: [mat_id * 8 + group_idx]  (2 mats, 8 groups each)
        # -------------------------------------------------------------------
        n_mats = 2
        sigma_t_flat  = np.zeros(n_mats * 8,  dtype=np.float32)
        sigma_s_flat  = np.zeros(n_mats * 64, dtype=np.float32)  # 8x8 per mat
        sigma_a_flat  = np.zeros(n_mats * 8,  dtype=np.float32)
        sigma_f_flat  = np.zeros(n_mats * 8,  dtype=np.float32)
        nu_sf_flat    = np.zeros(n_mats * 8,  dtype=np.float32)
        chi_flat      = np.zeros(n_mats * 8,  dtype=np.float32)
        is_fiss_flat  = np.zeros(n_mats,      dtype=np.float32)

        for mat_id, mat in materials.items():
            if mat_id < 0 or mat_id >= n_mats:
                continue
            off8  = mat_id * 8
            off64 = mat_id * 64
            sigma_t_flat[off8:off8 + 8]   = mat.sigma_t.astype(np.float32)
            sigma_a_flat[off8:off8 + 8]   = mat.sigma_a.astype(np.float32)
            sigma_f_flat[off8:off8 + 8]   = mat.sigma_f.astype(np.float32)
            nu_sf_flat[off8:off8 + 8]     = mat.nu_sigma_f.astype(np.float32)
            chi_flat[off8:off8 + 8]       = mat.chi.astype(np.float32)
            is_fiss_flat[mat_id]          = 1.0 if mat.is_fissile else 0.0
            # sigma_s: row = from_group, col = to_group
            sigma_s_flat[off64:off64 + 64] = mat.sigma_s.flatten().astype(np.float32)

        # -------------------------------------------------------------------
        # Particle SoA: downcast to float32
        # -------------------------------------------------------------------
        px_arr  = source_bank.x.astype(np.float32)
        py_arr  = source_bank.y.astype(np.float32)
        pz_arr  = source_bank.z.astype(np.float32)
        pux_arr = source_bank.ux.astype(np.float32)
        puy_arr = source_bank.uy.astype(np.float32)
        puz_arr = source_bank.uz.astype(np.float32)
        pg_arr  = source_bank.group.astype(np.int32)
        pw_arr  = source_bank.weight.astype(np.float32)

        # -------------------------------------------------------------------
        # Geometry + run parameters
        # -------------------------------------------------------------------
        seed_uint = int(rng.integers(0, 2**32))
        params = np.array([
            geometry.core_radius,
            geometry.core_half_height,
            geometry.outer_radius,
            geometry.outer_half_height,
            float(n),
            float(max_fiss),
            # Transmit seed as float bits preserved via reinterpret
            # The shader does as_type<uint>(params[6]) to recover the bits.
            np.float32(seed_uint).view(np.float32),
        ], dtype=np.float32)
        # Overwrite element 6 with the bit-exact float representation
        params[6] = np.array(seed_uint, dtype=np.uint32).view(np.float32)

        # -------------------------------------------------------------------
        # Allocate GPU output buffers (device.buffer(n_bytes))
        # metalcompute write-back: GPU writes to these, Python reads via
        # np.frombuffer() after dispatch returns.
        # -------------------------------------------------------------------
        fiss_x_buf   = self._device.buffer(max_fiss * 4)   # float32
        fiss_y_buf   = self._device.buffer(max_fiss * 4)
        fiss_z_buf   = self._device.buffer(max_fiss * 4)
        fiss_g_buf   = self._device.buffer(max_fiss * 4)   # int32
        fiss_cnt_buf = self._device.buffer(4)              # uint32 atomic counter

        # -------------------------------------------------------------------
        # Compile shader and dispatch
        # metalcompute API:
        #   fn = device.kernel(msl_source).function(name)
        #   fn(n_threads, buf0, buf1, ...)
        #
        # Input buffers: pass numpy arrays directly (zero-copy on unified memory)
        # Output buffers: pass device.buffer() objects; read back via np.frombuffer
        # n_threads: dispatch exactly this many threads (GPU may pad to threadgroup
        #            boundary internally, hence the tid < n_particles guard in shader)
        # -------------------------------------------------------------------
        try:
            kernel = self._device.kernel(METAL_SHADER)
            fn = kernel.function("transport_kernel")
        except Exception as e:
            raise RuntimeError(f"Metal shader compilation failed: {e}") from e

        # Round up dispatch count to nearest 256 (standard threadgroup size)
        # The shader guards with: if (tid >= n_particles) return;
        n_dispatch = ((n + 255) // 256) * 256

        fn(
            n_dispatch,
            # Particle SoA (buffers 0-7)
            px_arr, py_arr, pz_arr,
            pux_arr, puy_arr, puz_arr,
            pg_arr, pw_arr,
            # Fission bank output (buffers 8-12)
            fiss_x_buf, fiss_y_buf, fiss_z_buf, fiss_g_buf, fiss_cnt_buf,
            # Material data (buffers 13-19)
            sigma_t_flat, sigma_s_flat, sigma_a_flat, sigma_f_flat,
            nu_sf_flat, chi_flat, is_fiss_flat,
            # Geometry + params (buffer 20)
            params,
        )

        # -------------------------------------------------------------------
        # Read back results
        # np.frombuffer() gives a zero-copy view of the Metal buffer contents
        # -------------------------------------------------------------------
        count_raw = np.frombuffer(fiss_cnt_buf, dtype=np.uint32)[0]
        count = min(int(count_raw), max_fiss)

        fission_bank = FissionBank(max_sites=max(count + 1, 1))
        if count > 0:
            fx = np.frombuffer(fiss_x_buf, dtype=np.float32)[:count]
            fy = np.frombuffer(fiss_y_buf, dtype=np.float32)[:count]
            fz = np.frombuffer(fiss_z_buf, dtype=np.float32)[:count]
            fg = np.frombuffer(fiss_g_buf, dtype=np.int32)[:count]

            # Upcast to float64 for consistency with rest of code
            fission_bank.x[:count]     = fx.astype(np.float64)
            fission_bank.y[:count]     = fy.astype(np.float64)
            fission_bank.z[:count]     = fz.astype(np.float64)
            fission_bank.group[:count] = fg.copy()
            fission_bank.count         = count

        return fission_bank
