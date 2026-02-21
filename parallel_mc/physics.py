"""
Collision physics for 8-group MC neutron transport.

Implements:
- Free flight distance sampling: s = -ln(xi) / sigma_t[g]
- Collision type determination: scatter vs absorption
- Scattering: sample outgoing group from scatter_matrix[g_in, :]
- Absorption: capture vs fission
- Fission: sample number of secondaries from nu[g]
- Russian roulette variance reduction
"""
import numpy as np
from .constants import N_GROUPS, MAX_COLLISIONS, EPSILON, WEIGHT_CUTOFF, SURVIVAL_WEIGHT
from .materials import Material
from .particle import FissionBank, sample_isotropic, sample_from_cdf


def transport_particle(x, y, z, ux, uy, uz, group, weight,
                       geometry, materials, fission_bank,
                       tally_accumulator, rng):
    """Transport a single particle until death or leakage.

    This is the core transport loop, called once per source particle.

    Args:
        x, y, z: position (m)
        ux, uy, uz: direction (unit vector)
        group: energy group index [0..7]
        weight: statistical weight
        geometry: MCFRGeometry
        materials: dict {region_id: Material}
        fission_bank: FissionBank to accumulate fission sites
        tally_accumulator: TallyAccumulator for scoring
        rng: numpy random Generator

    Returns:
        None (modifies fission_bank and tally_accumulator in-place)
    """
    alive = True
    n_collisions = 0

    while alive and n_collisions < MAX_COLLISIONS:
        # 1. Determine current region
        reg = geometry.region(x, y, z)

        if reg < 0:  # VACUUM
            alive = False
            break

        mat = materials[reg]
        sigma_t_g = mat.sigma_t[group]

        if sigma_t_g < 1e-20:
            # Transparent medium - just move to boundary
            d_boundary, next_reg = geometry.distance_to_boundary(x, y, z, ux, uy, uz, reg)
            # Score track-length tally
            if tally_accumulator is not None:
                tally_accumulator.score_track(x, y, z, ux, uy, uz, d_boundary, weight, group, mat)
            x += ux * (d_boundary + EPSILON)
            y += uy * (d_boundary + EPSILON)
            z += uz * (d_boundary + EPSILON)
            continue

        # 2. Sample free flight distance
        s_collision = -np.log(rng.random()) / sigma_t_g

        # 3. Distance to boundary
        d_boundary, next_reg = geometry.distance_to_boundary(x, y, z, ux, uy, uz, reg)

        # 4. Determine if collision or boundary crossing
        if s_collision < d_boundary:
            # --- COLLISION ---
            # Score track-length tally for path to collision
            if tally_accumulator is not None:
                tally_accumulator.score_track(x, y, z, ux, uy, uz, s_collision, weight, group, mat)

            # Move to collision site
            x += ux * s_collision
            y += uy * s_collision
            z += uz * s_collision

            n_collisions += 1

            # 5. Determine reaction type
            xi = rng.random()
            sigma_s_total = np.sum(mat.sigma_s[group, :])

            if xi < sigma_s_total / sigma_t_g:
                # --- SCATTERING ---
                # Sample outgoing energy group from scattering matrix
                scatter_probs = mat.sigma_s[group, :]
                scatter_sum = np.sum(scatter_probs)
                if scatter_sum > 0:
                    group = int(sample_from_cdf(scatter_probs / scatter_sum, rng, 1)[0])

                # Sample new isotropic direction (P0 scattering)
                cos_theta = 2.0 * rng.random() - 1.0
                sin_theta = np.sqrt(max(0, 1.0 - cos_theta**2))
                phi = 2.0 * np.pi * rng.random()
                ux = sin_theta * np.cos(phi)
                uy = sin_theta * np.sin(phi)
                uz = cos_theta

            else:
                # --- ABSORPTION ---
                if mat.is_fissile and mat.sigma_f[group] > 0:
                    xi2 = rng.random()
                    if xi2 < mat.sigma_f[group] / mat.sigma_a[group]:
                        # --- FISSION ---
                        nu = mat.nu_sigma_f[group] / mat.sigma_f[group]  # nu for this group
                        n_new = int(nu)
                        if rng.random() < (nu - n_new):
                            n_new += 1

                        for _ in range(n_new):
                            # Sample fission neutron group from chi
                            fg = int(sample_from_cdf(mat.chi, rng, 1)[0])
                            fission_bank.add(x, y, z, fg)

                # Particle absorbed (capture or fission - particle dies)
                alive = False

        else:
            # --- BOUNDARY CROSSING ---
            # Score track-length tally for path to boundary
            if tally_accumulator is not None:
                tally_accumulator.score_track(x, y, z, ux, uy, uz, d_boundary, weight, group, mat)

            # Move to boundary + epsilon nudge
            x += ux * (d_boundary + EPSILON)
            y += uy * (d_boundary + EPSILON)
            z += uz * (d_boundary + EPSILON)

            if next_reg < 0:  # Leaking to vacuum
                if tally_accumulator is not None:
                    tally_accumulator.score_leakage(weight)
                alive = False

        # 6. Russian roulette for low-weight particles
        if alive and weight < WEIGHT_CUTOFF:
            if rng.random() < 0.5:
                weight = SURVIVAL_WEIGHT
            else:
                alive = False


def transport_batch_sequential(source_bank, geometry, materials, tally_accumulator, rng):
    """Transport all particles in a batch sequentially.

    This is the reference (non-parallel) implementation.

    Args:
        source_bank: ParticleBank with source particles
        geometry: MCFRGeometry
        materials: dict {region_id: Material}
        tally_accumulator: TallyAccumulator
        rng: numpy random Generator

    Returns:
        FissionBank containing new fission sites
    """
    n = source_bank.n_particles
    fission_bank = FissionBank(max_sites=4 * n)

    for i in range(n):
        if not source_bank.alive[i]:
            continue

        transport_particle(
            source_bank.x[i], source_bank.y[i], source_bank.z[i],
            source_bank.ux[i], source_bank.uy[i], source_bank.uz[i],
            int(source_bank.group[i]), source_bank.weight[i],
            geometry, materials, fission_bank,
            tally_accumulator, rng,
        )

    return fission_bank
