"""
Physical constants and 8-group energy structure for fast MCFR.
All units SI (meters, seconds, kg, eV for energy only).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AVOGADRO = 6.02214076e23          # 1/mol
BARN = 1.0e-28                     # m^2
BOLTZMANN = 1.380649e-23          # J/K
NEUTRON_MASS = 1.008665           # amu
SPEED_OF_LIGHT = 2.998e8          # m/s
EV_TO_JOULE = 1.602176634e-19    # J/eV

# ---------------------------------------------------------------------------
# 8-group energy boundaries (MeV) - fast spectrum structure
# ---------------------------------------------------------------------------
N_GROUPS = 8
GROUP_BOUNDARIES_MEV = np.array([
    20.0, 6.065, 2.231, 0.8208, 0.3020, 0.1111, 0.0409, 0.0150, 1e-5
])
# Group centers (geometric mean of boundaries) in MeV
GROUP_CENTERS_MEV = np.sqrt(GROUP_BOUNDARIES_MEV[:-1] * GROUP_BOUNDARIES_MEV[1:])
# Lethargy widths
GROUP_LETHARGY_WIDTHS = np.log(GROUP_BOUNDARIES_MEV[:-1] / GROUP_BOUNDARIES_MEV[1:])

# ---------------------------------------------------------------------------
# Transport parameters
# ---------------------------------------------------------------------------
MAX_COLLISIONS = 500              # max collisions per particle history
EPSILON = 1.0e-8                  # m, boundary nudge distance
WEIGHT_CUTOFF = 1.0e-4            # Russian roulette threshold
SURVIVAL_WEIGHT = 2.0 * WEIGHT_CUTOFF  # weight after survival

# ---------------------------------------------------------------------------
# Shannon entropy mesh
# ---------------------------------------------------------------------------
ENTROPY_NR = 5
ENTROPY_NZ = 5
