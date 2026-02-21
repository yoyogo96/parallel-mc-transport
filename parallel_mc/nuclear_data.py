"""
8-group microscopic cross-section library for fast MCFR Monte Carlo.

Nuclide data derived from ENDF/B-VIII.0, JENDL-4.0, and fast reactor physics
literature.  Cross-sections stored in m^2 (sigma * BARN).  Macroscopic XS are
computed by materials.py.

Energy group structure (MeV):
  G1: 6.065 - 20.0    G5: 0.1111 - 0.3020
  G2: 2.231 - 6.065   G6: 0.0409 - 0.1111
  G3: 0.8208 - 2.231   G7: 0.0150 - 0.0409
  G4: 0.3020 - 0.8208  G8: 1e-5  - 0.0150
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import N_GROUPS, BARN, GROUP_BOUNDARIES_MEV


# ===================================================================
# NuclideData dataclass
# ===================================================================

@dataclass(frozen=True)
class NuclideData:
    """Microscopic nuclear data for a single nuclide in 8-group structure."""

    name: str
    atomic_mass: float            # amu
    sigma_elastic: np.ndarray     # [N_GROUPS] in m^2
    sigma_inelastic: np.ndarray   # [N_GROUPS] in m^2
    sigma_capture: np.ndarray     # [N_GROUPS] in m^2
    sigma_fission: np.ndarray     # [N_GROUPS] in m^2
    nu: np.ndarray                # [N_GROUPS] neutrons per fission
    chi: np.ndarray               # [N_GROUPS] fission spectrum
    scatter_matrix: np.ndarray    # [N_GROUPS, N_GROUPS] in m^2

    # -- derived properties --------------------------------------------------

    @property
    def sigma_absorption(self) -> np.ndarray:
        """Microscopic absorption cross-section (capture + fission)."""
        return self.sigma_capture + self.sigma_fission

    @property
    def sigma_total(self) -> np.ndarray:
        """Microscopic total cross-section (absorption + total out-scatter)."""
        return self.sigma_absorption + np.sum(self.scatter_matrix, axis=1)

    @property
    def nu_sigma_fission(self) -> np.ndarray:
        """nu * sigma_f  (neutron production cross-section)."""
        return self.nu * self.sigma_fission


# ===================================================================
# Scatter-matrix builder
# ===================================================================

def _alpha_from_mass(A: float) -> float:
    """Maximum fractional energy retained after elastic collision.

    alpha = ((A-1)/(A+1))^2  where A is atomic mass number.
    A neutron scattering off mass-A nucleus cannot lose more energy than
    factor alpha in a single collision.
    """
    return ((A - 1.0) / (A + 1.0)) ** 2


def build_scatter_matrix(
    sigma_el: np.ndarray,
    sigma_inel: np.ndarray,
    atomic_mass: float,
) -> np.ndarray:
    """Build [N_GROUPS x N_GROUPS] scattering matrix from elastic + inelastic.

    The (g -> g') element gives the microscopic scattering cross-section for
    a neutron in group *g* scattering into group *g'*.

    Physics model
    -------------
    **Elastic scattering** from group g:
      - The minimum post-collision energy is E_min = alpha * E, where
        alpha = ((A-1)/(A+1))^2.  Heavier nuclei retain more energy
        (alpha -> 1) so scatter predominantly within the same group.
      - We determine which groups the neutron can land in by comparing
        alpha * E_upper(g) with the group boundaries.  Energy transfer is
        assumed uniform in lethargy within the reachable range.
      - The fraction deposited into each reachable group g' is proportional
        to the lethargy overlap between the reachable band and group g'.

    **Inelastic scattering** from group g:
      - Simplified: the neutron drops exactly one group (g -> g+1).
        For the lowest group (g = N_GROUPS-1), inelastic stays in-group.

    Parameters
    ----------
    sigma_el : (N_GROUPS,) array in m^2 -- elastic XS per group.
    sigma_inel : (N_GROUPS,) array in m^2 -- inelastic XS per group.
    atomic_mass : float -- atomic mass in amu.

    Returns
    -------
    scatter_matrix : (N_GROUPS, N_GROUPS) array in m^2.
    """
    smat = np.zeros((N_GROUPS, N_GROUPS), dtype=np.float64)
    alpha = _alpha_from_mass(atomic_mass)

    # Group boundaries in MeV (descending): E[0] > E[1] > ... > E[N_GROUPS]
    E = GROUP_BOUNDARIES_MEV  # length N_GROUPS + 1

    for g in range(N_GROUPS):
        # --- elastic component -------------------------------------------
        E_upper_g = E[g]       # upper boundary of group g
        E_lower_g = E[g + 1]   # lower boundary of group g

        # Minimum energy reachable by elastic scatter from this group:
        # A neutron at E_upper_g can scatter down to alpha * E_upper_g.
        # A neutron at E_lower_g can scatter down to alpha * E_lower_g.
        # The widest reachable band goes from alpha * E_upper_g (for a
        # neutron at the top of the group) down.  But the full source
        # distribution spans [E_lower_g, E_upper_g].
        # We approximate by using the geometric-mean energy of the source
        # group to define the reachable band.
        E_src = np.sqrt(E_upper_g * E_lower_g)
        E_min_reachable = alpha * E_src  # lowest energy after one elastic hit

        # The reachable post-scatter range is [E_min_reachable, E_src]
        # (maximum energy retained = E_src, i.e. glancing blow).
        # In lethargy space the scattering kernel is approximately uniform
        # between u_src and u_src + ln(1/alpha).

        if E_min_reachable >= E_src:
            # alpha ~ 1 (very heavy nucleus), all scatter stays in group
            smat[g, g] += sigma_el[g]
        else:
            lethargy_range = np.log(E_src / E_min_reachable)  # = ln(1/alpha)
            if lethargy_range < 1.0e-12:
                smat[g, g] += sigma_el[g]
            else:
                # Distribute elastic XS into reachable groups g' >= g
                total_frac = 0.0
                for gp in range(g, N_GROUPS):
                    E_upper_gp = E[gp]
                    E_lower_gp = E[gp + 1]

                    # Overlap of reachable band [E_min_reachable, E_src]
                    # with group g' [E_lower_gp, E_upper_gp]
                    ov_hi = min(E_src, E_upper_gp)
                    ov_lo = max(E_min_reachable, E_lower_gp)

                    if ov_lo >= ov_hi:
                        continue  # no overlap

                    # Fraction in lethargy space
                    frac = np.log(ov_hi / ov_lo) / lethargy_range
                    smat[g, gp] += sigma_el[g] * frac
                    total_frac += frac

                # Assign any residual (numerical) to in-group
                if total_frac < 1.0 - 1.0e-8:
                    smat[g, g] += sigma_el[g] * (1.0 - total_frac)

        # --- inelastic component -----------------------------------------
        # Simplified model: drop one group.  Lowest group stays in-group.
        if g < N_GROUPS - 1:
            smat[g, g + 1] += sigma_inel[g]
        else:
            smat[g, g] += sigma_inel[g]

    return smat


# ===================================================================
# Helper to construct a NuclideData from concise input arrays
# ===================================================================

def _make_nuclide(
    name: str,
    atomic_mass: float,
    sigma_el_b: list,
    sigma_inel_b: list,
    sigma_c_b: list,
    sigma_f_b: list,
    nu: list,
    chi: list,
) -> NuclideData:
    """Create NuclideData from cross-sections given in barns.

    All XS lists are length N_GROUPS.  They are converted to m^2 internally.
    The scatter matrix is built automatically from elastic + inelastic data
    and the atomic mass.
    """
    se = np.asarray(sigma_el_b, dtype=np.float64) * BARN
    si = np.asarray(sigma_inel_b, dtype=np.float64) * BARN
    sc = np.asarray(sigma_c_b, dtype=np.float64) * BARN
    sf = np.asarray(sigma_f_b, dtype=np.float64) * BARN
    nu_arr = np.asarray(nu, dtype=np.float64)
    chi_arr = np.asarray(chi, dtype=np.float64)

    smat = build_scatter_matrix(se, si, atomic_mass)

    return NuclideData(
        name=name,
        atomic_mass=atomic_mass,
        sigma_elastic=se,
        sigma_inelastic=si,
        sigma_capture=sc,
        sigma_fission=sf,
        nu=nu_arr,
        chi=chi_arr,
        scatter_matrix=smat,
    )


# ===================================================================
# Nuclide library -- ENDF/B-VIII.0 derived 8-group data
# ===================================================================

# Watt fission spectrum (shared by U-235 and U-238)
_WATT_CHI = [0.011, 0.167, 0.395, 0.286, 0.103, 0.033, 0.004, 0.001]
_ZERO_CHI = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_ZERO_NU = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_ZERO_SF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# --- U-235 (fissile) ----------------------------------------------------
_U235 = _make_nuclide(
    name="U235",
    atomic_mass=235.044,
    sigma_el_b=  [3.5, 4.2, 5.0, 6.5, 8.0, 9.5, 10.0, 11.0],
    sigma_inel_b=[1.9, 1.5, 0.8, 0.3, 0.1, 0.02, 0.0, 0.0],
    sigma_c_b=   [0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.80],
    sigma_f_b=   [1.95, 1.25, 1.30, 1.50, 1.75, 2.30, 3.20, 5.50],
    nu=          [2.70, 2.60, 2.52, 2.47, 2.44, 2.43, 2.43, 2.43],
    chi=_WATT_CHI,
)

# --- U-238 (fertile, fast fission above ~1.2 MeV) -----------------------
_U238 = _make_nuclide(
    name="U238",
    atomic_mass=238.051,
    sigma_el_b=  [3.8, 4.5, 5.5, 7.0, 8.5, 9.0, 10.0, 11.0],
    sigma_inel_b=[2.0, 1.8, 1.2, 0.5, 0.15, 0.02, 0.0, 0.0],
    sigma_c_b=   [0.02, 0.04, 0.06, 0.10, 0.30, 0.80, 2.50, 5.00],
    sigma_f_b=   [1.14, 0.55, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
    nu=          [2.80, 2.75, 2.70, 0.0, 0.0, 0.0, 0.0, 0.0],
    chi=_WATT_CHI,
)

# --- Na-23 (carrier cation) ---------------------------------------------
_Na23 = _make_nuclide(
    name="Na23",
    atomic_mass=22.990,
    sigma_el_b=  [2.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0],
    sigma_inel_b=[0.8, 0.5, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.02],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- K-39 (carrier cation) ----------------------------------------------
_K39 = _make_nuclide(
    name="K39",
    atomic_mass=38.964,
    sigma_el_b=  [2.0, 2.5, 3.0, 3.5, 3.5, 3.5, 3.5, 3.5],
    sigma_inel_b=[0.6, 0.4, 0.15, 0.03, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.04, 0.08],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- Cl-35 (carrier anion, 75.77% natural abundance) --------------------
_Cl35 = _make_nuclide(
    name="Cl35",
    atomic_mass=34.969,
    sigma_el_b=  [1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0],
    sigma_inel_b=[0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.001, 0.002, 0.005, 0.020, 0.040, 0.060, 0.100, 0.200],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- Cl-37 (carrier anion, 24.23% natural abundance) --------------------
_Cl37 = _make_nuclide(
    name="Cl37",
    atomic_mass=36.966,
    sigma_el_b=  [1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0],
    sigma_inel_b=[0.5, 0.2, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- Fe-56 (SS316H reflector, dominant) ----------------------------------
_Fe56 = _make_nuclide(
    name="Fe56",
    atomic_mass=55.845,
    sigma_el_b=  [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0, 10.0],
    sigma_inel_b=[1.2, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.025, 0.050],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- Cr-52 (SS316H reflector) -------------------------------------------
_Cr52 = _make_nuclide(
    name="Cr52",
    atomic_mass=51.996,
    sigma_el_b=  [2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 5.0, 7.0],
    sigma_inel_b=[0.8, 0.5, 0.3, 0.08, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.040],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- Ni-58 (SS316H reflector) -------------------------------------------
_Ni58 = _make_nuclide(
    name="Ni58",
    atomic_mass=58.693,
    sigma_el_b=  [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0],
    sigma_inel_b=[1.0, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.003, 0.005, 0.008, 0.012, 0.015, 0.020, 0.030, 0.060],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- Be-9 (BeO reflector) -----------------------------------------------
_Be9 = _make_nuclide(
    name="Be9",
    atomic_mass=9.012,
    sigma_el_b=  [1.5, 2.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
    sigma_inel_b=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)

# --- O-16 (BeO reflector) -----------------------------------------------
_O16 = _make_nuclide(
    name="O16",
    atomic_mass=15.999,
    sigma_el_b=  [1.5, 2.5, 3.5, 3.8, 3.8, 3.8, 3.8, 3.8],
    sigma_inel_b=[0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    sigma_c_b=   [0.0001, 0.0001, 0.0002, 0.0002, 0.0003, 0.0003, 0.0005, 0.001],
    sigma_f_b=_ZERO_SF,
    nu=_ZERO_NU,
    chi=_ZERO_CHI,
)


# ===================================================================
# Public nuclide library
# ===================================================================

NUCLIDE_LIBRARY: Dict[str, NuclideData] = {
    "U235": _U235,
    "U238": _U238,
    "Na23": _Na23,
    "K39":  _K39,
    "Cl35": _Cl35,
    "Cl37": _Cl37,
    "Fe56": _Fe56,
    "Cr52": _Cr52,
    "Ni58": _Ni58,
    "Be9":  _Be9,
    "O16":  _O16,
}
