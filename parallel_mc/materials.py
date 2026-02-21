"""
Material definitions with macroscopic cross-section computation.

Each material combines nuclide-level microscopic data (from nuclear_data.py)
with number densities to produce macroscopic cross-sections (Sigma = N * sigma)
in units of 1/m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .constants import N_GROUPS, AVOGADRO, BARN
from .nuclear_data import NUCLIDE_LIBRARY, NuclideData


# ===================================================================
# Material dataclass
# ===================================================================

@dataclass
class Material:
    """Macroscopic material data for Monte Carlo transport."""

    name: str
    mat_id: int                    # 0 = fuel core, 1 = reflector
    density: float                 # kg/m3
    temperature: float             # K
    sigma_t: np.ndarray           # [N_GROUPS] macroscopic total (1/m)
    sigma_s: np.ndarray           # [N_GROUPS, N_GROUPS] macro scattering (1/m)
    sigma_a: np.ndarray           # [N_GROUPS] macro absorption (1/m)
    sigma_f: np.ndarray           # [N_GROUPS] macro fission (1/m)
    nu_sigma_f: np.ndarray        # [N_GROUPS] production (1/m)
    chi: np.ndarray               # [N_GROUPS] fission spectrum
    is_fissile: bool


# ===================================================================
# Internal helper
# ===================================================================

def _accumulate_macroscopic(
    nuclides_and_densities: List[Tuple[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum N_i * sigma_i for each nuclide to get macroscopic XS arrays.

    Parameters
    ----------
    nuclides_and_densities : list of (nuclide_name, number_density_m3)

    Returns
    -------
    sigma_t, sigma_s, sigma_a, sigma_f, nu_sigma_f, chi
    """
    sigma_a = np.zeros(N_GROUPS, dtype=np.float64)
    sigma_f = np.zeros(N_GROUPS, dtype=np.float64)
    nu_sigma_f = np.zeros(N_GROUPS, dtype=np.float64)
    sigma_s = np.zeros((N_GROUPS, N_GROUPS), dtype=np.float64)
    chi_weighted = np.zeros(N_GROUPS, dtype=np.float64)

    for name, N_i in nuclides_and_densities:
        nuc = NUCLIDE_LIBRARY[name]
        sigma_a += N_i * nuc.sigma_absorption
        sigma_f += N_i * nuc.sigma_fission
        nu_sigma_f += N_i * nuc.nu_sigma_fission
        sigma_s += N_i * nuc.scatter_matrix
        # Weight chi by nu*sigma_f*N to get composite fission spectrum
        chi_weighted += N_i * nuc.nu_sigma_fission * nuc.chi

    sigma_s_total = np.sum(sigma_s, axis=1)  # total out-scatter per group
    sigma_t = sigma_a + sigma_s_total

    # Normalize chi
    chi_sum = np.sum(chi_weighted)
    if chi_sum > 0.0:
        chi = chi_weighted / chi_sum
    else:
        chi = np.zeros(N_GROUPS, dtype=np.float64)

    return sigma_t, sigma_s, sigma_a, sigma_f, nu_sigma_f, chi


# ===================================================================
# Fuel salt builder
# ===================================================================

def build_fuel_salt(temperature: float = 923.15, enrichment: float = 0.197) -> Material:
    """Build NaCl-KCl-UCl3 (42-20-38 mol%) fuel salt material.

    Number density calculation:
      MW_NaCl = 58.44 g/mol, MW_KCl = 74.55 g/mol, MW_UCl3 = 344.39 g/mol
      MW_avg = 0.42*58.44 + 0.20*74.55 + 0.38*344.39 = 170.11 g/mol
      rho ~ 3099 kg/m3 at 650 C  (from config.salt_density)
      N_mol = rho * N_A / MW_avg  (molecules per m3)
      Then per nuclide: N_i = mol_fraction * stoichiometry * N_mol

    Parameters
    ----------
    temperature : float
        Salt temperature in K (default 923.15 K = 650 C).
    enrichment : float
        U-235 weight fraction in uranium (default 0.197 = 19.7% HALEU).

    Returns
    -------
    Material with macroscopic cross-sections in 1/m.
    """
    # Salt density correlation for NaCl-KCl-UCl3 (42-20-38 mol%)
    # Linear fit: rho = 3450.0 - 0.54 * T_C (kg/m3)
    # Ref: Desyatnik et al., Atomnaya Energiya, 1975; Janz et al., J. Phys. Chem. Ref. Data, 1975
    def salt_density(T):
        T_C = T - 273.15
        return 3450.0 - 0.54 * T_C

    rho = salt_density(temperature)  # kg/m3

    # Molecular weights (kg/mol)
    MW_NaCl = 58.44e-3
    MW_KCl = 74.55e-3
    MW_UCl3 = 344.39e-3

    x_NaCl, x_KCl, x_UCl3 = 0.42, 0.20, 0.38
    MW_mix = x_NaCl * MW_NaCl + x_KCl * MW_KCl + x_UCl3 * MW_UCl3  # kg/mol

    N_molecules = rho * AVOGADRO / MW_mix  # molecules/m3

    # Number densities (atoms/m3)
    N_Na = x_NaCl * N_molecules            # 1 Na per NaCl
    N_K = x_KCl * N_molecules              # 1 K per KCl (approximated as K-39)
    N_U = x_UCl3 * N_molecules             # 1 U per UCl3
    N_U235 = N_U * enrichment
    N_U238 = N_U * (1.0 - enrichment)
    # Chlorine from all three compounds: NaCl(1) + KCl(1) + UCl3(3)
    N_Cl_total = (x_NaCl + x_KCl + 3.0 * x_UCl3) * N_molecules
    N_Cl35 = N_Cl_total * 0.7577           # natural Cl-35 abundance
    N_Cl37 = N_Cl_total * 0.2423           # natural Cl-37 abundance

    nuclides_and_densities = [
        ("U235", N_U235),
        ("U238", N_U238),
        ("Na23", N_Na),
        ("K39",  N_K),
        ("Cl35", N_Cl35),
        ("Cl37", N_Cl37),
    ]

    sigma_t, sigma_s, sigma_a, sigma_f, nu_sigma_f, chi = _accumulate_macroscopic(
        nuclides_and_densities
    )

    return Material(
        name="NaCl-KCl-UCl3 Fuel Salt",
        mat_id=0,
        density=rho,
        temperature=temperature,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        sigma_a=sigma_a,
        sigma_f=sigma_f,
        nu_sigma_f=nu_sigma_f,
        chi=chi,
        is_fissile=True,
    )


# ===================================================================
# BeO reflector builder
# ===================================================================

def build_reflector_beo(temperature: float = 923.15) -> Material:
    """Build BeO reflector material.

    BeO: density = 3010 kg/m3, MW = 25.01 g/mol
    N_BeO = 3010 * 6.022e23 / 0.02501 = 7.246e28 molecules/m3
    N_Be = N_O = N_BeO  (one of each per formula unit)

    Parameters
    ----------
    temperature : float
        Reflector temperature in K (default 923.15 K = 650 C).

    Returns
    -------
    Material with macroscopic cross-sections in 1/m.
    """
    rho = 3010.0       # kg/m3
    MW_BeO = 25.01e-3  # kg/mol

    N_BeO = rho * AVOGADRO / MW_BeO  # molecules/m3
    N_Be = N_BeO
    N_O = N_BeO

    nuclides_and_densities = [
        ("Be9", N_Be),
        ("O16", N_O),
    ]

    sigma_t, sigma_s, sigma_a, sigma_f, nu_sigma_f, chi = _accumulate_macroscopic(
        nuclides_and_densities
    )

    return Material(
        name="BeO Reflector",
        mat_id=1,
        density=rho,
        temperature=temperature,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        sigma_a=sigma_a,
        sigma_f=sigma_f,
        nu_sigma_f=nu_sigma_f,
        chi=chi,
        is_fissile=False,
    )


# ===================================================================
# SS316H reflector builder
# ===================================================================

def build_reflector_ss316h(temperature: float = 923.15) -> Material:
    """Build SS316H reflector (Fe-Cr-Ni alloy) material.

    SS316H composition (wt%): Fe ~ 66.8%, Cr ~ 17%, Ni ~ 12%
    (Mo ~ 2.5%, Mn ~ 1.7% -- minor constituents omitted).
    Density = 7900 kg/m3.

    Number densities from weight fractions:
      N_i = rho * wt_i * N_A / AM_i

    Parameters
    ----------
    temperature : float
        Reflector temperature in K (default 923.15 K = 650 C).

    Returns
    -------
    Material with macroscopic cross-sections in 1/m.
    """
    rho = 7900.0  # kg/m3

    # Weight fractions and atomic masses (kg/mol)
    wt_Fe, wt_Cr, wt_Ni = 0.668, 0.170, 0.120
    AM_Fe = 55.845e-3
    AM_Cr = 51.996e-3
    AM_Ni = 58.693e-3

    N_Fe = rho * wt_Fe * AVOGADRO / AM_Fe
    N_Cr = rho * wt_Cr * AVOGADRO / AM_Cr
    N_Ni = rho * wt_Ni * AVOGADRO / AM_Ni

    nuclides_and_densities = [
        ("Fe56", N_Fe),
        ("Cr52", N_Cr),
        ("Ni58", N_Ni),
    ]

    sigma_t, sigma_s, sigma_a, sigma_f, nu_sigma_f, chi = _accumulate_macroscopic(
        nuclides_and_densities
    )

    return Material(
        name="SS316H Reflector",
        mat_id=1,
        density=rho,
        temperature=temperature,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        sigma_a=sigma_a,
        sigma_f=sigma_f,
        nu_sigma_f=nu_sigma_f,
        chi=chi,
        is_fissile=False,
    )
