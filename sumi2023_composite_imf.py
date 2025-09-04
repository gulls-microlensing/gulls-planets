"""sumi2023_composite_imf

Reusable analytic composite IMF (planet + star/brown dwarf) based on Sumi (2023) style
parameterization. Provides per–star dN/dlog10 M (base-10) functions for:

    - Free–floating / wide orbit planetary population
      dN/dlog10 M = Z_planet * (M / Mnorm_planet)^(-alpha4)
      Valid over PLANET_MMIN_SOLAR .. PLANET_MMAX_SOLAR (in solar masses).

    - Low-mass stellar + brown dwarf rising segment
      dN/dlog10 M = A_bd * (M / STARBD_MNORM_SOLAR)^(-alpha3)
      Valid over STARBD_VALID_LO_SOLAR .. STARBD_VALID_HI_SOLAR (in solar masses).

The star+BD amplitude A_bd is solved from a target number of brown dwarfs per star in a
chosen interval [BD_LO_SOLAR, BD_HI_SOLAR]. This gives a fixed constant BD_AMPLITUDE so
users do not need to re-fit anything.

Units:
    Input masses for the public functions are in Earth masses (M_earth).
    Internal conversions use M_EARTH_TO_SOLAR.
    Returned values are per star per dex (dlog10 M units).

Main entry points:
    planetary_component_dndl_log10M(M_earth, ...)
    starbd_component_dndl_log10M(M_earth, ...)
    composite_imf(M_earth, planet_kwargs=None, starbd_kwargs=None)

Example:
    import numpy as np
    from sumi2023_composite_imf import composite_imf

    M = np.logspace(-1, 8, 500)  # Earth masses
    p, s, tot = composite_imf(M)

"""
from __future__ import annotations
import numpy as np

# --------------------------- Constants & Defaults ----------------------------
M_EARTH_TO_SOLAR = 3.00348959632e-6
LN10 = np.log(10.0)

# Planet (FFP) population parameters (published)
PLANET_ALPHA4 = 0.96
PLANET_Z = 2.18                 # dex^-1 star^-1
PLANET_MNORM_EARTH = 8.0        # pivot in Earth masses
PLANET_MMIN_SOLAR = 1e-6        # Msun ~0.33 M_earth
PLANET_MMAX_SOLAR = 0.02        # Msun ~6660 M_earth

# Star + BD low-mass IMF segment (single power law used for curvature)
STARBD_ALPHA3 = -0.82
STARBD_MNORM_SOLAR = 38.0 * M_EARTH_TO_SOLAR  # pivot 38 M_earth
STARBD_VALID_LO_SOLAR = 3e-4   # Msun (lower validity bound)
STARBD_VALID_HI_SOLAR = 0.8    # Msun (upper bound for this segment)

# Brown dwarf interval used to set amplitude
BD_LO_SOLAR = 0.012   # Msun (~13 M_Jup)
BD_HI_SOLAR = 0.08    # Msun
BD_TARGET_PER_STAR = 0.2

# --------------------------- Amplitude Solver -------------------------------

def solve_bd_amplitude(alpha3: float = STARBD_ALPHA3,
                        Mnorm_solar: float = STARBD_MNORM_SOLAR,
                        bd_lo: float = BD_LO_SOLAR,
                        bd_hi: float = BD_HI_SOLAR,
                        N_target: float = BD_TARGET_PER_STAR) -> float:
    """Solve for brown dwarf segment amplitude ``A_BD``.

    Computes the normalization such that integrating the analytic form
    ``dN/dlog10 M = A_BD * (M / Mnorm_solar)^(-alpha3)`` between ``bd_lo`` and
    ``bd_hi`` (solar masses) yields exactly ``N_target`` objects per star.

    Parameters
    ----------
    alpha3 : float, optional
        Power-law exponent (negative => rising with increasing mass).
    Mnorm_solar : float, optional
        Pivot (normalization) mass in solar masses.
    bd_lo, bd_hi : float, optional
        Integration bounds (solar masses). Must satisfy ``bd_lo < bd_hi``.
    N_target : float, optional
        Desired number of brown dwarfs per star in the interval.

    Returns
    -------
    float
        Amplitude ``A_BD`` in dex^-1 star^-1.
    """
    x1, x2 = np.log10(bd_lo), np.log10(bd_hi)
    numer = N_target * (-alpha3) * LN10
    denom = (Mnorm_solar ** alpha3) * (10 ** (-alpha3 * x2) - 10 ** (-alpha3 * x1))
    return numer / denom

BD_AMPLITUDE = solve_bd_amplitude()

# --------------------------- Component Functions -----------------------------

def planetary_component_dndl_log10M(M_earth: np.ndarray,
                                    Z: float = PLANET_Z,
                                    alpha4: float = PLANET_ALPHA4,
                                    Mnorm_earth: float = PLANET_MNORM_EARTH,
                                    mmin_solar: float = PLANET_MMIN_SOLAR,
                                    mmax_solar: float = PLANET_MMAX_SOLAR) -> np.ndarray:
    """Planetary (free-floating) component of the composite IMF.

    Implements a single power law in Earth-mass units:
    ``dN/dlog10 M = Z * (M / Mnorm_earth)^(-alpha4)`` within the validity
    interval expressed in solar masses (``mmin_solar`` .. ``mmax_solar``).

    Parameters
    ----------
    M_earth : array_like
        Masses at which to evaluate, in Earth masses.
    Z : float, optional
        Normalization (per star per dex).
    alpha4 : float, optional
        Power-law exponent.
    Mnorm_earth : float, optional
        Pivot mass (Earth masses).
    mmin_solar, mmax_solar : float, optional
        Validity bounds (solar masses); outside this range values are zero.

    Returns
    -------
    ndarray
        Per-star ``dN/dlog10 M`` values matching the shape of ``M_earth``.
    """
    M_earth = np.asarray(M_earth, dtype=float)
    M_solar = M_earth * M_EARTH_TO_SOLAR
    mask = (M_solar >= mmin_solar) & (M_solar <= mmax_solar)
    out = np.zeros_like(M_earth, dtype=float)
    if np.any(mask):
        out[mask] = Z * (M_earth[mask] / Mnorm_earth) ** (-alpha4)
    return out

def starbd_component_dndl_log10M(M_earth: np.ndarray,
                                 A_bd: float = BD_AMPLITUDE,
                                 alpha3: float = STARBD_ALPHA3,
                                 Mnorm_solar: float = STARBD_MNORM_SOLAR,
                                 valid_lo: float = STARBD_VALID_LO_SOLAR,
                                 valid_hi: float = STARBD_VALID_HI_SOLAR) -> np.ndarray:
    """Low-mass stellar + brown dwarf IMF segment.

    Parameters
    ----------
    M_earth : array_like
        Masses (Earth masses) where the function is evaluated.
    A_bd : float, optional
        Amplitude ``A_BD`` (per star per dex).
    alpha3 : float, optional
        Power-law exponent (negative implies rising with mass).
    Mnorm_solar : float, optional
        Pivot mass in solar masses.
    valid_lo, valid_hi : float, optional
        Validity interval in solar masses; values outside are zeroed.

    Returns
    -------
    ndarray
        Per-star ``dN/dlog10 M``; zeros where masses lie outside
        ``[valid_lo, valid_hi]`` after conversion to solar masses.
    """
    M_earth = np.asarray(M_earth, dtype=float)
    M_solar = M_earth * M_EARTH_TO_SOLAR
    mask = (M_solar >= valid_lo) & (M_solar <= valid_hi)
    out = np.zeros_like(M_earth, dtype=float)
    if np.any(mask):
        out[mask] = A_bd * (M_solar[mask] / Mnorm_solar) ** (-alpha3)
    return out

def composite_imf(M_earth: np.ndarray,
                  planet_kwargs: dict | None = None,
                  starbd_kwargs: dict | None = None):
    """Evaluate planetary, star/brown-dwarf, and total IMF components.

    Parameters
    ----------
    M_earth : array_like
        Masses (Earth) where the functions are evaluated.
    planet_kwargs : dict, optional
        Keyword overrides passed to :func:`planetary_component_dndl_log10M`.
    starbd_kwargs : dict, optional
        Keyword overrides passed to :func:`starbd_component_dndl_log10M`.

    Returns
    -------
    tuple of ndarray
        ``(planet_component, starbd_component, total)`` arrays.
    """
    planet_kwargs = planet_kwargs or {}
    starbd_kwargs = starbd_kwargs or {}
    p = planetary_component_dndl_log10M(M_earth, **planet_kwargs)
    s = starbd_component_dndl_log10M(M_earth, **starbd_kwargs)
    return p, s, p + s

# --------------------------- Utility Integrators ----------------------------

def integrate_number(M_earth: np.ndarray, dndl: np.ndarray) -> float:
    """Integrate counts over a grid of Earth masses.

    Uses trapezoidal integration in log10 mass space.

    Parameters
    ----------
    M_earth : array_like
        Mass grid (Earth masses).
    dndl : array_like
        Corresponding ``dN/dlog10 M`` values.

    Returns
    -------
    float
        Integrated number over the provided grid.
    """
    return float(np.trapz(dndl, np.log10(M_earth)))

def planet_canonical_count() -> float:
    """Compute reference integral of the planetary component.

    Integrates :func:`planetary_component_dndl_log10M` over 0.33–6660 Earth masses.

    Returns
    -------
    float
        Expected total (approx 21) planets per star in canonical interval.
    """
    M = np.logspace(np.log10(0.33), np.log10(6660.), 4000)
    d = planetary_component_dndl_log10M(M)
    return integrate_number(M, d)

def bd_count_in_interval(bd_lo: float = BD_LO_SOLAR, bd_hi: float = BD_HI_SOLAR) -> float:
    """Integrate brown dwarf counts over a solar-mass interval.

    Parameters
    ----------
    bd_lo, bd_hi : float, optional
        Lower and upper bounds in solar masses.

    Returns
    -------
    float
        Number of brown dwarfs per star in the interval.
    """
    M_solar = np.logspace(np.log10(bd_lo), np.log10(bd_hi), 3000)
    M_earth = M_solar / M_EARTH_TO_SOLAR
    d = starbd_component_dndl_log10M(M_earth)
    return integrate_number(M_earth, d)

# --------------------------- Public Exports ---------------------------------
__all__ = [
    'M_EARTH_TO_SOLAR', 'PLANET_ALPHA4', 'PLANET_Z', 'PLANET_MNORM_EARTH',
    'PLANET_MMIN_SOLAR', 'PLANET_MMAX_SOLAR', 'STARBD_ALPHA3', 'STARBD_MNORM_SOLAR',
    'STARBD_VALID_LO_SOLAR', 'STARBD_VALID_HI_SOLAR', 'BD_LO_SOLAR', 'BD_HI_SOLAR',
    'BD_TARGET_PER_STAR', 'BD_AMPLITUDE', 'solve_bd_amplitude',
    'planetary_component_dndl_log10M', 'starbd_component_dndl_log10M', 'composite_imf',
    'integrate_number', 'planet_canonical_count', 'bd_count_in_interval'
]

if __name__ == '__main__':
    # Quick self-checks when run as a script
    n_planets = planet_canonical_count()
    n_bd = bd_count_in_interval()
    print(f"Canonical planet count (0.33-6660 M_earth): {n_planets:.1f} (~21)")
    print(f"BD count in [{BD_LO_SOLAR},{BD_HI_SOLAR}] Msun: {n_bd:.3f} (target {BD_TARGET_PER_STAR})")
