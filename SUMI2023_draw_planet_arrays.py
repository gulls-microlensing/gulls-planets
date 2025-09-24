"""Generate planet property arrays using a composite (planet + star/BD) IMF.

This script samples planetary (and low-mass star / brown dwarf tail) masses
following a two–segment power-law in log10 mass space, together with
semi–major axis (log–uniform), isotropic inclination, and random orbital
phase. Output is written per field to text (or optionally ``.npy``) files.

Execution (CLI)
---------------
Run directly as a script::

        python SUMI2023_draw_planet_arrays.py

Environment / Configuration is governed by the module constants at the top of
the file (``nl``, ``nf``, ``rundes`` etc.). Set ``FIXED_BASE_SEED`` to an
integer for deterministic, reproducible draws across processes.

Notes
-----
* Masses are sampled in Earth masses and converted to solar masses only
    at output time. The composite distribution is entirely analytic; no
    rejection sampling is used.
* Two internal helper functions (``_segment_integral`` and
    ``_sample_segment``) operate in log10-space to avoid numerical issues.
* The component label (0=planet segment, 1=star/BD segment) is stored as the
    5th column in the output, enabling post–hoc filtering.
"""

import os
import math
import numpy as np
import multiprocessing as mp
import time

# Simulation mass truncation bounds (Earth masses). Sampling enforces these after
# applying each analytic segment's intrinsic validity range.
mmin = 0.1       # global minimum Earth mass to include in output
mmax = 25000     # global maximum Earth mass (~0.075 Msun)
amin = math.log10(0.3)
amax = math.log10(30)

rundes = 'test_Sumi2023'
sources_file = './gulls_surot2d_H2023.sources'  # this determines the number of fields
file_ext = ''
nl = 10000  # number of planets to generate per file
nf = 1  # number of files to generate per field
overwrite_existing = True  # set True to regenerate even if file already exists

# Text output formatting controls
delineator = ","
header = True
_HEADER_LINE = 'mass (M_Sun), a (au), inc (deg), p (deg), comp (0=planet,1=starbd)'

# Fixed seeding configuration: set to an integer for reproducible runs, or None (default) for nondeterministic.
FIXED_BASE_SEED = None

# ---------------- Piecewise mass function parameters ----------------
# Segment 1 (planets): dN/dlog10 M = Z * (M / PIVOT)^(-alpha4)
_ALPHA_PLANET = 0.96          # alpha4
_Z_PLANET = 2.18              # dex^-1 star^-1
_PIVOT_PLANET = 8.0           # Earth-mass pivot

# Segment 2 (star+BD rising tail): dN/dlog10 M = A_BD * (M / PIVOT_BD)^(-alpha3)
_ALPHA3 = -0.82               # negative => rises with mass
_A_BD = 2.22103e-3            # dex^-1 star^-1 (from BD target count solving)
_PIVOT_BD = 38.0              # Earth-mass pivot

_M_EARTH_TO_SOLAR = 3.00348959632e-6  # samples are drawn in earth masses and saved in solar masses

_STARBD_HI_SOLAR = 0.8        # Msun upper validity of segment
_STARBD_HI = _STARBD_HI_SOLAR / _M_EARTH_TO_SOLAR   # ~2.66e5 M_earth

# Pre-compute straight-line slopes/intercepts in log10 space:
# log10(dN/dlog10 M) = m * log10(M) + c
_m1 = -_ALPHA_PLANET
_c1 = math.log10(_Z_PLANET) + _ALPHA_PLANET * math.log10(_PIVOT_PLANET)
_m2 = -_ALPHA3
_c2 = math.log10(_A_BD) + _ALPHA3 * math.log10(_PIVOT_BD)

def get_field_numbers(sources_file):
    """Parse the survey sources file and extract field numbers.

    The file is expected to be ASCII text with at least one integer
    (field identifier) per non-empty line. Only the first whitespace-
    delimited token of each non-empty line is used.

    Parameters
    ----------
    sources_file : str or path-like
        Path to ``.sources`` file; first column contains integer field IDs.

    Returns
    -------
    list of int
        Ordered list of field numbers as they appear in the file.

    Raises
    ------
    FileNotFoundError
        If the provided file cannot be opened.
    ValueError
        If a token in first column cannot be converted to ``int``.
    """
    field_numbers = []
    with open(sources_file, 'r') as f:
        for line in f:
            if line.strip():
                field_numbers.append(int(line.split()[0]))
    return field_numbers

field_numbers = get_field_numbers(sources_file)

data_dir = './'
if data_dir[-1] != '/':
    data_dir += '/'
    

def _segment_integral(m, c, a, b):
    r"""Closed-form integral of a log10-space power law segment.

    Evaluates:

    .. math:: \int_a^b 10^{m x + c} dx

    where ``x = log10(M)``. This corresponds to integrating a pure power law
    in linear mass. The form is used to compute relative weights of segments.

    Parameters
    ----------
    m : float
        Slope in log10-space (``m = -alpha`` for a power law exponent ``alpha``).
    c : float
        Intercept so that ``log10(dN/dlog10M) = m * x + c``.
    a, b : float
        Lower / upper bounds in ``x = log10(M)`` (``b`` may exceed ``a``).

    Returns
    -------
    float
        Value of the definite integral. Returns 0 if ``b <= a``.
    """
    if b <= a:
        return 0.0
    if abs(m) < 1e-12:
        return 10**c * (b - a)
    k = m * np.log(10.0)
    K = 10**c
    return K * (np.exp(k * b) - np.exp(k * a)) / k

def _sample_segment(m, c, a, b, size, rng):
    """Sample log10-masses from a single analytic segment.

    Probability density is proportional to ``10^{m x + c}`` over ``[a, b]``.
    Uses inverse transform sampling with the analytical CDF.

    Parameters
    ----------
    m, c : float
        Slope and intercept (see :func:`_segment_integral`). ``c`` cancels for
        sampling and is unused except for interface symmetry.
    a, b : float
        Lower and upper bounds in log10 mass. If ``b <= a`` a degenerate array
        of ``a`` is returned.
    size : int
        Number of samples to draw.
    rng : numpy.random.Generator
        Numpy random generator (ensures reproducibility per worker).

    Returns
    -------
    ndarray of shape (size,)
        Sampled ``log10(M)`` values.
    """
    if b <= a:
        return np.full(size, a)
    u = rng.random(size)
    if abs(m) < 1e-12:
        return a + (b - a) * u
    k = m * np.log(10.0)
    ea = np.exp(k * a)
    eb = np.exp(k * b)
    return (1.0 / k) * np.log(ea + u * (eb - ea))

def sample_masses(N, *,
                  m_lo=mmin, m_hi=mmax,
                  m1=_m1, c1=_c1, m2=_m2, c2=_c2,
                  rng, return_component=False):
    """Sample composite IMF masses in Earth-mass units.

    A two-component mixture is constructed by integrating each analytic
    segment over the *same* interval ``[m_lo, m_hi]`` in linear mass and
    selecting a segment with probability proportional to its integral.
    Within a chosen segment, inverse-transform sampling in log10 mass is
    used via :func:`_sample_segment`.

    Parameters
    ----------
    N : int
        Number of masses to draw.
    m_lo, m_hi : float, optional
        Closed interval of allowed masses (Earth masses). Must satisfy
        ``0 < m_lo < m_hi <= _STARBD_HI``.
    m1, c1, m2, c2 : float, optional
        Slopes and intercepts of the two log10-linear segments.
    rng : numpy.random.Generator
        Random number generator to use.
    return_component : bool, optional
        If True, also return integer labels identifying the contributing
        segment (0=planet segment, 1=star/BD segment).

    Returns
    -------
    masses : ndarray, shape (N,)
        Sampled masses in Earth masses.
    labels : ndarray, shape (N,), dtype=int
        Only returned if ``return_component`` is True. 0 for first segment,
        1 for second.

    Raises
    ------
    ValueError
        If bounds are invalid or integral is zero.
    """
    if m_lo <= 0:
        raise ValueError("m_lo must be > 0")
    if m_lo >= m_hi:
        raise ValueError("m_lo must be < m_hi")
    if m_hi > _STARBD_HI:
        raise ValueError(f"Requested m_hi={m_hi} exceeds supported upper mass {_STARBD_HI:.3e} (Earth masses).")
    a = math.log10(m_lo); b = math.log10(m_hi)
    I1 = _segment_integral(m1, c1, a, b)
    I2 = _segment_integral(m2, c2, a, b)
    if I1 + I2 == 0:
        raise ValueError("Zero total integral over interval; check parameters.")
    p1 = I1 / (I1 + I2)
    comp = rng.random(N) >= p1  # True -> segment 2
    n2 = int(comp.sum()); n1 = N - n2
    logM = np.empty(N, dtype=float)
    if n1:
        logM[~comp] = _sample_segment(m1, c1, a, b, n1, rng)
    if n2:
        logM[comp] = _sample_segment(m2, c2, a, b, n2, rng)
    M = 10**logM
    if return_component:
        return M, comp.astype(int)
    return M

def worker(task):
    """Worker process function to generate a single output file.

    Parameters
    ----------
    task : tuple(int, int)
        ``(field_number, file_index)`` specifying which field and which file
        instance (out of ``nf``) to create.

    Notes
    -----
    The function *always* (re)computes data; existing files are removed only
    if ``overwrite_existing`` is True. Randomness is controlled per-task via
    a derived seed if ``FIXED_BASE_SEED`` is set.
    """
    field_number, file_index = task
    base = f"{data_dir}/planets/{rundes}/{rundes}.planets"
    pfile = f"{base}.{field_number}.{file_index}{file_ext}"

    if os.path.exists(pfile):
        if overwrite_existing:
            try:
                os.remove(pfile)
                print(f"Removed existing {pfile} (overwrite enabled).")
            except OSError as e:
                print(f"Could not remove {pfile}: {e}")
        else:
            print(f"File {pfile} exists; skipping (overwrite disabled).")
            return

    # Set up per-task RNG (deterministic if FIXED_BASE_SEED provided)
    if FIXED_BASE_SEED is not None:
        local_seed = FIXED_BASE_SEED + field_number * 100003 + file_index  # 100003 is a prime offset
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()

    # Semimajor axis (still log-uniform between amin, amax in AU)
    a_array = 10 ** (amin + (amax - amin) * rng.random(nl))

    # Draw masses from composite distribution
    masses_earth, labels = sample_masses(nl, return_component=True, m_lo=mmin, m_hi=mmax, rng=rng)
    mass_array = masses_earth * _M_EARTH_TO_SOLAR

    # Isotropic inclination distribution (signed degrees)
    rnd = rng.random(nl)
    arccos_arg = np.where(rnd < 0.5, 2 * rnd, 2 - 2 * rnd)
    safe_arg = np.clip(arccos_arg, -1.0, 1.0)
    angle = np.arccos(safe_arg)
    signed_angle = np.where(rnd < 0.5, angle, -angle)
    inc_array = 180 * signed_angle / np.pi

    # Random orbital phase (degrees)
    p_array = 360.0 * rng.random(nl)

    combined_array = np.empty((nl, 5))
    combined_array[:, 0] = mass_array
    combined_array[:, 1] = a_array
    combined_array[:, 2] = inc_array
    combined_array[:, 3] = p_array
    combined_array[:, 4] = labels  # 0=planet segment, 1=star/BD segment

    if file_ext == ".npy":
        np.save(pfile, combined_array)
    else:
        save_kwargs = {"delimiter": delineator}
        if header:
            save_kwargs["header"] = _HEADER_LINE
            save_kwargs["comments"] = "# "
        np.savetxt(pfile, combined_array, **save_kwargs)

def main():
    """Entry point: orchestrate multiprocessing generation across fields.

    Discovers field numbers from ``sources_file`` and submits all required
    ``(field, file_index)`` tasks to a process pool sized by available CPU
    count. Creates the target output directory if missing.

    Notes
    -----
    The function blocks until all tasks complete. Set ``FIXED_BASE_SEED`` for
    reproducibility.
    """
    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    dir_name = f"{data_dir}/planets/{rundes}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    field_numbers = get_field_numbers(sources_file)
    tasks = [(field, i) for field in field_numbers for i in range(nf)]
    print(f"Generated {len(tasks)} unique tasks. Handing them off to workers.")
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(worker, tasks)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
