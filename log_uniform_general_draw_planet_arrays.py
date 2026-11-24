"""Generate log-uniform single-planet catalogs for the GULLS ``general`` branch.

This script combines the sampling physics of ``uniform_draw_planet_arrays.py``
with the GULLS 3.0.0 (``general`` branch, ``gullsFish.x``) planet-file format
used by ``multiplanet_draw_planet_arrays.py``. It is intended for the Roman
eta-Earth Fisher simulation: every row is a single planet whose mass and
semi-major axis are drawn independently and log-uniformly.

Sampling
--------
* ``Mass``: log-uniform in Earth masses over ``[mass_min_earth, mass_max_earth]``
  (default 0.1-100 M_Earth), converted to solar masses with
  ``_M_EARTH_TO_SOLAR``.
* ``SemimajorAxis``: log-uniform in au over ``[a_min_au, a_max_au]``
  (default 0.3-30 au).
* ``Eccentricity``: fixed (default 0.0, circular orbits).
* ``Inclination``: signed isotropic in [-90, 90] degrees, using the same
  ``arccos(2u)`` / ``-arccos(2 - 2u)`` construction as
  ``uniform_draw_planet_arrays.py`` and GULLS ``generalPlanet.cpp``.
* ``LongitudePerihelion`` and ``LongitudeAscNode``: independent uniform draws
  on [0, 360) degrees. GULLS uses omega = varpi - Omega, so this yields a
  uniform argument of periapsis modulo 360 degrees.
* ``OrbitType``: 1 (planet orbiting the primary lens star).

Output format (GULLS ``readPlanets2.cpp`` + ``generalPlanet.cpp``)
-----------------------------------------------------------------
* Whitespace (single space) delimited text.
* The first line is an un-commented header (``readPlanets2.cpp`` skips lines
  starting with ``#`` and uses the first remaining line containing letters
  as the header)::

      Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType

* Files are named ``<rundes>.planets.<field>.<file_index>`` and, by default,
  written to the per-subrun layout ``planets/<rundes>/<file_index>/``
  (matching ``multiplanet_draw_planet_arrays.py``). ``--flat-layout`` writes
  to ``planets/<rundes>/`` instead; GULLS accepts both.

Execution (CLI)
---------------
Run directly (see ``--help`` for all options)::

    python log_uniform_general_draw_planet_arrays.py -n 10000 --files-per-field 1

Reproducibility
---------------
When a base seed is set (``FIXED_BASE_SEED`` or ``--seed``), each task derives
the deterministic seed ``base_seed + field_number * 100003 + file_index``
(same convention as the other samplers in this repository).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Resolve paths relative to this script
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------
_M_EARTH_TO_SOLAR = 3.00348959632e-6  # Earth -> Solar mass conversion

# -------------------------------------------------------------------------
# Sampling bounds (linear units; Roman eta-Earth white-paper defaults)
# -------------------------------------------------------------------------
MASS_MIN_EARTH = 0.1     # M_Earth
MASS_MAX_EARTH = 100.0   # M_Earth
A_MIN_AU = 0.3           # au
A_MAX_AU = 30.0          # au
ECCENTRICITY = 0.0       # circular orbits
ORBIT_TYPE = 1           # planet orbiting the primary lens

# -------------------------------------------------------------------------
# Run configuration
# -------------------------------------------------------------------------
rundes = 'log_uniform_general'
sources_file = './gulls_surot2d_H2023.sources'  # defines number of fields
data_dir = './'
nl = 10000                 # planets per file
nf = 1                     # files per field
overwrite_existing = True  # if False, existing files are kept
flat_layout = False        # if True, omit the per-subrun subdirectory

FIXED_BASE_SEED: int | None = 12345  # set to None for unseeded runs

# -------------------------------------------------------------------------
# GULLS 3.0.0 (general) output format
# -------------------------------------------------------------------------
COLUMNS: tuple[str, ...] = (
    'Mass',
    'SemimajorAxis',
    'Eccentricity',
    'Inclination',
    'LongitudePerihelion',
    'LongitudeAscNode',
    'OrbitType',
)
HEADER_LINE = ' '.join(COLUMNS)
DELIMITER = ' '
FMT: list[str] = ['%.6e', '%.6e', '%.6f', '%.4f', '%.4f', '%.4f', '%d']


@dataclass(frozen=True)
class GeneratorConfig:
    """Explicit run configuration passed to worker processes.

    Passing this object (rather than relying on mutated module globals) keeps
    workers correct under both ``fork`` and ``spawn`` start methods.

    Attributes
    ----------
    rundes : str
        Run identifier used in the output directory and filenames.
    data_dir : str
        Base directory under which ``planets/<rundes>/`` is created.
    n_per_file : int
        Number of planets (rows) per output file.
    mass_min_earth, mass_max_earth : float
        Log-uniform mass bounds in Earth masses.
    a_min_au, a_max_au : float
        Log-uniform semi-major axis bounds in au.
    eccentricity : float
        Fixed eccentricity assigned to every planet.
    base_seed : int or None
        Base seed for deterministic per-task generators.
    overwrite : bool
        Overwrite existing files when True; skip them otherwise.
    flat_layout : bool
        Write to ``planets/<rundes>/`` instead of ``planets/<rundes>/<file_index>/``.
    """

    rundes: str = rundes
    data_dir: str = data_dir
    n_per_file: int = nl
    mass_min_earth: float = MASS_MIN_EARTH
    mass_max_earth: float = MASS_MAX_EARTH
    a_min_au: float = A_MIN_AU
    a_max_au: float = A_MAX_AU
    eccentricity: float = ECCENTRICITY
    base_seed: int | None = FIXED_BASE_SEED
    overwrite: bool = overwrite_existing
    flat_layout: bool = flat_layout

    def validate(self) -> None:
        """Raise ``ValueError`` for physically invalid configurations."""
        if self.n_per_file < 1:
            raise ValueError('n_per_file must be >= 1')
        if not 0.0 < self.mass_min_earth < self.mass_max_earth:
            raise ValueError('require 0 < mass_min_earth < mass_max_earth')
        if not 0.0 < self.a_min_au < self.a_max_au:
            raise ValueError('require 0 < a_min_au < a_max_au')
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError('require 0 <= eccentricity < 1')


def get_field_numbers(sources_path: str | os.PathLike[str]) -> list[int]:
    """Extract integer field identifiers from a sources file.

    Parameters
    ----------
    sources_path : str or os.PathLike
        Path to a text file; the first whitespace-delimited token of each
        non-empty line is interpreted as an integer field number.

    Returns
    -------
    list of int
        Ordered list of parsed field numbers.
    """
    field_numbers: list[int] = []
    with open(sources_path, 'r') as fh:
        for line in fh:
            if line.strip():
                field_numbers.append(int(line.split()[0]))
    return field_numbers


def resolve_sources_file(path: str | os.PathLike[str]) -> str:
    """Return ``path`` if it exists, else fall back to the script directory.

    Parameters
    ----------
    path : str or os.PathLike
        Requested sources file.

    Returns
    -------
    str
        Existing path to use (the original path if no fallback is found).
    """
    path = os.fspath(path)
    if os.path.exists(path) or os.path.isabs(path):
        return path
    candidate = os.path.join(_BASE_DIR, os.path.basename(path))
    return candidate if os.path.exists(candidate) else path


def task_seed(base_seed: int, field_number: int, file_index: int) -> int:
    """Deterministic per-task seed (shared convention across samplers)."""
    return base_seed + field_number * 100003 + file_index


def output_path(config: GeneratorConfig, field_number: int, file_index: int) -> Path:
    """Path of the planet file for ``(field_number, file_index)``.

    Parameters
    ----------
    config : GeneratorConfig
        Run configuration (``data_dir``, ``rundes``, ``flat_layout``).
    field_number : int
        Field identifier from the sources file.
    file_index : int
        Subrun / file index.

    Returns
    -------
    pathlib.Path
        ``planets/<rundes>/[<file_index>/]<rundes>.planets.<field>.<file_index>``.
    """
    run_dir = Path(config.data_dir) / 'planets' / config.rundes
    if not config.flat_layout:
        run_dir = run_dir / str(file_index)
    return run_dir / f'{config.rundes}.planets.{field_number}.{file_index}'


def draw_log_uniform(size: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Draw samples log-uniformly distributed on ``[low, high]``.

    Parameters
    ----------
    size : int
        Number of samples.
    low, high : float
        Positive linear-space bounds.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(size,)``.
    """
    log_min = math.log10(low)
    log_max = math.log10(high)
    return 10.0 ** (log_min + (log_max - log_min) * rng.random(size))


def draw_isotropic_inclination(size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw signed isotropic inclinations in degrees on ``[-90, 90]``.

    Uses ``arccos(2u)`` for ``u < 0.5`` and ``-arccos(2 - 2u)`` otherwise,
    exactly as in ``uniform_draw_planet_arrays.py`` and GULLS
    ``generalPlanet.cpp``. ``|cos(inc)|`` is therefore uniform on [0, 1].

    Parameters
    ----------
    size : int
        Number of samples.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Inclinations in degrees, shape ``(size,)``.
    """
    rnd = rng.random(size)
    arccos_arg = np.where(rnd < 0.5, 2.0 * rnd, 2.0 - 2.0 * rnd)
    angle = np.arccos(np.clip(arccos_arg, -1.0, 1.0))
    return np.degrees(np.where(rnd < 0.5, angle, -angle))


def generate_planets(config: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate one catalog of single planets.

    Parameters
    ----------
    config : GeneratorConfig
        Sampling bounds, eccentricity and row count.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(config.n_per_file, 7)`` with columns ordered as
        :data:`COLUMNS` (mass in M_sun, angles in degrees).
    """
    n = config.n_per_file
    out = np.empty((n, len(COLUMNS)), dtype=float)
    # Draw order matches uniform_draw_planet_arrays.py: a, mass, inclination.
    out[:, 1] = draw_log_uniform(n, config.a_min_au, config.a_max_au, rng)
    out[:, 0] = draw_log_uniform(n, config.mass_min_earth, config.mass_max_earth, rng) * _M_EARTH_TO_SOLAR
    out[:, 3] = draw_isotropic_inclination(n, rng)
    out[:, 2] = config.eccentricity
    # GULLS uses omega = varpi - Omega; independent uniform varpi and Omega
    # give a uniform omega modulo 360 degrees.
    out[:, 4] = 360.0 * rng.random(n)
    out[:, 5] = 360.0 * rng.random(n)
    out[:, 6] = ORBIT_TYPE
    return out


def write_planet_file(path: str | os.PathLike[str], planets: np.ndarray) -> None:
    """Write a GULLS 3.0.0 (general) planet file.

    Parameters
    ----------
    path : str or os.PathLike
        Destination file.
    planets : numpy.ndarray
        Array of shape ``(n, 7)`` ordered as :data:`COLUMNS`.
    """
    np.savetxt(path, planets, delimiter=DELIMITER, header=HEADER_LINE, comments='', fmt=FMT)


def worker(task: tuple[int, int], config: GeneratorConfig) -> str | None:
    """Generate and write a single planet file.

    Parameters
    ----------
    task : tuple of (int, int)
        ``(field_number, file_index)`` pair.
    config : GeneratorConfig
        Explicit run configuration.

    Returns
    -------
    str or None
        Path written, or ``None`` if skipped because the file exists and
        overwriting is disabled.
    """
    field_number, file_index = task
    pfile = output_path(config, field_number, file_index)
    if pfile.exists() and not config.overwrite:
        return None
    pfile.parent.mkdir(parents=True, exist_ok=True)

    if config.base_seed is not None:
        rng = np.random.default_rng(task_seed(config.base_seed, field_number, file_index))
    else:
        rng = np.random.default_rng()

    write_planet_file(pfile, generate_planets(config, rng))
    return str(pfile)


def _worker_star(args: tuple[tuple[int, int], GeneratorConfig]) -> str | None:
    """Unpack helper for :meth:`multiprocessing.pool.Pool.map`."""
    return worker(*args)


def run(
    config: GeneratorConfig,
    field_numbers: Sequence[int],
    n_files: int,
    workers: int | None = None,
) -> list[str | None]:
    """Generate all ``(field, file_index)`` catalogs.

    Parameters
    ----------
    config : GeneratorConfig
        Explicit run configuration.
    field_numbers : sequence of int
        Fields to generate.
    n_files : int
        Number of files (subruns) per field.
    workers : int or None
        Process count; ``None`` uses ``mp.cpu_count()``, ``1`` runs serially.

    Returns
    -------
    list of (str or None)
        Written paths (``None`` for skipped files), in task order.
    """
    config.validate()
    tasks = [(field, i) for field in field_numbers for i in range(n_files)]
    if workers is None:
        workers = mp.cpu_count()
    workers = max(1, min(workers, len(tasks) or 1))
    if workers == 1:
        return [worker(task, config) for task in tasks]
    with mp.Pool(workers) as pool:
        return pool.map(_worker_star, [(task, config) for task in tasks])


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and return the namespace."""
    p = argparse.ArgumentParser(
        description='Generate log-uniform single-planet catalogs in GULLS 3.0.0 (general) format')
    p.add_argument('--systems-per-file', '-n', type=int, default=nl,
                   help='Number of planets per output file (default: %(default)s)')
    p.add_argument('--files-per-field', type=int, default=nf,
                   help='Number of files (subruns) per field (default: %(default)s)')
    p.add_argument('--seed', type=int, default=FIXED_BASE_SEED,
                   help='Base RNG seed for reproducible runs (default: %(default)s)')
    p.add_argument('--no-seed', action='store_true',
                   help='Ignore any base seed and draw unpredictable random numbers')
    p.add_argument('--rundes', type=str, default=rundes,
                   help='Run identifier used in output filenames (default: %(default)s)')
    p.add_argument('--sources-file', type=str, default=sources_file,
                   help='Path to sources file listing fields (default: %(default)s)')
    p.add_argument('--data-dir', type=str, default=data_dir,
                   help='Base directory for planets/<rundes>/ output (default: %(default)s)')
    p.add_argument('--mass-min-earth', type=float, default=MASS_MIN_EARTH,
                   help='Minimum planet mass in Earth masses (default: %(default)s)')
    p.add_argument('--mass-max-earth', type=float, default=MASS_MAX_EARTH,
                   help='Maximum planet mass in Earth masses (default: %(default)s)')
    p.add_argument('--a-min-au', type=float, default=A_MIN_AU,
                   help='Minimum semi-major axis in au (default: %(default)s)')
    p.add_argument('--a-max-au', type=float, default=A_MAX_AU,
                   help='Maximum semi-major axis in au (default: %(default)s)')
    p.add_argument('--eccentricity', type=float, default=ECCENTRICITY,
                   help='Fixed eccentricity for every planet (default: %(default)s)')
    p.add_argument('--no-overwrite', action='store_true', default=not overwrite_existing,
                   help='Do not overwrite existing files (default: overwrite)')
    p.add_argument('--flat-layout', action='store_true', default=flat_layout,
                   help='Write planets/<rundes>/<file> instead of planets/<rundes>/<file_index>/<file>')
    p.add_argument('--workers', type=int, default=None,
                   help='Worker processes (default: CPU count; 1 = serial)')
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[str | None]:
    """CLI entry point."""
    args = parse_cli_args(argv)
    config = GeneratorConfig(
        rundes=args.rundes,
        data_dir=args.data_dir,
        n_per_file=args.systems_per_file,
        mass_min_earth=args.mass_min_earth,
        mass_max_earth=args.mass_max_earth,
        a_min_au=args.a_min_au,
        a_max_au=args.a_max_au,
        eccentricity=args.eccentricity,
        base_seed=None if args.no_seed else args.seed,
        overwrite=not args.no_overwrite,
        flat_layout=args.flat_layout,
    )
    config.validate()

    field_ids = get_field_numbers(resolve_sources_file(args.sources_file))
    if config.base_seed is not None:
        print(f'Deterministic run with base seed {config.base_seed}')
    print(f'Mass: [{config.mass_min_earth}, {config.mass_max_earth}] M_Earth (log-uniform)')
    print(f'Semi-major axis: [{config.a_min_au}, {config.a_max_au}] au (log-uniform)')
    print(f'Eccentricity: {config.eccentricity}')
    print(f'{len(field_ids)} fields x {args.files_per_field} files x {config.n_per_file} planets')

    t0 = time.perf_counter()
    results = run(config, field_ids, args.files_per_field, workers=args.workers)
    n_written = sum(r is not None for r in results)
    print(f'Wrote {n_written} files ({len(results) - n_written} skipped) '
          f'under {Path(config.data_dir) / "planets" / config.rundes} '
          f'in {time.perf_counter() - t0:.2f}s')
    return results


if __name__ == '__main__':
    main()
