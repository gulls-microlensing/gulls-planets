"""Generate sensitivity-map planet arrays (Perl ``sensmap.pl`` port).

The original Perl script produced semi-grid-based planet catalogues for each
field/subrun combination using deterministic mass/semi-major axis grids along
with randomized inclination and orbital phase. This Python implementation keeps
that behaviour but mirrors the modern scripts in this repository, including
multi-processing, reproducible seeding, and dynamic field discovery from the
Roman source list.

Execution
---------
Run directly::

    python sensmap_draw_planet_arrays.py

Configuration is handled via module-level constants below. The script expects a
sources file (default: ``gulls_surot2d_H2023.sources``) whose first token per
line is an integer field identifier.

Output
------
Each generated text file ``smap.planets.<field>.<subrun>`` contains four columns::

    1. mass (M_sun)
    2. semi-major axis (au)
    3. inclination (deg)
    4. orbital phase (deg)

The mass/semi-major axis values follow the same grid as the Perl script, while
inclination and phase are freshly sampled per catalogue.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------ Simulation configuration (matches legacy script) ------------------------
rundes = "smap"
sources_file = "./gulls_surot2d_H2023.sources"
data_dir = Path("./")
file_ext = ""

n_subruns = 400  # corresponds to Perl's nsub
n_repeats_per_grid_point = 5  # corresponds to Perl's nrep

overwrite_existing = True
header = False
_delineator = " "
_fmt = "%.10e"

# Optional base seed for reproducibility (set to an integer to enable)
FIXED_BASE_SEED: Optional[int] = None

# ------------------------------- Static grids (log10 space) ------------------------------------
_M_EARTH_TO_SOLAR = 3.00374072e-6

_LOG_MASS_MIN = -2.0
_LOG_MASS_MAX = 1.0
_LOG_MASS_STEP = 0.25

_LOG_SMA_MIN = -1.0
_LOG_SMA_MAX = 1.0
_LOG_SMA_STEP = 0.125


def _inclusive_linspace(start: float, stop: float, step: float) -> np.ndarray:
    """Return evenly spaced values including *stop* despite floating error."""
    count = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, count)


_LOG_MASS_GRID = _inclusive_linspace(_LOG_MASS_MIN, _LOG_MASS_MAX, _LOG_MASS_STEP)
_LOG_SMA_GRID = _inclusive_linspace(_LOG_SMA_MIN, _LOG_SMA_MAX, _LOG_SMA_STEP)

_MASS_GRID_SOLAR = (10.0 ** _LOG_MASS_GRID) * _M_EARTH_TO_SOLAR
_SMA_GRID = 10.0 ** _LOG_SMA_GRID

# Mass/semi-major axis template for a single repeat over the grid points
_BASE_MASS_TEMPLATE = np.repeat(_MASS_GRID_SOLAR, _SMA_GRID.size)
_BASE_SMA_TEMPLATE = np.tile(_SMA_GRID, _MASS_GRID_SOLAR.size)

# Full templates with n_repeats_per_grid_point copies
_MASS_TEMPLATE = np.tile(_BASE_MASS_TEMPLATE, n_repeats_per_grid_point)
_SMA_TEMPLATE = np.tile(_BASE_SMA_TEMPLATE, n_repeats_per_grid_point)

_SAMPLES_PER_FILE = _MASS_TEMPLATE.size

# --------------------------------------- Helpers -----------------------------------------------
def get_field_numbers(sources_path: str) -> List[int]:
    """Extract integer field IDs from a sources file."""
    field_numbers: List[int] = []
    with open(sources_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            field_numbers.append(int(stripped.split()[0]))
    return field_numbers


def _resolve_tasks(fields: Sequence[int], subrun_count: int) -> List[Tuple[int, int]]:
    return [(field, subrun) for field in fields for subrun in range(subrun_count)]


# ------------------------------------- Worker logic ---------------------------------------------
def _build_rng(field_number: int, subrun: int) -> np.random.Generator:
    if FIXED_BASE_SEED is None:
        return np.random.default_rng()
    local_seed = FIXED_BASE_SEED + field_number * 100_003 + subrun
    return np.random.default_rng(local_seed)


def _save_catalogue(path: Path, payload: np.ndarray) -> None:
    if header:
        np.savetxt(
            path,
            payload,
            fmt=_fmt,
            delimiter=_delineator,
            header="mass (M_Sun) a (au) inc (deg) p (deg)",
            comments="# ",
        )
    else:
        np.savetxt(path, payload, fmt=_fmt, delimiter=_delineator)


def _prepare_payload(rng: np.random.Generator) -> np.ndarray:
    rnd = rng.random(_SAMPLES_PER_FILE)
    arccos_arg = np.where(rnd < 0.5, 2.0 * rnd, 2.0 - 2.0 * rnd)
    safe_arg = np.clip(arccos_arg, -1.0, 1.0)
    inclination = np.degrees(np.arccos(safe_arg))
    inclination = np.where(rnd < 0.5, inclination, -inclination)

    phase = 360.0 * rng.random(_SAMPLES_PER_FILE)

    payload = np.empty((_SAMPLES_PER_FILE, 4), dtype=float)
    payload[:, 0] = _MASS_TEMPLATE
    payload[:, 1] = _SMA_TEMPLATE
    payload[:, 2] = inclination
    payload[:, 3] = phase
    return payload


OUTPUT_DIR: Path = data_dir / "planets" / rundes


def _init_worker(output_dir_str: str) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir_str)


def worker(task: Tuple[int, int]) -> None:
    field_number, subrun = task
    target_path = OUTPUT_DIR / f"{rundes}.planets.{field_number}.{subrun}{file_ext}"

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        if overwrite_existing:
            try:
                target_path.unlink()
                print(f"Removed existing {target_path} (overwrite enabled).")
            except OSError as exc:
                print(f"Could not remove {target_path}: {exc}")
        else:
            print(f"File {target_path} exists; skipping (overwrite disabled).")
            return

    rng = _build_rng(field_number, subrun)
    payload = _prepare_payload(rng)
    _save_catalogue(target_path, payload)


# -------------------------------------- Pipeline ------------------------------------------------
def main(
    *,
    selected_fields: Optional[Iterable[int]] = None,
    max_subruns: Optional[int] = None,
    processes: Optional[int] = None,
    output_dir: Optional[os.PathLike[str] | str] = None,
) -> None:
    """Entry point for generating sensitivity-map catalogues.

    Parameters
    ----------
    selected_fields : iterable of int, optional
        If provided, only these field numbers are processed (useful for tests).
    max_subruns : int, optional
        If provided, limits the number of subruns per field (0-indexed inclusive of 0).
    processes : int, optional
        Size of the multiprocessing pool. Defaults to ``mp.cpu_count()``.
    """

    if selected_fields is None:
        fields = get_field_numbers(sources_file)
    else:
        fields = list(selected_fields)

    if not fields:
        raise ValueError("No fields specified or found in sources file.")

    subrun_count = max_subruns if max_subruns is not None else n_subruns
    if subrun_count <= 0:
        raise ValueError("max_subruns must be positive when provided.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else data_dir / "planets" / rundes
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _resolve_tasks(fields, subrun_count)
    print(
        f"Generating {len(tasks)} catalogues in {resolved_output_dir}."
        f" (fields={len(fields)}, subruns={subrun_count})"
    )

    pool_size = processes or mp.cpu_count()
    if pool_size <= 1:
        _init_worker(str(resolved_output_dir))
        for task in tasks:
            worker(task)
        return

    with mp.Pool(pool_size, initializer=_init_worker, initargs=(str(resolved_output_dir),)) as pool:
        pool.map(worker, tasks)


if __name__ == "__main__":
    start_time = time.time()
    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    main()
    duration = time.time() - start_time
    print(f"Execution time: {duration:.2f} seconds")
