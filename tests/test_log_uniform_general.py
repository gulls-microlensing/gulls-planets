"""Tests for ``log_uniform_general_draw_planet_arrays.py``.

Covers GULLS 3.0.0 (``general`` / ``gullsFish.x``) format compatibility as
parsed by ``readPlanets2.cpp`` + ``generalPlanet.cpp``, the sampled
distributions, determinism, and the CLI / output layout.
"""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "log_uniform_general_draw_planet_arrays.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

gen = importlib.import_module("log_uniform_general_draw_planet_arrays")

M_EARTH_TO_SOLAR = 3.00348959632e-6
REQUIRED_COLUMNS = [
    "Mass",
    "SemimajorAxis",
    "Eccentricity",
    "Inclination",
    "LongitudePerihelion",
    "LongitudeAscNode",
    "OrbitType",
]
SAMPLE_SIZE = 20_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_sample_ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic."""
    a_sorted = np.sort(sample_a)
    b_sorted = np.sort(sample_b)
    merged = np.concatenate([a_sorted, b_sorted])
    merged.sort()
    cdf_a = np.searchsorted(a_sorted, merged, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, merged, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ks_critical_value(n_a: int, n_b: int, alpha: float = 1e-3) -> float:
    factor = -0.5 * math.log(alpha / 2.0)
    return math.sqrt(factor * (n_a + n_b) / (n_a * n_b))


def _one_sample_uniform_ks(sample: np.ndarray, low: float, high: float) -> float:
    """One-sample KS statistic of ``sample`` against U(low, high)."""
    x = np.sort((np.asarray(sample) - low) / (high - low))
    n = x.size
    cdf_hi = np.arange(1, n + 1) / n
    cdf_lo = np.arange(0, n) / n
    return float(max(np.max(cdf_hi - x), np.max(x - cdf_lo)))


def _one_sample_critical_value(n: int, alpha: float = 1e-3) -> float:
    return math.sqrt(-0.5 * math.log(alpha / 2.0) / n)


def _assert_uniform(sample: np.ndarray, low: float, high: float, *, label: str) -> None:
    """KS + mean + quantile checks that ``sample`` ~ U(low, high)."""
    n = sample.size
    stat = _one_sample_uniform_ks(sample, low, high)
    crit = _one_sample_critical_value(n)
    assert stat < crit, f"{label}: KS {stat:.4f} >= {crit:.4f} against U({low}, {high})"

    width = high - low
    # Mean of U(a,b) has std width/sqrt(12 n); allow 5 sigma.
    mean_tol = 5.0 * width / math.sqrt(12.0 * n)
    assert abs(sample.mean() - (low + high) / 2.0) < mean_tol, f"{label}: mean off"

    probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    expected = low + probs * width
    q_tol = 5.0 * width * np.sqrt(probs * (1 - probs) / n)
    observed = np.quantile(sample, probs)
    assert np.all(np.abs(observed - expected) < q_tol), f"{label}: quantiles {observed} vs {expected}"


def _write_sources(path: Path, fields: list[int]) -> Path:
    path.write_text("".join(f"{f} 0.0 0.0 0.2 0.2 1e-3 dummy.csv\n" for f in fields), encoding="ascii")
    return path


def _parse_gulls_planet_file(path: Path) -> tuple[list[str], list[list[str]], np.ndarray]:
    """Parse a planet file the way GULLS ``readPlanets2.cpp`` does.

    Lines starting with ``#`` are skipped; the first remaining line that
    contains letters is the header; every subsequent non-empty line is split
    on whitespace and parsed as floats.

    Returns
    -------
    header_tokens, raw_rows, data
    """
    header: list[str] | None = None
    raw_rows: list[list[str]] = []
    with path.open("r", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            if header is None and any(ch.isalpha() for ch in line):
                header = line.split()
                continue
            raw_rows.append(line.split())
    assert header is not None, "readPlanets2.cpp would find no header line"
    data = np.array([[float(tok) for tok in row] for row in raw_rows], dtype=float)
    return header, raw_rows, data


def _generate(tmp_path: Path, *, seed: int = 2024, n: int = SAMPLE_SIZE, field: int = 0,
              index: int = 0, **overrides) -> Path:
    config = gen.GeneratorConfig(
        rundes="pytest_lugen",
        data_dir=str(tmp_path),
        n_per_file=n,
        base_seed=seed,
        **overrides,
    )
    written = gen.worker((field, index), config)
    assert written is not None
    return Path(written)


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def default_sample(tmp_path_factory) -> np.ndarray:
    out = _generate(tmp_path_factory.mktemp("default_sample"), seed=314159)
    _, _, data = _parse_gulls_planet_file(out)
    return data


# ---------------------------------------------------------------------------
# 1. GULLS readPlanets2.cpp + generalPlanet.cpp format compatibility
# ---------------------------------------------------------------------------

def test_gulls_general_format(tmp_path):
    out = _generate(tmp_path, n=500)
    first_line = out.read_text(encoding="ascii").splitlines()[0]
    assert not first_line.startswith("#"), "Header must not be #-commented (readPlanets2.cpp skips it)"
    assert "," not in first_line

    header, raw_rows, data = _parse_gulls_planet_file(out)
    assert len(header) == 7 and len(header) % 7 == 0
    for token, required in zip(header, REQUIRED_COLUMNS):
        assert token.rfind(required, 0) == 0, f"header token {token!r} lacks prefix {required!r}"
    assert header == REQUIRED_COLUMNS

    assert len(raw_rows) == 500
    assert all(len(row) == 7 and len(row) % 7 == 0 for row in raw_rows)
    assert all("," not in tok for row in raw_rows for tok in row)
    assert np.all(np.isfinite(data))

    mass, sma, ecc, inc, varpi, node, otype = data.T
    assert np.all(mass > 0)
    assert np.all(sma > 0)
    assert np.all(ecc == 0.0)
    assert np.all((inc >= -90.0) & (inc <= 90.0))
    # %.4f formatting can round draws in [359.99995, 360) to 360.0000.
    assert np.all((varpi >= 0.0) & (varpi <= 360.0))
    assert np.all((node >= 0.0) & (node <= 360.0))
    assert np.all(otype == 1)
    assert all(row[6] == "1" for row in raw_rows), "OrbitType must be written as the integer token '1'"


# ---------------------------------------------------------------------------
# 2. Distribution checks
# ---------------------------------------------------------------------------

def test_module_defaults():
    assert gen._M_EARTH_TO_SOLAR == M_EARTH_TO_SOLAR
    assert (gen.MASS_MIN_EARTH, gen.MASS_MAX_EARTH) == (0.1, 100.0)
    assert (gen.A_MIN_AU, gen.A_MAX_AU) == (0.3, 30.0)
    assert gen.ECCENTRICITY == 0.0
    assert gen.FIXED_BASE_SEED == 12345
    assert gen.sources_file == "./gulls_surot2d_H2023.sources"
    assert gen.HEADER_LINE.split() == REQUIRED_COLUMNS


def test_default_bounds(default_sample):
    mass_earth = default_sample[:, 0] / M_EARTH_TO_SOLAR
    sma = default_sample[:, 1]
    # %.6e output carries ~1e-6 relative rounding.
    assert mass_earth.min() >= 0.1 * (1 - 1e-6) and mass_earth.max() <= 100.0 * (1 + 1e-6)
    assert sma.min() >= 0.3 * (1 - 1e-6) and sma.max() <= 30.0 * (1 + 1e-6)
    # The sample should actually span the range.
    assert mass_earth.min() < 0.11 and mass_earth.max() > 90.0
    assert sma.min() < 0.32 and sma.max() > 28.0


def test_log_mass_uniform(default_sample):
    log_mass = np.log10(default_sample[:, 0] / M_EARTH_TO_SOLAR)
    _assert_uniform(log_mass, math.log10(0.1), math.log10(100.0), label="log10 mass (Earth)")


def test_log_sma_uniform(default_sample):
    log_a = np.log10(default_sample[:, 1])
    _assert_uniform(log_a, math.log10(0.3), math.log10(30.0), label="log10 semi-major axis")


def test_inclination_isotropic(default_sample):
    inc = default_sample[:, 3]
    _assert_uniform(np.cos(np.radians(inc)), 0.0, 1.0, label="cos(inclination)")
    frac_pos = np.mean(inc > 0)
    assert abs(frac_pos - 0.5) < 5.0 * 0.5 / math.sqrt(inc.size), f"sign fraction {frac_pos:.4f}"


def test_inclination_matches_uniform_sampler(tmp_path, monkeypatch, default_sample):
    # uniform_draw_planet_arrays reads its sources file relative to cwd on import.
    monkeypatch.chdir(REPO_ROOT)
    uni = importlib.reload(importlib.import_module("uniform_draw_planet_arrays"))
    for name, value in {
        "nl": SAMPLE_SIZE, "nf": 1, "rundes": "pytest_uniform_ref", "data_dir": str(tmp_path),
        "file_ext": "", "overwrite_existing": True, "FIXED_BASE_SEED": 777,
        "delineator": " ", "header": False,
    }.items():
        monkeypatch.setattr(uni, name, value)
    (tmp_path / "planets" / "pytest_uniform_ref").mkdir(parents=True)
    uni.worker((0, 0))
    ref = np.loadtxt(tmp_path / "planets" / "pytest_uniform_ref" / "pytest_uniform_ref.planets.0.0")

    stat = _two_sample_ks_statistic(ref[:, 2], default_sample[:, 3])
    crit = _ks_critical_value(ref.shape[0], default_sample.shape[0])
    assert stat < crit, f"inclination KS {stat:.4f} >= {crit:.4f}"

    # Same seed + same draw order (a, mass, inc) => identical draws.
    ours = _parse_gulls_planet_file(_generate(tmp_path, seed=777, n=SAMPLE_SIZE))[2]
    np.testing.assert_allclose(ours[:, 1], ref[:, 1], rtol=1e-6)
    np.testing.assert_allclose(ours[:, 0], ref[:, 0], rtol=1e-6)
    np.testing.assert_allclose(ours[:, 3], ref[:, 2], atol=1e-4)


def test_longitudes_uniform(default_sample):
    varpi = default_sample[:, 4]
    node = default_sample[:, 5]
    _assert_uniform(varpi, 0.0, 360.0, label="LongitudePerihelion")
    _assert_uniform(node, 0.0, 360.0, label="LongitudeAscNode")
    _assert_uniform((varpi - node) % 360.0, 0.0, 360.0, label="omega = varpi - Omega")
    assert abs(np.corrcoef(varpi, node)[0, 1]) < 0.05


# ---------------------------------------------------------------------------
# 3. Determinism, CLI, layout, --no-overwrite
# ---------------------------------------------------------------------------

def test_deterministic_and_distinct(tmp_path):
    a = _generate(tmp_path / "a", seed=99, n=200).read_bytes()
    b = _generate(tmp_path / "b", seed=99, n=200).read_bytes()
    assert a == b, "same seed must give byte-identical output"

    # Note: the shared seed convention (base + field*100003 + index) means
    # adjacent base seeds alias neighbouring subruns, so use a distant seed.
    other_seed = _generate(tmp_path / "c", seed=99 + 50_000, n=200).read_bytes()
    other_field = _generate(tmp_path / "d", seed=99, n=200, field=1).read_bytes()
    other_index = _generate(tmp_path / "e", seed=99, n=200, index=1).read_bytes()
    assert len({a, other_seed, other_field, other_index}) == 4


def test_cli_default_layout(tmp_path):
    sources = _write_sources(tmp_path / "test.sources", [0, 7])
    _run_cli(tmp_path, "-n", "50", "--files-per-field", "2", "--seed", "5",
             "--rundes", "clirun", "--sources-file", str(sources), "--workers", "2")

    run_dir = tmp_path / "planets" / "clirun"
    files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())
    assert files == [
        "0/clirun.planets.0.0",
        "0/clirun.planets.7.0",
        "1/clirun.planets.0.1",
        "1/clirun.planets.7.1",
    ]
    for f in files:
        header, rows, data = _parse_gulls_planet_file(run_dir / f)
        assert header == REQUIRED_COLUMNS
        assert data.shape == (50, 7)

    # Pool output must equal the serial in-process result for the same task.
    config = gen.GeneratorConfig(rundes="clirun", data_dir=str(tmp_path / "serial"),
                                 n_per_file=50, base_seed=5)
    serial = Path(gen.worker((7, 1), config))
    assert serial.read_bytes() == (run_dir / "1" / "clirun.planets.7.1").read_bytes()


def test_cli_flat_layout_and_ranges(tmp_path):
    sources = _write_sources(tmp_path / "test.sources", [3])
    _run_cli(tmp_path, "-n", "2000", "--files-per-field", "2", "--rundes", "flat",
             "--sources-file", str(sources), "--data-dir", str(tmp_path / "out"),
             "--flat-layout", "--workers", "1",
             "--mass-min-earth", "1", "--mass-max-earth", "10",
             "--a-min-au", "1", "--a-max-au", "5", "--eccentricity", "0.25")

    run_dir = tmp_path / "out" / "planets" / "flat"
    assert sorted(p.name for p in run_dir.iterdir()) == ["flat.planets.3.0", "flat.planets.3.1"]
    _, _, data = _parse_gulls_planet_file(run_dir / "flat.planets.3.0")
    assert data.shape == (2000, 7)
    mass_earth = data[:, 0] / M_EARTH_TO_SOLAR
    assert mass_earth.min() >= 1 - 1e-5 and mass_earth.max() <= 10 + 1e-5
    assert data[:, 1].min() >= 1 - 1e-6 and data[:, 1].max() <= 5 + 1e-5
    assert np.all(data[:, 2] == 0.25)


def test_cli_no_overwrite(tmp_path):
    sources = _write_sources(tmp_path / "test.sources", [0])
    common = ["-n", "20", "--rundes", "keep", "--sources-file", str(sources), "--workers", "1"]
    _run_cli(tmp_path, *common, "--seed", "1")
    target = tmp_path / "planets" / "keep" / "0" / "keep.planets.0.0"
    original = target.read_bytes()

    result = _run_cli(tmp_path, *common, "--seed", "2", "--no-overwrite")
    assert target.read_bytes() == original
    assert "1 skipped" in result.stdout

    _run_cli(tmp_path, *common, "--seed", "2")
    assert target.read_bytes() != original


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        gen.GeneratorConfig(mass_min_earth=10.0, mass_max_earth=1.0).validate()
    with pytest.raises(ValueError):
        gen.GeneratorConfig(a_min_au=0.0).validate()
    with pytest.raises(ValueError):
        gen.GeneratorConfig(eccentricity=1.0).validate()
