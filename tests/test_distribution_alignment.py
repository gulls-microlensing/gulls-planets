from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _two_sample_ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic."""
    a_sorted = np.sort(sample_a)
    b_sorted = np.sort(sample_b)
    merged = np.concatenate([a_sorted, b_sorted])
    merged.sort()
    cdf_a = np.searchsorted(a_sorted, merged, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, merged, side="right") / b_sorted.size
    return np.max(np.abs(cdf_a - cdf_b))


def _ks_critical_value(n_a: int, n_b: int, alpha: float = 1e-3) -> float:
    factor = -0.5 * math.log(alpha / 2.0)
    return math.sqrt(factor * (n_a + n_b) / (n_a * n_b))


def _load_perl_sample(tmp_path: Path, sample_size: int, seed: int) -> np.ndarray:
    sources_text = "0 0.0 -1.0 0 0 0 0\n"
    sources_path = tmp_path / "gulls_surot2d_H2023.sources"
    sources_path.write_text(sources_text, encoding="ascii")

    perl_script = Path(__file__).resolve().parents[1] / "kosh_hz_uniform.pl"
    perl_str = str(perl_script).replace("\\", "\\\\").replace('"', '\\"')
    command = [
        "perl",
        "-we",
        f'srand({seed}); do "{perl_str}"; die $@ if $@;',
    ]
    subprocess.run(command, cwd=tmp_path, check=True)

    output_path = tmp_path / "pearl_test_uniform_draw" / "pearl_test_uniform_draw.planets.0.0"
    data = np.loadtxt(output_path)
    if data.shape[0] < sample_size:
        raise AssertionError("Perl sample smaller than requested subset")
    return data[:sample_size]


def _load_python_sample(
    tmp_path: Path,
    sample_size: int,
    seed: int,
    *,
    delineator_override: str | None = None,
    header_override: bool | None = None,
    rundes_override: str = "python_uniform_test",
) -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.import_module("uniform_draw_planet_arrays")
    module = importlib.reload(module)

    cached = {
        "nl": module.nl,
        "nf": module.nf,
        "rundes": module.rundes,
        "data_dir": module.data_dir,
        "file_ext": module.file_ext,
        "overwrite_existing": module.overwrite_existing,
        "FIXED_BASE_SEED": module.FIXED_BASE_SEED,
        "delineator": getattr(module, "delineator", ","),
        "header": getattr(module, "header", True),
    }

    try:
        module.nl = sample_size
        module.nf = 1
        module.rundes = rundes_override
        module.data_dir = str(tmp_path)
        module.file_ext = ""
        module.overwrite_existing = True
        module.FIXED_BASE_SEED = seed
        if delineator_override is not None:
            module.delineator = delineator_override
        if header_override is not None:
            module.header = header_override

        target_dir = Path(module.data_dir) / "planets" / module.rundes
        target_dir.mkdir(parents=True, exist_ok=True)

        module.worker((0, 0))

        output_path = target_dir / f"{module.rundes}.planets.0.0"
        load_kwargs: dict[str, object] = {}
        delim_token = getattr(module, "delineator", ",")
        if delim_token and delim_token.strip():
            stripped = delim_token.strip()
            if stripped != "":
                load_kwargs["delimiter"] = stripped if stripped != " " else None
        delimiter_value = load_kwargs.get("delimiter")
        if delimiter_value is None:
            load_kwargs.pop("delimiter", None)
        data = np.loadtxt(output_path, **load_kwargs)
    finally:
        for name, value in cached.items():
            setattr(module, name, value)

    return data


def _run_sensmap_perl(tmp_path: Path, sources_path: Path, seed: int, name: str = "perl_test_sensmap", nsub: int = 1) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    perl_script = repo_root / "sensmap.pl"
    env = os.environ.copy()
    env["SENSMAP_SEED"] = str(seed)
    command = [
        "perl",
        str(perl_script),
        str(sources_path),
        name,
        str(nsub),
    ]
    subprocess.run(command, cwd=tmp_path, check=True, env=env)
    return tmp_path / name


def _load_sensmap_perl_sample(tmp_path: Path, sources_path: Path, field_number: int, seed: int) -> np.ndarray:
    output_dir = _run_sensmap_perl(tmp_path, sources_path, seed)
    output_file = output_dir / f"{output_dir.name}.planets.{field_number}.0"
    data = np.loadtxt(output_file)
    return data


def _load_sensmap_python_sample(
    tmp_path: Path,
    sources_path: Path,
    field_number: int,
    seed: int,
    *,
    rundes_override: str = "python_test_sensmap",
) -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.import_module("sensmap_draw_planet_arrays")
    module = importlib.reload(module)

    cached = {
        "rundes": module.rundes,
        "sources_file": module.sources_file,
        "overwrite_existing": module.overwrite_existing,
        "FIXED_BASE_SEED": module.FIXED_BASE_SEED,
        "header": module.header,
        "file_ext": module.file_ext,
        "OUTPUT_DIR": module.OUTPUT_DIR,
    }

    try:
        module.rundes = rundes_override
        module.sources_file = str(sources_path)
        module.overwrite_existing = True
        module.FIXED_BASE_SEED = seed
        module.header = False
        module.file_ext = ""

        output_dir = tmp_path / "python_sensmap_output"
        module.main(
            selected_fields=[field_number],
            max_subruns=1,
            processes=1,
            output_dir=output_dir,
        )

        output_file = output_dir / f"{module.rundes}.planets.{field_number}.0"
        data = np.loadtxt(output_file)
    finally:
        for name, value in cached.items():
            setattr(module, name, value)
        module.OUTPUT_DIR = cached["OUTPUT_DIR"]

    return data


def _assert_same_distribution(array_a: np.ndarray, array_b: np.ndarray, *, label: str) -> None:
    statistic = _two_sample_ks_statistic(array_a, array_b)
    threshold = _ks_critical_value(array_a.size, array_b.size)
    if statistic >= threshold:
        msg = (
            f"KS statistic for {label} distributions is {statistic:.4f} "
            f"(threshold {threshold:.4f}); distributions diverge"
        )
        raise AssertionError(msg)


def _save_overlay_plot(
    array_a: np.ndarray,
    array_b: np.ndarray,
    *,
    label: str,
    perl_kwargs: dict[str, object] | None = None,
    python_kwargs: dict[str, object] | None = None,
) -> None:
    """Persist an overlaid histogram comparison for visual inspection."""
    artifact_dir = Path(__file__).resolve().parents[1] / "tests" / "artifacts" / "distribution_alignment"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    label_slug = "".join(ch if ch.isalnum() else "_" for ch in label.lower())
    filename = artifact_dir / f"{label_slug}_comparison.png"

    combined = np.concatenate([array_a, array_b])
    if np.ptp(combined) == 0:
        bins = 10
    else:
        bins = np.linspace(combined.min(), combined.max(), 60)

    fig, ax = plt.subplots(figsize=(6, 4))
    perl_kwargs = {"histtype": "step", "label": "Perl", "color": "tab:blue", "linewidth": 2.5} | (perl_kwargs or {})
    python_kwargs = {"histtype": "step", "label": "Python", "color": "tab:orange", "linewidth": 1.5} | (python_kwargs or {})

    ax.hist(array_a, bins=bins, density=True, **perl_kwargs)
    ax.hist(array_b, bins=bins, density=True, **python_kwargs)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def test_uniform_draw_matches_perl_distribution(tmp_path):
    sample_size = 4096
    seed = 314159

    perl_sample = _load_perl_sample(tmp_path, sample_size, seed)
    python_sample = _load_python_sample(tmp_path, sample_size, seed)

    perl_mass_log = np.log10(perl_sample[:, 0] / 3.00374072e-6)
    python_mass_log = np.log10(python_sample[:, 0] / 3.00348959632e-6)
    _assert_same_distribution(perl_mass_log, python_mass_log, label="log10 mass (Earth units)")
    _save_overlay_plot(perl_mass_log, python_mass_log, label="log10 mass (Earth units)")

    perl_a_log = np.log10(perl_sample[:, 1])
    python_a_log = np.log10(python_sample[:, 1])
    _assert_same_distribution(perl_a_log, python_a_log, label="log10 semi-major axis")
    _save_overlay_plot(perl_a_log, python_a_log, label="log10 semi-major axis")

    _assert_same_distribution(perl_sample[:, 2], python_sample[:, 2], label="inclination")
    _assert_same_distribution(perl_sample[:, 3], python_sample[:, 3], label="orbital phase")
    _save_overlay_plot(perl_sample[:, 2], python_sample[:, 2], label="inclination")
    _save_overlay_plot(perl_sample[:, 3], python_sample[:, 3], label="orbital phase")


def test_output_layout_and_headers(tmp_path):
    """Ensure both samplers emit expected directories, filenames, and headers."""
    seed = 271828
    sample_size = 8

    _load_perl_sample(tmp_path, sample_size, seed)
    _load_python_sample(tmp_path, sample_size, seed)

    perl_dir = tmp_path / "pearl_test_uniform_draw"
    perl_file = perl_dir / "pearl_test_uniform_draw.planets.0.0"
    assert perl_dir.is_dir(), "Perl sampler did not create expected directory"
    assert perl_file.is_file(), "Perl sampler missing planet output file"

    with perl_file.open("r", encoding="ascii") as handle:
        first_line = handle.readline().strip()
        second_line = handle.readline().strip()

    assert not first_line.startswith("#"), "Perl sampler unexpectedly writes a header"
    perl_columns = len(first_line.split())
    assert perl_columns == 4, f"Perl sampler should emit four columns, saw {perl_columns}"
    assert len(second_line.split()) == 4, "Perl sampler second line must keep four columns"

    python_dir = tmp_path / "planets" / "python_uniform_test"
    python_file = python_dir / "python_uniform_test.planets.0.0"
    assert python_dir.is_dir(), "Python sampler did not create expected directory"
    assert python_file.is_file(), "Python sampler missing planet output file"

    with python_file.open("r", encoding="ascii") as handle:
        header_line = handle.readline().strip()
        second_line = handle.readline().strip()

    assert header_line.startswith("# mass (M_Sun), a (au), inc (deg), p (deg)"), (
        "Python sampler header did not match expected format"
    )
    python_columns = len([c for c in second_line.split(",") if c])
    assert python_columns == 4, f"Python sampler should emit four comma-delimited columns, saw {python_columns}"

    ascii_rundes = "python_uniform_ascii"
    _load_python_sample(
        tmp_path,
        sample_size,
        seed,
        delineator_override=" ",
        header_override=False,
        rundes_override=ascii_rundes,
    )
    ascii_dir = tmp_path / "planets" / ascii_rundes
    ascii_file = ascii_dir / f"{ascii_rundes}.planets.0.0"
    assert ascii_file.is_file(), "Whitespace-delimited run did not produce expected file"

    with ascii_file.open("r", encoding="ascii") as handle:
        ascii_first_line = handle.readline().strip()
    assert not ascii_first_line.startswith("#"), "Whitespace-delimited run unexpectedly wrote a header"
    assert "," not in ascii_first_line, "Whitespace-delimited run should not include commas"
    assert len(ascii_first_line.split()) == 4, "Whitespace-delimited run should emit four space-separated columns"


def test_sensmap_draw_matches_perl_distribution(tmp_path):
    seed = 424242
    sources_path = tmp_path / "sensmap.sources"
    sources_path.write_text("0 0.0 0.0 0 0 0 0\n", encoding="ascii")

    perl_sample = _load_sensmap_perl_sample(tmp_path, sources_path, field_number=0, seed=seed)
    python_sample = _load_sensmap_python_sample(tmp_path, sources_path, field_number=0, seed=seed)

    assert perl_sample.shape == python_sample.shape == (1105, 4)

    assert np.allclose(perl_sample[:, 0], python_sample[:, 0])
    assert np.allclose(perl_sample[:, 1], python_sample[:, 1])

    perl_mass_log = np.log10(perl_sample[:, 0] / 3.00374072e-6)
    python_mass_log = np.log10(python_sample[:, 0] / 3.00374072e-6)
    _save_overlay_plot(perl_mass_log, python_mass_log, label="sensmap log10 mass (Earth units)")

    perl_a_log = np.log10(perl_sample[:, 1])
    python_a_log = np.log10(python_sample[:, 1])
    _save_overlay_plot(perl_a_log, python_a_log, label="sensmap log10 semi-major axis")

    _assert_same_distribution(perl_sample[:, 2], python_sample[:, 2], label="sensmap inclination")
    _assert_same_distribution(perl_sample[:, 3], python_sample[:, 3], label="sensmap orbital phase")

    _save_overlay_plot(perl_sample[:, 2], python_sample[:, 2], label="sensmap inclination")
    _save_overlay_plot(perl_sample[:, 3], python_sample[:, 3], label="sensmap orbital phase")


def test_sensmap_output_layout(tmp_path):
    seed = 202311
    sources_path = tmp_path / "sensmap.sources"
    sources_path.write_text("0 0.0 0.0 0 0 0 0\n1 0.0 -0.2 0 0 0 0\n", encoding="ascii")

    perl_dir = _run_sensmap_perl(tmp_path, sources_path, seed, name="perl_layout_sensmap", nsub=1)
    perl_file = perl_dir / "perl_layout_sensmap.planets.1.0"
    assert perl_file.is_file(), "Perl sensmap run missing expected field output"

    perl_lines = perl_file.read_text(encoding="ascii").strip().splitlines()
    assert len(perl_lines) == 1105, "Perl sensmap file should contain grid * repeats rows"
    assert len(perl_lines[0].split()) == 4

    python_sample = _load_sensmap_python_sample(
        tmp_path,
        sources_path,
        field_number=1,
        seed=seed,
        rundes_override="python_layout_sensmap",
    )

    python_dir = tmp_path / "python_sensmap_output"
    python_file = python_dir / "python_layout_sensmap.planets.1.0"
    assert python_file.is_file(), "Python sensmap run missing expected field output"

    python_lines = python_file.read_text(encoding="ascii").strip().splitlines()
    assert len(python_lines) == 1105
    assert len(python_lines[0].split()) == 4

    assert not python_lines[0].startswith("#"), "Python sensmap should not prepend a header"
