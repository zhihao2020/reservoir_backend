"""Compile the ``src`` package to a Linux ``.so`` with Nuitka.

Only ``src/`` is compiled. numpy / scipy / PyYAML stay as pip dependencies.
Must run on Linux (GitHub Actions ``ubuntu-latest``, or ``./build_linux.sh``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BUILD_DIR = REPO_ROOT / "build" / "src_so"
DIST_DIR = REPO_ROOT / "dist" / "linux"


def library_version() -> str:
    ns: dict[str, object] = {}
    exec((SRC_DIR / "version.py").read_text(encoding="utf-8"), ns)
    return str(ns["__version__"])


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def zip_name(version: str | None = None) -> str:
    return f"reservoir-backend-{version or library_version()}-linux-so.zip"


def require_linux() -> None:
    if sys.platform != "linux":
        raise SystemExit(
            "src.so is a Linux extension. Run this on Linux or via "
            ".github/workflows/pack-linux.yml (GitHub Actions)."
        )


def require_nuitka() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "nuitka", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("Nuitka is missing. pip install -e '.[pack]'")


def nuitka_command(*, out: Path) -> list[str]:
    jobs = os.cpu_count() or 1
    return [
        sys.executable,
        "-m",
        "nuitka",
        "--module",
        "src",
        "--include-package=src",
        "--include-package=src.cli",
        "--include-package=src.core",
        "--include-package=src.programs",
        "--include-module=src.version",
        "--follow-import-to=src",
        "--nofollow-import-to=numpy",
        "--nofollow-import-to=scipy",
        "--nofollow-import-to=yaml",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=pytest",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--noinclude-unittest-mode=nofollow",
        "--assume-yes-for-downloads",
        "--remove-output",
        f"--jobs={jobs}",
        f"--output-dir={out}",
    ]


def find_built_so(out: Path) -> Path:
    matches = sorted(out.glob("src*.so")) + sorted(out.glob("src.so"))
    if not matches:
        raise SystemExit(f"Nuitka finished but no src*.so under {out}")
    return matches[0]


def dist_readme(version: str) -> str:
    tag = python_tag()
    return (
        f"reservoir-backend {version}  Linux module (src.so)\n"
        "\n"
        f"library: reservoir-backend\n"
        f"version: {version}\n"
        f"python:  CPython {sys.version_info.major}.{sys.version_info.minor} ({tag})\n"
        "abi:     x86_64-linux-gnu\n"
        "\n"
        "Deliverable is a single src.so. numpy / scipy / PyYAML stay as pip packages.\n"
        "Do not keep a src/ folder next to src.so — a package directory would win.\n"
        "\n"
        "  pip install numpy scipy pyyaml\n"
        "  PYTHONPATH=/path/containing/src.so python3 -c 'import src; print(src.__version__)'\n"
        "  PYTHONPATH=/path/containing/src.so python3 -m src --version\n"
        "  PYTHONPATH=/path/containing/src.so python3 -m src case.yaml --tcp-port 9000 \\\n"
        "      --ip 127.0.0.1 --lab-port 9001 --field-port 9002\n"
    )


def stage_dist(so_path: Path, dest: Path, *, version: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    extra = dest / "reservoir_backend"
    if extra.exists():
        shutil.rmtree(extra)
    staged_so = dest / "src.so"
    shutil.copy2(so_path, staged_so)
    (dest / "VERSION").write_text(version + "\n", encoding="utf-8")
    (dest / "README.txt").write_text(dist_readme(version), encoding="utf-8")
    return staged_so


def zip_dist(dest: Path, *, version: str) -> Path:
    zip_path = dest / zip_name(version)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dest.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            zf.write(path, path.relative_to(dest).as_posix())
    return zip_path


def smoke_so(so_path: Path, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    isolated = scratch / "import_src"
    if isolated.exists():
        shutil.rmtree(isolated)
    isolated.mkdir(parents=True)
    shutil.copy2(so_path, isolated / "src.so")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src; print('reservoir-backend', src.__version__); assert hasattr(src, 'run_pipeline')",
        ],
        cwd=str(isolated),
        env={**os.environ, "PYTHONPATH": str(isolated)},
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("smoke import of src.so failed")
    print(f"smoke ok: {isolated / 'src.so'}", flush=True)


def run_build(*, skip_smoke: bool = False) -> Path:
    require_linux()
    require_nuitka()
    if not (SRC_DIR / "__init__.py").is_file():
        raise SystemExit(f"missing {SRC_DIR}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = nuitka_command(out=BUILD_DIR)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Nuitka failed with exit {proc.returncode}")
    so_path = find_built_so(BUILD_DIR)
    version = library_version()
    staged = stage_dist(so_path, DIST_DIR, version=version)
    zipped = zip_dist(DIST_DIR, version=version)
    print(f"library: reservoir-backend {version}", flush=True)
    print(f"so: {staged}", flush=True)
    print(f"zip: {zipped}", flush=True)
    if not skip_smoke:
        smoke_so(staged, REPO_ROOT / "build" / "src_so_smoke")
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_src_so.py",
        description="Compile only src/ to a Linux src.so (Nuitka --module).",
    )
    parser.add_argument("--print-command", action="store_true", help="print Nuitka argv and exit")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.print_command:
        print(" ".join(nuitka_command(out=BUILD_DIR)))
        return 0
    run_build(skip_smoke=bool(args.skip_smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
