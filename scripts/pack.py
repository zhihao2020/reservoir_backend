"""Nuitka standalone packer for the laboratory reconstruction CLI.

Compiles ``main.py`` to C++ and links ``reservoir.exe`` plus a folder of
DLLs (CPython, numpy, scipy, yaml). Not a one-file build. Not encryption.
Case YAML stays on disk and is passed as an argument.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_REL = Path("main.py")
OUTPUT_DIR_REL = Path("dist") / "nuitka"
SMOKE_CASE_REL = Path("examples") / "lab_cube" / "case.yaml"
EXE_STEM = "reservoir"
ZIP_NAME = "reservoir-win64.zip"
ZIP_ROOT_NAME = "reservoir"
SHARE_FILES = (
    ("examples/lab_cube/case.yaml", "share/lab_cube/case.yaml"),
    ("examples/lab_cube/probes.csv", "share/lab_cube/probes.csv"),
    ("examples/lab_cube/wells.csv", "share/lab_cube/wells.csv"),
    ("examples/lab_cube/observations.csv", "share/lab_cube/observations.csv"),
    ("examples/lab_cube/well_series.csv", "share/lab_cube/well_series.csv"),
    ("docs/cmg模型.md", "share/cmg模型.md"),
    ("docs/目标的四个程序.md", "share/目标的四个程序.md"),
)

MSVC_MISSING = (
    "Nuitka standalone needs MSVC (same ABI as official CPython/numpy/scipy wheels). "
    "Install Visual Studio Build Tools with the Desktop development with C++ workload, "
    "then re-run from a Developer Command Prompt, or put cl.exe on PATH. "
    "Do not use MinGW for this package."
)
NUITKA_MISSING = 'Nuitka is not installed. Run: python -m pip install -e ".[pack]"'
ENTRY_MISSING = "pack entry is missing: main.py (must call reservoir_backend.cli.main:main)"
SMOKE_MISSING_EXE = (
    "No packed exe at dist/nuitka/main.dist/reservoir.exe. Run: python scripts/pack.py"
)


def is_windows() -> bool:
    return sys.platform == "win32"


def exe_name(*, windows: bool | None = None) -> str:
    if windows is None:
        windows = is_windows()
    return f"{EXE_STEM}.exe" if windows else EXE_STEM


def entry_path(root: Path | None = None) -> Path:
    return (root or REPO_ROOT) / ENTRY_REL


def output_dir(root: Path | None = None) -> Path:
    return (root or REPO_ROOT) / OUTPUT_DIR_REL


def smoke_case_path(root: Path | None = None) -> Path:
    return (root or REPO_ROOT) / SMOKE_CASE_REL


def dist_folder(out: Path) -> Path:
    """Nuitka standalone dir for compiling ``main.py`` into ``out``."""
    return out / "main.dist"


def exe_path(out: Path | None = None, *, windows: bool | None = None) -> Path:
    return dist_folder(out or output_dir()) / exe_name(windows=windows)


def resolve_packed_exe(out: Path, *, windows: bool | None = None) -> Path | None:
    """Locate the standalone exe; Nuitka names the folder after the entry module."""
    name = exe_name(windows=windows)
    for path in (out / "main.dist" / name, out / "reservoir.dist" / name):
        if path.is_file():
            return path
    matches = sorted(p for p in out.glob(f"**/{name}") if p.is_file())
    return matches[0] if matches else None


def _vswhere() -> Path | None:
    for key in ("ProgramFiles(x86)", "ProgramFiles"):
        raw = os.environ.get(key)
        if not raw:
            continue
        candidate = Path(raw) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if candidate.is_file():
            return candidate
    which = shutil.which("vswhere")
    return Path(which) if which else None


def msvc_available() -> bool:
    if shutil.which("cl"):
        return True
    vswhere = _vswhere()
    if vswhere is None:
        return False
    proc = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool((proc.stdout or "").strip())


def nuitka_available() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "nuitka", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def require_msvc(*, windows: bool | None = None) -> None:
    if windows is None:
        windows = is_windows()
    if not windows:
        return
    if not msvc_available():
        raise SystemExit(MSVC_MISSING)


def require_nuitka() -> None:
    if not nuitka_available():
        raise SystemExit(NUITKA_MISSING)


def require_entry(root: Path | None = None) -> Path:
    path = entry_path(root)
    if not path.is_file():
        raise SystemExit(ENTRY_MISSING)
    return path


def share_readme_text() -> str:
    return (
        "实验室岩样场重建（Windows）\n"
        "\n"
        "请拷走整个文件夹，不要只复制 reservoir.exe。\n"
        "\n"
        "reservoir.exe share\\lab_cube\\case.yaml -o results\\lab_cube\n"
        "\n"
        "输入：几何、测点坐标与时序、注采压力与流量。\n"
        "省略 --output 时写入 results\\<案例名>。先看 fields.npz 和 summary.json。\n"
        "需求：share\\目标的四个程序.md ，试样：share\\cmg模型.md\n"
    )


def stage_user_share(dist: Path, *, root: Path | None = None) -> Path:
    """Copy the lab_cube starter case next to the packed exe."""
    root = root or REPO_ROOT
    dist = Path(dist)
    dist.mkdir(parents=True, exist_ok=True)
    for src_rel, dest_rel in SHARE_FILES:
        src = root / src_rel
        if not src.is_file():
            raise SystemExit(f"pack payload missing: {src}")
        dest = dist / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    readme = dist / "share" / "README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(share_readme_text(), encoding="utf-8")
    return dist / "share"


def zip_user_dist(dist: Path, zip_path: Path | None = None) -> Path:
    """Zip the standalone folder as reservoir/reservoir.exe + DLLs + share/."""
    import zipfile

    dist = Path(dist)
    if zip_path is None:
        zip_path = dist.parent / ZIP_NAME
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dist.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(dist).as_posix()
            zf.write(path, f"{ZIP_ROOT_NAME}/{rel}")
    return zip_path


def nuitka_command(
    *,
    root: Path | None = None,
    out: Path | None = None,
    windows: bool | None = None,
) -> list[str]:
    """Default Nuitka argv: standalone folder, never --onefile.

    Compiles ``reservoir_backend`` and copies numpy/scipy wheels as DLLs.
    Do not ``--include-package=numpy`` / ``scipy`` (that compiles their tests).
    """
    if windows is None:
        windows = is_windows()
    root = root or REPO_ROOT
    out = out or output_dir(root)
    jobs = os.cpu_count() or 1
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--show-progress",
        f"--output-dir={out}",
        f"--output-filename={exe_name(windows=windows)}",
        "--include-package=reservoir_backend",
        "--include-package=yaml",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=numpy.tests",
        "--nofollow-import-to=numpy.random.tests",
        "--nofollow-import-to=scipy.tests",
        "--nofollow-import-to=setuptools",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--noinclude-unittest-mode=nofollow",
        "--assume-yes-for-downloads",
        "--remove-output",
        f"--jobs={jobs}",
    ]
    if windows:
        cmd.append("--msvc=latest")
    cmd.append(str(entry_path(root)))
    return cmd


def run_pack(
    *,
    root: Path | None = None,
    out: Path | None = None,
) -> Path:
    root = root or REPO_ROOT
    require_entry(root)
    require_msvc()
    require_nuitka()
    out = out or output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    cmd = nuitka_command(root=root, out=out)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Nuitka failed with exit {proc.returncode}")
    packed = resolve_packed_exe(out) or exe_path(out)
    if not packed.is_file():
        raise SystemExit(f"Nuitka finished but {packed} is missing")
    stage_user_share(packed.parent, root=root)
    zipped = zip_user_dist(packed.parent, out.parent / ZIP_NAME)
    print(f"packed: {packed}", flush=True)
    print(f"zip: {zipped}", flush=True)
    return packed


def run_smoke(
    *,
    root: Path | None = None,
    exe: Path | None = None,
    case: Path | None = None,
    scratch: Path | None = None,
) -> Path:
    root = root or REPO_ROOT
    if exe is not None:
        packed = Path(exe)
    else:
        packed = resolve_packed_exe(output_dir(root)) or exe_path(output_dir(root))
    if not packed.is_file():
        raise SystemExit(SMOKE_MISSING_EXE)
    case_path = Path(case) if case is not None else smoke_case_path(root)
    if not case_path.is_file():
        raise SystemExit(f"smoke case missing: {case_path}")
    scratch = Path(scratch) if scratch is not None else root / "build" / "pack_smoke"
    out_dir = scratch / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(packed), str(case_path), "--output", str(out_dir)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"smoke failed with exit {proc.returncode}")
    written = out_dir / "summary.json"
    if not written.is_file():
        raise SystemExit(f"smoke did not write {written}")
    print(f"smoke ok: {packed}", flush=True)
    return packed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/pack.py",
        description=(
            "Nuitka --standalone: compile the CLI to C++ and link reservoir.exe "
            "plus a folder of DLLs. Compilation is not encryption. Not --onefile."
        ),
        epilog=(
            "YAML cases stay outside the binary; pass them as arguments. "
            "Ship the zip (whole reservoir/ folder), not the exe alone. "
            "Default install remains pip install -e ."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("smoke",),
        default=None,
            help="omit to pack; 'smoke' runs the packed exe on the lab_cube case",
    )
    parser.add_argument("--exe", type=Path, default=None, help="packed exe for smoke")
    parser.add_argument("--output-dir", type=Path, default=None, help="override dist/nuitka")
    parser.add_argument("--case", type=Path, default=None, help="YAML for smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        exe = args.exe
        if exe is None and args.output_dir is not None:
            exe = exe_path(args.output_dir)
        run_smoke(exe=exe, case=args.case)
        return 0
    run_pack(out=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
