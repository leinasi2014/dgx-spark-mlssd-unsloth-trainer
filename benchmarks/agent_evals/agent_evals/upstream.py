from __future__ import annotations

import os
import subprocess
import venv
from pathlib import Path

from .config import ConfigError
from .shell import current_python, quote, run_shell


def repo_dir(layout: dict[str, Path], bench_cfg: dict[str, object]) -> Path:
    repo_subdir = str(bench_cfg.get("repo_subdir") or bench_cfg["name"])
    return layout["external_root"] / repo_subdir


def venv_dir(repo_root: Path) -> Path:
    return repo_root / ".venv"


def venv_python(repo_root: Path) -> Path:
    subpath = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return venv_dir(repo_root) / subpath


def venv_bin(repo_root: Path) -> Path:
    subpath = Path("Scripts") if os.name == "nt" else Path("bin")
    return venv_dir(repo_root) / subpath


def ensure_checkout(layout: dict[str, Path], bench_cfg: dict[str, object], *, dry_run: bool = False) -> Path:
    root = repo_dir(layout, bench_cfg)
    root.parent.mkdir(parents=True, exist_ok=True)
    repo_url = str(bench_cfg["repo_url"])
    ref = str(bench_cfg["ref"])

    if dry_run:
        return root
    if not root.exists():
        subprocess.run(["git", "clone", repo_url, str(root)], check=True)
    else:
        subprocess.run(["git", "-C", str(root), "fetch", "--tags", "origin"], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", ref], check=True)
    return root


def ensure_upstream_venv(repo_root: Path) -> Path:
    python_path = venv_python(repo_root)
    if python_path.exists():
        return python_path
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(str(venv_dir(repo_root)))
    if not python_path.exists():
        raise ConfigError(f"Failed to create virtualenv for upstream repo: {repo_root}")
    return python_path


def install_upstream(
    repo_root: Path,
    bench_cfg: dict[str, object],
    *,
    log_path: Path,
    dry_run: bool,
) -> list[str]:
    python_path = venv_python(repo_root) if dry_run else ensure_upstream_venv(repo_root)
    env = os.environ.copy()
    context = {
        "python": quote(current_python()),
        "venv_python": quote(python_path),
        "venv_bin": quote(venv_bin(repo_root)),
        "repo_dir": quote(repo_root),
    }
    commands = [str(cmd).format(**context) for cmd in bench_cfg.get("install_commands", [])]
    for command in commands:
        exit_code = run_shell(command, cwd=repo_root, env=env, log_path=log_path, dry_run=dry_run)
        if exit_code != 0:
            raise ShellInstallError(f"Install command failed for {bench_cfg['name']}: {command}")
    return commands


class ShellInstallError(RuntimeError):
    pass
