from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


class ShellError(RuntimeError):
    pass


def quote(value: object) -> str:
    return shlex.quote(str(value))


def shell_join(items: list[str]) -> str:
    return " ".join(quote(item) for item in items)


def shell_executable() -> list[str]:
    if os.name == "nt":
        return ["powershell", "-NoProfile", "-Command"]
    return ["bash", "-lc"]


def run_shell(command: str, *, cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {command}\n")
        if dry_run:
            handle.write("[dry-run] command not executed.\n")
            return 0
        process = subprocess.run(
            shell_executable() + [command],
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"[exit={process.returncode}]\n")
    return process.returncode


def current_python() -> str:
    return sys.executable
