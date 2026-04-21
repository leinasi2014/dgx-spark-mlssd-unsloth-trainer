from __future__ import annotations

import os
import platform
import shutil
from typing import Any


def run_doctor(cfg: dict[str, Any]) -> dict[str, Any]:
    benchmarks = cfg["benchmarks"]
    required_commands = {"git"}
    for bench_cfg in benchmarks.values():
        for command in bench_cfg.get("requires", {}).get("commands", []):
            required_commands.add(str(command))
    checks = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "commands": {command: bool(shutil.which(command)) for command in sorted(required_commands)},
        "openai_api_key_present": bool(os.environ.get(cfg.get("defaults", {}).get("openai_api_key_env", "OPENAI_API_KEY"))),
    }
    checks["ok"] = checks["platform"] == "Linux" and all(checks["commands"].values())
    return checks
