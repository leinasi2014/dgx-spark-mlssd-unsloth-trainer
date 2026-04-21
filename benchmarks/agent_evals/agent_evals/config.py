from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    pass


def subproject_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return subproject_root() / "configs" / "benchmarks.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_config_path()
    raw = load_yaml(cfg_path)
    version = int(raw.get("version", 1))
    if version != 1:
        raise ConfigError(f"Unsupported config version: {version}")

    root = cfg_path.resolve().parent.parent
    cfg = copy.deepcopy(raw)
    cfg["_config_path"] = str(cfg_path.resolve())
    cfg["_root"] = str(root)

    paths = cfg.setdefault("paths", {})
    for key in ("external_root", "cache_root", "runs_root"):
        paths[key] = _resolve_path(root, str(paths.get(key, key.replace("_root", ""))))

    benchmarks = cfg.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ConfigError("Config must define non-empty benchmarks mapping.")
    return cfg


def project_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {
        "root": Path(cfg["_root"]),
        "external_root": Path(cfg["paths"]["external_root"]),
        "cache_root": Path(cfg["paths"]["cache_root"]),
        "runs_root": Path(cfg["paths"]["runs_root"]),
    }


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise ConfigError(
            "run_id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ and stay within the benchmark workspace."
        )
    return run_id


def get_benchmark_config(cfg: dict[str, Any], benchmark: str) -> dict[str, Any]:
    benchmarks = cfg["benchmarks"]
    if benchmark not in benchmarks:
        raise ConfigError(f"Unknown benchmark: {benchmark}")
    bench_cfg = copy.deepcopy(benchmarks[benchmark])
    bench_cfg["name"] = benchmark
    return bench_cfg


def resolve_profile(
    cfg: dict[str, Any],
    benchmark: str,
    profile: str | None,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    bench_cfg = get_benchmark_config(cfg, benchmark)
    profiles = bench_cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError(f"Benchmark {benchmark} is missing profile definitions.")

    profile_name = profile or bench_cfg.get("default_profile") or cfg.get("default_profile")
    if profile_name not in profiles:
        raise ConfigError(f"Unknown profile for {benchmark}: {profile_name}")

    merged = _merge_dicts(bench_cfg, profiles[profile_name])
    merged["profile_name"] = profile_name

    defaults = cfg.get("defaults", {})
    merged.setdefault("max_workers", max_workers or defaults.get("max_workers", 4))
    if max_workers is not None:
        merged["max_workers"] = max_workers

    merged.setdefault("env", {})
    merged.setdefault("install_commands", [])
    merged.setdefault("post_run_commands", [])
    merged.setdefault("summary_candidates", [])
    merged.setdefault("requires", {})
    merged.setdefault("notes", "")
    merged.setdefault("openai_api_key_env", defaults.get("openai_api_key_env", "OPENAI_API_KEY"))
    merged.setdefault("openai_base_url_env", defaults.get("openai_base_url_env", "OPENAI_BASE_URL"))
    return merged


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def effective_api_key(api_key_env: str) -> str | None:
    value = os.environ.get(api_key_env)
    return value.strip() if isinstance(value, str) and value.strip() else None
