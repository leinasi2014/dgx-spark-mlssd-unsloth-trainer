from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import dump_json, effective_api_key
from ..shell import current_python, quote, run_shell
from ..summary import copy_candidates, find_candidate_files, summarize_candidate_files
from ..upstream import ensure_checkout, install_upstream, repo_dir, venv_bin, venv_python


def render_command(template: str, context: dict[str, object]) -> str:
    return template.format(**{key: str(value) for key, value in context.items()})


def common_context(
    *,
    cfg: dict[str, Any],
    bench_cfg: dict[str, Any],
    layout: dict[str, Path],
    run_id: str,
    benchmark_dir: Path,
    model_name: str,
    base_url: str,
    max_workers: int,
    predictions_path: str | None,
) -> dict[str, object]:
    repo_root = repo_dir(layout, bench_cfg)
    env_file = benchmark_dir / ".env.resolved"
    upstream_run_id = f"{run_id}-{bench_cfg['name']}-{bench_cfg['profile_name']}"
    return {
        "python": current_python(),
        "repo_dir": repo_root,
        "repo_name": bench_cfg["name"],
        "venv_python": venv_python(repo_root),
        "venv_bin": venv_bin(repo_root),
        "output_dir": benchmark_dir,
        "raw_dir": benchmark_dir / "raw",
        "log_dir": benchmark_dir / "logs",
        "model_name": model_name,
        "base_url": base_url,
        "max_workers": max_workers,
        "run_id": run_id,
        "upstream_run_id": upstream_run_id,
        "predictions_path": predictions_path or "",
        "api_key_env": bench_cfg["openai_api_key_env"],
        "base_url_env": bench_cfg["openai_base_url_env"],
        "env_file": env_file,
        "cache_root": layout["cache_root"],
    }


def prepare_workspace(
    *,
    cfg: dict[str, Any],
    bench_cfg: dict[str, Any],
    layout: dict[str, Path],
    benchmark_dir: Path,
    dry_run: bool,
) -> tuple[Path, list[str]]:
    repo_root = ensure_checkout(layout, bench_cfg, dry_run=dry_run)
    install_log = benchmark_dir / "logs" / "install.log"
    install_commands = install_upstream(repo_root, bench_cfg, log_path=install_log, dry_run=dry_run)
    return repo_root, install_commands


def benchmark_env(
    *,
    bench_cfg: dict[str, Any],
    base_url: str,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env[str(bench_cfg["openai_base_url_env"])] = base_url
    api_key = effective_api_key(str(bench_cfg["openai_api_key_env"]))
    if api_key:
        env[str(bench_cfg["openai_api_key_env"])] = api_key
    for key, value in (bench_cfg.get("env") or {}).items():
        env[str(key)] = str(value)
    if extra_env:
        env.update(extra_env)
    return env


def execute_commands(
    commands: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> list[dict[str, object]]:
    executed: list[dict[str, object]] = []
    for command in commands:
        exit_code = run_shell(command, cwd=cwd, env=env, log_path=log_path, dry_run=dry_run)
        executed.append({"command": command, "exit_code": exit_code})
        if exit_code != 0:
            raise RuntimeError(f"Command failed with exit={exit_code}: {command}")
    return executed


def finalize_summary(
    *,
    bench_cfg: dict[str, Any],
    benchmark_dir: Path,
    raw_search_roots: list[Path],
    metadata: dict[str, Any],
    render_context: dict[str, object] | None = None,
) -> dict[str, Any]:
    patterns = []
    for pattern in list(bench_cfg.get("summary_candidates") or []):
        if render_context:
            patterns.append(str(pattern).format(**{key: str(value) for key, value in render_context.items()}))
        else:
            patterns.append(str(pattern))
    candidates = find_candidate_files(raw_search_roots, patterns)
    raw_dir = benchmark_dir / "raw"
    copied_files = copy_candidates(candidates, raw_dir)
    metric_name, metric_value, metric_source = summarize_candidate_files(candidates)
    summary = {
        "benchmark": bench_cfg["name"],
        "display_name": bench_cfg.get("display_name", bench_cfg["name"]),
        "profile": bench_cfg["profile_name"],
        "status": "succeeded",
        "primary_metric_name": metric_name,
        "primary_metric_value": metric_value,
        "metric_source": metric_source,
        "raw_artifact_paths": copied_files,
        "notes": bench_cfg.get("notes", ""),
        "upstream_ref": bench_cfg["ref"],
        "metadata_path": str(benchmark_dir / "metadata.json"),
    }
    dump_json(metadata, benchmark_dir / "metadata.json")
    dump_json(summary, benchmark_dir / "summary.json")
    return summary
