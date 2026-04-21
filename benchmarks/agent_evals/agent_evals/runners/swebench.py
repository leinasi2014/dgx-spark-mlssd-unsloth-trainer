from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from ..config import ConfigError
from .common import benchmark_env, common_context, execute_commands, finalize_summary, prepare_workspace, render_command


def _run_swebench_like(
    *,
    cfg: dict[str, Any],
    bench_cfg: dict[str, Any],
    layout: dict[str, Path],
    benchmark_dir: Path,
    run_id: str,
    model_name: str,
    base_url: str,
    max_workers: int,
    predictions_path: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if not predictions_path:
        raise ConfigError(
            f"{bench_cfg['display_name']} expects --predictions-path pointing to a JSON/JSONL patch file."
        )
    repo_root, install_commands = prepare_workspace(
        cfg=cfg,
        bench_cfg=bench_cfg,
        layout=layout,
        benchmark_dir=benchmark_dir,
        dry_run=dry_run,
    )
    context = common_context(
        cfg=cfg,
        bench_cfg=bench_cfg,
        layout=layout,
        run_id=run_id,
        benchmark_dir=benchmark_dir,
        model_name=model_name,
        base_url=base_url,
        max_workers=max_workers,
        predictions_path=predictions_path,
    )
    commands = [render_command(str(bench_cfg["evaluation_command"]), context)]
    env = benchmark_env(bench_cfg=bench_cfg, base_url=base_url)
    execution_log = benchmark_dir / "logs" / "run.log"
    executed = execute_commands(commands, cwd=repo_root, env=env, log_path=execution_log, dry_run=dry_run)
    metadata = {
        "benchmark": bench_cfg["name"],
        "display_name": bench_cfg.get("display_name"),
        "profile": bench_cfg["profile_name"],
        "repo_dir": str(repo_root),
        "predictions_path": predictions_path,
        "model_name": model_name,
        "base_url": base_url,
        "install_commands": install_commands,
        "run_commands": executed,
    }
    return finalize_summary(
        bench_cfg=bench_cfg,
        benchmark_dir=benchmark_dir,
        raw_search_roots=[repo_root, benchmark_dir],
        metadata=metadata,
        render_context=context,
    )


def run_verified(**kwargs: Any) -> dict[str, Any]:
    return _run_swebench_like(**kwargs)


def run_multilingual(
    *,
    cfg: dict[str, Any],
    bench_cfg: dict[str, Any],
    layout: dict[str, Path],
    benchmark_dir: Path,
    run_id: str,
    model_name: str,
    base_url: str,
    max_workers: int,
    predictions_path: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if not predictions_path:
        raise ConfigError(
            f"{bench_cfg['display_name']} expects --predictions-path pointing to a JSONL patch file."
        )
    repo_root, install_commands = prepare_workspace(
        cfg=cfg,
        bench_cfg=bench_cfg,
        layout=layout,
        benchmark_dir=benchmark_dir,
        dry_run=dry_run,
    )
    context = common_context(
        cfg=cfg,
        bench_cfg=bench_cfg,
        layout=layout,
        run_id=run_id,
        benchmark_dir=benchmark_dir,
        model_name=model_name,
        base_url=base_url,
        max_workers=max_workers,
        predictions_path=predictions_path,
    )
    config_payload = {
        "mode": "evaluation",
        "workdir": str(benchmark_dir / "workdir"),
        "patch_files": [predictions_path],
        "dataset_files": [str(_resolve_dataset_file(bench_cfg["dataset_file"], benchmark_dir, dry_run=dry_run))],
        "force_build": False,
        "output_dir": str(benchmark_dir / "raw" / "evaluation_output"),
        "specifics": bench_cfg.get("specifics", []),
        "skips": bench_cfg.get("skips", []),
        "repo_dir": str(benchmark_dir / "raw" / "repos"),
        "need_clone": False,
        "global_env": [],
        "clear_env": True,
        "stop_on_error": True,
        "max_workers": max_workers,
        "max_workers_build_image": max_workers,
        "max_workers_run_instance": max_workers,
        "log_dir": str(benchmark_dir / "logs"),
        "log_level": "INFO",
    }
    multi_cfg_path = benchmark_dir / "raw" / "multi_swe_eval_config.json"
    multi_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    multi_cfg_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    context["evaluation_config_path"] = multi_cfg_path
    commands = [render_command(str(bench_cfg["evaluation_command"]), context)]
    env = benchmark_env(bench_cfg=bench_cfg, base_url=base_url)
    execution_log = benchmark_dir / "logs" / "run.log"
    executed = execute_commands(commands, cwd=repo_root, env=env, log_path=execution_log, dry_run=dry_run)
    metadata = {
        "benchmark": bench_cfg["name"],
        "display_name": bench_cfg.get("display_name"),
        "profile": bench_cfg["profile_name"],
        "repo_dir": str(repo_root),
        "predictions_path": predictions_path,
        "dataset_file": str(bench_cfg["dataset_file"]),
        "model_name": model_name,
        "base_url": base_url,
        "install_commands": install_commands,
        "run_commands": executed,
    }
    return finalize_summary(
        bench_cfg=bench_cfg,
        benchmark_dir=benchmark_dir,
        raw_search_roots=[benchmark_dir, repo_root],
        metadata=metadata,
        render_context=context,
    )


def _resolve_dataset_file(dataset_file: object, benchmark_dir: Path, *, dry_run: bool) -> Path:
    raw_value = str(dataset_file)
    if raw_value.startswith("http://") or raw_value.startswith("https://"):
        target = benchmark_dir / "raw" / "datasets" / Path(raw_value).name
        if dry_run:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve(raw_value, target)
        return target
    return Path(raw_value)
