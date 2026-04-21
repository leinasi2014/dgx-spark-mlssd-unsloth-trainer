from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import benchmark_env, common_context, execute_commands, finalize_summary, prepare_workspace, render_command


def run(
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
    commands = [render_command(str(bench_cfg["run_command"]), context)]
    env = benchmark_env(bench_cfg=bench_cfg, base_url=base_url)
    execution_log = benchmark_dir / "logs" / "run.log"
    executed = execute_commands(commands, cwd=repo_root, env=env, log_path=execution_log, dry_run=dry_run)
    metadata = {
        "benchmark": bench_cfg["name"],
        "display_name": bench_cfg.get("display_name"),
        "profile": bench_cfg["profile_name"],
        "repo_dir": str(repo_root),
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
