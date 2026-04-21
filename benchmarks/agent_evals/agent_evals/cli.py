from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .aggregate import aggregate_run
from .config import dump_json, load_config, project_paths, resolve_profile, validate_run_id
from .doctor import run_doctor
from .runners import RUNNERS
from .upstream import ensure_checkout, install_upstream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run external coding-agent benchmarks from one subproject.")
    parser.add_argument("--config", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Validate Linux tooling and env prerequisites.")

    fetch_parser = subparsers.add_parser("fetch", help="Clone pinned benchmark repos and install their Python envs.")
    fetch_parser.add_argument("--benchmark", default="all")
    fetch_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run a benchmark wrapper.")
    run_parser.add_argument("--benchmark", required=True)
    run_parser.add_argument("--profile", default=None)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--model-name", required=True)
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--predictions-path", default=None)
    run_parser.add_argument("--max-workers", type=int, default=None)
    run_parser.add_argument("--dry-run", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate one run_id across benchmark summaries.")
    aggregate_parser.add_argument("--run-id", required=True)
    return parser


def _benchmarks_to_process(cfg: dict[str, Any], benchmark: str) -> list[str]:
    if benchmark == "all":
        return sorted(cfg["benchmarks"].keys())
    return [benchmark]


def _run_fetch(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    layout = project_paths(cfg)
    results: list[dict[str, Any]] = []
    for benchmark in _benchmarks_to_process(cfg, args.benchmark):
        bench_cfg = resolve_profile(cfg, benchmark, None)
        repo_root = ensure_checkout(layout, bench_cfg, dry_run=args.dry_run)
        install_log = layout["cache_root"] / "logs" / f"{benchmark}.install.log"
        commands = install_upstream(repo_root, bench_cfg, log_path=install_log, dry_run=args.dry_run)
        results.append(
            {
                "benchmark": benchmark,
                "repo_dir": str(repo_root),
                "ref": bench_cfg["ref"],
                "install_commands": commands,
                "dry_run": args.dry_run,
            }
        )
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))


def _run_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    layout = project_paths(cfg)
    validate_run_id(args.run_id)
    bench_cfg = resolve_profile(cfg, args.benchmark, args.profile, max_workers=args.max_workers)
    benchmark_dir = layout["runs_root"] / args.run_id / args.benchmark
    if benchmark_dir.exists():
        raise FileExistsError(f"Benchmark output already exists: {benchmark_dir}")
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    runner = RUNNERS[args.benchmark]
    summary = runner(
        cfg=cfg,
        bench_cfg=bench_cfg,
        layout=layout,
        benchmark_dir=benchmark_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        base_url=args.base_url,
        max_workers=int(bench_cfg["max_workers"]),
        predictions_path=args.predictions_path,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _run_aggregate(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    layout = project_paths(cfg)
    validate_run_id(args.run_id)
    aggregate = aggregate_run(layout["runs_root"], args.run_id)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        print(json.dumps(run_doctor(cfg), indent=2, ensure_ascii=False))
        return
    if args.command == "fetch":
        _run_fetch(cfg, args)
        return
    if args.command == "run":
        _run_benchmark(cfg, args)
        return
    if args.command == "aggregate":
        _run_aggregate(cfg, args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
