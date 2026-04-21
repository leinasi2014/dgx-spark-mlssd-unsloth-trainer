from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBPROJECT = ROOT / "benchmarks" / "agent_evals"
if str(SUBPROJECT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT))

from agent_evals.aggregate import aggregate_run  # type: ignore  # noqa: E402
from agent_evals.config import ConfigError, load_config, project_paths, resolve_profile, validate_run_id  # type: ignore  # noqa: E402
from agent_evals.summary import infer_primary_metric  # type: ignore  # noqa: E402


class AgentEvalsTests(unittest.TestCase):
    def test_load_config_resolves_subproject_paths(self) -> None:
        cfg = load_config(SUBPROJECT / "configs" / "benchmarks.yaml")
        paths = project_paths(cfg)
        self.assertEqual(paths["root"], SUBPROJECT.resolve())
        self.assertTrue(str(paths["runs_root"]).endswith(str(Path("benchmarks") / "agent_evals" / "runs")))

    def test_validate_run_id_rejects_escape_like_values(self) -> None:
        with self.assertRaises(ConfigError):
            validate_run_id("../escape")

    def test_resolve_profile_merges_defaults_and_profile(self) -> None:
        cfg = load_config(SUBPROJECT / "configs" / "benchmarks.yaml")
        bench_cfg = resolve_profile(cfg, "mcpmark", "smoke", max_workers=3)
        self.assertEqual(bench_cfg["profile_name"], "smoke")
        self.assertEqual(bench_cfg["max_workers"], 3)
        self.assertIn("run_command", bench_cfg)

    def test_infer_primary_metric_handles_resolved_ratio(self) -> None:
        name, value = infer_primary_metric({"resolved_instances": 12, "total_instances": 20})
        self.assertEqual(name, "resolved_instances/total_instances")
        self.assertEqual(value, 0.6)

    def test_aggregate_run_collects_benchmark_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir)
            run_dir = runs_root / "demo"
            for name in ("terminal_bench", "mcpmark"):
                bench_dir = run_dir / name
                bench_dir.mkdir(parents=True)
                (bench_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "benchmark": name,
                            "display_name": name,
                            "profile": "smoke",
                            "status": "succeeded",
                            "primary_metric_name": "pass_rate",
                            "primary_metric_value": 0.5,
                        }
                    ),
                    encoding="utf-8",
                )
            aggregate = aggregate_run(runs_root, "demo")
            self.assertEqual(len(aggregate["benchmarks"]), 2)
            self.assertTrue((run_dir / "aggregate.json").exists())
            self.assertTrue((run_dir / "aggregate.md").exists())


if __name__ == "__main__":
    unittest.main()
