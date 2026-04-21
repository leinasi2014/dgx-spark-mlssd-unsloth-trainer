from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def install_stubs() -> None:
    if 'datasets' not in sys.modules:
        datasets = types.ModuleType('datasets')

        class Dataset:
            @staticmethod
            def from_list(rows):
                return rows

        datasets.Dataset = Dataset
        datasets.load_dataset = lambda *args, **kwargs: []
        sys.modules['datasets'] = datasets

    if 'peft' not in sys.modules:
        peft = types.ModuleType('peft')

        class PeftModel:
            @staticmethod
            def from_pretrained(model, path, is_trainable=False):
                return model

        peft.PeftModel = PeftModel
        sys.modules['peft'] = peft

    if 'transformers' not in sys.modules:
        transformers = types.ModuleType('transformers')
        transformers.set_seed = lambda seed: None

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        transformers.AutoTokenizer = AutoTokenizer
        sys.modules['transformers'] = transformers

    if 'trl' not in sys.modules:
        trl = types.ModuleType('trl')

        class SFTConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class SFTTrainer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        trl.SFTConfig = SFTConfig
        trl.SFTTrainer = SFTTrainer
        sys.modules['trl'] = trl

    if 'torch' not in sys.modules:
        torch = types.ModuleType('torch')

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch.cuda = _Cuda()
        torch.no_grad = lambda: _NoGrad()
        sys.modules['torch'] = torch

    if 'jinja2' not in sys.modules:
        jinja2 = types.ModuleType('jinja2')

        class Template:
            def __init__(self, text: str):
                self.text = text

            def render(self, **kwargs):
                rendered = self.text
                for key, value in kwargs.items():
                    rendered = rendered.replace(f'{{{{ {key} }}}}', str(value))
                return rendered

        jinja2.Template = Template
        sys.modules['jinja2'] = jinja2

    if 'numpy' not in sys.modules:
        numpy = types.ModuleType('numpy')

        class ndarray(list):
            def __gt__(self, other):
                return ndarray([item > other for item in self])

            def __ge__(self, other):
                return ndarray([item >= other for item in self])

            def __rtruediv__(self, other):
                return ndarray([other / item for item in self])

            def __rsub__(self, other):
                return ndarray([other - item for item in self])

            def all(self):
                return all(self)

            def mean(self):
                return sum(self) / len(self) if self else 0.0

            def tolist(self):
                return list(self)

        def array(values):
            return ndarray(list(values))

        def arange(start, stop=None):
            if stop is None:
                start, stop = 0, start
            return ndarray(list(range(start, stop)))

        def prod(values):
            result = 1.0
            for item in values:
                result *= item
            return result

        numpy.ndarray = ndarray
        numpy.array = array
        numpy.arange = arange
        numpy.prod = prod
        numpy.all = lambda values: all(values)
        sys.modules['numpy'] = numpy


install_stubs()

import build_skill0_dataset  # type: ignore  # noqa: E402
import build_mixed_dataset  # type: ignore  # noqa: E402
import common  # type: ignore  # noqa: E402
import evaluate_livecodebench  # type: ignore  # noqa: E402
import evaluate_codegen  # type: ignore  # noqa: E402
import generate_ssd_local  # type: ignore  # noqa: E402
import livecodebench_utils  # type: ignore  # noqa: E402
import prepare_ssd_data  # type: ignore  # noqa: E402
import recover_after_squeeze  # type: ignore  # noqa: E402
import run_training_plan  # type: ignore  # noqa: E402
import source_adapters  # type: ignore  # noqa: E402
import train_unsloth_lora  # type: ignore  # noqa: E402


class RegressionTests(unittest.TestCase):
    def test_load_config_normalizes_all_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'configs').mkdir()
            cfg_path = root / 'configs' / 'train.yaml'
            cfg_path.write_text(
                '\n'.join([
                    'run_name: demo',
                    'paths:',
                    '  project_root: ..',
                    '  output_root: runs',
                    '  skill_task_dataset: data/skill0/tasks.jsonl',
                    '  ssd_train_jsonl: ${paths.output_root}/${run_name}/ssd_train.jsonl',
                    '  skill0_train_jsonl: ${paths.output_root}/${run_name}/skill0_train.jsonl',
                    '  mixed_train_jsonl: ${paths.output_root}/${run_name}/mixed_train.jsonl',
                ]),
                encoding='utf-8',
            )
            cfg = common.load_config(str(cfg_path))
            self.assertEqual(cfg['paths']['project_root'], str(root))
            self.assertEqual(cfg['paths']['skill_task_dataset'], str(root / 'data' / 'skill0' / 'tasks.jsonl'))
            self.assertEqual(cfg['paths']['skill0_train_jsonl'], str(root / 'runs' / 'demo' / 'skill0_train.jsonl'))
            self.assertEqual(cfg['paths']['mixed_train_jsonl'], str(root / 'runs' / 'demo' / 'mixed_train.jsonl'))

    def test_load_config_rejects_unsafe_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'configs').mkdir()
            cfg_path = root / 'configs' / 'train.yaml'
            cfg_path.write_text('run_name: ../escape\npaths:\n  project_root: ..\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                common.load_config(str(cfg_path))

    def test_load_config_fails_on_missing_template_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'configs').mkdir()
            cfg_path = root / 'configs' / 'train.yaml'
            cfg_path.write_text(
                '\n'.join([
                    'run_name: demo',
                    'paths:',
                    '  project_root: ..',
                    '  eval_dataset: ${paths.missing}/eval.jsonl',
                ]),
                encoding='utf-8',
            )
            with self.assertRaises(KeyError):
                common.load_config(str(cfg_path))

    def test_load_config_rejects_drive_ambiguous_rooted_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'configs').mkdir()
            cfg_path = root / 'configs' / 'train.yaml'
            cfg_path.write_text(
                '\n'.join([
                    'run_name: demo',
                    'paths:',
                    '  project_root: ..',
                    '  output_root: /outside-runs',
                ]),
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                common.load_config(str(cfg_path))

    def test_load_config_normalizes_additional_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'configs').mkdir()
            cfg_path = root / 'configs' / 'train.yaml'
            cfg_path.write_text(
                '\n'.join([
                    'run_name: demo',
                    'paths:',
                    '  project_root: ..',
                    '  eval_dataset: data/eval/code_eval.jsonl',
                    'ssd:',
                    '  templates:',
                    '    template_root: ${paths.project_root}/scripts/ml_ssd_templates',
                    'evaluation:',
                    '  local_smoke:',
                    '    dataset_path: ${paths.eval_dataset}',
                ]),
                encoding='utf-8',
            )
            cfg = common.load_config(str(cfg_path))
            self.assertEqual(cfg['ssd']['templates']['template_root'], str(root / 'scripts' / 'ml_ssd_templates'))
            self.assertEqual(cfg['evaluation']['local_smoke']['dataset_path'], str(root / 'data' / 'eval' / 'code_eval.jsonl'))

    def test_response_only_fails_closed_when_markers_missing(self) -> None:
        trainer = object()

        class Tokenizer:
            chat_template = '<|im_start|>system\n'

        cfg = {
            'training': {
                'response_only': True,
                'response_markers': {
                    'instruction_part': '<|im_start|>user\n',
                    'response_part': '<|im_start|>assistant\n',
                },
            },
        }
        with self.assertRaises(RuntimeError):
            common.maybe_enable_response_only(trainer, Tokenizer(), cfg)

    def test_prepare_problem_rows_renders_upstream_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            templates_dir = root / 'templates'
            templates_dir.mkdir()
            (templates_dir / 'self_distillation_prompt_stdin.j2').write_text('STDIN {{ question }}', encoding='utf-8')
            (templates_dir / 'self_distillation_prompt_function.j2').write_text('FUNC {{ starter_code }} // {{ question }}', encoding='utf-8')
            source_cfg = {
                'family': 'problem_code',
                'adapter': 'rstar_coder',
                'dataset': {'name': 'demo', 'config': 'seed_sft', 'split': 'train'},
            }
            original = source_adapters.load_hf_rows
            source_adapters.load_hf_rows = lambda cfg, limit=0: [  # type: ignore[assignment]
                {'question_id': 'q1', 'question': 'Solve me', 'starter_code': 'def solve():\n    pass'},
                {'question_id': 'q2', 'question': 'Use stdin', 'starter_code': ''},
            ]
            try:
                rows = source_adapters.format_problem_code_rows('rstar_coder_seed_sft', source_cfg, templates_dir)
            finally:
                source_adapters.load_hf_rows = original  # type: ignore[assignment]
            self.assertEqual(rows[0]['problem_type'], 'function')
            self.assertIn('FUNC', rows[0]['prompt'])
            self.assertEqual(rows[1]['problem_type'], 'stdin')
            self.assertIn('STDIN', rows[1]['prompt'])

    def test_build_generated_train_rows_supports_legacy_messages_only_raw_rows(self) -> None:
        rows = prepare_ssd_data.build_generated_train_rows([
            {
                'prompt_id': 'legacy-1',
                'messages': [{'role': 'user', 'content': 'Write a function'}],
                'raw_outputs': ['def solve():\n    return 1'],
            },
        ])
        self.assertEqual(rows[0]['messages'][0]['content'], 'Write a function')

    def test_build_generated_train_rows_rejects_missing_prompt_payload(self) -> None:
        with self.assertRaises(ValueError):
            prepare_ssd_data.build_generated_train_rows([
                {'prompt_id': 'bad-row', 'raw_outputs': ['def solve():\n    return 1']},
            ])

    def test_get_source_config_rejects_disabled_source(self) -> None:
        cfg = {
            'data_sources': {
                'coderforge_preview': {
                    'enabled': False,
                    'family': 'agent_trajectory',
                },
            },
        }
        with self.assertRaises(ValueError):
            source_adapters.get_source_config(cfg, 'coderforge_preview')

    def test_load_skill_views_requires_missing_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                build_skill0_dataset.load_skill_views(skill_dir, {'full'})

    def test_default_recovery_dataset_uses_correct_dataset(self) -> None:
        cfg = {
            'training_plan': {'plan': 'code_then_skill0', 'mode': 'sequential', 'agent': {'source': 'coderforge_preview'}},
            'paths': {
                'output_root': '/tmp/runs',
                'skill0_train_jsonl': '/tmp/skill0.jsonl',
                'mixed_train_jsonl': '/tmp/mixed.jsonl',
                'ssd_train_jsonl': '/tmp/code.jsonl',
            },
            'run_name': 'demo',
        }
        self.assertEqual(
            recover_after_squeeze.default_recovery_dataset(cfg, 'adapter_skill0_squeezed'),
            Path('/tmp/skill0.jsonl'),
        )
        cfg['training_plan']['plan'] = 'mixed_sources'
        self.assertEqual(
            recover_after_squeeze.default_recovery_dataset(cfg, 'adapter_squeezed'),
            Path('/tmp/mixed.jsonl'),
        )
        cfg['training_plan']['plan'] = 'code_only'
        self.assertEqual(
            recover_after_squeeze.default_recovery_dataset(cfg, 'adapter_code_squeezed'),
            Path('/tmp/code.jsonl'),
        )

    def test_default_recovery_dataset_uses_default_agent_source(self) -> None:
        cfg = {
            'training_plan': {'plan': 'code_then_agent', 'agent': {}},
            'paths': {
                'output_root': '/tmp/runs',
                'skill0_train_jsonl': '/tmp/skill0.jsonl',
                'mixed_train_jsonl': '/tmp/mixed.jsonl',
                'ssd_train_jsonl': '/tmp/code.jsonl',
            },
            'run_name': 'demo',
        }
        self.assertEqual(
            recover_after_squeeze.default_recovery_dataset(cfg, 'adapter_agent_squeezed'),
            Path('/tmp/runs/demo/prepared_sources/coderforge_preview.jsonl'),
        )
        cfg['training_plan']['plan'] = 'code_then_agent'
        self.assertEqual(
            recover_after_squeeze.default_recovery_dataset(cfg, 'adapter_agent_squeezed'),
            Path('/tmp/runs/demo/prepared_sources/coderforge_preview.jsonl'),
        )
        cfg['training_plan']['plan'] = 'code_then_skill0'
        self.assertEqual(
            recover_after_squeeze.default_recovery_dataset(cfg, 'adapter_code_squeezed'),
            Path('/tmp/code.jsonl'),
        )

    def test_resolve_adapter_fails_for_missing_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                evaluate_codegen.resolve_adapter(Path(tmpdir), 'missing-adapter')

    def test_resolve_adapter_prefers_mode_specific_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for subdir in ['adapter_skill0_recovered', 'adapter_mixed_recovered']:
                adapter_dir = run_dir / subdir
                adapter_dir.mkdir()
                (adapter_dir / 'adapter_config.json').write_text('{}', encoding='utf-8')
            adapter_dir, tag = evaluate_codegen.resolve_adapter(run_dir, None, 'mixed')
            self.assertEqual(tag, 'adapter_mixed_recovered')
            self.assertEqual(adapter_dir, run_dir / 'adapter_mixed_recovered')

    def test_resolve_adapter_prefers_agent_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for subdir in ['adapter_skill0_recovered', 'adapter_agent_recovered']:
                adapter_dir = run_dir / subdir
                adapter_dir.mkdir()
                (adapter_dir / 'adapter_config.json').write_text('{}', encoding='utf-8')
            adapter_dir, tag = evaluate_codegen.resolve_adapter(run_dir, None, 'agent')
            self.assertEqual(tag, 'adapter_agent_recovered')
            self.assertEqual(adapter_dir, run_dir / 'adapter_agent_recovered')

    def test_preferred_eval_family_uses_plan_before_legacy_mode(self) -> None:
        cfg = {'training_plan': {'plan': 'mixed_sources', 'mode': 'sequential'}}
        self.assertEqual(evaluate_codegen.preferred_eval_family(cfg), 'mixed')
        cfg = {'training_plan': {'plan': 'code_then_agent', 'mode': 'mixed'}}
        self.assertEqual(evaluate_codegen.preferred_eval_family(cfg), 'agent')

    def test_code_then_skill0_commands_use_current_python_and_resolved_paths(self) -> None:
        cfg = {
            'paths': {
                'project_root': str(ROOT),
                'ssd_train_jsonl': '/abs/ssd_train.jsonl',
                'skill0_train_jsonl': '/abs/skill0_train.jsonl',
            },
            'training_plan': {
                'code': {
                    'output_subdir': 'adapter_code_high_rank',
                    'squeezed_output_subdir': 'adapter_code_squeezed',
                    'recovered_output_subdir': 'adapter_code_recovered',
                },
                'skill0': {
                    'output_subdir': 'adapter_skill0_high_rank',
                    'squeezed_output_subdir': 'adapter_skill0_squeezed',
                    'recovered_output_subdir': 'adapter_skill0_recovered',
                },
            },
            'evaluation': {'public': {}},
        }
        commands = run_training_plan.code_then_skill0_commands('/abs/config.yaml', cfg, Path('/abs/run-dir'), sys.executable)
        self.assertTrue(all(sys.executable in command for command in commands))
        self.assertFalse(any('build_skill_views.py' in command for command in commands))
        self.assertIn('/abs/ssd_train.jsonl', commands[0])
        self.assertIn('/abs/skill0_train.jsonl', commands[1])
        self.assertIn('adapter_code_high_rank', commands[1])
        self.assertNotIn('runs/', '\n'.join(commands))

    def test_code_only_commands_respect_disabled_squeeze_and_recovery(self) -> None:
        cfg = {
            'paths': {
                'project_root': str(ROOT),
                'ssd_train_jsonl': '/abs/ssd_train.jsonl',
            },
            'training_plan': {
                'code': {
                    'output_subdir': 'adapter_code_high_rank',
                    'squeezed_output_subdir': 'adapter_code_squeezed',
                    'recovered_output_subdir': 'adapter_code_recovered',
                },
            },
            'lora_squeeze': {'enabled': False},
            'recovery': {'enabled': False},
        }
        commands = run_training_plan.code_only_commands('/abs/config.yaml', cfg, sys.executable)
        self.assertEqual(len(commands), 2)
        self.assertIn('train_unsloth_lora.py', commands[0])
        self.assertIn('evaluate_livecodebench.py', commands[1])
        self.assertIn('adapter_code_high_rank', commands[1])

    def test_code_only_commands_reject_recovery_without_squeeze(self) -> None:
        cfg = {
            'paths': {
                'project_root': str(ROOT),
                'ssd_train_jsonl': '/abs/ssd_train.jsonl',
            },
            'training_plan': {
                'code': {
                    'output_subdir': 'adapter_code_high_rank',
                    'squeezed_output_subdir': 'adapter_code_squeezed',
                    'recovered_output_subdir': 'adapter_code_recovered',
                },
            },
            'lora_squeeze': {'enabled': False},
            'recovery': {'enabled': True},
        }
        with self.assertRaises(ValueError):
            run_training_plan.code_only_commands('/abs/config.yaml', cfg, sys.executable)

    def test_training_plan_section_defaults_support_legacy_mode_only_config(self) -> None:
        cfg = {
            'paths': {
                'project_root': str(ROOT),
                'ssd_train_jsonl': '/abs/ssd_train.jsonl',
                'skill0_train_jsonl': '/abs/skill0_train.jsonl',
            },
            'training_plan': {
                'mode': 'sequential',
            },
        }
        commands = run_training_plan.code_then_skill0_commands('/abs/config.yaml', cfg, Path('/abs/run-dir'), sys.executable)
        self.assertIn('adapter_code_high_rank', commands[0])
        self.assertIn('adapter_skill0_high_rank', commands[1])

    def test_resolve_plan_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            run_training_plan.resolve_plan('seq')

    def test_resolve_plan_maps_legacy_mode(self) -> None:
        cfg = {'training_plan': {'mode': 'mixed'}}
        self.assertEqual(run_training_plan.resolve_plan(None, cfg), 'mixed_sources')
        cfg = {'training_plan': {'mode': 'sequential'}}
        self.assertEqual(run_training_plan.resolve_plan(None, cfg), 'code_then_skill0')

    def test_run_training_plan_parser_accepts_legacy_plan_aliases(self) -> None:
        parser = run_training_plan.build_parser()
        parsed = parser.parse_args(['--config', 'demo.yaml', '--plan', 'mixed'])
        self.assertEqual(parsed.plan, 'mixed')
        parsed = parser.parse_args(['--config', 'demo.yaml', '--plan', 'sequential'])
        self.assertEqual(parsed.plan, 'sequential')

    def test_referenced_mixed_external_sources_finds_named_sources(self) -> None:
        cfg = {
            'training_plan': {
                'mixed': {
                    'sources': [
                        {'source': 'code', 'weight': 0.7},
                        {'source': 'coderforge_preview', 'weight': 0.3},
                        {'source': 'coderforge_preview', 'weight': 0.1},
                    ],
                },
            },
        }
        self.assertEqual(run_training_plan.referenced_mixed_external_sources(cfg), ['coderforge_preview'])

    def test_referenced_mixed_external_sources_skip_zero_weight_sources(self) -> None:
        cfg = {
            'training_plan': {
                'mixed': {
                    'sources': [
                        {'source': 'missing_agent', 'weight': 0.0},
                        {'source': 'coderforge_preview', 'weight': 0.3},
                    ],
                },
            },
        }
        self.assertEqual(run_training_plan.referenced_mixed_external_sources(cfg), ['coderforge_preview'])

    def test_prepare_external_mixed_sources_prepares_agent_sources(self) -> None:
        cfg = {
            'training_plan': {
                'code': {'source': 'rstar_coder_seed_sft'},
                'mixed': {
                    'sources': [
                        {'source': 'code', 'weight': 0.7},
                        {'source': 'coderforge_preview', 'weight': 0.3},
                    ],
                },
            },
            'data_sources': {
                'coderforge_preview': {'enabled': True, 'family': 'agent_trajectory'},
            },
        }
        original_prepare = run_training_plan.prepare_agent_source
        calls: list[str] = []
        run_training_plan.prepare_agent_source = lambda cfg_path, source_name: calls.append(source_name)  # type: ignore[assignment]
        try:
            run_training_plan.prepare_external_mixed_sources('/abs/config.yaml', cfg)
        finally:
            run_training_plan.prepare_agent_source = original_prepare  # type: ignore[assignment]
        self.assertEqual(calls, ['coderforge_preview'])

    def test_resolve_named_dataset_path_uses_default_code_source_name(self) -> None:
        cfg = {
            'training_plan': {},
            'paths': {
                'ssd_train_jsonl': '/abs/ssd_train.jsonl',
                'skill0_train_jsonl': '/abs/skill0_train.jsonl',
                'mixed_train_jsonl': '/abs/mixed_train.jsonl',
            },
        }
        self.assertEqual(
            source_adapters.resolve_named_dataset_path(cfg, Path('/abs/run-dir'), 'rstar_coder_seed_sft'),
            Path('/abs/ssd_train.jsonl'),
        )

    def test_normalized_mix_targets_allows_zero_weight(self) -> None:
        targets = build_mixed_dataset.normalized_mix_targets({'code': [{'id': 'code'}], 'skill0': [{'id': 'skill'}]}, {'code': 1.0, 'skill0': 0.0})
        self.assertEqual(targets['code'], 1)
        self.assertEqual(targets['skill0'], 0)

    def test_normalized_mix_targets_preserve_skewed_weights(self) -> None:
        targets = build_mixed_dataset.normalized_mix_targets({'code': [{'id': 'code'}], 'skill0': [{'id': 'skill'}]}, {'code': 0.99, 'skill0': 0.01})
        self.assertEqual(targets['code'], 99)
        self.assertEqual(targets['skill0'], 1)

    def test_normalized_mix_targets_rejects_positive_weight_with_empty_dataset(self) -> None:
        with self.assertRaises(ValueError):
            build_mixed_dataset.normalized_mix_targets({'code': [], 'skill0': [{'id': 'skill'}]}, {'code': 0.7, 'skill0': 0.3})

    def test_resolve_mixed_sources_rejects_duplicate_source_entries(self) -> None:
        cfg = {
            'training_plan': {
                'mixed': {
                    'sources': [
                        {'source': 'code', 'weight': 0.5},
                        {'source': 'code', 'weight': 0.5},
                    ],
                },
            },
            'paths': {
                'ssd_train_jsonl': '/abs/ssd_train.jsonl',
                'skill0_train_jsonl': '/abs/skill0_train.jsonl',
                'mixed_train_jsonl': '/abs/mixed_train.jsonl',
            },
        }
        with self.assertRaises(ValueError):
            build_mixed_dataset.resolve_mixed_sources(cfg, Path('/abs/run-dir'))

    def test_build_mixed_dataset_ignores_missing_zero_weight_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / 'ssd_train.jsonl'
            common.write_jsonl(code_path, [{'id': 'code-1', 'messages': [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}]}])
            cfg = {
                'run_name': 'demo',
                'training_plan': {
                    'mixed': {
                        'sources': [
                            {'source': 'code', 'weight': 1.0},
                            {'source': 'missing_agent', 'weight': 0.0},
                        ],
                    },
                },
                'paths': {
                    'ssd_train_jsonl': str(code_path),
                    'skill0_train_jsonl': str(root / 'skill0.jsonl'),
                    'mixed_train_jsonl': str(root / 'mixed.jsonl'),
                },
            }
            source_specs = build_mixed_dataset.resolve_mixed_sources(cfg, root)
            source_rows: dict[str, list[dict[str, object]]] = {}
            source_weights: dict[str, float] = {}
            for source_name, path, weight in source_specs:
                if weight <= 0:
                    source_rows[source_name] = []
                    source_weights[source_name] = weight
                    continue
                source_rows[source_name] = common.load_jsonl(path)
                source_weights[source_name] = weight
            targets = build_mixed_dataset.normalized_mix_targets(source_rows, source_weights)
            self.assertEqual(targets['code'], 1)
            self.assertEqual(targets['missing_agent'], 0)

    def test_normalize_coderforge_messages_handles_nested_json(self) -> None:
        raw_messages = json.dumps(json.dumps([
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': 'thinking', 'tool_calls': [{'name': 'bash'}]},
            {'role': 'tool', 'name': 'bash', 'content': 'ls'},
            {'role': 'assistant', 'content': 'done'},
        ]))
        normalized = source_adapters.normalize_coderforge_messages(raw_messages)
        self.assertEqual(normalized[0]['role'], 'system')
        self.assertIn('[tool_calls]', normalized[2]['content'])
        self.assertEqual(normalized[3]['role'], 'user')
        self.assertIn('[bash]', normalized[3]['content'])
        self.assertEqual(normalized[-1]['role'], 'assistant')

    def test_normalize_coderforge_messages_rejects_truncated_tool_trajectory(self) -> None:
        raw_messages = json.dumps([
            {'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': 'thinking', 'tool_calls': [{'name': 'bash'}]},
            {'role': 'tool', 'name': 'bash', 'content': 'ls'},
        ])
        self.assertEqual(source_adapters.normalize_coderforge_messages(raw_messages), [])

    def test_filter_by_contest_month(self) -> None:
        self.assertTrue(evaluate_livecodebench.filter_by_contest_month({'contest_date': '2025-03-12'}, {'2025-03'}))
        self.assertFalse(evaluate_livecodebench.filter_by_contest_month({'contest_date': '2024-12-01'}, {'2025-03'}))

    def test_ensure_nonempty_examples_rejects_empty_livecodebench_slice(self) -> None:
        cfg = {'evaluation': {'public': {'version_tag': 'release_v5', 'contest_months': ['2025-02']}}}
        with self.assertRaises(ValueError):
            evaluate_livecodebench.ensure_nonempty_examples([], cfg)

    def test_load_livecodebench_examples_passes_version_tag(self) -> None:
        captured: dict[str, object] = {}
        datasets_module = sys.modules['datasets']
        original = datasets_module.load_dataset

        def fake_load_dataset(name, **kwargs):
            captured['name'] = name
            captured['kwargs'] = kwargs
            return []

        datasets_module.load_dataset = fake_load_dataset
        cfg = {'evaluation': {'public': {'dataset_name': 'livecodebench/code_generation_lite', 'version_tag': 'release_v5', 'split': 'test'}}}
        try:
            rows = evaluate_livecodebench.load_livecodebench_examples(cfg)
        finally:
            datasets_module.load_dataset = original
        self.assertEqual(rows, [])
        self.assertEqual(captured['name'], 'livecodebench/code_generation_lite')
        self.assertEqual(captured['kwargs']['version_tag'], 'release_v5')

    def test_mock_buffer_consumes_lines_progressively(self) -> None:
        buffer = livecodebench_utils.MockBuffer('a\nb\n')
        self.assertEqual(buffer.readline(), b'a\n')
        self.assertEqual(buffer.readline(), b'b\n')
        self.assertEqual(buffer.read(), b'')

    def test_mock_stdin_with_buffer_consumes_read_calls(self) -> None:
        stdin = livecodebench_utils.MockStdinWithBuffer('abc')
        self.assertEqual(stdin.read(1), 'a')
        self.assertEqual(stdin.read(), 'bc')
        self.assertEqual(stdin.read(), '')

    def test_prompt_token_budgets_cap_requested_generation(self) -> None:
        class Tokenizer:
            def __call__(self, prompt: str, add_special_tokens: bool = False):
                return {'input_ids': list(range(len(prompt.split())))}

        budgets = generate_ssd_local.prompt_token_budgets(Tokenizer(), ['a b c d', 'a b c d e f g h i'], 10, 8)
        self.assertEqual(budgets, [6, 1])

    def test_prompt_token_budgets_reject_prompt_that_fills_context(self) -> None:
        class Tokenizer:
            def __call__(self, prompt: str, add_special_tokens: bool = False):
                return {'input_ids': list(range(len(prompt.split())))}

        with self.assertRaises(ValueError):
            generate_ssd_local.prompt_token_budgets(Tokenizer(), ['a b c d e'], 5, 3)

    def test_ssd_max_model_len_prefers_generation_config(self) -> None:
        cfg = {
            'model': {'max_seq_length': 16384},
            'ssd': {'max_model_len': 8192},
        }
        self.assertEqual(generate_ssd_local.ssd_max_model_len(cfg), 8192)

    def test_local_smoke_dataset_path_prefers_evaluation_override(self) -> None:
        cfg = {
            'paths': {'eval_dataset': '/abs/default.jsonl'},
            'evaluation': {'local_smoke': {'dataset_path': '/abs/override.jsonl'}},
        }
        self.assertEqual(evaluate_codegen.local_smoke_dataset_path(cfg), Path('/abs/override.jsonl'))

    def test_stage_assignments_do_not_wrap_back_to_early_stages(self) -> None:
        cfg = {
            'skill0': {
                'stages': [
                    {'name': 'stage_a', 'mixture': {'full': 1.0}},
                    {'name': 'stage_b', 'mixture': {'summary': 1.0}},
                    {'name': 'stage_c', 'mixture': {'zero': 1.0}},
                ],
            },
        }
        assignments = build_skill0_dataset.stage_assignments(cfg, 6)
        self.assertEqual([stage for stage, _ in assignments], ['stage_a', 'stage_a', 'stage_b', 'stage_b', 'stage_c', 'stage_c'])
        self.assertEqual([level for _, level in assignments], ['full', 'full', 'summary', 'summary', 'zero', 'zero'])

    def test_run_tests_timeout_returns_failure_instead_of_raising(self) -> None:
        ok, output = evaluate_codegen.run_tests('while True:\n    pass', [], timeout_seconds=1)
        self.assertFalse(ok)
        self.assertIn('Timed out', output)

    def test_validate_training_guardrails_rejects_router_modules(self) -> None:
        cfg = {
            'paths': {'ssd_train_jsonl': str(ROOT / 'runs' / 'demo' / 'ssd_train.jsonl')},
            'training': {'response_only': True},
        }
        with self.assertRaises(ValueError):
            train_unsloth_lora.validate_training_guardrails(cfg, Path(cfg['paths']['ssd_train_jsonl']), ['router', 'q_proj'])

    def test_validate_training_guardrails_rejects_code_stage_without_response_only(self) -> None:
        dataset_path = ROOT / 'runs' / 'demo' / 'ssd_train.jsonl'
        cfg = {
            'paths': {'ssd_train_jsonl': str(dataset_path)},
            'training': {'response_only': False},
        }
        with self.assertRaises(ValueError):
            train_unsloth_lora.validate_training_guardrails(cfg, dataset_path, ['q_proj'])

    def test_validate_training_guardrails_rejects_any_training_without_response_only(self) -> None:
        dataset_path = ROOT / 'custom' / 'code_data.jsonl'
        cfg = {
            'paths': {'ssd_train_jsonl': str(ROOT / 'runs' / 'demo' / 'ssd_train.jsonl')},
            'training': {'response_only': False},
        }
        with self.assertRaises(ValueError):
            train_unsloth_lora.validate_training_guardrails(cfg, dataset_path, ['q_proj'])

    def test_validate_init_adapter_guardrails_rejects_router_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / 'adapter_config.json').write_text(
                json.dumps({'target_modules': ['router', 'q_proj']}),
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                train_unsloth_lora.validate_init_adapter_guardrails(str(adapter_dir))

    def test_validate_init_adapter_guardrails_rejects_router_target_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / 'adapter_config.json').write_text(
                json.dumps({'target_modules': 'router'}),
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                train_unsloth_lora.validate_init_adapter_guardrails(str(adapter_dir))

    def test_validate_recovery_adapter_guardrails_rejects_router_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / 'adapter_config.json').write_text(
                json.dumps({'target_modules': ['router', 'q_proj']}),
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                recover_after_squeeze.validate_recovery_adapter_guardrails(adapter_dir)

    def test_recovery_requires_response_only_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / 'adapter_config.json').write_text(
                json.dumps({'target_modules': ['q_proj']}),
                encoding='utf-8',
            )
            cfg = {
                'training': {'response_only': False},
            }
            with self.assertRaises(ValueError):
                recover_after_squeeze.validate_recovery_training_guardrails(cfg, adapter_dir)

    def test_effective_optimizer_steps_account_for_world_size(self) -> None:
        self.assertEqual(common.effective_optimizer_steps(64, 2, 4, world_size=4), 2)

    def test_recovery_max_steps_accounts_for_num_train_epochs(self) -> None:
        cfg = {
            'training': {
                'per_device_train_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'num_train_epochs': 1.5,
            },
            'recovery': {'max_steps_ratio': 0.08},
        }
        steps_per_epoch, max_steps = recover_after_squeeze.recovery_max_steps(cfg, 64, world_size=4)
        self.assertEqual(steps_per_epoch, 2)
        self.assertEqual(max_steps, 1)

    def test_ensure_path_is_new_rejects_existing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'exists.json'
            path.write_text(json.dumps({'ok': True}), encoding='utf-8')
            with self.assertRaises(FileExistsError):
                common.ensure_path_is_new(path, 'test artifact')
if __name__ == '__main__':
    unittest.main()
