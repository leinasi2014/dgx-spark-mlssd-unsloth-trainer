from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
ENV_PATTERN = re.compile(r'\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}')
TEMPLATE_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_\.]*)\}')
RUN_NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


def setup_logging(level: str = 'INFO') -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)


logger = logging.getLogger('dgx-spark-trainer')
setup_logging()


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path).resolve()
    data = yaml.safe_load(config_file.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise TypeError(f'Config at {config_file} must be a mapping')
    data.setdefault('_meta', {})
    data['_meta']['config_path'] = str(config_file)
    data['_meta']['config_dir'] = str(config_file.parent)
    resolved = data
    for _ in range(8):
        updated = _resolve_templates(resolved, resolved)
        if updated == resolved:
            break
        resolved = updated
    _validate_run_name(resolved.get('run_name', ''))
    _normalize_paths(resolved)
    final = _resolve_templates(data, resolved)
    _validate_run_name(final.get('run_name', ''))
    _normalize_paths(final)
    _normalize_additional_paths(final)
    return final


def _resolve_templates(value: Any, root: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_templates(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_templates(v, root) for v in value]
    if isinstance(value, str):
        def env_repl(match: re.Match[str]) -> str:
            env_name = match.group(1)
            if env_name not in os.environ:
                raise KeyError(f'Missing required environment variable template: env:{env_name}')
            return os.environ[env_name]

        out = ENV_PATTERN.sub(env_repl, value)

        def dotted_repl(match: re.Match[str]) -> str:
            token = match.group(1)
            resolved = _lookup_template_value(root, token)
            if resolved is None:
                raise KeyError(f'Missing required config template: {token}')
            return str(resolved)

        return TEMPLATE_PATTERN.sub(dotted_repl, out)
    return value


def _lookup_template_value(root: Dict[str, Any], token: str) -> Any:
    current: Any = root
    for part in token.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _validate_run_name(run_name: Any) -> None:
    value = str(run_name or '')
    if not RUN_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            'run_name must be a simple path-safe slug containing only letters, digits, ".", "_" or "-". '
            f'Got: {value!r}'
        )


def _normalize_paths(cfg: Dict[str, Any]) -> None:
    cfg_dir = Path(cfg['_meta']['config_dir'])
    paths = cfg.setdefault('paths', {})
    project_root_value = paths.get('project_root', '.')
    project_root = Path(project_root_value)
    if not project_root.is_absolute():
        if str(project_root_value) == '.' and cfg_dir.name == 'configs':
            project_root = cfg_dir.parent.resolve()
        else:
            project_root = (cfg_dir / project_root).resolve()
    paths['project_root'] = str(project_root)
    for key, raw_value in list(paths.items()):
        if key == 'project_root' or not isinstance(raw_value, str):
            continue
        value = Path(raw_value)
        if not value.is_absolute() and raw_value[:1] in {'/', '\\'}:
            raise ValueError(f'Path {key} must be relative to project_root or fully qualified, not drive-ambiguous rooted path: {raw_value!r}')
        paths[key] = str((project_root / value).resolve()) if not value.is_absolute() else str(value)


def _normalize_additional_paths(cfg: Dict[str, Any]) -> None:
    project_root = Path(cfg['paths']['project_root'])
    nested_paths = [
        ('ssd', 'templates', 'template_root'),
        ('evaluation', 'local_smoke', 'dataset_path'),
    ]
    for path_parts in nested_paths:
        current: Any = cfg
        for part in path_parts[:-1]:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if not isinstance(current, dict):
            continue
        key = path_parts[-1]
        raw_value = current.get(key)
        if not isinstance(raw_value, str):
            continue
        value = Path(raw_value)
        if not value.is_absolute() and raw_value[:1] in {'/', '\\'}:
            dotted = '.'.join(path_parts)
            raise ValueError(f'Path {dotted} must be relative to project_root or fully qualified, not drive-ambiguous rooted path: {raw_value!r}')
        current[key] = str((project_root / value).resolve()) if not value.is_absolute() else str(value)


def ensure_run_dirs(cfg: Dict[str, Any]) -> Path:
    output_root = Path(cfg['paths']['output_root'])
    run_dir = output_root / cfg['run_name']
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_path_is_new(path: Path, description: str) -> None:
    if path.exists():
        raise FileExistsError(
            f'Refusing to overwrite existing {description}: {path}. '
            'Choose a new run_name or remove the artifact explicitly.'
        )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL in {path} at line {line_no}: {exc}') from exc
            if not isinstance(item, dict):
                raise TypeError(f'JSONL item at {path}:{line_no} must be an object')
            rows.append(item)
    return rows


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def is_moe_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return 'moe' in lowered or '-a3b' in lowered or '-a22b' in lowered or '-a35b' in lowered


def choose_unsloth_loader(model_name: str, explicit: str | None = None) -> tuple[Any, str]:
    import unsloth

    if explicit == 'fastlanguage':
        return unsloth.FastLanguageModel, 'FastLanguageModel'
    if explicit == 'fastmodel':
        return unsloth.FastModel, 'FastModel'
    if explicit and explicit != 'auto':
        raise ValueError(f'Unknown loader setting: {explicit}')
    if is_moe_model_name(model_name) and hasattr(unsloth, 'FastModel'):
        return unsloth.FastModel, 'FastModel'
    return unsloth.FastLanguageModel, 'FastLanguageModel'


def response_markers(cfg: Dict[str, Any]) -> tuple[str, str]:
    markers = cfg.get('training', {}).get('response_markers', {})
    instruction = markers.get('instruction_part', '<|im_start|>user\n')
    response = markers.get('response_part', '<|im_start|>assistant\n')
    return instruction, response


def maybe_enable_response_only(trainer: Any, tokenizer: Any, cfg: Dict[str, Any]) -> Any:
    if not cfg.get('training', {}).get('response_only', True):
        return trainer
    instruction_part, response_part = response_markers(cfg)
    chat_template = getattr(tokenizer, 'chat_template', None) or ''
    if chat_template and instruction_part.strip() not in chat_template:
        raise RuntimeError('Response-only masking marker for user not found in tokenizer chat template.')
    if chat_template and response_part.strip() not in chat_template:
        raise RuntimeError('Response-only masking marker for assistant not found in tokenizer chat template.')
    try:
        from unsloth.chat_templates import train_on_responses_only
        return train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    except Exception as exc:
        raise RuntimeError(f'Response-only masking unavailable: {exc}') from exc


def effective_optimizer_steps(n_examples: int, per_device_batch_size: int, grad_accum: int, world_size: int = 1) -> int:
    per_step = max(1, per_device_batch_size) * max(1, grad_accum) * max(1, world_size)
    return max(1, math.ceil(n_examples / per_step))


def append_run_note(run_dir: Path, lines: list[str]) -> None:
    notes_path = run_dir / 'notes.md'
    existing = notes_path.read_text(encoding='utf-8') if notes_path.exists() else '# Run notes\n\n'
    notes_path.write_text(existing + ''.join(f'- {line}\n' for line in lines), encoding='utf-8')
