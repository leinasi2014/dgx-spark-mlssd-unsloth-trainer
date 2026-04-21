from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from common import ensure_path_is_new, logger, write_jsonl


def get_source_config(cfg: dict[str, Any], source_name: str, *, allow_disabled: bool = False) -> dict[str, Any]:
    sources = cfg.get('data_sources', {})
    if source_name not in sources:
        raise KeyError(f'Unknown data source: {source_name}')
    source_cfg = sources[source_name]
    if not isinstance(source_cfg, dict):
        raise TypeError(f'Data source config must be a mapping: {source_name}')
    if not allow_disabled and not bool(source_cfg.get('enabled', True)):
        raise ValueError(f'Data source {source_name} is disabled. Set data_sources.{source_name}.enabled=true before using it.')
    return source_cfg


def source_limit(source_cfg: dict[str, Any], cli_limit: int = 0) -> int:
    configured_limit = int(source_cfg.get('limit', 0) or 0)
    if cli_limit > 0 and configured_limit > 0:
        return min(cli_limit, configured_limit)
    return cli_limit if cli_limit > 0 else configured_limit


def prepared_sources_root(run_dir: Path) -> Path:
    return run_dir / 'prepared_sources'


def prepared_source_dataset_path(run_dir: Path, source_name: str) -> Path:
    return prepared_sources_root(run_dir) / f'{source_name}.jsonl'


def prepared_source_metadata_path(run_dir: Path, source_name: str) -> Path:
    return prepared_sources_root(run_dir) / f'{source_name}.meta.json'


def render_template(template_path: Path, **values: Any) -> str:
    template = Template(template_path.read_text(encoding='utf-8'))
    return template.render(**values).strip()


def load_hf_rows(source_cfg: dict[str, Any], limit: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset_cfg = source_cfg.get('dataset', {})
    dataset_name = str(dataset_cfg.get('name') or '').strip()
    dataset_config = dataset_cfg.get('config')
    dataset_split = str(dataset_cfg.get('split') or 'train').strip()
    if not dataset_name:
        raise ValueError('Data source is missing dataset.name')
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if limit > 0 and idx >= limit:
            break
        rows.append(dict(row))
    return rows


def infer_problem_type(starter_code: Any) -> str:
    return 'function' if isinstance(starter_code, str) and starter_code.strip() else 'stdin'


def format_problem_code_rows(
    source_name: str,
    source_cfg: dict[str, Any],
    templates_dir: Path,
    limit: int = 0,
) -> list[dict[str, Any]]:
    adapter = str(source_cfg.get('adapter') or '').strip()
    if adapter != 'rstar_coder':
        raise ValueError(f'Unsupported problem_code adapter for ml-ssd flow: {adapter}')
    rows = load_hf_rows(source_cfg, limit=limit)
    stdin_template = templates_dir / 'self_distillation_prompt_stdin.j2'
    function_template = templates_dir / 'self_distillation_prompt_function.j2'
    if not stdin_template.exists() or not function_template.exists():
        raise FileNotFoundError(f'Missing ml-ssd template files in {templates_dir}')

    formatted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get('question') or '').strip()
        if not question:
            continue
        starter_code = str(row.get('starter_code') or '')
        problem_type = infer_problem_type(starter_code)
        template_path = function_template if problem_type == 'function' else stdin_template
        prompt = render_template(template_path, question=question, starter_code=starter_code)
        prompt_id = str(row.get('question_id') or f'{source_name}-{idx:06d}')
        formatted.append({
            'prompt_id': prompt_id,
            'source_name': source_name,
            'family': source_cfg.get('family', 'problem_code'),
            'problem_type': problem_type,
            'question': question,
            'starter_code': starter_code,
            'prompt': prompt,
        })
    return formatted


def _parse_json_blob(value: Any) -> Any:
    current = value
    for _ in range(4):
        if not isinstance(current, str):
            return current
        stripped = current.strip()
        if not stripped:
            return stripped
        try:
            current = json.loads(stripped)
        except json.JSONDecodeError:
            return current
    return current


def _stringify_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ''
    try:
        rendered = json.dumps(tool_calls, ensure_ascii=False)
    except TypeError:
        rendered = str(tool_calls)
    return f'\n\n[tool_calls]\n{rendered}'


def normalize_coderforge_messages(raw_messages: Any) -> list[dict[str, str]]:
    parsed = _parse_json_blob(raw_messages)
    if not isinstance(parsed, list):
        raise TypeError('CoderForge messages payload must decode to a list')
    normalized: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = str(item.get('role') or 'assistant').strip().lower()
        content = item.get('content')
        if isinstance(content, list):
            content_text = '\n'.join(str(part.get('text') or part) for part in content)
        else:
            content_text = str(content or '')
        content_text = content_text.strip()
        tool_suffix = _stringify_tool_calls(item.get('tool_calls'))
        if role == 'tool':
            tool_name = str(item.get('name') or item.get('tool_name') or 'tool').strip()
            tool_text = content_text or str(item.get('result') or '').strip()
            content_text = f'[{tool_name}]\n{tool_text}'.strip()
            role = 'user'
        if not content_text and not tool_suffix:
            continue
        if role not in {'system', 'user', 'assistant'}:
            role = 'assistant'
        normalized.append({'role': role, 'content': (content_text + tool_suffix).strip()})
    if normalized and normalized[-1]['role'] != 'assistant':
        return []
    return [message for message in normalized if message.get('content')]


def format_agent_trajectory_rows(
    source_name: str,
    source_cfg: dict[str, Any],
    limit: int = 0,
) -> list[dict[str, Any]]:
    adapter = str(source_cfg.get('adapter') or '').strip()
    if adapter == 'coderforge':
        rows = load_hf_rows(source_cfg, limit=limit)
        min_reward = float(source_cfg.get('min_reward', 0.0))
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            reward = float(row.get('reward') or 0.0)
            if reward < min_reward:
                continue
            messages = normalize_coderforge_messages(row.get('messages'))
            if len(messages) < 2:
                continue
            if messages[-1]['role'] != 'assistant':
                continue
            out.append({
                'id': str(row.get('trajectory_id') or f'{source_name}-{idx:06d}'),
                'source_name': source_name,
                'messages': messages,
                'metadata': {
                    'reward': reward,
                    'finish_reason': row.get('finish_reason'),
                    'image': row.get('image'),
                    'license': row.get('license'),
                },
            })
        return out
    if adapter == 'litecoder_terminal':
        raise NotImplementedError(
            'LiteCoder-Terminal-RL-preview currently ships Harbor environments rather than directly trainable SFT trajectories. '
            'Use it for environment/export workflows, not this SFT data-prep path.'
        )
    raise ValueError(f'Unsupported agent trajectory adapter: {adapter}')


def resolve_named_dataset_path(cfg: dict[str, Any], run_dir: Path, source_name: str) -> Path:
    code_source = str(cfg.get('training_plan', {}).get('code', {}).get('source') or 'rstar_coder_seed_sft').strip()
    if source_name == 'code':
        return Path(cfg['paths']['ssd_train_jsonl'])
    if code_source and source_name == code_source:
        return Path(cfg['paths']['ssd_train_jsonl'])
    if source_name == 'skill0':
        return Path(cfg['paths']['skill0_train_jsonl'])
    if source_name == 'mixed':
        return Path(cfg['paths']['mixed_train_jsonl'])
    return prepared_source_dataset_path(run_dir, source_name)


def ensure_prepared_source(run_dir: Path, source_name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
    dataset_path = prepared_source_dataset_path(run_dir, source_name)
    meta_path = prepared_source_metadata_path(run_dir, source_name)
    ensure_path_is_new(dataset_path, f'prepared dataset for source {source_name}')
    ensure_path_is_new(meta_path, f'prepared metadata for source {source_name}')
    write_jsonl(dataset_path, rows)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info('Prepared source %s at %s (%d rows)', source_name, dataset_path, len(rows))
    return dataset_path
