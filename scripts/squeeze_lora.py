#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from common import append_run_note, ensure_run_dirs, load_config, logger


def randomized_svd(matrix: torch.Tensor, rank: int, oversample: int = 8, n_iter: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = matrix.device
    m, n = matrix.shape
    q_rank = min(rank + oversample, min(m, n))
    omega = torch.randn(n, q_rank, device=device, dtype=matrix.dtype)
    y = matrix @ omega
    q, _ = torch.linalg.qr(y, mode='reduced')
    for _ in range(n_iter):
        y = matrix.transpose(0, 1) @ q
        q, _ = torch.linalg.qr(y, mode='reduced')
        y = matrix @ q
        q, _ = torch.linalg.qr(y, mode='reduced')
    b = q.transpose(0, 1) @ matrix
    u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_hat
    return u[:, :rank], s[:rank], vh[:rank, :]


def squeeze_pair(a: torch.Tensor, b: torch.Tensor, target_rank: int, oversample: int, n_iter: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    delta = b @ a
    orig_norm = float(torch.linalg.norm(delta).item())
    u, s, vh = randomized_svd(delta, target_rank, oversample=oversample, n_iter=n_iter)
    sqrt_s = torch.sqrt(s)
    new_b = u * sqrt_s.unsqueeze(0)
    new_a = sqrt_s.unsqueeze(1) * vh
    approx = new_b @ new_a
    error = float(torch.linalg.norm(delta - approx).item())
    energy_kept = float((s.square().sum() / delta.square().sum()).item()) if torch.any(delta) else 1.0
    report = {
        'original_fro_norm': orig_norm,
        'approx_fro_norm': float(torch.linalg.norm(approx).item()),
        'fro_error': error,
        'relative_fro_error': error / max(orig_norm, 1e-8),
        'energy_kept': energy_kept,
    }
    return new_a.contiguous(), new_b.contiguous(), report


def main() -> None:
    parser = argparse.ArgumentParser(description='Compress a LoRA adapter via RSVD while preserving scaling.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-subdir', default='adapter_high_rank')
    parser.add_argument('--output-subdir', default='adapter_squeezed')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    src_dir = run_dir / args.source_subdir
    dst_dir = run_dir / args.output_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = src_dir / 'adapter_model.safetensors'
    if not adapter_path.exists():
        raise FileNotFoundError(f'Missing adapter: {adapter_path}')

    tensors = load_file(str(adapter_path))
    target_rank = int(cfg['training']['target_rank'])
    oversample = int(cfg['lora_squeeze']['oversample_rank'])
    n_iter = int(cfg['lora_squeeze']['power_iterations'])

    out: dict[str, torch.Tensor] = {}
    handled_prefixes: set[str] = set()
    report: dict[str, Any] = {'source_subdir': args.source_subdir, 'target_rank': target_rank, 'layers': {}}

    for key in sorted(tensors.keys()):
        if '.lora_A.' in key:
            prefix = key.split('.lora_A.')[0]
            if prefix in handled_prefixes:
                continue
            a_key = key
            b_key = key.replace('.lora_A.', '.lora_B.')
            if b_key not in tensors:
                out[a_key] = tensors[a_key]
                continue
            a = tensors[a_key].float()
            b = tensors[b_key].float()
            new_a, new_b, layer_report = squeeze_pair(a, b, target_rank, oversample, n_iter)
            out[a_key] = new_a.to(tensors[a_key].dtype)
            out[b_key] = new_b.to(tensors[b_key].dtype)
            report['layers'][prefix] = layer_report
            handled_prefixes.add(prefix)
        elif '.lora_B.' in key:
            if key.replace('.lora_B.', '.lora_A.') not in tensors:
                out[key] = tensors[key]
        else:
            out[key] = tensors[key]

    save_file(out, str(dst_dir / 'adapter_model.safetensors'))
    config_src = src_dir / 'adapter_config.json'
    if config_src.exists():
        adapter_cfg = json.loads(config_src.read_text(encoding='utf-8'))
        orig_rank = int(adapter_cfg.get('r', cfg['training']['source_rank']))
        orig_alpha = float(adapter_cfg.get('lora_alpha', cfg['training']['lora_alpha']))
        adapter_cfg['r'] = target_rank
        adapter_cfg['lora_alpha'] = max(1, int(round(orig_alpha * target_rank / max(orig_rank, 1))))
        report['alpha_preservation'] = {
            'orig_rank': orig_rank,
            'orig_alpha': orig_alpha,
            'target_rank': target_rank,
            'new_alpha': adapter_cfg['lora_alpha'],
            'orig_scaling': orig_alpha / max(orig_rank, 1),
            'new_scaling': adapter_cfg['lora_alpha'] / max(target_rank, 1),
        }
        (dst_dir / 'adapter_config.json').write_text(json.dumps(adapter_cfg, indent=2), encoding='utf-8')

    for aux_name in ['tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json', 'tokenizer.model']:
        aux = src_dir / aux_name
        if aux.exists():
            (dst_dir / aux_name).write_bytes(aux.read_bytes())

    (dst_dir / 'compression_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    append_run_note(run_dir, [f'Squeezed adapter from {args.source_subdir} to {args.output_subdir} at rank {target_rank}.'])
    logger.info('Wrote squeezed adapter to %s', dst_dir)


if __name__ == '__main__':
    main()
