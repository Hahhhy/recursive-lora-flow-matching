#!/usr/bin/env python3
"""One-batch Scale-RAE native flow-matching backward smoke test.

This is an engineering gate, not a quality benchmark. It uses synthetic tensors
with checkpoint-derived dimensions to verify the real DiT, LoRA targets, loop
semantics, gradients and memory before any dataset training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from recursive_lora import LoRAConfig, apply_lora_config, freeze_except_lora, lora_parameters
from target_audit import audit_scale_rae_targets, candidate_paths, resolve_scale_rae_dit
from training_loops import training_loop_patch


TARGET_CATEGORY = {
    "attn_proj": "attention_output",
    "adaln": "conditioning",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-rae-root", type=Path, required=True)
    parser.add_argument("--model-path", default="nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B")
    parser.add_argument("--target", choices=tuple(TARGET_CATEGORY), required=True)
    parser.add_argument("--granularity", choices=("layerwise", "rangewise"), required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12, 28)))
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lambda-total", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this real-checkpoint smoke test requires CUDA")
    root = args.scale_rae_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "inference"))
    from utils.load_model import load_scale_rae_model

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    _, model, _, _ = load_scale_rae_model(args.model_path, device="cuda", dtype=dtype)
    model.train()
    root_path, dit = resolve_scale_rae_dit(model)
    report = audit_scale_rae_targets(model, args.layers, rank=args.rank)
    paths = candidate_paths(report, TARGET_CATEGORY[args.target], relative_to_dit=True)
    if len(paths) != len(args.layers):
        raise RuntimeError(
            f"expected one {args.target} Linear per block, found {len(paths)} for {len(args.layers)} blocks: {paths}"
        )

    config = LoRAConfig(tuple(paths), rank=args.rank, alpha=args.alpha)
    apply_lora_config(dit, config)
    freeze_except_lora(model)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_trainable = 2 * len(paths)
    if len(trainable) != expected_trainable:
        raise RuntimeError(f"expected {expected_trainable} LoRA tensors, found {list(trainable)}")

    head = model.diff_head
    device = next(dit.parameters()).device
    tensor_dtype = next(dit.parameters()).dtype
    z = torch.randn(1, head.diffusion_tokens, head.z_channels, device=device, dtype=tensor_dtype)
    x = torch.randn(1, head.diffusion_tokens, head.diffusion_channels, device=device, dtype=tensor_dtype)
    torch.cuda.reset_peak_memory_stats()

    with training_loop_patch(
        dit,
        args.layers,
        granularity=args.granularity,
        num_loops=args.num_loops,
        lambda_total=args.lambda_total,
    ) as stats:
        loss = head.training_loss(z, x).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss.item()}")
        loss.backward()

    gradients = [parameter.grad for parameter in lora_parameters(model)]
    missing = sum(gradient is None for gradient in gradients)
    nonfinite = sum(gradient is not None and not torch.isfinite(gradient).all() for gradient in gradients)
    frozen_with_grad = [name for name, parameter in model.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
    result = {
        "status": "ok" if missing == 0 and nonfinite == 0 and not frozen_with_grad else "failed",
        "model_path": args.model_path,
        "dit_root": root_path,
        "target": args.target,
        "target_paths": paths,
        "granularity": args.granularity,
        "layers": args.layers,
        "train_k": args.num_loops,
        "lambda_total": args.lambda_total,
        "rank": args.rank,
        "alpha": args.alpha,
        "loss": float(loss.detach().cpu()),
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "missing_lora_gradients": missing,
        "nonfinite_lora_gradients": nonfinite,
        "frozen_parameters_with_gradient": frozen_with_grad,
        "patched_invocations": stats.patched_invocations,
        "actual_block_calls": stats.actual_block_calls,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "seed": args.seed,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] != "ok":
        raise RuntimeError("single-batch validation failed; inspect JSON report")


if __name__ == "__main__":
    main()
