#!/usr/bin/env python3
"""Short fixed-objective Scale-RAE LoRA optimization check.

This deliberately reuses one synthetic checkpoint-shaped batch and resets the
diffusion RNG before every step.  It answers whether the selected LoRA and
recursive forward can optimize a stable native flow-matching objective.  It is
not a data-training run and its loss is not a generation-quality metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from recursive_lora import (
    LoRAConfig,
    apply_lora_config,
    freeze_except_lora,
    load_lora_checkpoint,
    lora_parameters,
    lora_state_dict,
    save_lora_checkpoint,
)
from target_audit import audit_scale_rae_targets, candidate_paths, resolve_scale_rae_dit
from training_loops import training_loop_patch


TARGET_CATEGORY = {
    "attn_proj": "attention_output",
    "adaln": "conditioning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-rae-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGET_CATEGORY), default="attn_proj")
    parser.add_argument("--granularity", choices=("layerwise", "rangewise"), default="layerwise")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12, 28)))
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lambda-total", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective-seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline"), default="disabled")
    parser.add_argument("--wandb-project", default="recursive-lora-scale-rae")
    parser.add_argument("--run-name")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_checkpoint(path: Path, model, config: LoRAConfig, metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_lora_checkpoint(temporary, model, config, metadata=metadata)
    temporary.replace(path)


def state_checksum(state: dict[str, torch.Tensor]) -> tuple[float, float]:
    total = sum(float(value.float().sum()) for value in state.values())
    squared = sum(float(value.float().square().sum()) for value in state.values())
    return total, squared


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.steps < 1 or args.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    if args.log_every < 1 or args.checkpoint_every < 1:
        raise ValueError("logging/checkpoint intervals must be positive")

    root = args.scale_rae_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "inference"))
    from utils.load_model import load_scale_rae_model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to mix runs in existing metrics file: {metrics_path}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    _, model, _, _ = load_scale_rae_model(args.model_path, device="cuda", dtype=dtype)
    model.train()
    dit_root, dit = resolve_scale_rae_dit(model)
    report = audit_scale_rae_targets(model, args.layers, rank=args.rank)
    paths = candidate_paths(report, TARGET_CATEGORY[args.target], relative_to_dit=True)
    if len(paths) != len(args.layers):
        raise RuntimeError(f"expected {len(args.layers)} targets, found {paths}")

    config = LoRAConfig(tuple(paths), rank=args.rank, alpha=args.alpha)
    apply_lora_config(dit, config)
    freeze_except_lora(model)
    parameters = list(lora_parameters(model))
    trainable_parameters = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    head = model.diff_head
    device = next(dit.parameters()).device
    tensor_dtype = next(dit.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    z = torch.randn(
        1, head.diffusion_tokens, head.z_channels,
        device=device, dtype=tensor_dtype, generator=generator,
    )
    x = torch.randn(
        1, head.diffusion_tokens, head.diffusion_channels,
        device=device, dtype=tensor_dtype, generator=generator,
    )

    run_name = args.run_name or f"{args.target}-{args.granularity}-traink{args.num_loops}"
    run_config = {
        "run_name": run_name,
        "model_path": args.model_path,
        "dit_root": dit_root,
        "target": args.target,
        "target_paths": paths,
        "granularity": args.granularity,
        "layers": args.layers,
        "train_k": args.num_loops,
        "lambda_total": args.lambda_total,
        "rank": args.rank,
        "alpha": args.alpha,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "objective_seed": args.objective_seed,
        "dtype": args.dtype,
        "trainable_parameters": trainable_parameters,
        "data_kind": "fixed_synthetic_checkpoint_shaped_batch",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    wandb_run = None
    if args.wandb_mode == "offline":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("--wandb-mode offline requires the wandb package") from error
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=run_config,
            mode="offline",
            dir=str(args.output_dir),
            tags=[args.target, args.granularity, f"train_k={args.num_loops}", "fixed-overfit"],
        )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    exponential_loss = None
    initial_loss = None
    final_loss = None

    with training_loop_patch(
        dit,
        args.layers,
        granularity=args.granularity,
        num_loops=args.num_loops,
        lambda_total=args.lambda_total,
    ) as stats:
        for step in range(1, args.steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(args.objective_seed)
            torch.cuda.manual_seed_all(args.objective_seed)
            loss = head.training_loss(z, x).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()

            missing = sum(parameter.grad is None for parameter in parameters)
            nonfinite = sum(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in parameters
            )
            if missing or nonfinite:
                raise RuntimeError(
                    f"bad gradients at step {step}: missing={missing}, nonfinite={nonfinite}"
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite grad norm at step {step}")
            optimizer.step()
            torch.cuda.synchronize()

            loss_value = float(loss.detach().cpu())
            initial_loss = loss_value if initial_loss is None else initial_loss
            final_loss = loss_value
            exponential_loss = (
                loss_value if exponential_loss is None
                else 0.9 * exponential_loss + 0.1 * loss_value
            )
            row = {
                "step": step,
                "loss": loss_value,
                "smoothed_loss": exponential_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step_seconds": time.perf_counter() - step_started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "patched_invocations": stats.patched_invocations,
                "actual_block_calls": stats.actual_block_calls,
            }
            append_jsonl(metrics_path, row)
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps(row), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                save_checkpoint(
                    args.output_dir / f"lora_step_{step:06d}.pt",
                    model,
                    config,
                    metadata={**run_config, "completed_step": step, "loss": loss_value},
                )

    checkpoint = args.output_dir / f"lora_step_{args.steps:06d}.pt"
    before = state_checksum(lora_state_dict(model))
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    restored_metadata = load_lora_checkpoint(checkpoint, model, config)
    after = state_checksum(lora_state_dict(model))
    if before != after:
        raise RuntimeError(f"checkpoint round-trip mismatch: before={before}, after={after}")

    summary = {
        **run_config,
        "status": "ok",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "loss_decreased": final_loss < initial_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint": str(checkpoint),
        "checkpoint_roundtrip_ok": True,
        "restored_completed_step": restored_metadata["completed_step"],
    }
    if not math.isfinite(summary["loss_ratio"]):
        raise RuntimeError("non-finite loss ratio")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
