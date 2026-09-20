#!/usr/bin/env python3
"""CUDA training of Recursive LoRA on real caption/image pairs.

Scale-RAE's released query-mode data path is guarded by IS_XLA_AVAILABLE.  This
runner ports only that path to CUDA: frozen Qwen hidden states condition the
native diffusion head and frozen SigLIP2 features are its clean target.  Only
the explicitly injected LoRA tensors are optimized.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from recursive_lora import LoRAConfig, apply_lora_config, freeze_except_lora, lora_parameters
from scale_rae_overfit import TARGET_CATEGORY, append_jsonl, save_checkpoint
from target_audit import audit_scale_rae_targets, candidate_paths, resolve_scale_rae_dit
from training_loops import training_loop_patch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale-rae-root", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True, help="JSONL: image and caption per row")
    p.add_argument("--image-root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", choices=tuple(TARGET_CATEGORY), default="attn_proj")
    p.add_argument("--granularity", choices=("layerwise", "rangewise"), default="layerwise")
    p.add_argument("--layers", nargs="+", type=int, default=list(range(12, 28)))
    p.add_argument("--num-loops", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lambda-total", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty dataset")
    for index, row in enumerate(rows):
        if not isinstance(row.get("image"), str) or not isinstance(row.get("caption"), str):
            raise ValueError(f"row {index} must contain string image and caption")
    return rows


def module_device_dtype(module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


@torch.no_grad()
def encode_pair(model, tokenizer, image_processor, build_prompt, row: dict, image_root: Path, dtype):
    image_path = Path(row["image"])
    if not image_path.is_absolute():
        image_path = image_root / image_path
    image = Image.open(image_path).convert("RGB")
    vision_tower = model.get_vision_tower_aux_list()[0]
    vision_device, vision_dtype = module_device_dtype(vision_tower)
    pixels = image_processor[0].preprocess(image, return_tensors="pt")["pixel_values"].to(
        device=vision_device, dtype=vision_dtype
    )

    # Clean RAE target before the multimodal projector, matching prediction_target
    # in the official query-mode branch.
    x = model.encode_images([pixels])[0].detach()
    expected_tokens = int(model.diff_head.diffusion_tokens)
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != expected_tokens:
        raise RuntimeError(f"unexpected vision target shape {tuple(x.shape)}")

    text = "Generate an image of " + row["caption"].strip()
    prompt = build_prompt(text, model_config=model.config, with_image=False)
    language_model = model.get_model()
    language_device, language_dtype = module_device_dtype(language_model.embed_tokens)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(language_device)
    start_id = tokenizer.convert_tokens_to_ids("<im_start>")
    if start_id is None or start_id < 0 or start_id == tokenizer.unk_token_id:
        raise RuntimeError("checkpoint tokenizer has no <im_start> token")
    start = torch.tensor([[start_id]], device=language_device, dtype=ids.dtype)
    text_embeds = language_model.embed_tokens(torch.cat((ids, start), dim=1))
    queries = language_model.latent_queries
    if queries is None or queries.shape[0] != expected_tokens:
        raise RuntimeError(f"unexpected latent_queries shape {None if queries is None else tuple(queries.shape)}")
    inputs_embeds = torch.cat(
        (text_embeds, queries.unsqueeze(0).to(device=language_device, dtype=language_dtype)), dim=1
    )
    outputs = language_model(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
    query_states = outputs.last_hidden_state[:, -expected_tokens:, :]
    if model.use_diff_head_projector:
        projector_device, projector_dtype = module_device_dtype(model.diff_head_projector)
        query_states = query_states.to(device=projector_device, dtype=projector_dtype)
        z = model.diff_head_projector(query_states)
    else:
        z = query_states
    if z.shape[-1] != model.diff_head.z_channels:
        raise RuntimeError(f"unexpected condition shape {tuple(z.shape)}")
    head_device, head_dtype = module_device_dtype(model.diff_head)
    return (
        z.to(device=head_device, dtype=head_dtype).detach(),
        x.to(device=head_device, dtype=head_dtype).detach(),
        str(image_path),
        text,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("steps and checkpoint interval must be positive")
    root = args.scale_rae_root.resolve()
    sys.path[:0] = [str(root), str(root / "inference")]
    from cli import build_prompt
    from utils.load_model import load_scale_rae_model

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = load_rows(args.data)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    tokenizer, model, image_processor, _ = load_scale_rae_model(args.model_path, device="cuda", dtype=dtype)
    model.eval()
    dit_root, dit = resolve_scale_rae_dit(model)
    report = audit_scale_rae_targets(model, args.layers, rank=args.rank)
    paths = candidate_paths(report, TARGET_CATEGORY[args.target], relative_to_dit=True)
    if len(paths) != len(args.layers):
        raise RuntimeError(f"expected {len(args.layers)} LoRA targets, found {len(paths)}")
    config = LoRAConfig(tuple(paths), rank=args.rank, alpha=args.alpha)
    apply_lora_config(dit, config)
    freeze_except_lora(model)
    parameters = list(lora_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    run_config = vars(args).copy()
    run_config.update(model_path=args.model_path, dit_root=dit_root, target_paths=paths,
                      trainable_parameters=sum(p.numel() for p in parameters),
                      data_kind="real_caption_image_query_mode_cuda_port")
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in run_config.items()}
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    metrics = args.output_dir / "metrics.jsonl"
    order = list(range(len(rows)))
    smoothed = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with training_loop_patch(dit, args.layers, granularity=args.granularity,
                             num_loops=args.num_loops, lambda_total=args.lambda_total) as stats:
        for step in range(1, args.steps + 1):
            if (step - 1) % len(order) == 0:
                random.shuffle(order)
            sample_index = order[(step - 1) % len(order)]
            step_started = time.perf_counter()
            z, x, image_path, prompt = encode_pair(
                model, tokenizer, image_processor, build_prompt, rows[sample_index], args.image_root, dtype
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.diff_head.training_loss(z=z, x=x).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            missing = sum(p.grad is None for p in parameters)
            nonfinite = sum(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters)
            if missing or nonfinite:
                raise RuntimeError(f"bad LoRA gradients: missing={missing}, nonfinite={nonfinite}")
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            torch.cuda.synchronize()
            value = float(loss.detach().cpu())
            smoothed = value if smoothed is None else 0.9 * smoothed + 0.1 * value
            row = dict(step=step, sample_index=sample_index, image=image_path, prompt=prompt,
                       loss=value, smoothed_loss=smoothed, grad_norm=float(grad_norm.cpu()),
                       step_seconds=time.perf_counter() - step_started,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                       patched_invocations=stats.patched_invocations,
                       actual_block_calls=stats.actual_block_calls)
            append_jsonl(metrics, row)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(row), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                save_checkpoint(args.output_dir / f"lora_step_{step:06d}.pt", model, config,
                                {**run_config, "completed_step": step, "loss": value})

    summary = {**run_config, "status": "ok", "steps_completed": args.steps,
               "final_loss": value, "final_smoothed_loss": smoothed,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
