#!/usr/bin/env python3
"""Execute one shard of a prepared Scale-RAE benchmark plan.

The model and decoder are loaded once. Each output has a JSONL record, and the
same script handles B0/Dense/Sparse/Loop-Guidance from the resolved plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-rae-root", type=Path, required=True)
    parser.add_argument("--loop-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-path", default="nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B")
    parser.add_argument("--decoder-repo", default="nyu-visionx/siglip2_decoder")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lora-checkpoint", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def counter_delta(before: dict, after: dict) -> dict:
    result = {}
    for key, value in after.items():
        previous = before.get(key)
        result[key] = value - previous if isinstance(value, (int, float)) and isinstance(previous, (int, float)) else value
    return result


def format_scale_rae_generation_prompt(prompt: str) -> str:
    """Match the official Scale-RAE benchmark generation instruction."""
    return "Generate an image of " + prompt


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_lora_checkpoint(model, checkpoint: Path) -> dict:
    from recursive_lora import LoRAConfig, apply_lora_config, load_lora_checkpoint
    from target_audit import resolve_scale_rae_dit

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = payload.get("lora_config", {})
    config = LoRAConfig(
        tuple(raw.get("target_paths", ())),
        rank=int(raw.get("rank", 0)),
        alpha=float(raw.get("alpha", 0.0)),
    )
    _, dit = resolve_scale_rae_dit(model)
    apply_lora_config(dit, config)
    metadata = load_lora_checkpoint(checkpoint, model, config)
    return {
        "path": str(checkpoint.resolve()),
        "sha256": file_sha256(checkpoint),
        "config": {
            "target_paths": list(config.target_paths),
            "rank": config.rank,
            "alpha": config.alpha,
        },
        "training_metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Scale-RAE benchmark generation requires CUDA")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")

    scale_root = args.scale_rae_root.resolve()
    loop_root = args.loop_root.resolve()
    sys.path.insert(0, str(scale_root))
    sys.path.insert(0, str(scale_root / "inference"))
    sys.path.insert(0, str(loop_root / "src"))

    import cli as scale_cli
    from diffusion_loop import LoopConfig
    from diffusion_loop.scale_rae import (
        apply_scale_rae_loop_patch,
        apply_scale_rae_sampler_step_hook,
        reset_scale_rae_sampler_step_hook,
        set_scale_rae_cfg_config,
        set_scale_rae_loopguidance_config,
        set_scale_rae_sampler_step_count,
    )

    all_rows = read_jsonl(args.plan)
    if not all_rows:
        raise ValueError("plan is empty")
    identity = {(row["method_id"], row["benchmark"], json.dumps(row["resolved_config"], sort_keys=True)) for row in all_rows}
    if len(identity) != 1:
        raise ValueError("one plan must contain exactly one method, benchmark and resolved config")
    rows = all_rows[args.shard_index :: args.num_shards]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    completed = set()
    if args.records.exists():
        completed = {(row["method_id"], row["benchmark"], row["prompt_id"]) for row in read_jsonl(args.records)}

    resolved = all_rows[0]["resolved_config"]
    generation = resolved["generation"]
    loop_config = LoopConfig.from_dict(resolved["loop"])
    dtype = getattr(torch, args.dtype)
    tokenizer, model, _, _ = scale_cli.load_scale_rae_model(args.model_path, device="cuda", dtype=dtype)
    lora_provenance = None
    if args.lora_checkpoint is not None:
        lora_provenance = install_lora_checkpoint(model, args.lora_checkpoint.resolve())
    model.eval()
    start_id, end_id, eos_id = scale_cli.prepare_special_token_ids(tokenizer)
    decoder = scale_cli.build_decoder(model, model_path=args.model_path, decoder_repo_id=args.decoder_repo)

    stats = apply_scale_rae_loop_patch(model, loop_config)
    step_protocol = set_scale_rae_sampler_step_count(model, int(generation["num_inference_steps"]))
    apply_scale_rae_sampler_step_hook(model)
    set_scale_rae_loopguidance_config(model, loop_config)
    cfg_protocol = set_scale_rae_cfg_config(model, float(generation["model_guidance"]))

    for row in rows:
        key = (row["method_id"], row["benchmark"], row["prompt_id"])
        output = Path(row["output_image"])
        if key in completed:
            if not output.exists():
                raise FileNotFoundError(f"record exists but image is missing: {output}")
            continue
        if output.exists():
            raise FileExistsError(f"image exists without a completed record: {output}")

        scale_cli.set_seed(int(row["seed"]))
        reset_scale_rae_sampler_step_hook(model)
        generation_prompt = format_scale_rae_generation_prompt(row["prompt"])
        prompt = scale_cli.build_prompt(generation_prompt, model_config=model.config, with_image=False)
        input_ids = scale_cli.tokenize_prompt(prompt, tokenizer, device=model.device)
        before = stats.to_record()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids, image_embeds = model.generate(
                input_ids,
                images=None,
                **scale_cli._common_gen_kwargs(
                    start_id,
                    end_id,
                    eos_id,
                    float(generation["model_guidance"]),
                    512,
                ),
            )
            images = scale_cli.decode_image_embeds(model, image_embeds, decoder)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if len(images) != 1:
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            raise RuntimeError(
                f"expected one decoded image, received {len(images)}; "
                f"generation_prompt={generation_prompt!r}; output_text={output_text!r}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        images[0].save(temporary, format="PNG")
        temporary.replace(output)
        after = stats.to_record()
        append_record(
            args.records,
            {
                **row,
                "output_image": str(output.resolve()),
                "model_path": args.model_path,
                "decoder_repo": args.decoder_repo,
                "generation_prompt": generation_prompt,
                "dtype": args.dtype,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "seconds_generate_and_decode": seconds,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "loop_stats_delta": counter_delta(before, after),
                "step_protocol": step_protocol,
                "cfg_protocol": cfg_protocol,
                "lora_checkpoint": lora_provenance,
            },
        )
        print(json.dumps({"completed": row["prompt_id"], "output": str(output), "seconds": seconds}), flush=True)


if __name__ == "__main__":
    main()
