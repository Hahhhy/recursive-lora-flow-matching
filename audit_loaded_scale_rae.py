#!/usr/bin/env python3
"""Load an official Scale-RAE checkpoint and write its LoRA target audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from target_audit import audit_scale_rae_targets, candidate_paths, write_audit_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-rae-root", type=Path, required=True)
    parser.add_argument("--model-path", default="nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12, 28)))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, default=Path("scale_rae_lora_targets.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.scale_rae_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "inference"))
    from utils.load_model import load_scale_rae_model

    dtype = getattr(torch, args.dtype)
    _, model, _, _ = load_scale_rae_model(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    report = audit_scale_rae_targets(model, args.layers, rank=args.rank)
    write_audit_report(report, args.output)
    summary = {
        "output": str(args.output.resolve()),
        "dit_root": report["dit_root"],
        "depth": report["depth"],
        "audited_blocks": report["audited_blocks"],
        "category_totals": report["category_totals"],
        "first_attention_output_targets_relative_to_dit": candidate_paths(report, "attention_output")[:3],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
