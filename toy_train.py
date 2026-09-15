#!/usr/bin/env python3
"""CPU sanity check: train shared LoRA through a fixed-depth recursive path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from recursive_lora import (
    AdaptedBlockRunner,
    LoRAConfig,
    apply_lora_config,
    freeze_except_lora,
    lora_parameters,
    save_lora_checkpoint,
)


class ToyResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ordinary", "recursive"), default="recursive")
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/toy_recursive_lora"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    width = 16
    block = ToyResidualBlock(width)
    lora_config = LoRAConfig(("proj",), rank=args.rank, alpha=args.rank)
    apply_lora_config(block, lora_config)
    freeze_except_lora(block)
    runner = AdaptedBlockRunner(block, default_num_loops=args.num_loops, lambda_total=1.0)
    trainable = list(lora_parameters(block))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    inputs = torch.randn(32, 8, width)
    # A deterministic synthetic teacher transformation, not a flow-matching loss.
    target = 1.25 * inputs
    losses: list[float] = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = runner(
            inputs,
            mode=args.mode,
            num_loops=1 if args.mode == "ordinary" else args.num_loops,
        )
        loss = (prediction - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_k = 1 if args.mode == "ordinary" else args.num_loops
    checkpoint = args.output_dir / f"lora_{args.mode}_k{effective_k}.pt"
    save_lora_checkpoint(
        checkpoint,
        block,
        lora_config,
        metadata={"mode": args.mode, "num_loops": effective_k, "lambda_total": 1.0},
    )
    report = {
        "purpose": "engineering_sanity_check_only",
        "seed": args.seed,
        "mode": args.mode,
        "num_loops": effective_k,
        "lambda_total": 1.0,
        "rank": args.rank,
        "steps": args.steps,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction_ratio": losses[-1] / losses[0],
        "checkpoint": str(checkpoint),
    }
    report_path = args.output_dir / f"report_{args.mode}_k{effective_k}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
