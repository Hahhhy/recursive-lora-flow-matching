"""Inspect a loaded Scale-RAE model and enumerate exact LoRA target paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from torch import nn


@dataclass(frozen=True)
class LinearTarget:
    block_index: int
    path: str
    relative_path: str
    category: str
    in_features: int
    out_features: int
    base_parameters: int
    lora_parameters: int


def resolve_scale_rae_dit(model: nn.Module) -> tuple[str, nn.Module]:
    """Resolve the unique module that owns Scale-RAE's ``dit_blocks``."""
    preferred = (
        ("diff_head.model", lambda value: value.diff_head.model),
        ("module.diff_head.model", lambda value: value.module.diff_head.model),
        ("diff_head", lambda value: value.diff_head),
        ("", lambda value: value),
    )
    for path, getter in preferred:
        try:
            candidate = getter(model)
        except AttributeError:
            continue
        if isinstance(getattr(candidate, "dit_blocks", None), nn.ModuleList):
            return path, candidate

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "dit_blocks", None), nn.ModuleList)
    ]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise ValueError(f"expected one Scale-RAE DiT owner, found {len(matches)}: {names}")
    return matches[0]


def classify_linear(relative_path: str) -> str:
    if relative_path == "attn.qkv":
        return "attention_qkv"
    if relative_path == "attn.proj":
        return "attention_output"
    if relative_path.startswith("mlp."):
        return "mlp"
    if relative_path.startswith("adaLN_modulation."):
        return "conditioning"
    return "other"


def audit_scale_rae_targets(
    model: nn.Module,
    block_indices: Sequence[int],
    *,
    rank: int = 8,
) -> dict:
    if rank <= 0:
        raise ValueError("rank must be positive")
    root_path, dit = resolve_scale_rae_dit(model)
    blocks = dit.dit_blocks
    indices = [int(index) for index in block_indices]
    if not indices:
        raise ValueError("at least one block index is required")
    invalid = [index for index in indices if not 0 <= index < len(blocks)]
    if invalid:
        raise IndexError(f"block indices out of range for depth {len(blocks)}: {invalid}")

    targets: list[LinearTarget] = []
    for index in indices:
        for relative_path, module in blocks[index].named_modules():
            if not relative_path or not isinstance(module, nn.Linear):
                continue
            local_path = f"dit_blocks.{index}.{relative_path}"
            full_path = f"{root_path}.{local_path}" if root_path else local_path
            targets.append(
                LinearTarget(
                    block_index=index,
                    path=full_path,
                    relative_path=relative_path,
                    category=classify_linear(relative_path),
                    in_features=module.in_features,
                    out_features=module.out_features,
                    base_parameters=module.weight.numel(),
                    lora_parameters=rank * (module.in_features + module.out_features),
                )
            )

    category_totals: dict[str, dict[str, int]] = {}
    for target in targets:
        values = category_totals.setdefault(target.category, {"linear_count": 0, "base_parameters": 0, "lora_parameters": 0})
        values["linear_count"] += 1
        values["base_parameters"] += target.base_parameters
        values["lora_parameters"] += target.lora_parameters

    return {
        "dit_root": root_path,
        "depth": len(blocks),
        "audited_blocks": indices,
        "rank": rank,
        "targets": [asdict(target) for target in targets],
        "category_totals": category_totals,
    }


def write_audit_report(report: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def candidate_paths(report: dict, category: str, *, relative_to_dit: bool = True) -> list[str]:
    key = "path"
    if relative_to_dit:
        root = report["dit_root"]
        prefix = f"{root}." if root else ""
        return [target[key][len(prefix):] for target in report["targets"] if target["category"] == category]
    return [target[key] for target in report["targets"] if target["category"] == category]
