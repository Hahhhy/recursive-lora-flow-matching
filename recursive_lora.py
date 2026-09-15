"""Minimal, backend-agnostic prototype for fixed-depth Recursive LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen Linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if rank > min(base.in_features, base.out_features):
            raise ValueError("rank cannot exceed the smaller linear dimension")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        update = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_output + self.scaling * update


def _resolve_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid module path: {path!r}")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_paths: Sequence[str],
    *,
    rank: int = 8,
    alpha: float = 8.0,
) -> list[str]:
    """Replace exact Linear module paths with LoRALinear wrappers."""
    injected: list[str] = []
    for path in target_paths:
        parent, leaf = _resolve_parent(model, path)
        current = parent[int(leaf)] if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)) else getattr(parent, leaf)
        if isinstance(current, LoRALinear):
            raise ValueError(f"target is already LoRA-wrapped: {path}")
        if not isinstance(current, nn.Linear):
            raise TypeError(f"target must be nn.Linear, got {type(current).__name__}: {path}")
        wrapped = LoRALinear(current, rank=rank, alpha=alpha)
        if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
        injected.append(path)
    return injected


def freeze_except_lora(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = set(lora_state_dict(model))
    received = set(state)
    if expected != received:
        raise ValueError(f"LoRA state mismatch; missing={sorted(expected - received)}, unexpected={sorted(received - expected)}")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected LoRA keys: {unexpected}")


@dataclass(frozen=True)
class RecursiveConfig:
    num_loops: int = 4
    lambda_total: float = 1.0

    def __post_init__(self) -> None:
        if self.num_loops <= 0:
            raise ValueError("num_loops must be positive")
        if self.lambda_total < 0:
            raise ValueError("lambda_total must be non-negative")

    @property
    def step_size(self) -> float:
        return self.lambda_total / self.num_loops


def recursive_euler(block: nn.Module, hidden_states: torch.Tensor, *args, config: RecursiveConfig, **kwargs) -> torch.Tensor:
    """Differentiate through K calls to one shared, possibly LoRA-adapted block."""
    state = hidden_states
    for _ in range(config.num_loops):
        proposal = block(state, *args, **kwargs)
        if not torch.is_tensor(proposal) or proposal.shape != state.shape:
            raise ValueError("recursive block must return a tensor with the input shape")
        state = state + config.step_size * (proposal - state)
    return state
