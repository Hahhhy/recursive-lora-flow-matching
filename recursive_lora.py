"""Minimal, backend-agnostic prototype for fixed-depth Recursive LoRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

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


@dataclass(frozen=True)
class LoRAConfig:
    """Portable description of exact LoRA insertion targets."""

    target_paths: tuple[str, ...]
    rank: int = 8
    alpha: float = 8.0

    def __post_init__(self) -> None:
        if not self.target_paths:
            raise ValueError("target_paths must not be empty")
        if len(set(self.target_paths)) != len(self.target_paths):
            raise ValueError("target_paths must be unique")
        if self.rank <= 0:
            raise ValueError("rank must be positive")


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
    """Replace exact Linear paths, after validating every target."""
    if len(set(target_paths)) != len(target_paths):
        raise ValueError("target_paths must be unique")
    resolved: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for path in target_paths:
        parent, leaf = _resolve_parent(model, path)
        current = parent[int(leaf)] if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)) else getattr(parent, leaf)
        if isinstance(current, LoRALinear):
            raise ValueError(f"target is already LoRA-wrapped: {path}")
        if not isinstance(current, nn.Linear):
            raise TypeError(f"target must be nn.Linear, got {type(current).__name__}: {path}")
        if rank > min(current.in_features, current.out_features):
            raise ValueError(f"rank exceeds dimensions for target: {path}")
        resolved.append((path, parent, leaf, current))

    injected: list[str] = []
    for path, parent, leaf, current in resolved:
        wrapped = LoRALinear(current, rank=rank, alpha=alpha)
        if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
        injected.append(path)
    return injected


def apply_lora_config(model: nn.Module, config: LoRAConfig) -> list[str]:
    return inject_lora(
        model,
        config.target_paths,
        rank=config.rank,
        alpha=config.alpha,
    )


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


CHECKPOINT_SCHEMA_VERSION = 1


def save_lora_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: LoRAConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lora_config": asdict(config),
        "state_dict": lora_state_dict(model),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output)
    return output


def load_lora_checkpoint(
    path: str | Path,
    model: nn.Module,
    expected_config: LoRAConfig,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    received = payload.get("lora_config", {})
    normalized = {
        "target_paths": tuple(received.get("target_paths", ())),
        "rank": received.get("rank"),
        "alpha": received.get("alpha"),
    }
    if normalized != asdict(expected_config):
        raise ValueError(f"LoRA config mismatch; expected={asdict(expected_config)}, received={normalized}")
    load_lora_state_dict(model, payload["state_dict"])
    return dict(payload.get("metadata", {}))


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


@dataclass
class LoopStepTrace:
    """Graph-connected tensors for diagnosing each recursive step."""

    index: int
    input_state: torch.Tensor
    proposal: torch.Tensor
    output_state: torch.Tensor


def ordinary_forward(block: nn.Module, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """The independent, non-recursive LoRA comparison path."""
    output = block(hidden_states, *args, **kwargs)
    if not torch.is_tensor(output) or output.shape != hidden_states.shape:
        raise ValueError("ordinary block must return a tensor with the input shape")
    return output


def recursive_euler_with_trace(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *args,
    config: RecursiveConfig,
    **kwargs,
) -> tuple[torch.Tensor, list[LoopStepTrace]]:
    """Differentiate through K calls and expose graph-connected step tensors."""
    state = hidden_states
    trace: list[LoopStepTrace] = []
    for index in range(config.num_loops):
        input_state = state
        proposal = block(input_state, *args, **kwargs)
        if not torch.is_tensor(proposal) or proposal.shape != input_state.shape:
            raise ValueError("recursive block must return a tensor with the input shape")
        state = input_state + config.step_size * (proposal - input_state)
        trace.append(LoopStepTrace(index, input_state, proposal, state))
    return state, trace


def recursive_euler(block: nn.Module, hidden_states: torch.Tensor, *args, config: RecursiveConfig, **kwargs) -> torch.Tensor:
    """Differentiate through K calls to one shared, possibly LoRA-adapted block."""
    output, _ = recursive_euler_with_trace(block, hidden_states, *args, config=config, **kwargs)
    return output


class AdaptedBlockRunner(nn.Module):
    """Explicit ordinary/recursive paths with an Elastic-ready K override."""

    def __init__(
        self,
        block: nn.Module,
        *,
        default_num_loops: int = 4,
        lambda_total: float = 1.0,
    ) -> None:
        super().__init__()
        self.block = block
        self.default_num_loops = int(default_num_loops)
        self.lambda_total = float(lambda_total)
        RecursiveConfig(self.default_num_loops, self.lambda_total)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        mode: Literal["ordinary", "recursive"] = "recursive",
        num_loops: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mode == "ordinary":
            if num_loops not in (None, 1):
                raise ValueError("ordinary mode accepts only num_loops=None or 1")
            return ordinary_forward(self.block, hidden_states, *args, **kwargs)
        if mode != "recursive":
            raise ValueError(f"unknown forward mode: {mode!r}")
        config = RecursiveConfig(
            num_loops=self.default_num_loops if num_loops is None else num_loops,
            lambda_total=self.lambda_total,
        )
        return recursive_euler(self.block, hidden_states, *args, config=config, **kwargs)
