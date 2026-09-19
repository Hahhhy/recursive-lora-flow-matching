"""Differentiable training-time loop patches for Scale-RAE style DiT blocks.

Unlike the inference patch, this module does not depend on sampler step metadata.
It is intentionally small: dense Euler loops only, with explicit layerwise and
rangewise semantics, and a reversible context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

from torch import nn

from recursive_lora import RecursiveConfig, recursive_euler


LoopGranularity = Literal["layerwise", "rangewise"]


@dataclass
class TrainingLoopStats:
    granularity: LoopGranularity
    block_indices: tuple[int, ...]
    num_loops: int
    patched_invocations: int = 0
    actual_block_calls: int = 0


def _validate_indices(blocks: nn.ModuleList, block_indices: Sequence[int], granularity: LoopGranularity) -> tuple[int, ...]:
    indices = tuple(int(index) for index in block_indices)
    if not indices:
        raise ValueError("block_indices must not be empty")
    if len(set(indices)) != len(indices) or tuple(sorted(indices)) != indices:
        raise ValueError("block_indices must be unique and sorted")
    invalid = [index for index in indices if not 0 <= index < len(blocks)]
    if invalid:
        raise IndexError(f"block indices out of range for depth {len(blocks)}: {invalid}")
    if granularity == "rangewise" and indices != tuple(range(indices[0], indices[-1] + 1)):
        raise ValueError("rangewise block_indices must form one contiguous interval")
    return indices


@contextmanager
def training_loop_patch(
    dit: nn.Module,
    block_indices: Sequence[int],
    *,
    granularity: LoopGranularity,
    num_loops: int,
    lambda_total: float = 1.0,
) -> Iterator[TrainingLoopStats]:
    """Temporarily replace selected block forwards with differentiable loops.

    ``layerwise`` loops every selected block independently. ``rangewise`` treats
    the selected contiguous block interval as one composite function and loops
    that complete interval. Original forwards are restored even after errors.
    """
    if granularity not in ("layerwise", "rangewise"):
        raise ValueError(f"unknown granularity: {granularity!r}")
    blocks = getattr(dit, "dit_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("dit must expose an nn.ModuleList named dit_blocks")
    indices = _validate_indices(blocks, block_indices, granularity)
    config = RecursiveConfig(num_loops=num_loops, lambda_total=lambda_total)
    originals = {index: blocks[index].forward for index in indices}
    stats = TrainingLoopStats(granularity, indices, num_loops)

    try:
        if granularity == "layerwise":
            for index in indices:
                original = originals[index]

                def layer_forward(hidden_states, *args, _original=original, **kwargs):
                    stats.patched_invocations += 1
                    stats.actual_block_calls += config.num_loops
                    return recursive_euler(_original, hidden_states, *args, config=config, **kwargs)

                blocks[index].forward = layer_forward
        else:
            start = indices[0]
            followers = set(indices[1:])
            remaining_to_skip: set[int] = set()

            def range_once(hidden_states, *args, **kwargs):
                state = hidden_states
                for index in indices:
                    state = originals[index](state, *args, **kwargs)
                return state

            def range_start_forward(hidden_states, *args, **kwargs):
                stats.patched_invocations += 1
                stats.actual_block_calls += config.num_loops * len(indices)
                output = recursive_euler(range_once, hidden_states, *args, config=config, **kwargs)
                remaining_to_skip.clear()
                remaining_to_skip.update(followers)
                return output

            blocks[start].forward = range_start_forward
            for index in indices[1:]:
                original = originals[index]

                def range_follower_forward(hidden_states, *args, _index=index, _original=original, **kwargs):
                    stats.patched_invocations += 1
                    if _index in remaining_to_skip:
                        remaining_to_skip.remove(_index)
                        return hidden_states
                    stats.actual_block_calls += 1
                    return _original(hidden_states, *args, **kwargs)

                blocks[index].forward = range_follower_forward

        yield stats
    finally:
        for index, original in originals.items():
            blocks[index].forward = original

