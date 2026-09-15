import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recursive_lora import (  # noqa: E402
    AdaptedBlockRunner,
    LoRAConfig,
    RecursiveConfig,
    apply_lora_config,
    freeze_except_lora,
    inject_lora,
    load_lora_checkpoint,
    load_lora_state_dict,
    lora_state_dict,
    ordinary_forward,
    recursive_euler,
    recursive_euler_with_trace,
    save_lora_checkpoint,
)


class TinyBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, width)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.proj(self.norm(x))


class RecursiveLoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = TinyBlock()
        self.model = copy.deepcopy(self.base)
        inject_lora(self.model, ["proj"], rank=2, alpha=2)
        freeze_except_lora(self.model)
        self.x = torch.randn(2, 5, 8)

    def test_zero_initialized_lora_preserves_base(self):
        torch.testing.assert_close(self.model(self.x), self.base(self.x))

    def test_k_calls_share_one_parameter_set(self):
        output = recursive_euler(self.model, self.x, config=RecursiveConfig(num_loops=4))
        self.assertEqual(self.model.calls, 4)
        self.assertEqual(len(lora_state_dict(self.model)), 2)
        self.assertEqual(output.shape, self.x.shape)

    def test_only_lora_receives_gradient(self):
        target = torch.zeros_like(self.x)
        loss = (recursive_euler(self.model, self.x, config=RecursiveConfig(num_loops=4)) - target).square().mean()
        loss.backward()
        trainable = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, {"proj.lora_a", "proj.lora_b"})
        self.assertIsNotNone(self.model.proj.lora_b.grad)
        self.assertTrue(torch.isfinite(self.model.proj.lora_b.grad).all())
        self.assertIsNone(self.model.proj.base.weight.grad)

    def test_checkpoint_round_trip(self):
        with torch.no_grad():
            self.model.proj.lora_b.normal_()
        expected = recursive_euler(self.model, self.x, config=RecursiveConfig(num_loops=4))

        restored = copy.deepcopy(self.base)
        inject_lora(restored, ["proj"], rank=2, alpha=2)
        freeze_except_lora(restored)
        load_lora_state_dict(restored, lora_state_dict(self.model))
        actual = recursive_euler(restored, self.x, config=RecursiveConfig(num_loops=4))
        torch.testing.assert_close(actual, expected)

    def test_k1_matches_single_block_call(self):
        direct = self.model(self.x)
        self.model.calls = 0
        recursive = recursive_euler(self.model, self.x, config=RecursiveConfig(num_loops=1))
        torch.testing.assert_close(recursive, direct)
        self.assertEqual(self.model.calls, 1)

    def test_unified_config_and_transactional_validation(self):
        model = TinyBlock()
        config = LoRAConfig(("proj",), rank=2, alpha=4)
        self.assertEqual(apply_lora_config(model, config), ["proj"])
        untouched = TinyBlock()
        with self.assertRaises(AttributeError):
            inject_lora(untouched, ["proj", "missing"], rank=2)
        self.assertIsInstance(untouched.proj, nn.Linear)

    def test_every_recursive_step_is_in_the_graph(self):
        output, trace = recursive_euler_with_trace(
            self.model,
            self.x,
            config=RecursiveConfig(num_loops=4),
        )
        self.assertEqual(len(trace), 4)
        for step in trace:
            step.output_state.retain_grad()
        output.square().mean().backward()
        for index, step in enumerate(trace):
            self.assertEqual(step.index, index)
            self.assertIsNotNone(step.output_state.grad)
            self.assertTrue(torch.isfinite(step.output_state.grad).all())

    def test_explicit_ordinary_and_recursive_runtime_paths(self):
        runner = AdaptedBlockRunner(self.model, default_num_loops=4)
        runner(self.x, mode="ordinary")
        self.assertEqual(self.model.calls, 1)
        self.model.calls = 0
        runner(self.x, mode="recursive", num_loops=2)
        self.assertEqual(self.model.calls, 2)
        with self.assertRaises(ValueError):
            runner(self.x, mode="ordinary", num_loops=2)

    def test_metadata_checkpoint_and_config_guard(self):
        config = LoRAConfig(("proj",), rank=2, alpha=2)
        with torch.no_grad():
            self.model.proj.lora_b.normal_()
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = save_lora_checkpoint(
                Path(directory) / "lora.pt",
                self.model,
                config,
                metadata={"num_loops": 4},
            )
            restored = copy.deepcopy(self.base)
            apply_lora_config(restored, config)
            freeze_except_lora(restored)
            metadata = load_lora_checkpoint(path, restored, config)
            self.assertEqual(metadata, {"num_loops": 4})
            torch.testing.assert_close(
                ordinary_forward(restored, self.x),
                ordinary_forward(self.model, self.x),
            )
            with self.assertRaises(ValueError):
                load_lora_checkpoint(path, restored, LoRAConfig(("proj",), rank=1, alpha=2))


if __name__ == "__main__":
    unittest.main()
