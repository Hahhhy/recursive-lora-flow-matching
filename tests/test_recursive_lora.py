import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recursive_lora import (  # noqa: E402
    RecursiveConfig,
    freeze_except_lora,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    recursive_euler,
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


if __name__ == "__main__":
    unittest.main()
