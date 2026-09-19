import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recursive_lora import freeze_except_lora, inject_lora  # noqa: E402
from training_loops import training_loop_patch  # noqa: E402


class AddBlock(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.calls = 0

    def forward(self, x, *args, **kwargs):
        self.calls += 1
        return x + self.proj(x)


class TinyDiT(nn.Module):
    def __init__(self, depth=4, width=4):
        super().__init__()
        self.dit_blocks = nn.ModuleList(AddBlock(width) for _ in range(depth))

    def forward(self, x):
        for block in self.dit_blocks:
            x = block(x)
        return x


class TrainingLoopsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.base = TinyDiT()
        self.x = torch.randn(2, 3, 4)

    def test_layerwise_calls_each_selected_block_k_times(self):
        model = copy.deepcopy(self.base)
        with training_loop_patch(model, [1, 2], granularity="layerwise", num_loops=3) as stats:
            model(self.x)
        self.assertEqual([block.calls for block in model.dit_blocks], [1, 3, 3, 1])
        self.assertEqual(stats.actual_block_calls, 6)

    def test_rangewise_loops_complete_window_and_skips_followers(self):
        model = copy.deepcopy(self.base)
        with training_loop_patch(model, [1, 2], granularity="rangewise", num_loops=3) as stats:
            model(self.x)
        self.assertEqual([block.calls for block in model.dit_blocks], [1, 3, 3, 1])
        self.assertEqual(stats.actual_block_calls, 6)

    def test_granularities_have_different_computation(self):
        layerwise = copy.deepcopy(self.base)
        rangewise = copy.deepcopy(self.base)
        with training_loop_patch(layerwise, [1, 2], granularity="layerwise", num_loops=3):
            layer_output = layerwise(self.x)
        with training_loop_patch(rangewise, [1, 2], granularity="rangewise", num_loops=3):
            range_output = rangewise(self.x)
        self.assertFalse(torch.allclose(layer_output, range_output))

    def test_lora_gradient_flows_for_both_granularities(self):
        for granularity in ("layerwise", "rangewise"):
            model = copy.deepcopy(self.base)
            inject_lora(model, ["dit_blocks.1.proj", "dit_blocks.2.proj"], rank=2, alpha=2)
            freeze_except_lora(model)
            with training_loop_patch(model, [1, 2], granularity=granularity, num_loops=3):
                model(self.x).square().mean().backward()
            for index in (1, 2):
                self.assertIsNotNone(model.dit_blocks[index].proj.lora_b.grad)
                self.assertTrue(torch.isfinite(model.dit_blocks[index].proj.lora_b.grad).all())

    def test_patch_restores_original_forward_after_error(self):
        model = copy.deepcopy(self.base)
        original = model.dit_blocks[1].forward
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with training_loop_patch(model, [1], granularity="layerwise", num_loops=2):
                raise RuntimeError("boom")
        self.assertEqual(model.dit_blocks[1].forward, original)

    def test_rangewise_requires_contiguous_interval(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            with training_loop_patch(self.base, [0, 2], granularity="rangewise", num_loops=2):
                pass


if __name__ == "__main__":
    unittest.main()

