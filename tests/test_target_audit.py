import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from target_audit import audit_scale_rae_targets, candidate_paths, write_audit_report  # noqa: E402


class TinyAttention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)


class TinyBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.attn = TinyAttention(width)
        self.mlp = nn.Sequential(nn.Linear(width, 16), nn.GELU(), nn.Linear(16, width))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit_blocks = nn.ModuleList([TinyBlock() for _ in range(4)])


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.diff_head = nn.Module()
        self.diff_head.model = TinyDiT()


class TargetAuditTest(unittest.TestCase):
    def test_finds_exact_scale_rae_hierarchy_and_categories(self):
        report = audit_scale_rae_targets(Wrapper(), [1, 2], rank=2)
        self.assertEqual(report["dit_root"], "diff_head.model")
        self.assertEqual(report["depth"], 4)
        self.assertEqual(report["category_totals"]["attention_qkv"]["linear_count"], 2)
        self.assertEqual(report["category_totals"]["attention_output"]["linear_count"], 2)
        self.assertEqual(report["category_totals"]["mlp"]["linear_count"], 4)
        self.assertEqual(report["category_totals"]["conditioning"]["linear_count"], 2)
        self.assertEqual(
            candidate_paths(report, "attention_output"),
            ["dit_blocks.1.attn.proj", "dit_blocks.2.attn.proj"],
        )

    def test_parameter_estimate(self):
        report = audit_scale_rae_targets(Wrapper(), [0], rank=2)
        target = next(item for item in report["targets"] if item["relative_path"] == "attn.proj")
        self.assertEqual(target["base_parameters"], 64)
        self.assertEqual(target["lora_parameters"], 32)

    def test_writes_json(self):
        report = audit_scale_rae_targets(Wrapper(), [0], rank=2)
        with tempfile.TemporaryDirectory() as directory:
            output = write_audit_report(report, Path(directory) / "audit.json")
            self.assertTrue(output.is_file())
            self.assertIn('"attention_output"', output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
