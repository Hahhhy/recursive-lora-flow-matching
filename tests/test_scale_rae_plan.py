import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_scale_rae_plan import format_scale_rae_generation_prompt  # noqa: E402


class ScaleRAEPlanTest(unittest.TestCase):
    def test_official_generation_instruction_is_added_once(self):
        self.assertEqual(
            format_scale_rae_generation_prompt("a photo of a bench"),
            "Generate an image of a photo of a bench",
        )


if __name__ == "__main__":
    unittest.main()
