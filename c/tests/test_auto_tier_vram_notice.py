"""--auto-tier must say when it is leaving the plan's VRAM tier on the floor.

Dropping the tier without --gpu is the contract and stays the contract: --gpu
and --vram are what select a CUDA-capable build, so asking for a plan must not
turn a CPU-only sibling binary into an attempted CUDA launch.

Doing it in silence was the defect. `coli plan` and `coli doctor` print the
VRAM tier, --auto-tier says it applies the plan, and the tier then disappears
with nothing on screen tying the two together -- which on the box in #1581 was
the difference between 11.8 and 21 tok/s with no clue as to why.

So this checks two things and they pull in opposite directions: the notice
appears when the tier was real, and stays quiet when it was not. A warning that
fires on every CPU-only launch is a warning people learn to scroll past.

It also checks that a plan of the wrong shape prints nothing and raises
nothing. This function only writes to a terminal; it must never be the reason a
launch fails.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

_loader = importlib.machinery.SourceFileLoader("coli_cli_vram", str(HERE / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli_vram", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


def plan(devices, budget_bytes):
    return {"tiers": {"vram": {"devices": devices, "budget_bytes": budget_bytes}}}


def notice_for(p):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        coli.report_unapplied_vram_tier(p)
    return err.getvalue()


QUALIFIED = [{"index": 0, "name": "NVIDIA GeForce RTX 4060 Ti",
              "free_bytes": 8 * 1000 ** 3}]


class VramNoticeTest(unittest.TestCase):
    def test_a_real_tier_is_announced(self):
        out = notice_for(plan(QUALIFIED, 5_200_000_000))
        self.assertIn("VRAM tier", out)
        self.assertIn("5.2 GB", out, "the notice should say how much is unused")
        self.assertIn("NVIDIA GeForce RTX 4060 Ti", out, "and on which device")
        self.assertIn("--gpu auto", out, "and what to do about it")

    def test_an_unqualified_device_is_not_worth_a_warning(self):
        """free_bytes None means the planner never qualified that memory as a
        budget, so the tier was not going to be applied with the GPU on either."""
        out = notice_for(plan([{"index": 0, "name": "AMD", "free_bytes": None}],
                              5_200_000_000))
        self.assertEqual(out, "")

    def test_a_zero_budget_is_not_worth_a_warning(self):
        self.assertEqual(notice_for(plan(QUALIFIED, 0)), "")

    def test_no_devices_is_not_worth_a_warning(self):
        self.assertEqual(notice_for(plan([], 5_200_000_000)), "")

    def test_a_plan_of_the_wrong_shape_is_silent_and_harmless(self):
        """It only prints. It must never be why a launch stops."""
        for broken in ({}, {"tiers": {}}, {"tiers": {"vram": {}}},
                       {"tiers": {"vram": {"devices": ["not a dict"],
                                           "budget_bytes": 1}}},
                       {"tiers": {"vram": {"devices": None, "budget_bytes": 1}}}):
            with self.subTest(plan=broken):
                self.assertEqual(notice_for(broken), "")


if __name__ == "__main__":
    unittest.main()
