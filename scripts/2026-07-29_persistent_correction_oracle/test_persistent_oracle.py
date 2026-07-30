import importlib.util
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("persistent_oracle.py")
SPEC = importlib.util.spec_from_file_location(
    "persistent_oracle", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def boundary(chunk, safe):
    return {
        "chunk": chunk,
        "correction_actions": chunk * 8,
        "absolute_handoff_step": 100 + chunk * 8,
        "safe_recoverable_handoff": safe,
        "continuation": {
            "outcome": "safe_success" if safe else "ncr"
        },
    }


class TestPersistentOracle(unittest.TestCase):
    def test_seed_schedule_is_unique(self):
        seeds = {
            MODULE.correction_seed(123, trajectory, chunk)
            for trajectory in range(4)
            for chunk in range(20)
        }
        self.assertEqual(len(seeds), 80)
        self.assertEqual(MODULE.continuation_seed(123), 124)

    def test_earliest_safe_handoff(self):
        trajectory = {
            "boundaries": [
                boundary(1, False),
                boundary(4, True),
                boundary(3, True),
            ]
        }
        selected = MODULE.earliest_safe_handoff(trajectory)
        self.assertEqual(selected["chunk"], 3)

    def test_summary_passes_on_one_state(self):
        states = [
            {
                "trajectories": [
                    {
                        "boundaries": [
                            boundary(1, False),
                            boundary(2, True),
                        ],
                        "direct_success": False,
                        "correction_collision_any": False,
                    },
                    {
                        "boundaries": [boundary(1, False)],
                        "direct_success": False,
                        "correction_collision_any": True,
                    },
                ]
            },
            {
                "trajectories": [
                    {
                        "boundaries": [boundary(1, False)],
                        "direct_success": False,
                        "correction_collision_any": False,
                    }
                ]
            },
        ]
        summary = MODULE.summarize_states(states)
        self.assertTrue(summary["pilot_pass"])
        self.assertEqual(
            summary["states_with_safe_recoverable_handoff"], 1
        )
        self.assertEqual(
            summary["trajectories_with_safe_recoverable_handoff"], 1
        )
        self.assertEqual(summary["safe_recoverable_handoffs"], 1)
        self.assertEqual(summary["total_safe_prefix_boundaries"], 4)

    def test_compact_chunk_ranges(self):
        self.assertEqual(MODULE.compact_chunk_ranges([]), "--")
        self.assertEqual(MODULE.compact_chunk_ranges([5]), "5")
        self.assertEqual(
            MODULE.compact_chunk_ranges([5, 6, 7, 11, 13, 14]),
            "5-7, 11, 13-14",
        )

    def test_select_saved_states_applies_offset_after_global_ordering(self):
        with mock.patch.object(
            MODULE,
            "load_saved_states",
            return_value=["s0", "s1", "s2", "s3", "s4"],
        ) as loader:
            selected = MODULE.select_saved_states(["bank"], 3, 2)
        loader.assert_called_once_with(["bank"], 5)
        self.assertEqual(selected, ["s3", "s4"])

    def test_select_saved_states_rejects_short_bank(self):
        with mock.patch.object(
            MODULE,
            "load_saved_states",
            return_value=["s0", "s1", "s2"],
        ):
            with self.assertRaisesRegex(ValueError, "only 3 states"):
                MODULE.select_saved_states(["bank"], 2, 2)


if __name__ == "__main__":
    unittest.main()
