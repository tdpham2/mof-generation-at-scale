"""Check fidelity of the saved-input replay and failure classification offline."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("replay", ROOT / "cp2k-test/replay-elpa.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class ELPAReplayTests(unittest.TestCase):
    def test_variants_preserve_physics_and_exact_coordinates(self):
        from ase.calculators.cp2k import parse_input
        import numpy as np
        for path in replay.CASES.glob("*.inp"):
            original = path.read_text()
            baseline = parse_input(original)
            with tempfile.TemporaryDirectory() as tmp:
                for variant in ("original", "cpu-elpa", "gpu-block32"):
                    text = replay.variant_input(original, variant)
                    root = parse_input(text)
                    self.assertEqual(root.get_subsection("FORCE_EVAL").write(),
                                     baseline.get_subsection("FORCE_EVAL").write())
                    copy = Path(tmp) / "copy.inp"
                    copy.write_text(text)
                    np.testing.assert_array_equal(replay.diag.input_atoms(copy).positions,
                                                  replay.diag.input_atoms(path).positions)
                    if variant == "cpu-elpa":
                        self.assertIn("ELPA_KERNEL GENERIC", root.get_subsection("GLOBAL").keywords)
                    elif variant == "gpu-block32":
                        self.assertEqual(root.get_subsection("GLOBAL/FM").keywords,
                                         ["NROW_BLOCKS 32", "NCOL_BLOCKS 32"])
                    self.assertIn("IGNORE_CONVERGENCE_FAILURE", text)
            self.assertEqual(path.read_text(), original)

    def test_crashes_and_unconverged_scf_are_not_passing_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "launcher.stderr").write_text("SIGFPE\nunpack_row_group_real_gpu_double\n")
            self.assertEqual(replay.inspect_output(folder, {"status": "failed"})["outcome"], "elpa_sigfpe")
            (folder / "launcher.stderr").write_text("")
            (folder / "cp2k.out").write_text(
                "GLOBAL| Total number of message passing processes 8\n"
                "GLOBAL| Number of threads for this process 8\n"
                "SCF run NOT converged\nTotal energy: -600.0\n"
            )
            result = replay.inspect_output(folder, {"status": "passed"})
            self.assertEqual(result["outcome"], "completed_scf_unconverged")
            self.assertEqual(result["scf_unconverged"], 1)


if __name__ == "__main__":
    unittest.main()
