"""Exercise standalone CP2K diagnostic startup without PBS or a simulation."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("block32", ROOT / "cp2k-test/check-block32.py")
block32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(block32)


class Block32StartupTests(unittest.TestCase):
    def test_fresh_checkout_creates_run_directory_and_records_launch_failure(self):
        with tempfile.TemporaryDirectory(prefix="cp2k fresh checkout ") as tmp:
            root = Path(tmp)
            inputs = root / "cp2k-test/block32-inputs"
            inputs.mkdir(parents=True)
            contents = (ROOT / "cp2k-test/block32-inputs/standalone-112.inp").read_bytes()
            (inputs / "case.inp").write_bytes(contents)
            manifest = {"cases": [{"name": "case", "input": "case.inp",
                                   "sha256": hashlib.sha256(contents).hexdigest()}]}
            (inputs / "manifest.json").write_text(json.dumps(manifest))
            hosts = root / "hosts"
            hosts.write_text("node1\nnode2\n")
            with patch.object(block32, "ROOT", root), \
                    patch.dict(os.environ, PBS_JOBID="test", PBS_NODEFILE=str(hosts)), \
                    patch.object(block32.signal, "signal"), \
                    patch.object(block32.diag, "execute_stage", return_value={
                        "status": "failed", "exit_code": 1,
                    }) as launch, redirect_stdout(StringIO()):
                self.assertEqual(block32.main(), 1)

            launch.assert_called_once()
            output, = (root / "run").iterdir()
            self.assertEqual((output / "hosts").read_text(), "node1\nnode2\n")
            self.assertEqual((output / "case/cp2k.inp").read_bytes(), contents)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["stages"][0]["validation_error"], "CP2K launch: failed")


if __name__ == "__main__":
    unittest.main()
