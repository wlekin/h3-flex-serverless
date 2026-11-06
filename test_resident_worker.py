import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from resident_worker import GenerationProcess, eligible


class ResidentWorkerTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.spec = self.root / "custom-loras.json"
        self.spec.write_text("[]")
        self.output = self.root / "video.mp4"
        self.args = ["python", "/app/ref_worker.py", "--child", str(self.spec),
                     "--task", "ref2va", "--resolution", "480p",
                     "--duration", "5", "--output", str(self.output)]

    def test_only_standard_480p_reference_jobs_without_loras_use_resident_process(self):
        self.assertTrue(eligible(self.args))
        self.assertFalse(eligible([*self.args[:1], "/app/turbo_worker.py", *self.args[2:]]))
        self.assertFalse(eligible([a if a != "480p" else "768p" for a in self.args]))
        self.spec.write_text(json.dumps([{"name": "test.safetensors"}]))
        self.assertFalse(eligible(self.args))

    def test_two_requests_use_existing_process(self):
        runner = GenerationProcess()
        runner.process = Mock()
        runner.connection = Mock()
        with patch.object(runner, "start") as start:
            for _ in range(2):
                self.assertEqual(runner.run(self.args, check=True, timeout=30).returncode, 0)
        start.assert_not_called()
        self.assertEqual(runner.connection.send.call_count, 2)

    def test_failed_resident_process_cleans_partial_output_before_fallback(self):
        runner = GenerationProcess()
        runner.process = Mock()
        runner.process.is_alive.side_effect = [True, False, False]
        runner.connection = Mock()
        runner.connection.recv.side_effect = EOFError
        self.output.touch()
        self.output.with_suffix(".json").touch()

        def staged(argv, **kwargs):
            self.assertFalse(self.output.exists())
            self.assertFalse(self.output.with_suffix(".json").exists())
            self.assertIsNone(runner.process)
            return subprocess.CompletedProcess(argv, 0)

        with patch("resident_worker.subprocess.run", side_effect=staged) as run:
            runner.run(self.args, check=True, timeout=30)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
