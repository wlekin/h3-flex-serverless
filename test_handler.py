import base64
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import handler


class WorkerTests(unittest.TestCase):
    def test_invalid_requests_do_not_start_cli(self):
        invalid = [None, [], {}, {"prompt": " "}, {"prompt": "x", "duration": True},
                   {"prompt": "x", "duration": 16}, {"prompt": "x", "seed": -1},
                   {"prompt": "x", "task": "i2va"},
                   {"prompt": "x", "cpu_offload": "false"},
                   {"prompt": "x", "revision": "untrusted"},
                   {"prompt": "x", "task": "ref2va", "references": [{"kind": "audio", "base64": "YQ=="}]}]
        with patch("handler.subprocess.run") as run:
            for data in invalid:
                with self.subTest(data=data):
                    self.assertIn("error", handler.handler({"input": data}))
            run.assert_not_called()

    def test_prompt_never_enters_shell_arguments(self):
        prompt = '$(touch /tmp/nope); `echo no` & --revision evil'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = handler.validate({"prompt": prompt})
            args = handler.command(request, root, root / "result.mp4")
            self.assertNotIn(prompt, args)
            self.assertEqual((root / "prompt.txt").read_text(), prompt)
            self.assertIn(handler.RELEASE["model_revision"], args)

    def test_media_limits_and_invalid_base64(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            for value in ("%%%%", "", "data:image/png,abc", base64.b64encode(b"abcd").decode()):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    handler.decode_media(value, path, 3)
            self.assertFalse(path.exists())

    def test_health_does_not_download_or_generate(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RUNPOD_VOLUME_PATH": directory}), patch("handler.subprocess.run") as run:
            result = handler.handler({"input": {"action": "health"}})
            self.assertEqual(result["status"], "wrapper-ready")
            self.assertFalse(result["generation_verified"])
            run.assert_not_called()

    def test_generation_returns_video_and_persists_output(self):
        def fake_cli(args, **kwargs):
            self.assertNotIn("shell", kwargs)
            self.assertTrue(kwargs["check"])
            output = Path(args[args.index("--output") + 1])
            output.write_bytes(b"test-video")
            output.with_suffix(".json").write_text(json.dumps({"seed": 42}))

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RUNPOD_VOLUME_PATH": directory, "OUTPUT_S3_BUCKET": ""}), patch("handler.subprocess.run", side_effect=fake_cli):
            result = handler.handler({"input": {"prompt": "A boat"}})
            self.assertEqual(base64.b64decode(result["video"]["data"]), b"test-video")
            self.assertEqual(result["metadata"]["seed"], 42)
            self.assertEqual(len(list(Path(directory).rglob("*.mp4"))), 1)

    def test_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RUNPOD_VOLUME_PATH": directory}), patch("handler.subprocess.run", side_effect=subprocess.TimeoutExpired("pika-h3-pro", 1)):
            self.assertEqual(handler.handler({"input": {"prompt": "A boat"}})["error"], "Generation timed out")


if __name__ == "__main__":
    unittest.main()
