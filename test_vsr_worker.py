from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import vsr_worker


class VsrWorkerTests(unittest.TestCase):
    def test_factors_and_matching_checkpoint_selection(self):
        for scale in (1, 2, 4):
            self.assertEqual(vsr_worker.factor(scale), scale)
        for scale in (True, "4", 3, None):
            with self.assertRaises(ValueError):
                vsr_worker.factor(scale)
        token = vsr_worker.FACTOR.set(4)
        try:
            graph = vsr_worker.selected_graph(24)
            self.assertIn("x4_", graph["4"]["inputs"]["unet_name"])
            self.assertIn("x4_", graph["7"]["inputs"]["vae_name"])
        finally:
            vsr_worker.FACTOR.reset(token)

    def test_upscale_preserves_original_and_checks_audio_before_publishing(self):
        source_info = {"width": 832, "height": 480, "frames": 124, "fps": "24/1", "audio": 1}
        output_info = {**source_info, "width": 3328, "height": 1920}
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / ("a" * 32 + ".mp4")
            source.write_bytes(b"original")
            def render(_source, output, *_):
                output.write_bytes(b"upscaled")
                return 40
            token = vsr_worker.FACTOR.set(4)
            try:
                with patch.object(vsr_worker, "probe", side_effect=[source_info, output_info]), patch.object(
                    vsr_worker, "render", side_effect=render
                ), patch.object(vsr_worker, "audio_digest", return_value="same"):
                    output, metadata = vsr_worker.upscale(source, {"resident_until_ms": 123})
                self.assertEqual(source.read_bytes(), b"original")
                self.assertEqual(output.read_bytes(), b"upscaled")
                self.assertEqual(metadata["upscale_dimensions"], [3328, 1920])
                self.assertNotIn("resident_until_ms", metadata)
                output.unlink()
                output.with_suffix(".json").unlink()
                with patch.object(vsr_worker, "probe", side_effect=[source_info, output_info]), patch.object(
                    vsr_worker, "render", side_effect=render
                ), patch.object(vsr_worker, "audio_digest", side_effect=["source", "changed"]):
                    with self.assertRaisesRegex(ValueError, "audio"):
                        vsr_worker.upscale(source, {})
                self.assertEqual(list(Path(tmp).glob("*.mp4")), [source])
            finally:
                vsr_worker.FACTOR.reset(token)


if __name__ == "__main__":
    unittest.main()
