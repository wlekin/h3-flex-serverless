import hashlib
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import lora_compat


class MysticV5Test(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.source = Path(folder.name) / "input.safetensors"
        self.dest = Path(folder.name) / "converted.safetensors"
        code = Path(__file__).with_name("i2v_slim_bundle.py").read_text(encoding="utf-8")
        self.worker = {"__name__": "test_worker"}
        exec(compile(lora_compat.patch_worker(code), "lora_worker.py", "exec"), self.worker)

    def fixture(self, include_output=True, base=lora_compat.MYSTIC_V5_METADATA):
        header = {"__metadata__": {"ss_base_model_version": base, "ss_network_alpha": "420"}}
        data = bytearray()
        targets = {"audio_patch_proj": (32, 8)}
        if include_output:
            targets["final_layer.audio_out"] = (8, 32)
        for target, (inputs, outputs) in targets.items():
            for part, shape in {"A": [16, inputs], "B": [outputs, 16]}.items():
                start = len(data)
                data.extend(struct.pack("<f", 0.25) * (shape[0] * shape[1]))
                header[f"diffusion_model.{target}.lora_{part}.weight"] = {
                    "shape": shape, "dtype": "F32", "data_offsets": [start, len(data)]}
        encoded = json.dumps(header).encode()
        self.source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def convert(self):
        return self.worker["convert"](self.source, self.dest, dimensions=(8, 4, 16))

    def test_both_audio_projections_use_rank16_unity_scaling(self):
        with patch.object(lora_compat, "MYSTIC_V5_SHA256", self.fixture()):
            report = self.convert()
        self.assertEqual(report["rank_pattern"], {"audio_proj_in": 16, "audio_proj_out": 16})
        self.assertEqual(report["audio_targets"], ["audio_proj_in", "audio_proj_out"])
        options = self.worker["adapter_options"](report)
        self.assertEqual(options["alpha_pattern"], report["rank_pattern"])
        with self.dest.open("rb") as f:
            count = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(count))
            self.assertEqual(header["audio_proj_in.lora_A.weight"]["shape"], [16, 32])
            self.assertEqual(f.read(2), struct.pack("<H", 0x3e80))

    def test_same_metadata_cannot_bypass_hash_or_allow_partial_audio(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "different base model"):
            self.convert()
        with patch.object(lora_compat, "MYSTIC_V5_SHA256", self.fixture(include_output=False)):
            with self.assertRaisesRegex(ValueError, "both audio projections"):
                self.convert()

    def test_generic_audio_targets_remain_rejected(self):
        self.fixture(base="minimax_h3")
        with self.assertRaisesRegex(ValueError, "Unsupported LoRA tensor"):
            self.convert()


if __name__ == "__main__":
    unittest.main()
