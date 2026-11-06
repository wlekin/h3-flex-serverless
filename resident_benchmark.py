"""Isolated standard Ref2VA benchmark with two passes in one GPU process."""
import json
from copy import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path("/runpod-volume/h3-flex")


def compare(generate, args):
    import torch
    from pika_h3.core import conditioning, loader, vae

    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 90 * 1024**3:
        raise RuntimeError("Resident H3 benchmark requires a full 96 GB GPU")

    encoder_type = conditioning.H3ConditionEncoder
    decoder_type = vae.H3Decoders
    build_transformer = loader.build_transformer
    encoder = decoder = transformer = None

    class ResidentEncoder(encoder_type):
        def free(self, what="all"):
            pass  # This benchmark owns the encoder until both passes finish.

    def get_encoder(*args, **kwargs):
        nonlocal encoder
        if encoder is None:
            encoder = ResidentEncoder(*args, **kwargs, keep_loaded=True)
        return encoder

    def get_decoder(*args, **kwargs):
        nonlocal decoder
        if decoder is None:
            decoder = decoder_type(*args, **kwargs)
        return decoder

    def get_transformer(*args, **kwargs):
        nonlocal transformer
        if transformer is None:
            transformer = build_transformer(*args, **kwargs)
        return transformer

    # Only the first pass attaches the immutable standard Flex adapter.
    with patch.object(conditioning, "H3ConditionEncoder", get_encoder), patch.object(
        vae, "H3Decoders", get_decoder
    ), patch.object(loader, "build_transformer", get_transformer):
        cold_start = time.monotonic()
        generate(args)
        cold_wall = time.monotonic() - cold_start
        path = args.output.with_suffix(".json")
        result = json.loads(path.read_text())
        warm_output = args.output.with_name("warm.mp4")
        warm_args = copy(args)
        warm_args.output = warm_output
        with patch.object(loader, "apply_lora"), patch.object(loader, "load_lora_weights"):
            warm_start = time.monotonic()
            generate(warm_args)
            warm_wall = time.monotonic() - warm_start
        warm = json.loads(warm_output.with_suffix(".json").read_text())
    result["resident_benchmark"] = {
        "cold_seconds": cold_wall,
        "warm_seconds": warm_wall,
        "cold_timings": result["timings"],
        "warm_timings": warm["timings"],
        "cold_sampling_peak_bytes": result["sampling_peak_allocated_bytes"],
        "warm_sampling_peak_bytes": warm["sampling_peak_allocated_bytes"],
    }
    path.write_text(json.dumps(result))


def child():
    from pika_h3 import pipeline
    import ref_worker

    generate = pipeline.generate
    pipeline.generate = lambda args: compare(generate, args)
    ref_worker.ref_child("nf4")


def parent():
    import handler as base
    import runpod

    command = base.command

    def benchmark_command(request, workdir):
        args = command(request, workdir, workdir / "cold.mp4")
        return [sys.executable, __file__, "--child", *args[1:],
                "--base-path", str(ROOT / "native-assets/base"),
                "--adapter-path", str(ROOT / "native-assets/adapter/ref2va")]

    # Reuse the request validation and media staging. Return timing only.
    def handler(job):
        raw = job.get("input")
        if not isinstance(raw, dict) or raw.get("task") != "ref2va" or raw.get("action", "generate") != "generate":
            return {"error": "Benchmark accepts standard Ref2VA generation only"}
        if raw.get("resolution") != "480p" or raw.get("duration") != 5 or raw.get("cpu_offload"):
            return {"error": "Benchmark requires 480p, five seconds and no CPU offload"}
        raw = {**raw, "cpu_offload": False}
        try:
            request = base.validate(raw)
            from tempfile import TemporaryDirectory
            import subprocess

            with TemporaryDirectory(prefix="h3-resident-bench-") as tmp:
                folder = Path(tmp)
                subprocess.run(benchmark_command(request, folder), check=True, timeout=540)
                metadata = json.loads((folder / "cold.json").read_text())
                return {"gpu": metadata["gpu"], **metadata["resident_benchmark"]}
        except subprocess.TimeoutExpired:
            return {"error": "Benchmark exceeded 540 seconds"}
        except subprocess.CalledProcessError as exc:
            return {"error": f"Benchmark process exited {exc.returncode}. Inspect worker logs."}
        except (ValueError, OSError) as exc:
            return {"error": str(exc)}

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        del sys.argv[1]
        child()
    else:
        parent()
