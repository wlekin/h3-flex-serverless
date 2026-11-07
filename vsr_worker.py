"""Optional RelayVSR postprocessing and standalone upscale jobs."""
import base64
from contextvars import ContextVar
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock
import time
import uuid

from relayvsr_test_worker import audio_digest, probe, render
import relayvsr_test_worker as vsr

FACTOR = ContextVar("vsr_factor", default=1)
ORIGINAL_GRAPH = vsr.graph
LIMIT = 20 * 1024**2


def factor(value):
    if type(value) is not int or value not in (1, 2, 4):
        raise ValueError("Choose no upscale, 2x or 4x")
    return value


def checked_source(source):
    before = probe(source)
    fps = Fraction(before["fps"])
    if before["width"] * before["height"] > 1408 * 768 or before["frames"] > 132 or not 1 <= fps <= 30:
        raise ValueError("Upscaling currently supports 768p-class clips up to 132 frames at 30 fps")
    return before


def selected_graph(fps):
    graph = ORIGINAL_GRAPH(fps)
    if FACTOR.get() == 4:
        graph["4"]["inputs"]["unet_name"] = "relayvsr_x4_int8_convrot.safetensors"
        graph["7"]["inputs"]["vae_name"] = "relayvsr_flash_vae_x4_fp32.safetensors"
    return graph


def upscale(source, metadata):
    before = checked_source(source)
    scale = FACTOR.get()
    started = time.monotonic()
    output = source.with_name(uuid.uuid4().hex + ".mp4")
    try:
        with TemporaryDirectory(prefix="i2v-vsr-") as tmp:
            folder = Path(tmp)
            shutil.copyfile(source, folder / "input.mp4")
            temporary = folder / "upscaled.mp4"
            graph_seconds = render(source, temporary, folder, Fraction(before["fps"]))
            after = probe(temporary)
            if (after["width"], after["height"]) != (before["width"] * scale, before["height"] * scale):
                raise ValueError("Upscale dimensions did not match the requested factor")
            if after["frames"] != before["frames"] or Fraction(after["fps"]) != Fraction(before["fps"]):
                raise ValueError("Upscale changed the video timing")
            if before["audio"] != after["audio"] or before["audio"] and audio_digest(source) != audio_digest(temporary):
                raise ValueError("Upscale did not preserve the original audio")
            shutil.copyfile(temporary, output.with_suffix(".tmp"))
            output.with_suffix(".tmp").replace(output)
        details = {**metadata, "upscale_factor": scale, "source_output_id": source.stem,
                   "upscale_seconds": time.monotonic() - started, "upscale_graph_seconds": graph_seconds,
                   "upscale_dimensions": [after["width"], after["height"]], "upscale_audio_preserved": True}
        details.pop("resident_until_ms", None)
        output.with_suffix(".json").write_text(json.dumps(details, indent=2))
        return output, details
    except Exception:
        output.with_suffix(".tmp").unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        output.with_suffix(".json").unlink(missing_ok=True)
        raise


def main():
    import handler as base
    import resident_worker
    import runpod

    vsr.graph = selected_graph
    publish = base.publish
    start = runpod.serverless.start
    lock = Lock()

    def release_models():
        runner = base.subprocess
        if hasattr(runner, "close"):
            runner.close()

    def publish_result(source, metadata):
        scale = FACTOR.get()
        if scale == 1:
            return publish(source, metadata)
        release_models()
        metadata = {**metadata, "upscale_factor": 1}
        metadata.pop("resident_until_ms", None)
        source.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        try:
            output, details = upscale(source, metadata)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            # A failed optional upscale still returns the completed original.
            fallback = {**metadata, "requested_upscale_factor": scale, "upscale_error": str(exc)}
            source.with_suffix(".json").write_text(json.dumps(fallback, indent=2))
            return publish(source, fallback)
        return publish(output, details)

    def standalone(data):
        if set(data) - {"action", "factor", "video_base64"}:
            raise ValueError("Unsupported upscale input")
        scale = factor(data.get("factor"))
        if scale == 1:
            raise ValueError("Choose 2x or 4x")
        encoded = data.get("video_base64")
        if not isinstance(encoded, str) or len(encoded) > (LIMIT * 4 + 2) // 3 + 4:
            raise ValueError("Choose a cached MP4 video up to 20 MiB")
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > LIMIT:
            raise ValueError("Choose a cached MP4 video up to 20 MiB")
        release_models()
        volume = Path(base.os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
        output_dir = volume / "outputs/h3-flex"
        output_dir.mkdir(parents=True, exist_ok=True)
        token = FACTOR.set(scale)
        try:
            with TemporaryDirectory(prefix="i2v-upscale-") as tmp:
                source = Path(tmp) / (uuid.uuid4().hex + ".mp4")
                source.write_bytes(raw)
                checked_source(source)
                gpu = subprocess.check_output(
                    ["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
                    timeout=30).decode().strip()
                metadata = {"operation": "upscale", "gpu": gpu}
                # upscale() creates its validated result beside the source.
                output, details = upscale(source, metadata)
                destination = output_dir / output.name
                shutil.copyfile(output, destination)
                details.pop("source_output_id", None)
                destination.with_suffix(".json").write_text(json.dumps(details, indent=2))
                return publish(destination, details)
        finally:
            FACTOR.reset(token)

    def install(config):
        original = config["handler"]

        def handler(job):
            with lock:
                data = job.get("input")
                if not isinstance(data, dict):
                    return {"error": "Input must be an object"}
                try:
                    if data.get("action") == "upscale":
                        return standalone(data)
                    data = dict(data)
                    scale = factor(data.pop("upscale_factor", 1))
                    if scale != 1 and (data.get("resolution", "480p") != "480p" or data.get("duration", 5) != 5):
                        raise ValueError("H3 upscaling currently requires 480p and 5 seconds")
                    token = FACTOR.set(scale)
                    try:
                        return original({**job, "input": data})
                    finally:
                        FACTOR.reset(token)
                except (ValueError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    return {"error": str(exc)}

        base.publish = publish_result
        return start({**config, "handler": handler})

    runpod.serverless.start = install
    resident_worker.main()


if __name__ == "__main__":
    main()
