"""Test H3 generation and RelayVSR in one isolated RunPod request."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import time

import handler as h3
from relayvsr_test_worker import audio_digest, probe, render
import signed_input_worker


def publish(original, metadata):
    from fractions import Fraction

    started = time.monotonic()
    source = probe(original)
    report = {
        "source": source,
        "original_name": original.name,
        "gpu": metadata.get("gpu"),
        "h3_timings": metadata.get("timings"),
    }
    try:
        fps = Fraction(source["fps"])
        if source["width"] * source["height"] > 480 * 864 or source["frames"] > 132:
            raise ValueError("Combined test requires a short 480p clip")
        with TemporaryDirectory(prefix="h3-relayvsr-") as tmp:
            folder = Path(tmp)
            shutil.copyfile(original, folder / "input.mp4")
            upscaled = folder / "upscaled.mp4"
            graph_seconds = render(original, upscaled, folder, fps)
            result = probe(upscaled)
            if (result["width"], result["height"]) != (source["width"] * 2, source["height"] * 2):
                raise ValueError("RelayVSR output is not exactly 2x")
            if result["frames"] != source["frames"] or Fraction(result["fps"]) != fps:
                raise ValueError("RelayVSR changed the video timing")
            if not source["audio"] or source["audio"] != result["audio"] or audio_digest(original) != audio_digest(upscaled):
                raise ValueError("RelayVSR did not preserve the original audio")
            report.update(result=result, audio_pcm_identical=True, graph_seconds=graph_seconds,
                          upscale_seconds=time.monotonic() - started, result_bytes=upscaled.stat().st_size)
        if not original.is_file():
            raise RuntimeError("The original H3 clip was lost")
    except Exception as exc:
        report["error"] = str(exc)
    return report


if __name__ == "__main__":
    # The staged H3 subprocess has exited and released VRAM before publish runs.
    h3.publish = publish
    signed_input_worker.main()
