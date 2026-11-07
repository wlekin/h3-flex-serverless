"""Isolated RelayVSR 2x benchmark for short, inline MP4 clips."""
import base64
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

COMFY = Path("/opt/ComfyUI")
LIMIT = 8 * 1024 * 1024
ADDRESS = "http://127.0.0.1:8188"


def request(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = Request(ADDRESS + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as response:
        return json.load(response)


def graph(fps):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    return {
        "1": node("LoadVideo", file="input.mp4"),
        "2": node("GetVideoComponents", video=["1", 0]),
        "4": node("UNETLoader", unet_name="relayvsr_x2_lightx2v_int8_convrot.safetensors", weight_dtype="default"),
        "5": node("RelayVSRConditioning", model=["4", 0], images=["2", 0]),
        "6": node("KSampler", model=["4", 0], positive=["5", 0], negative=["5", 0], latent_image=["5", 1],
                  seed=0, steps=1, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0),
        "7": node("VAELoader", vae_name="relayvsr_flash_vae_x2_fp32.safetensors"),
        "104": node("VSRDecode", vae=["7", 0], samples=["6", 0], images=["2", 0], detail=1.0),
        "105": node("CreateVideo", images=["104", 0], fps=float(fps)),
        "106": node("SaveVideo", video=["105", 0], filename_prefix="relayvsr", format="auto"),
    }


def probe(path):
    data = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)], timeout=30))
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    audio = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    return {
        "width": video["width"], "height": video["height"], "fps": video["avg_frame_rate"],
        "frames": int(video["nb_read_frames"]), "audio": len(audio),
    }


def audio_digest(path):
    return subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "hash",
         "-hash", "sha256", "-"], timeout=30).decode().strip()


def render(source, output, folder, fps):
    out_dir = folder / "comfy-output"
    with subprocess.Popen(
        ["python", "-u", str(COMFY / "main.py"), "--listen", "127.0.0.1", "--disable-auto-launch",
         "--disable-all-custom-nodes", "--offline", "--disable-metadata", "--input-directory", str(folder),
         "--output-directory", str(out_dir)], cwd=COMFY
    ) as server:
        try:
            ready_by = time.monotonic() + 70
            while True:
                if server.poll() is not None or time.monotonic() >= ready_by:
                    raise RuntimeError("ComfyUI did not start")
                try:
                    request("/system_stats")
                    break
                except OSError:
                    time.sleep(1)
            started = time.monotonic()
            pid = request("/prompt", {"prompt": graph(fps)})["prompt_id"]
            while time.monotonic() - started < 440:
                entry = request("/history/" + pid).get(pid)
                if entry:
                    if entry["status"]["status_str"] != "success":
                        raise RuntimeError("RelayVSR graph failed. Inspect worker logs.")
                    files = list(out_dir.glob("relayvsr_*.mp4"))
                    if len(files) != 1:
                        raise RuntimeError("RelayVSR produced an unexpected number of files")
                    break
                time.sleep(2)
            else:
                raise RuntimeError("RelayVSR graph timed out")
            elapsed = time.monotonic() - started
        finally:
            if server.poll() is None:
                server.kill()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(files[0]), "-i", str(source),
                    "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", str(output)],
                   check=True, timeout=30)
    return elapsed


def handler(job):
    started = time.monotonic()
    try:
        encoded = job["input"]["video_base64"]
        if not isinstance(encoded, str) or len(encoded) > (LIMIT * 4 + 2) // 3 + 4:
            raise ValueError("Input exceeds test size limit")
        content = base64.b64decode(encoded, validate=True)
        if not content or len(content) > LIMIT:
            raise ValueError("Invalid test video size")
        with tempfile.TemporaryDirectory(prefix="relayvsr-") as temp:
            folder = Path(temp)
            source, output = folder / "input.mp4", folder / "upscaled.mp4"
            source.write_bytes(content)
            before = probe(source)
            fps = Fraction(before["fps"])
            if before["width"] * before["height"] > 480 * 864 or before["frames"] > 132 or not 1 <= fps <= 30:
                raise ValueError("Test requires a 480p-class short clip")
            elapsed = render(source, output, folder, fps)
            after = probe(output)
            assert (after["width"], after["height"]) == (2 * before["width"], 2 * before["height"])
            assert after["frames"] == before["frames"] and Fraction(after["fps"]) == fps
            assert after["audio"] == before["audio"]
            if before["audio"]:
                assert audio_digest(source) == audio_digest(output), "Audio changed"
            return {"source": before, "result": after, "result_bytes": output.stat().st_size,
                    "audio_pcm_identical": bool(before["audio"]), "graph_seconds": elapsed,
                    "total_seconds": time.monotonic() - started}
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, AssertionError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
