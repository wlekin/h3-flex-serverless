"""RunPod queue worker for the unmodified official Pika CLI.

This worker is for the full native runtime, not quantized ComfyUI checkpoints.
No model downloads or GPU initialization happen at module import.
"""
import base64
import binascii
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

RELEASE = json.loads((Path(__file__).parent / "release.json").read_text())
ASPECTS = {"21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"}
TASKS = {"t2va": (False, False), "i2va": (True, False), "l2va": (False, True), "fl2va": (True, True), "ref2va": (False, False)}


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return value


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("input must be a JSON object")
    allowed = {"action", "task", "prompt", "resolution", "aspect", "duration", "seed", "cpu_offload", "first_image_base64", "last_image_base64", "references"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError("Unsupported fields: " + ", ".join(sorted(unknown)))
    action = data.get("action", "generate")
    if action not in {"generate", "health"}:
        raise ValueError("action must be generate or health")
    if action == "health":
        return {"action": action}
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("prompt must contain 1–8000 characters")
    task = data.get("task", "t2va")
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError("Unsupported task")
    resolution = data.get("resolution", "480p")
    aspect = data.get("aspect", "16:9")
    if resolution not in ("480p", "768p") or not isinstance(aspect, str) or aspect not in ASPECTS:
        raise ValueError("Unsupported resolution or aspect")
    first = data.get("first_image_base64")
    last = data.get("last_image_base64")
    if (bool(first), bool(last)) != TASKS[task]:
        raise ValueError("first/last images must match the task")
    refs = data.get("references", [])
    if not isinstance(refs, list) or (task == "ref2va") != bool(refs):
        raise ValueError("references are required exclusively for ref2va")
    if len(refs) > 12:
        raise ValueError("At most 12 reference files are allowed")
    counts = {"image": 0, "video": 0, "audio": 0}
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"kind", "base64"} or ref.get("kind") not in counts:
            raise ValueError("Each reference requires kind (image/video/audio) and base64")
        counts[ref["kind"]] += 1
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("Reference limits are 9 images, 3 videos and 3 audio files")
    if task == "ref2va" and not (counts["image"] or counts["video"]):
        raise ValueError("ref2va needs at least one image or video")
    offload = data.get("cpu_offload", os.environ.get("CPU_OFFLOAD", "true").lower() == "true")
    if type(offload) is not bool:
        raise ValueError("cpu_offload must be a JSON boolean")
    return dict(action=action, task=task, prompt=prompt, resolution=resolution, aspect=aspect,
                duration=integer(data.get("duration", 5), "duration", 5, 15),
                seed=integer(data.get("seed", 42), "seed", 0, 2**63 - 1),
                cpu_offload=offload, first=first, last=last, references=refs)


def decode_media(value, path, remaining):
    if not isinstance(value, str):
        raise ValueError("Media must be a base64 string")
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError("Expected a base64 data URI")
    if len(value) > ((remaining + 2) // 3) * 4:
        raise ValueError("Input media exceeds the worker's size limit")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 media") from exc
    if not raw or len(raw) > remaining:
        raise ValueError("Media is empty or exceeds the worker's size limit")
    path.write_bytes(raw)
    return remaining - len(raw)


def command(request, workdir, output):
    # A fixed executable and argument array: prompts never enter a shell.
    prompt_path = workdir / "prompt.txt"
    prompt_path.write_text(request["prompt"], encoding="utf-8")
    args = ["pika-h3-pro", "--task", request["task"], "--prompt-file", str(prompt_path),
            "--resolution", request["resolution"], "--aspect", request["aspect"],
            "--duration", str(request["duration"]), "--seed", str(request["seed"]),
            "--model-repo", RELEASE["model_repo"], "--revision", RELEASE["model_revision"],
            "--output", str(output)]
    if request["cpu_offload"]:
        args.append("--cpu-offload")
    remaining = int(os.environ.get("MAX_INPUT_MEDIA_BYTES", "7000000"))
    for key, flag in (("first", "--first-image"), ("last", "--last-image")):
        if request[key]:
            path = workdir / f"{key}.png"
            remaining = decode_media(request[key], path, remaining)
            args.extend([flag, str(path)])
    suffix = {"image": ".png", "video": ".mp4", "audio": ".wav"}
    for index, ref in enumerate(request["references"]):
        path = workdir / f"ref-{index}{suffix[ref['kind']]}"
        remaining = decode_media(ref["base64"], path, remaining)
        args.extend(["--ref", f"{ref['kind']}:{path}"])
    return args


def publish(output, metadata):
    bucket = os.environ.get("OUTPUT_S3_BUCKET")
    if bucket:
        import boto3
        client = boto3.client("s3", endpoint_url=os.environ.get("OUTPUT_S3_ENDPOINT_URL") or None,
                              region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        prefix = os.environ.get("OUTPUT_S3_PREFIX", "h3-flex").strip("/")
        key = f"{prefix}/{output.name}" if prefix else output.name
        client.upload_file(str(output), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                           ExpiresIn=3600)
        return {"video": {"type": "url", "url": url, "expires_in": 3600}, "metadata": metadata}
    limit = int(os.environ.get("MAX_INLINE_VIDEO_BYTES", "6000000"))
    if output.stat().st_size > limit:
        return {"error": "Video exceeds inline response limit. Configure OUTPUT_S3_BUCKET for download URLs.",
                "volume_output_path": str(output), "bytes": output.stat().st_size}
    return {"video": {"type": "base64", "mime_type": "video/mp4",
                      "data": base64.b64encode(output.read_bytes()).decode("ascii")}, "metadata": metadata}


def handler(job):
    try:
        request = validate(job.get("input"))
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}
    volume = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    if not volume.is_dir():
        return {"error": "Network volume is not mounted at RUNPOD_VOLUME_PATH"}
    if request["action"] == "health":
        return {"status": "wrapper-ready", "runtime": "official-native", "generation_verified": False,
                "model_revision": RELEASE["model_revision"],
                "free_disk_gb": round(shutil.disk_usage(volume).free / 10**9, 2)}
    output_dir = volume / "outputs" / "h3-flex"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{uuid.uuid4().hex}.mp4"
    try:
        with tempfile.TemporaryDirectory(prefix="h3-job-") as tmp:
            args = command(request, Path(tmp), output)
            # Stream operational progress to RunPod logs. Never print the request or credentials.
            subprocess.run(args, check=True, timeout=int(os.environ.get("GENERATION_TIMEOUT_SECONDS", "3600")))
        metadata_path = output.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return publish(output, metadata)
    except subprocess.TimeoutExpired:
        return {"error": "Generation timed out"}
    except subprocess.CalledProcessError as exc:
        return {"error": f"Pika CLI failed (exit {exc.returncode}); inspect the worker logs"}
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
