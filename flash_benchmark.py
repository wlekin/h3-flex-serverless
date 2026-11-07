import base64
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import runpod
import vsr_worker

PROMPT = "Start with the living room composition in <Picture 1> and preserve the face in <Picture 2>. Static camera. The woman in red looks up from her book, smiles, and resumes reading. Quiet room ambience."


def flash(data):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        refs = []
        for index, ref in enumerate(data["references"]):
            path = folder / f"reference-{index}.png"
            path.write_bytes(base64.b64decode(ref["base64"], validate=True))
            refs.extend(["--reference", str(path)])
        env = {**os.environ, "HF_HOME": "/tmp/hyperflow-hf", "TRITON_CACHE_DIR": "/tmp/hyperflow-triton"}
        result = {}
        for steps in (4, 6):
            path = folder / f"flash-{steps}.mp4"
            start = time.monotonic()
            subprocess.run([sys.executable, "-m", "hyperflow_h3_flash", "--prompt", PROMPT, *refs,
                "--height", "768", "--width", "1344", "--num-frames", "124", "--seed", "192310423",
                "--steps", str(steps), "--text-encoder-precision", "fp8", "--offload", "stream",
                "--output", str(path)], env=env, timeout=1500, check=True)
            from vsr_worker import probe
            result[str(steps)] = {"seconds": time.monotonic()-start, "media": probe(path),
                                  "video": base64.b64encode(path.read_bytes()).decode()}
        return {"flash": result}


def main():
    start = runpod.serverless.start
    def install(config):
        baseline = config["handler"]
        def handler(job):
            data = job["input"]
            if data.get("action") == "flash":
                return flash(data)
            return baseline(job)
        return start({**config, "handler": handler})
    runpod.serverless.start = install
    vsr_worker.main()


if __name__ == "__main__":
    main()
