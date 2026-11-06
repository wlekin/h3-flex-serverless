"""Keep the standard 480p Ref2VA pipeline resident in a dedicated GPU process."""
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time
from unittest.mock import patch

IDLE_SECONDS = 60


def serve(connection):
    import torch

    device = torch.cuda.get_device_properties(0)
    supported = device.total_memory >= 90 * 1024**3 and "RTX PRO 6000" in device.name
    connection.send(supported)
    if not supported:
        return

    from pika_h3 import cli, pipeline
    from pika_h3.core import conditioning, loader, vae
    import ref_worker

    encoder_type = conditioning.H3ConditionEncoder
    apply_lora = loader.apply_lora
    load_lora = loader.load_lora_weights
    build_transformer = lru_cache(maxsize=1)(loader.build_transformer)
    decoder = lru_cache(maxsize=1)(vae.H3Decoders)
    cli_main = cli.main
    warmed = False

    class ResidentEncoder(encoder_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, keep_loaded=True)

        def free(self, what="all"):
            pass

    def apply_once(*args, **kwargs):
        if not warmed:
            apply_lora(*args, **kwargs)

    def load_once(*args, **kwargs):
        if not warmed:
            load_lora(*args, **kwargs)

    def loop():
        nonlocal warmed
        with (
            patch.object(conditioning, "H3ConditionEncoder", lru_cache(maxsize=1)(ResidentEncoder)),
            patch.object(loader, "build_transformer", build_transformer),
            patch.object(loader, "apply_lora", apply_once),
            patch.object(loader, "load_lora_weights", load_once),
            patch.object(vae, "H3Decoders", decoder),
        ):
            while connection.poll(IDLE_SECONDS):
                argv = connection.recv()
                sys.argv = [__file__, *argv]
                cli_main()
                output = Path(argv[argv.index("--output") + 1]).with_suffix(".json")
                metadata = json.loads(output.read_text())
                metadata["resident_reused"] = warmed
                metadata["resident_until_ms"] = int((time.time() + IDLE_SECONDS) * 1000)
                output.write_text(json.dumps(metadata))
                warmed = True
                connection.send(True)

    # Install the same verified NF4 loaders and runtime setup as the CLI.
    with patch.object(cli, "main", loop):
        ref_worker.ref_child("nf4")


def eligible(argv):
    if len(argv) < 5 or argv[1:3] != ["/app/ref_worker.py", "--child"]:
        return False

    def option(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    return (
        option("--task") == "ref2va"
        and option("--resolution") == "480p"
        and option("--duration") == "5"
        and json.loads(Path(argv[3]).read_text()) == []
    )


class GenerationProcess:
    # The base handler uses this subprocess-compatible interface.
    TimeoutExpired = subprocess.TimeoutExpired
    CalledProcessError = subprocess.CalledProcessError

    def __init__(self):
        self.connection = None
        self.process = None
        self.unsupported = False

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None

    def receive(self, timeout):
        if not self.connection.poll(max(0, timeout)):
            raise TimeoutError("Resident H3 process timed out")
        return self.connection.recv()

    def start(self, timeout):
        ctx = get_context("spawn")
        self.connection, child = ctx.Pipe()
        process = ctx.Process(target=serve, args=(child,), daemon=True)
        try:
            process.start()
            self.process = process
        finally:
            child.close()
        self.unsupported = not self.receive(min(30, timeout))

    def run(self, argv, *, check, timeout):
        if self.unsupported or not eligible(argv):
            self.close()
            return subprocess.run(argv, check=check, timeout=timeout)

        deadline = time.monotonic() + timeout
        try:
            if self.process is None or not self.process.is_alive():
                self.close()
                self.start(deadline - time.monotonic())
            if not self.unsupported:
                self.connection.send(argv[4:])
                if self.receive(deadline - time.monotonic()):
                    return subprocess.CompletedProcess(argv, 0)
        except (OSError, EOFError):
            print("[resident] GPU process failed, falling back to staged generation", flush=True)

        self.close()
        output = Path(argv[argv.index("--output") + 1])
        output.unlink(missing_ok=True)
        output.with_suffix(".json").unlink(missing_ok=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.run(argv, check=check, timeout=remaining)


def main():
    import handler as base
    import runpod
    import signed_input_worker

    runner = GenerationProcess()
    start = runpod.serverless.start
    lock = Lock()

    def install(config):
        original = config["handler"]

        def handler(job):
            with lock:
                return original(job)

        base.subprocess = runner
        return start({**config, "handler": handler})

    runpod.serverless.start = install
    signed_input_worker.main()


if __name__ == "__main__":
    main()
