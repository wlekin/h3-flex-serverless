"""Resolve only I2V-signed original images before invoking the existing H3 handler."""
import base64
import os
import re
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

PATH = re.compile(r"/I2V/private/r2/i2v-client%2[fF][a-f0-9]{64}\.(?:jpg|png|webp)")
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch_image(url, opener=None, limit=None):
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Invalid signed H3 image URL")
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "ggggggg.uk" or parts.fragment
            or not parts.query or not PATH.fullmatch(parts.path)):
        raise ValueError("Invalid signed H3 image URL")
    if limit is None:
        limit = int(os.environ.get("MAX_INPUT_MEDIA_BYTES", "7000000"))
    opener = opener or build_opener(NoRedirect())
    try:
        with opener.open(Request(url), timeout=20) as response:
            if response.status != 200 or response.headers.get_content_type() not in IMAGE_TYPES:
                raise ValueError("Signed H3 input must be a JPEG, PNG, or WebP image")
            content = response.read(limit + 1)
    except (OSError, URLError) as exc:
        raise ValueError("Could not download signed H3 input") from exc
    if not content or len(content) > limit:
        raise ValueError("Signed H3 input exceeds the media limit or is empty")
    return base64.b64encode(content).decode("ascii")


def resolve_input(data, fetch=fetch_image):
    if "original_image_url" not in data:
        return data
    resolved = dict(data)
    url = resolved.pop("original_image_url")
    task = resolved.get("task")
    if task == "i2va" and "first_image_base64" not in resolved:
        resolved["first_image_base64"] = fetch(url)
    elif task == "ref2va" and isinstance(resolved.get("references", []), list):
        refs = resolved.get("references", [])
        if len(refs) > 8:
            raise ValueError("Too many H3 image references")
        remaining = int(os.environ.get("MAX_INPUT_MEDIA_BYTES", "7000000"))

        def read_reference(link):
            nonlocal remaining
            encoded = fetch(link, limit=remaining)
            remaining -= len(encoded) * 3 // 4 - encoded.count("=")
            if remaining < 0:
                raise ValueError("H3 references exceed the media limit")
            return {"kind": "image", "base64": encoded}

        references = [read_reference(url)]
        for ref in refs:
            if isinstance(ref, dict) and set(ref) == {"kind", "url"} and ref["kind"] == "image":
                references.append(read_reference(ref["url"]))
            else:
                references.append(ref)
        resolved["references"] = references
    else:
        raise ValueError("Signed H3 input requires image-to-video or reference mode")
    return resolved


def main():
    import runpod
    import turbo_worker

    start = runpod.serverless.start

    def install(config):
        original = config["handler"]

        def handler(job):
            try:
                data = job.get("input")
                if isinstance(data, dict):
                    job = {**job, "input": resolve_input(data)}
            except (ValueError, TypeError):
                return {"error": "Could not load the signed H3 reference image"}
            return original(job)

        return start({**config, "handler": handler})

    runpod.serverless.start = install
    turbo_worker.parent()


if __name__ == "__main__":
    main()

