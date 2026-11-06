"""Normalize known H3 LoRA encodings, then use the worker's strict shape validator."""
import hashlib
import json
import math
import re
import struct
import tempfile
from pathlib import Path

KOHYA = re.compile(
    r"lora_unet_(blocks_\d+|token_refiner_blocks_\d+)_(attn_(?:out_proj|qkv_proj)|mlp_fc[12])"
    r"\.(alpha|lora_down\.weight|lora_up\.weight)"
)
MAX_BYTES = 1024**3
MYSTIC_V5_SHA256 = "f57c76e761f7a5113e541ad58e84c648a5fe4d3cfbc277d0c44ae9763da29afe"
MYSTIC_V5_METADATA = "Schwifty-9K-Morty-Proof (Chuck Norris Edition)"


def read_header(source):
    with source.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid LoRA safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 8 * 1024**2 or length + 8 > source.stat().st_size:
            raise ValueError("Invalid LoRA safetensors header")
        header = json.loads(file.read(length))
    metadata = header.pop("__metadata__", {})
    verified_mystic = False
    if metadata.get("ss_base_model_version") == MYSTIC_V5_METADATA:
        with source.open("rb") as file:
            verified_mystic = hashlib.file_digest(file, "sha256").hexdigest() == MYSTIC_V5_SHA256
    if not verified_mystic and metadata.get("ss_base_model_version") not in (None, "minimax_h3"):
        raise ValueError("LoRA metadata identifies a different base model")
    if not header:
        raise ValueError("LoRA has no tensors")
    return header, metadata, length + 8, verified_mystic


def normalize(source, destination, header, metadata, offset, kohya):
    import numpy as np

    dtype = {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}
    output = {"__metadata__": dict(metadata, ss_base_model_version="minimax_h3")}
    chunks = []
    output_offset = 0
    groups = {}
    end = 0
    ordered = sorted(header.items(), key=lambda pair: pair[1]["data_offsets"][0])
    with source.open("rb") as file:
        file.seek(offset)
        for name, spec in ordered:
            start, stop = spec["data_offsets"]
            shape = spec["shape"]
            kind = spec["dtype"]
            if kind not in dtype or len(shape) not in (0, 2) or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("Unsupported LoRA tensor dtype or shape")
            if start != end or stop - start != math.prod(shape) * np.dtype(dtype[kind]).itemsize:
                raise ValueError("Invalid LoRA tensor offsets")
            end = stop
            data = file.read(stop - start)
            if len(data) != stop - start:
                raise ValueError("Truncated LoRA tensor data")
            if kohya:
                match = KOHYA.fullmatch(name)
                if not match:
                    raise ValueError("Unsupported Kohya H3 tensor name")
                block, module, part = match.groups()
                groups.setdefault((block, module), {})[part] = (shape, kind, data)
            else:
                groups[name] = (shape, kind, data)
        if file.read(1):
            raise ValueError("Unexpected data after LoRA tensors")

    def values(kind, data):
        raw = np.frombuffer(data, dtype=dtype[kind])
        if kind == "BF16":
            return (raw.astype("<u4") << 16).view("<f4")
        return raw.astype("<f4")

    def add(name, shape, kind, data, scale=1.0):
        nonlocal output_offset
        if len(shape) != 2:
            raise ValueError("LoRA weights must be two-dimensional")
        if kind != "BF16" or scale != 1.0:
            numbers = values(kind, data) * np.float32(scale)
            if not np.isfinite(numbers).all():
                raise ValueError("LoRA contains non-finite weights")
            bits = numbers.view("<u4")
            data = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()
        output[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [output_offset, output_offset + len(data)]}
        output_offset += len(data)
        chunks.append(data)

    if kohya:
        for (block, module), group in groups.items():
            if set(group) != {"alpha", "lora_down.weight", "lora_up.weight"}:
                raise ValueError("Kohya H3 LoRA requires alpha and paired weights")
            shape, kind, data = group["alpha"]
            if shape != [] or not group["lora_down.weight"][0]:
                raise ValueError("Invalid Kohya H3 alpha or rank")
            alpha = float(values(kind, data)[0])
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("Kohya H3 alpha must be positive and finite")
            rank = group["lora_down.weight"][0][0]
            prefix = "diffusion_model." + block.replace("token_refiner_blocks_", "token_refiner.blocks.").replace("blocks_", "blocks.")
            prefix += "." + module.replace("attn_", "attn.").replace("mlp_", "mlp.")
            add(prefix + ".lora_A.weight", *group["lora_down.weight"])
            add(prefix + ".lora_B.weight", *group["lora_up.weight"], scale=alpha / rank)
    else:
        for name, spec in groups.items():
            add(name, *spec)

    encoded = json.dumps(output, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with destination.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        file.writelines(chunks)


def convert_compatible(source, destination, strict, **kwargs):
    source = Path(source)
    if not 8 < source.stat().st_size <= MAX_BYTES:
        raise ValueError("LoRA file must be at most 1024 MiB")
    try:
        header, metadata, offset, verified_mystic = read_header(source)
        kohya = any(name.startswith("lora_unet_") for name in header)
        if not kohya and all(item["dtype"] == "BF16" for item in header.values()):
            return strict(source, destination, **kwargs)
        if not kohya and not all(name.startswith("diffusion_model.") for name in header):
            raise ValueError("Unsupported H3 LoRA key format")
        with tempfile.TemporaryDirectory(prefix="h3-lora-") as tmp:
            normalized = Path(tmp) / "normalized.safetensors"
            normalize(source, normalized, header, metadata, offset, kohya)
            if verified_mystic:
                info = strict(normalized, destination, allow_audio=True, **kwargs)
                info["audio_targets"] = ["audio_proj_in", "audio_proj_out"]
                info["adapter_scale_assumption"] = "alpha equals rank (16)"
            else:
                info = strict(normalized, destination, **kwargs)
        with source.open("rb") as file:
            info["source_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
        return info
    except (KeyError, TypeError, IndexError, struct.error) as exc:
        raise ValueError("Invalid H3 LoRA tensor metadata") from exc


def patch_worker(code):
    def replace_once(before, after):
        nonlocal code
        if code.count(before) != 1:
            raise RuntimeError("Unexpected H3 LoRA worker source: " + before)
        code = code.replace(before, after)

    before = "def convert(source, destination, dimensions=(5376, 7168, 14336)):"
    replace_once(before, "def _convert_bf16(source, destination, dimensions=(5376, 7168, 14336), *, allow_audio=False):")
    replace_once("    groups = {}\n    for key, item in header.items():", """    audio_names = {
        "audio_patch_proj": "audio_proj_in",
        "final_layer.audio_out": "audio_proj_out",
    }
    groups = {}
    for key, item in header.items():
        if allow_audio:
            audio = re.fullmatch(r"diffusion_model\\.(audio_patch_proj|final_layer\\.audio_out)\\.lora_([AB])\\.weight", key)
            if audio:
                groups.setdefault(("", audio_names[audio.group(1)]), {})[audio.group(2)] = item
                continue""")
    replace_once("    hidden, inner, ffn = dimensions", """    if allow_audio and not all(("", name) in groups for name in audio_names.values()):
        raise ValueError("Verified Mystic V5 requires both audio projections")
    hidden, inner, ffn = dimensions""")
    replace_once("'fc2':(ffn,hidden)}[module]", "'fc2':(ffn,hidden),'audio_proj_in':(32,hidden),'audio_proj_out':(hidden,32)}[module]")
    replace_once("        if module == 'qkv_proj':", """        if module in audio_names.values():
            if rank != 16:
                raise ValueError("Verified Mystic V5 audio projections must be rank 16")
            targets = [(module, bd, expected[1])]
        elif module == 'qkv_proj':""")
    replace_once("            name=prefix+'.'+target", "            name=prefix+'.'+target if prefix else target")
    replace_once("        state=load_file(converted)", """        if info.get("audio_targets") and ("--task" not in sys.argv or sys.argv[sys.argv.index("--task")+1] != "ref2va"):
            raise ValueError("Mystic V5 audio adapter requires Ref2VA")
        state=load_file(converted)""")
    hook = """from lora_compat import convert_compatible

def convert(source, destination, dimensions=(5376, 7168, 14336)):
    return convert_compatible(source, destination, _convert_bf16, dimensions=dimensions)


"""
    replace_once("def adapter_options(", hook + "def adapter_options(")
    return code


if __name__ == "__main__":
    target = Path("/app/lora_worker.py")
    # The parent image already installed the compatibility wrapper.
    code = target.read_text()
    start = code.index("from lora_compat import convert_compatible")
    end = code.index("def adapter_options(", start)
    code = (code[:start] + code[end:]).replace("def _convert_bf16(", "def convert(", 1)
    target.write_text(patch_worker(code))

