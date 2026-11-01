"""Isolated loading/quantization experiment; original image remains available."""
import contextvars
import json
import os
from pathlib import Path
import sys
import time


def child(profile):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers import MiniMaxH3Transformer3DModel
    from pika_h3 import pipeline

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    stats = {"profile": profile, "threads": 8, "loads": {}}
    cache = Path('/runpod-volume/bench-cache/nf4-v1')

    def patch(cls, component, library):
        original = cls.from_pretrained

        def load(_cls, path, *args, **kwargs):
            started = time.monotonic()
            target = cache / component
            ready = target / 'COMPLETE'
            hit = profile == 'nf4' and ready.is_file()
            kwargs.update(low_cpu_mem_usage=True, device_map={"": "cuda:0"})
            if profile == 'nf4':
                if hit:
                    path = str(target)
                    kwargs.pop('subfolder', None)
                else:
                    if library == 'transformers':
                        from transformers import BitsAndBytesConfig
                    else:
                        from diffusers import BitsAndBytesConfig
                    kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)
            model = original(path, *args, **kwargs)
            torch.cuda.synchronize()
            loaded = time.monotonic() - started
            saved = 0.0
            if profile == 'nf4' and not hit:
                save_start = time.monotonic()
                target.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(target, safe_serialization=True, max_shard_size='4GB')
                ready.write_text('NF4 base weights, before Pika adapter injection\n')
                saved = time.monotonic() - save_start
            stats['loads'][component] = dict(load_seconds=loaded,
                cache_save_seconds=saved, quantized_cache_hit=hit)
            print('[benchmark] ' + component + ' ' + json.dumps(stats['loads'][component]), flush=True)
            return model

        cls.from_pretrained = classmethod(load)

    patch(Qwen3VLForConditionalGeneration, 'text_encoder', 'transformers')
    patch(MiniMaxH3Transformer3DModel, 'transformer', 'diffusers')
    original_generate = pipeline.generate

    def generate(args):
        if args.cpu_offload or args.task != 't2va':
            raise ValueError('Benchmark supports t2va without CPU offload only')
        original_generate(args)
        metadata_path = args.output.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        metadata['benchmark'] = stats
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print('[benchmark] timings ' + json.dumps(metadata['timings']), flush=True)

    pipeline.generate = generate
    from pika_h3.cli import main
    main()


def parent():
    import handler as base
    import runpod
    profile_context = contextvars.ContextVar('benchmark_profile', default='direct')
    original_command = base.command

    def command(request, workdir, output):
        argv = original_command(request, workdir, output)
        return [sys.executable, __file__, '--child', profile_context.get(), *argv[1:]]

    base.command = command

    def handler(job):
        data = job.get('input')
        if not isinstance(data, dict):
            return {'error': 'input must be an object'}
        data = dict(data)
        profile = data.pop('benchmark_profile', 'direct')
        if profile not in ('direct', 'nf4'):
            return {'error': 'benchmark_profile must be direct or nf4'}
        token = profile_context.set(profile)
        try:
            return base.handler({**job, 'input': data})
        finally:
            profile_context.reset(token)

    runpod.serverless.start({'handler': handler})


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--child':
        profile = sys.argv[2]
        if profile not in ('direct', 'nf4'):
            raise ValueError('Invalid benchmark profile')
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        child(profile)
    else:
        parent()
