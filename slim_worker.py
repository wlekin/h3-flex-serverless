"""NF4-only H3 worker using migrated local assets; no base-weight downloads."""
import json
from pathlib import Path
import sys

ROOT = Path('/runpod-volume/h3-flex')


def child():
    for component in ('text_encoder', 'transformer'):
        if not (ROOT / 'nf4-v1' / component / 'COMPLETE').is_file():
            raise RuntimeError('Migrated NF4 cache is incomplete: ' + component)
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers import MiniMaxH3Transformer3DModel
    from pika_h3 import pipeline
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    def patch(cls, component):
        original = cls.from_pretrained
        def load(_cls, path, *args, **kwargs):
            kwargs.pop('subfolder', None)
            kwargs.update(low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, local_files_only=True)
            return original(str(ROOT / 'nf4-v1' / component), *args, **kwargs)
        cls.from_pretrained = classmethod(load)
    patch(Qwen3VLForConditionalGeneration, 'text_encoder')
    patch(MiniMaxH3Transformer3DModel, 'transformer')
    original_generate = pipeline.generate
    def generate(args):
        if args.cpu_offload or args.task != 't2va':
            raise ValueError('Qualified slim worker supports t2va without CPU offload only')
        original_generate(args)
        path = args.output.with_suffix('.json')
        metadata = json.loads(path.read_text())
        metadata['worker_profile'] = 'nf4-slim-v1'
        path.write_text(json.dumps(metadata, indent=2))
        print('[slim] timings ' + json.dumps(metadata['timings']), flush=True)
    pipeline.generate = generate
    from pika_h3.cli import main
    main()


def parent():
    import handler as base
    import runpod
    original_command = base.command
    def command(request, workdir, output):
        argv = original_command(request, workdir, output)
        argv += ['--base-path', str(ROOT / 'native-assets/base'),
                 '--adapter-path', str(ROOT / 'native-assets/adapter/fl2va')]
        return [sys.executable, __file__, '--child', *argv[1:]]
    base.command = command
    def handler(job):
        data = job.get('input')
        if not isinstance(data, dict):
            return {'error': 'input must be an object'}
        data = dict(data)
        data.setdefault('cpu_offload', False)
        if data.get('action', 'generate') != 'health' and data.get('task', 't2va') != 't2va':
            return {'error': 'Slim NF4 runtime is qualified for t2va only; other tasks need qualification'}
        result = base.handler({**job, 'input': data})
        if data.get('action') == 'health' and isinstance(result, dict) and 'error' not in result:
            result.update(runtime='nf4-slim-v1', cache_ready=all(
                (ROOT / 'nf4-v1' / component / 'COMPLETE').is_file()
                for component in ('text_encoder', 'transformer')))
        return result
    runpod.serverless.start({'handler': handler})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        child()
    else:
        parent()
