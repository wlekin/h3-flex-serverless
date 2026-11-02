import time
"""Strict AI Toolkit MiniMax H3 LoRA conversion and additive PEFT loading."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def convert(source, destination, dimensions=(5376, 7168, 14336)):
    source = Path(source)
    if not 8 < source.stat().st_size <= 1024 * 1024**2:
        raise ValueError('LoRA file must be at most 1024 MiB')
    raw = source.read_bytes()
    if not 8 < len(raw) <= 1024 * 1024**2:
        raise ValueError('LoRA file must be at most 1024 MiB')
    length = struct.unpack('<Q', raw[:8])[0]
    if not 2 <= length <= min(len(raw)-8, 8*1024**2):
        raise ValueError('Invalid safetensors header')
    header = json.loads(raw[8:8+length])
    metadata = header.pop('__metadata__', {})
    if metadata.get('ss_base_model_version') not in (None, 'minimax_h3'):
        raise ValueError('LoRA metadata identifies a different base model')
    payload = memoryview(raw)[8+length:]
    end = 0
    for start, stop, key in sorted((v['data_offsets'][0], v['data_offsets'][1], k) for k,v in header.items()):
        item = header[key]
        if item['dtype'] != 'BF16' or len(item['shape']) != 2 or start != end or stop-start != math.prod(item['shape'])*2:
            raise ValueError('Unsupported or invalid tensor layout')
        end = stop
    if end != len(payload):
        raise ValueError('Tensor payload length mismatch')
    pattern = re.compile(r'diffusion_model\.(blocks\.(\d+)|token_refiner\.blocks\.(\d+))\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))\.lora_([AB])\.weight')
    groups = {}
    for key, item in header.items():
        match = pattern.fullmatch(key)
        if not match:
            raise ValueError('Unsupported LoRA tensor: '+key)
        main, ref = match.group(2), match.group(3)
        if (main is not None and int(main)>=50) or (ref is not None and int(ref)>=2):
            raise ValueError('LoRA block is outside the H3 model')
        prefix = 'transformer_blocks.'+main if main is not None else 'token_refiner.refiner_blocks.'+ref
        module = match.group(5) or match.group(6)
        groups.setdefault((prefix,module), {})[match.group(7)] = item
    hidden, inner, ffn = dimensions
    converted = {}
    ranks = set()
    rank_pattern = {}
    for (prefix,module), pair in groups.items():
        if set(pair) != {'A','B'}:
            raise ValueError('Incomplete LoRA A/B pair')
        a,b = pair['A'],pair['B']
        rank = a['shape'][0]
        expected = {'qkv_proj':(hidden,3*inner),'out_proj':(inner,hidden),'fc1':(hidden,2*ffn),'fc2':(ffn,hidden)}[module]
        if not 1<=rank<=256 or a['shape'] != [rank,expected[0]] or b['shape'] != [expected[1],rank]:
            raise ValueError('LoRA shapes do not match MiniMax H3')
        ranks.add(rank)
        ad = bytes(payload[slice(*a['data_offsets'])]); bd = bytes(payload[slice(*b['data_offsets'])])
        if module == 'qkv_proj':
            targets = [('attn.to_'+name,bd[i*inner*rank*2:(i+1)*inner*rank*2],inner) for i,name in enumerate('qkv')]
        elif module == 'fc1':
            cut=ffn*rank*2
            targets=[('ff.net.0.proj',bd[cut:]+bd[:cut],2*ffn)]
        else:
            targets=[({'out_proj':'attn.to_out.0','fc2':'ff.net.2'}[module],bd,expected[1])]
        for target, data, rows in targets:
            name=prefix+'.'+target
            rank_pattern[name]=rank
            converted[name+'.lora_A.weight']=([rank,expected[0]],ad)
            converted[name+'.lora_B.weight']=([rows,rank],data)
    if not converted:
        raise ValueError('Expected a nonempty H3 LoRA')
    rank=max(ranks)
    output={'__metadata__':{'source_sha256':hashlib.sha256(raw).hexdigest(),'conversion':'ai-toolkit-h3-to-diffusers-v2','rank':str(rank),'rank_pattern':json.dumps(rank_pattern),'default_scaling':'alpha equals each module rank'}}
    chunks=[]; offset=0
    for key,(shape,data) in sorted(converted.items()):
        output[key]={'dtype':'BF16','shape':shape,'data_offsets':[offset,offset+len(data)]}
        offset+=len(data);chunks.append(data)
    encoded=json.dumps(output,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp')
    temporary.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(chunks))
    temporary.replace(destination)
    return dict(rank=rank,ranks=sorted(ranks),rank_pattern=rank_pattern,source_sha256=output['__metadata__']['source_sha256'],modules=len(converted)//2)



def adapter_options(info, alpha=None):
    # PEFT uses alpha/r scaling per target. Missing source alpha means unity,
    # including merged adapters whose layers have different ranks.
    pattern=info['rank_pattern']
    return dict(r=info['rank'],lora_alpha=alpha if alpha is not None else info['rank'],
                target_modules=sorted(pattern),rank_pattern=dict(pattern),
                alpha_pattern={name:alpha if alpha is not None else rank for name,rank in pattern.items()},
                init_lora_weights=False)


def attach(model, specs, volume):
    import torch
    from peft import LoraConfig
    from safetensors.torch import load_file
    reports=[]; names=['default']; weights=[1.0]
    for index,spec in enumerate(specs):
        folder=(Path(volume)/'h3-flex/loras').resolve()
        source=(folder/spec['name']).resolve()
        if source.parent!=folder or not source.is_file():
            raise ValueError('LoRA file not found in h3-flex/loras: '+spec['name'])
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        converted=Path(volume)/'h3-flex/converted-loras'/(digest+'.safetensors')
        info=convert(source,converted)
        state=load_file(converted)
        modules=sorted({key.rsplit('.lora_',1)[0] for key in state})
        own_modules=dict(model.named_modules())
        if any(name not in own_modules for name in modules):
            raise ValueError('Converted LoRA targets missing model modules')
        name='user_'+str(index)
        alpha=spec.get('alpha')
        model.add_adapter(LoraConfig(**adapter_options(info,alpha)),adapter_name=name)
        own=dict(model.named_parameters())
        with torch.no_grad():
            for key,value in state.items():
                target=key.replace('.lora_A.weight','.lora_A.'+name+'.weight').replace('.lora_B.weight','.lora_B.'+name+'.weight')
                if target not in own or own[target].shape!=value.shape:
                    raise ValueError('Converted adapter parameter mismatch: '+target)
                own[target].copy_(value.to(device=own[target].device,dtype=own[target].dtype))
                own[target].requires_grad_(False)
        names.append(name);weights.append(spec['strength'])
        reports.append({**{k:v for k,v in info.items() if k!='rank_pattern'},'name':spec['name'],'strength':spec['strength'],'alpha':alpha if alpha is not None else 'per-module rank'})
    if specs:
        model.set_adapters(names,weights=weights)
    return reports

def nf4_child(profile):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers import MiniMaxH3Transformer3DModel
    from pika_h3 import pipeline

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    stats = {"profile": profile, "threads": 8, "loads": {}}
    cache = Path('/runpod-volume/h3-flex/nf4-v1')

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
                    raise RuntimeError('Slim volume requires migrated NF4 cache at h3-flex/nf4-v1/'+component)
            if hit:
                kwargs['local_files_only'] = True
            model = original(path, *args, **kwargs)
            torch.cuda.synchronize()
            loaded = time.monotonic() - started
            stats['loads'][component] = dict(load_seconds=loaded,
                cache_save_seconds=0.0, quantized_cache_hit=hit, layout='h3-flex')
            print('[benchmark] ' + component + ' ' + json.dumps(stats['loads'][component]), flush=True)
            return model

        cls.from_pretrained = classmethod(load)

    patch(Qwen3VLForConditionalGeneration, 'text_encoder', 'transformers')
    patch(MiniMaxH3Transformer3DModel, 'transformer', 'diffusers')
    original_generate = pipeline.generate

    def generate(args):
        if args.cpu_offload or args.task not in ('t2va', 'i2va'):
            raise ValueError('NF4 worker supports t2va/i2va without CPU offload only')
        original_generate(args)
        metadata_path = args.output.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        metadata['benchmark'] = stats
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print('[benchmark] timings ' + json.dumps(metadata['timings']), flush=True)

    pipeline.generate = generate
    from pika_h3.cli import main
    main()
"""NF4 H3 worker with additional AI Toolkit H3 LoRAs from network storage."""
import contextvars
import json
import math
from pathlib import Path
import re
import sys


def validate_loras(specs):
    if not isinstance(specs,list) or len(specs)>3:
        raise ValueError('loras must be a list of at most3 adapters')
    result=[]
    for spec in specs:
        if not isinstance(spec,dict) or not {'name'}<=set(spec) or set(spec)-{'name','strength','alpha'}:
            raise ValueError('Each LoRA accepts name, strength and optional alpha')
        name=spec['name']; strength=spec.get('strength',1.0)
        if not isinstance(name,str) or len(name)>180 or not re.fullmatch(r'[\w .()-]+\.safetensors',name) or '..' in name:
            raise ValueError('LoRA name must be a safetensors filename without a path')
        if type(strength) not in (int,float) or not math.isfinite(strength) or not -2<=strength<=2:
            raise ValueError('LoRA strength must be finite and between-2 and2')
        item={'name':name,'strength':float(strength)}
        if 'alpha' in spec:
            alpha=spec['alpha']
            if type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<alpha<=128:
                raise ValueError('LoRA alpha must be finite and in(0,128]')
            item['alpha']=alpha
        result.append(item)
    if len({x['name'] for x in result})!=len(result):
        raise ValueError('Duplicate LoRA names are not allowed')
    return result


def child(specs):
    from pika_h3 import pipeline
    from pika_h3.core import loader
    reports=[]
    original_load=loader.load_lora_weights
    def load(model,path,**kwargs):
        original_load(model,path,**kwargs)
        reports.extend(attach(model,specs,'/runpod-volume'))
        print('[custom-lora] '+json.dumps(reports),flush=True)
    loader.load_lora_weights=load
    original_generate=pipeline.generate
    def generate(args):
        original_generate(args)
        path=args.output.with_suffix('.json')
        metadata=json.loads(path.read_text())
        metadata['custom_loras']=reports
        path.write_text(json.dumps(metadata,indent=2))
    pipeline.generate=generate
    nf4_child('nf4')


def parent():
    import handler as base
    import runpod
    context=contextvars.ContextVar('loras',default=[])
    original_command=base.command
    def command(request,workdir,output):
        config=workdir/'custom-loras.json'
        config.write_text(json.dumps(context.get()))
        argv=original_command(request,workdir,output)
        argv += ['--base-path','/runpod-volume/h3-flex/native-assets/base',
                 '--adapter-path','/runpod-volume/h3-flex/native-assets/adapter/fl2va']
        return [sys.executable,__file__,'--child',str(config),*argv[1:]]
    base.command=command
    def handler(job):
        data=job.get('input')
        if not isinstance(data,dict):
            return {'error':'input must be an object'}
        data=dict(data)
        try:
            specs=validate_loras(data.pop('loras',[]))
        except ValueError as exc:
            return {'error':str(exc)}
        data.setdefault('cpu_offload',False)
        if data.get('action','generate')!='health' and (data.get('task','t2va') not in ('t2va','i2va') or data['cpu_offload']):
            return {'error':'NF4 custom LoRA worker supports t2va/i2va without CPU offload'}
        token=context.set(specs)
        try:
            result=base.handler({**job,'input':data})
            if data.get('action')=='health' and 'error' not in result:
                result['runtime']='nf4-i2v-slim-v1'
                root=Path('/runpod-volume/h3-flex')
                result['cache_ready']=all((root/'nf4-v1'/c/'COMPLETE').is_file() for c in ('text_encoder','transformer'))
                result['native_assets']=all((root/'native-assets'/p).exists() for p in ('base','adapter/fl2va'))
            return result
        finally:
            context.reset(token)
    runpod.serverless.start({'handler':handler})



import json
from pathlib import Path
import struct
import tempfile
import unittest


def write(path,tensors):
    header={'__metadata__':{'ss_base_model_version':'minimax_h3'}};chunks=[];offset=0
    for key,(shape,values) in tensors.items():
        data=b''.join(struct.pack('<f',float(v))[2:] for v in values)
        header[key]={'dtype':'BF16','shape':shape,'data_offsets':[offset,offset+len(data)]}
        offset+=len(data);chunks.append(data)
    encoded=json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(chunks))


def read(path):
    raw=path.read_bytes();n=struct.unpack('<Q',raw[:8])[0];h=json.loads(raw[8:8+n]);p=raw[8+n:]
    return {k:[struct.unpack('<f',b'\0\0'+p[i:i+2])[0] for i in range(v['data_offsets'][0],v['data_offsets'][1],2)] for k,v in h.items() if k!='__metadata__'}


class ConversionTests(unittest.TestCase):
    def test_i2v_first_image_reaches_cli_unchanged(self):
        import base64
        import handler
        image=b'PNG-test-payload'
        request=handler.validate({'task':'i2va','prompt':'Move gently',
            'first_image_base64':base64.b64encode(image).decode(),'cpu_offload':False})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            argv=handler.command(request,root,root/'out.mp4')
            self.assertEqual(argv[argv.index('--task')+1],'i2va')
            self.assertEqual(Path(argv[argv.index('--first-image')+1]).read_bytes(),image)
            self.assertNotIn('--last-image',argv)

    def test_i2v_rejects_missing_first_or_unexpected_last_image(self):
        import handler
        for data in ({'task':'i2va','prompt':'x'},
                     {'task':'i2va','prompt':'x','first_image_base64':'eA==','last_image_base64':'eA=='},
                     {'task':'t2va','prompt':'x','first_image_base64':'eA=='}):
            with self.assertRaises(ValueError):handler.validate(data)

    def test_qkv_delta_is_split_without_changing_low_rank_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in';dst=Path(tmp)/'out';prefix='diffusion_model.blocks.0.attn.qkv_proj'
            write(src,{prefix+'.lora_A.weight':([1,2],[2,3]),prefix+'.lora_B.weight':([6,1],[1,2,3,4,5,6])})
            info=convert(src,dst,dimensions=(2,2,3));state=read(dst)
            self.assertEqual(info['modules'],3)
            x=[7,11];ax=2*x[0]+3*x[1]
            reconstructed=[]
            for part in 'qkv':
                a=state['transformer_blocks.0.attn.to_'+part+'.lora_A.weight']
                b=state['transformer_blocks.0.attn.to_'+part+'.lora_B.weight']
                reconstructed.extend(v*sum(i*j for i,j in zip(a,x)) for v in b)
            self.assertEqual(reconstructed,[v*ax for v in range(1,7)])

    def test_swiglu_gate_up_halves_are_reversed(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in';dst=Path(tmp)/'out';prefix='diffusion_model.token_refiner.blocks.1.mlp.fc1'
            write(src,{prefix+'.lora_A.weight':([1,2],[2,3]),prefix+'.lora_B.weight':([6,1],[1,2,3,4,5,6])})
            convert(src,dst,dimensions=(2,2,3));state=read(dst)
            self.assertEqual(state['token_refiner.refiner_blocks.1.ff.net.0.proj.lora_B.weight'],[4,5,6,1,2,3])

    def test_rejects_incomplete_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in';write(src,{'diffusion_model.blocks.0.mlp.fc2.lora_A.weight':([1,3],[1,2,3])})
            with self.assertRaises(ValueError):convert(src,Path(tmp)/'out',dimensions=(2,2,3))

    def test_slim_volume_paths_are_h3_flex(self):
        source = Path(__file__).read_text(encoding='utf-8')
        runtime = source.split('class ConversionTests', 1)[0]
        self.assertIn("cache = Path('/runpod-volume/h3-flex/nf4-v1')", runtime)
        self.assertNotIn('bench-cache', runtime)
        self.assertIn("'--base-path','/runpod-volume/h3-flex/native-assets/base'", runtime)
        self.assertIn("'--adapter-path','/runpod-volume/h3-flex/native-assets/adapter/fl2va'", runtime)
        self.assertIn("nf4-i2v-slim-v1", runtime)

    def test_missing_metadata_and_mixed_ranks_preserve_layer_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in';dst=Path(tmp)/'out'
            q='diffusion_model.blocks.0.attn.qkv_proj';f='diffusion_model.blocks.0.mlp.fc2'
            write(src,{q+'.lora_A.weight':([80,2],[1]*160),q+'.lora_B.weight':([6,80],[1]*480),
                       f+'.lora_A.weight':([48,3],[1]*144),f+'.lora_B.weight':([2,48],[1]*96)})
            raw=src.read_bytes();n=struct.unpack('<Q',raw[:8])[0];h=json.loads(raw[8:8+n]);h['__metadata__']={'format':'pt'}
            encoded=json.dumps(h).encode();src.write_bytes(struct.pack('<Q',len(encoded))+encoded+raw[8+n:])
            info=convert(src,dst,dimensions=(2,2,3));opts=adapter_options(info)
            self.assertEqual(info['ranks'],[48,80]);self.assertEqual(info['modules'],4)
            self.assertEqual(opts['rank_pattern']['transformer_blocks.0.attn.to_q'],80)
            self.assertEqual(opts['rank_pattern']['transformer_blocks.0.ff.net.2'],48)
            self.assertEqual(opts['alpha_pattern'],opts['rank_pattern'])
            self.assertTrue(all(v==24 for v in adapter_options(info,24)['alpha_pattern'].values()))
            self.assertEqual(len(read(dst)['transformer_blocks.0.attn.to_q.lora_B.weight']),160)

    def test_wrong_declared_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in';h=json.dumps({'__metadata__':{'ss_base_model_version':'wan'}}).encode()
            src.write_bytes(struct.pack('<Q',len(h))+h)
            with self.assertRaisesRegex(ValueError,'different base model'):convert(src,Path(tmp)/'out')

    def test_peft_mixed_rank_cpu_injection(self):
        try:
            import torch
            from peft import LoraConfig,inject_adapter_in_model
        except ImportError:
            self.skipTest('PEFT/torch integration runs in the container build')
        model=torch.nn.Module();model.transformer_blocks=torch.nn.ModuleList([torch.nn.Module()])
        block=model.transformer_blocks[0];block.attn=torch.nn.Module();block.attn.to_q=torch.nn.Linear(2,2,bias=False)
        block.ff=torch.nn.Module();block.ff.net=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity(),torch.nn.Linear(3,2,bias=False)])
        pattern={'transformer_blocks.0.attn.to_q':80,'transformer_blocks.0.ff.net.2':48}
        inject_adapter_in_model(LoraConfig(**adapter_options({'rank':80,'rank_pattern':pattern})),model,adapter_name='mixed')
        self.assertEqual(block.attn.to_q.lora_A['mixed'].weight.shape,(80,2))
        self.assertEqual(block.ff.net[2].lora_A['mixed'].weight.shape,(48,3))
        self.assertEqual(block.attn.to_q.scaling['mixed'],1)
        self.assertEqual(block.ff.net[2].scaling['mixed'],1)

    def test_rejects_paths_nan_and_duplicate_specs(self):
        for specs in [[{'name':'../a.safetensors'}],[{'name':'a.safetensors','strength':float('nan')}],[{'name':'a.safetensors'}]*2]:
            with self.assertRaises(ValueError):validate_loras(specs)
        self.assertEqual(validate_loras([{'name':'impact.safetensors','strength':-1}])[0]['strength'],-1)



if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':
        unittest.main(argv=[sys.argv[0]])
    elif len(sys.argv)>2 and sys.argv[1]=='--child':
        specs=validate_loras(json.loads(Path(sys.argv[2]).read_text()))
        sys.argv=[sys.argv[0],*sys.argv[3:]]
        child(specs)
    else:
        parent()
