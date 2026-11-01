"""Strict AI Toolkit MiniMax H3 LoRA conversion and additive PEFT loading."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def convert(source, destination, dimensions=(5376, 7168, 14336)):
    raw = Path(source).read_bytes()
    if not 8 < len(raw) <= 256 * 1024**2:
        raise ValueError('LoRA file must be at most256MiB')
    length = struct.unpack('<Q', raw[:8])[0]
    if not 2 <= length <= min(len(raw)-8, 8*1024**2):
        raise ValueError('Invalid safetensors header')
    header = json.loads(raw[8:8+length])
    metadata = header.pop('__metadata__', {})
    if metadata.get('ss_base_model_version') != 'minimax_h3':
        raise ValueError('Expected an AI Toolkit minimax_h3 LoRA')
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
    for (prefix,module), pair in groups.items():
        if set(pair) != {'A','B'}:
            raise ValueError('Incomplete LoRA A/B pair')
        a,b = pair['A'],pair['B']
        rank = a['shape'][0]
        expected = {'qkv_proj':(hidden,3*inner),'out_proj':(inner,hidden),'fc1':(hidden,2*ffn),'fc2':(ffn,hidden)}[module]
        if not 1<=rank<=64 or a['shape'] != [rank,expected[0]] or b['shape'] != [expected[1],rank]:
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
            converted[name+'.lora_A.weight']=([rank,expected[0]],ad)
            converted[name+'.lora_B.weight']=([rows,rank],data)
    if not converted or len(ranks)!=1:
        raise ValueError('Expected a nonempty, uniform-rank LoRA')
    rank=ranks.pop()
    output={'__metadata__':{'source_sha256':hashlib.sha256(raw).hexdigest(),'conversion':'ai-toolkit-h3-to-diffusers-v1','rank':str(rank),'default_alpha':str(rank)}}
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
    return dict(rank=rank,source_sha256=output['__metadata__']['source_sha256'],modules=len(converted)//2)


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
        alpha=spec.get('alpha',info['rank'])
        model.add_adapter(LoraConfig(r=info['rank'],lora_alpha=alpha,target_modules=modules,init_lora_weights=False),adapter_name=name)
        own=dict(model.named_parameters())
        with torch.no_grad():
            for key,value in state.items():
                target=key.replace('.lora_A.weight','.lora_A.'+name+'.weight').replace('.lora_B.weight','.lora_B.'+name+'.weight')
                if target not in own or own[target].shape!=value.shape:
                    raise ValueError('Converted adapter parameter mismatch: '+target)
                own[target].copy_(value.to(device=own[target].device,dtype=own[target].dtype))
                own[target].requires_grad_(False)
        names.append(name);weights.append(spec['strength'])
        reports.append({**info,'name':spec['name'],'strength':spec['strength'],'alpha':alpha})
    if specs:
        model.set_adapters(names,weights=weights)
    return reports

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
    import benchmark_worker
    benchmark_worker.child('nf4')


def parent():
    import handler as base
    import runpod
    context=contextvars.ContextVar('loras',default=[])
    original_command=base.command
    def command(request,workdir,output):
        config=workdir/'custom-loras.json'
        config.write_text(json.dumps(context.get()))
        argv=original_command(request,workdir,output)
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
        if data.get('action','generate')!='health' and (data.get('task','t2va')!='t2va' or data['cpu_offload']):
            return {'error':'NF4 custom LoRA worker currently supports t2va without CPU offload'}
        token=context.set(specs)
        try:
            result=base.handler({**job,'input':data})
            if data.get('action')=='health' and 'error' not in result:
                result['runtime']='nf4-custom-lora-v1'
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
