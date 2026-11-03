"""Strict conversion of the pinned rank-32 Turbo Ref2VA adapter, without torch."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

SOURCE_SHA='c9482ccfffd2580b57a0b994035a94379afa97bc97dc0b05b5753e8ab906b9bd'
REVISION='dcacd93a50f714ba1cd61c7ca2cc5f65e9a1ec60'

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def convert(source,destination):
    source=Path(source);destination=Path(destination)
    if source.stat().st_size!=310189392 or sha256(source)!=SOURCE_SHA:
        raise ValueError('Expected the pinned Ref2VA Turbo Lite file')
    with source.open('rb') as handle:
        length=struct.unpack('<Q',handle.read(8))[0]
        if not 2<=length<=1024**2:raise ValueError('Invalid header')
        header=json.loads(handle.read(length));header.pop('__metadata__',None)
        end=0
        for start,stop,key in sorted((v['data_offsets'][0],v['data_offsets'][1],k) for k,v in header.items()):
            v=header[key];size={'BF16':2,'F32':4}.get(v['dtype'])
            if size is None or start!=end or stop-start!=math.prod(v['shape'])*size:raise ValueError('Invalid tensor layout')
            end=stop
        if 8+length+end!=source.stat().st_size:raise ValueError('Invalid payload size')
        pattern=re.compile(r'diffusion_model\.(blocks\.(\d+)|token_refiner\.blocks\.(\d+))\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))\.(alpha|lora_([AB])\.weight)')
        groups={}
        for key,v in header.items():
            m=pattern.fullmatch(key)
            if not m:raise ValueError('Unexpected tensor '+key)
            block,ref=m.group(2),m.group(3)
            if (block is not None and int(block)>=50) or (ref is not None and int(ref)>=2):raise ValueError('Invalid block')
            prefix='transformer_blocks.'+block if block is not None else 'token_refiner.refiner_blocks.'+ref
            groups.setdefault((prefix,m.group(5) or m.group(6)),{})[m.group(8) or 'alpha']=v
        if len(groups)!=208:raise ValueError('Expected all 208 adapter modules')
        entries=[]
        def tensor(name,shape,segments):entries.append((name,shape,segments))
        for (prefix,module),g in sorted(groups.items()):
            if set(g)!={'A','B','alpha'}:raise ValueError('Incomplete adapter group')
            a,b,alpha=g['A'],g['B'],g['alpha'];rank=32
            ins,outs={'qkv_proj':(5376,21504),'out_proj':(7168,5376),'fc1':(5376,28672),'fc2':(14336,5376)}[module]
            if a['dtype']!='BF16' or b['dtype']!='BF16' or a['shape']!=[rank,ins] or b['shape']!=[outs,rank]:raise ValueError('Unexpected rank or shape')
            if alpha['dtype']!='F32' or alpha['shape']!=[]:raise ValueError('Expected scalar FP32 alpha')
            handle.seek(8+length+alpha['data_offsets'][0]);value=struct.unpack('<f',handle.read(4))[0]
            if value!=rank:raise ValueError('This pinned converter requires alpha=rank')
            aa=tuple(a['data_offsets']);bb=tuple(b['data_offsets'])
            if module=='qkv_proj':
                for i,name in enumerate('qkv'):
                    target=prefix+'.attn.to_'+name
                    tensor(target+'.lora_A.default.weight',[rank,ins],[aa])
                    tensor(target+'.lora_B.default.weight',[7168,rank],[(bb[0]+i*7168*rank*2,bb[0]+(i+1)*7168*rank*2)])
            else:
                target=prefix+'.'+{'out_proj':'attn.to_out.0','fc1':'ff.net.0.proj','fc2':'ff.net.2'}[module]
                tensor(target+'.lora_A.default.weight',[rank,ins],[aa])
                segments=[bb] if module!='fc1' else [(bb[0]+14336*rank*2,bb[1]),(bb[0],bb[0]+14336*rank*2)]
                tensor(target+'.lora_B.default.weight',[outs,rank],segments)
        out={'__metadata__':{'source_sha256':SOURCE_SHA,'source_revision':REVISION,'conversion':'h3-turbo-ref-lite-r32-v1','alpha_scaling':'1'}};offset=0
        for name,shape,segments in entries:
            size=sum(stop-start for start,stop in segments);out[name]={'dtype':'BF16','shape':shape,'data_offsets':[offset,offset+size]};offset+=size
        encoded=json.dumps(out,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%8)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.tmp')
        with temporary.open('wb') as f:
            f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
            for name,shape,segments in entries:
                for start,stop in segments:
                    handle.seek(8+length+start);remaining=stop-start
                    while remaining:
                        chunk=handle.read(min(remaining,1024**2))
                        if not chunk:raise ValueError('Truncated source')
                        f.write(chunk);remaining-=len(chunk)
        temporary.replace(destination)
    return {'source_sha256':SOURCE_SHA,'sha256':sha256(destination),'bytes':destination.stat().st_size,'rank':32,'modules':312,'revision':REVISION}

if __name__=='__main__':
    import sys
    print(json.dumps(convert(sys.argv[1],sys.argv[2])))
