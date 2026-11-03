"""Experimental Ref2VA worker. Separate cache; existing FL2VA worker is untouched."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/runpod-volume/h3-flex')
BASE_REV='42ed227ee7df40d41602854ae760620d6eb651fe'
ADAPTER_REV='0ddf55ad7067ca1e76812253bd7b4600df87f759'

def convert():
    import torch
    from diffusers import MiniMaxH3Transformer3DModel, BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    started=time.monotonic()
    target=ROOT/'nf4-v1/transformer_ref'
    if (target/'COMPLETE').is_file():
        return {'cached':True,'path':str(target)}
    raw=snapshot_download('MiniMaxAI/MiniMax-H3',revision=BASE_REV,allow_patterns=['transformer_ref/*'])
    adapter=snapshot_download('Pika-Labs/Pika-H3-Flex',revision=ADAPTER_REV,allow_patterns=['ref2va/*'])
    import shutil
    destination=ROOT/'native-assets/adapter/ref2va'
    shutil.copytree(Path(adapter)/'ref2va',destination,dirs_exist_ok=True)
    manifest=json.loads((destination/'manifest.json').read_text())
    if manifest['trunk']!='transformer_ref' or manifest['numerics_policy']!='fp32_conditioning_v1':
        raise ValueError('Unexpected reference adapter contract')
    model=MiniMaxH3Transformer3DModel.from_pretrained(raw,subfolder='transformer_ref',torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True))
    target.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(target,safe_serialization=True,max_shard_size='4GB')
    del model;gc.collect();torch.cuda.empty_cache()
    # Verify a fresh load of the serialized quantization before marking ready.
    model=MiniMaxH3Transformer3DModel.from_pretrained(target,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True)
    if not getattr(model,'is_loaded_in_4bit',False):
        raise RuntimeError('Serialized reference cache did not reload as 4-bit')
    report={'base_revision':BASE_REV,'adapter_revision':ADAPTER_REV,'seconds':time.monotonic()-started,'cache_bytes':sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),'quantized_reload':True}
    (target/'COMPLETE').write_text(json.dumps(report))
    return report

def ref_child(_profile):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers import MiniMaxH3Transformer3DModel
    from pika_h3 import pipeline
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    def patch(cls,component):
        original=cls.from_pretrained
        def load(_cls,path,*args,**kwargs):
            folder=kwargs.get('subfolder')
            selected='transformer_ref' if component=='transformer' and folder=='transformer_ref' else component
            target=ROOT/'nf4-v1'/selected
            if not (target/'COMPLETE').is_file():raise RuntimeError('Missing verified NF4 cache: '+str(target))
            kwargs.pop('subfolder',None)
            kwargs.update(local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda:0'})
            return original(str(target),*args,**kwargs)
        cls.from_pretrained=classmethod(load)
    patch(Qwen3VLForConditionalGeneration,'text_encoder');patch(MiniMaxH3Transformer3DModel,'transformer')
    from pika_h3.cli import main
    main()

def parent():
    import handler as base
    import runpod
    import contextvars
    import lora_worker as slim
    context=contextvars.ContextVar('loras',default=[])
    original=base.command
    def command(request,workdir,output):
        config=workdir/'custom-loras.json';config.write_text(json.dumps(context.get()))
        argv=original(request,workdir,output)
        adapter='ref2va' if request['task']=='ref2va' else 'fl2va'
        argv+=['--base-path',str(ROOT/'native-assets/base'),'--adapter-path',str(ROOT/'native-assets/adapter'/adapter)]
        return [sys.executable,__file__,'--child',str(config),*argv[1:]]
    base.command=command
    def handler(job):
        data=job.get('input')
        if not isinstance(data,dict):return {'error':'input must be an object'}
        if data.get('action')=='convert_ref':
            # Endpoint execution timeout is independently capped. A child failure
            # never leaves COMPLETE behind and can be retried explicitly.
            result=subprocess.run([sys.executable,__file__,'--convert'],capture_output=True,text=True,timeout=1800)
            if result.returncode:return {'error':'Ref conversion failed','log':result.stderr[-4000:]}
            return json.loads(result.stdout.strip().splitlines()[-1])
        data=dict(data)
        try:specs=slim.validate_loras(data.pop('loras',[]))
        except ValueError as e:return {'error':str(e)}
        data.setdefault('cpu_offload',False)
        if data.get('action')!='health' and (data.get('task','t2va') not in ('t2va','i2va','ref2va') or data['cpu_offload']):return {'error':'Supported tasks: t2va/i2va/ref2va without CPU offload'}
        token=context.set(specs)
        try:return base.handler({**job,'input':data})
        finally:context.reset(token)
    runpod.serverless.start({'handler':handler})

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--convert':print(json.dumps(convert()))
    elif len(sys.argv)>2 and sys.argv[1]=='--child':
        import lora_worker as slim
        specs=slim.validate_loras(json.loads(Path(sys.argv[2]).read_text()))
        sys.argv=[sys.argv[0],*sys.argv[3:]]
        slim.nf4_child=ref_child
        slim.child(specs)
    else:parent()
