"""Isolated qualification worker; production defaults and files are untouched."""
import contextvars
import json
from pathlib import Path
import sys
import time

ROOT=Path('/runpod-volume/h3-flex')
TURBO=ROOT/'turbo/ref2va-lite-r32-v1'

def euler_update(state,prediction,sigma,next_sigma):
    return state+(prediction-state)*(1-next_sigma/sigma)

def euler_sample(wrapper,*,conditional_dict,video_shape,audio_shape,sigmas_v,sigmas_a,seed,device,**kwargs):
    import torch
    from pika_h3.core.evaluation import evaluation_conditions,evaluation_noise_scope
    generator=torch.Generator(device=device).manual_seed(int(seed))
    video=torch.randn(video_shape,device=device,dtype=torch.float32,generator=generator)
    audio=torch.randn(audio_shape,device=device,dtype=torch.float32,generator=generator)
    cond=evaluation_conditions(conditional_dict)
    with torch.inference_mode(),evaluation_noise_scope(seed,device):
        for i in range(len(sigmas_v)-1):
            sv=sigmas_v[i].float();sa=sigmas_a[i].float()
            pv,pa=wrapper(noisy_image_or_video=video,conditional_dict=cond,
                timestep=sv.expand(video.shape[0],video.shape[1]),noisy_audio=audio,
                audio_timestep=sa.expand(audio.shape[0],audio.shape[1]))
            video=euler_update(video,pv.float(),sv,sigmas_v[i+1])
            audio=euler_update(audio,pa.float(),sa,sigmas_a[i+1])
    return video.float(),audio.float()

def turbo_child(_profile):
    from pika_h3 import pipeline
    from pika_h3.core import evaluation
    import ref_worker
    from turbo_lite import sha256,REVISION,SOURCE_SHA
    info=json.loads((TURBO/'manifest.json').read_text())
    adapter=TURBO/'adapter.safetensors'
    if info['source_sha256']!=SOURCE_SHA or sha256(adapter)!=info['sha256']:raise ValueError('Turbo cache checksum mismatch')
    sampling={'method':'euler','knots':[1000,750,500,250,0],'video_shift':12,'audio_shift':3,'cfg':1.0,'steps':4}
    def assets(args):
        if args.task!='ref2va':raise ValueError('Qualification profile supports only Ref2VA')
        return ROOT/'native-assets/base',adapter,{'rank':32,'alpha':32,'trunk':'transformer_ref',
            'numerics_policy':'fp32_conditioning_v1','sha256':info['sha256'],'sampling':sampling,
            'base':{'repo_id':pipeline.BASE_REPO,'revision':pipeline.BASE_REVISION}}
    pipeline.resolve_assets=assets;pipeline.KNOTS=sampling['knots'];evaluation.sample_evaluation_ladder=euler_sample
    original=pipeline.generate
    def generate(args):
        original(args)
        path=args.output.with_suffix('.json');data=json.loads(path.read_text())
        data.update(profile='turbo-ref-lite-4',adapter_revision=REVISION,turbo_source_sha256=SOURCE_SHA)
        path.write_text(json.dumps(data,indent=2))
    pipeline.generate=generate
    ref_worker.ref_child('nf4')

def parent():
    import handler as base
    import lora_worker as slim
    import runpod
    context=contextvars.ContextVar('profile',default=('turbo',[]));original=base.command
    def command(request,workdir,output):
        profile,specs=context.get();config=workdir/'custom-loras.json';config.write_text(json.dumps(specs))
        argv=original(request,workdir,output)
        argv+=['--base-path',str(ROOT/'native-assets/base'),'--adapter-path',str(ROOT/'native-assets/adapter/ref2va')]
        entry=__file__ if profile=='turbo' else '/app/ref_worker.py'
        return [sys.executable,entry,'--child',str(config),*argv[1:]]
    base.command=command
    def run(data,profile):
        token=context.set((profile,slim.validate_loras(data.pop('loras',[]))))
        try:
            started=time.monotonic();out=base.handler({'input':data})
            if isinstance(out,dict):out['qualification_wall_seconds']=time.monotonic()-started
            return out
        finally:context.reset(token)
    def handler(job):
        raw=job.get('input')
        if not isinstance(raw,dict):return {'error':'input must be an object'}
        data=dict(raw);profile=data.pop('profile','turbo')
        if profile not in ('turbo','baseline','compare'):return {'error':'Unsupported qualification profile'}
        if data.get('task')!='ref2va' or data.get('cpu_offload') is True:return {'error':'Ref2VA without CPU offload required'}
        data.setdefault('cpu_offload',False)
        try:
            if profile=='compare':
                baseline=run(dict(data),'baseline')
                if baseline.get('error'):return {'baseline':baseline,'error':'Baseline failed; Turbo not run'}
                return {'baseline':baseline,'turbo':run(dict(data),'turbo')}
            return run(data,profile)
        except (ValueError,TypeError) as e:return {'error':str(e)}
    runpod.serverless.start({'handler':handler})

if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--child':
        import lora_worker as slim
        specs=slim.validate_loras(json.loads(Path(sys.argv[2]).read_text()))
        sys.argv=[sys.argv[0],*sys.argv[3:]];slim.nf4_child=turbo_child;slim.child(specs)
    else:parent()
