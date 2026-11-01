"""Exploratory target closed-loop pilot with frozen source-selected masks.

Uses the production evaluate_native loop, adds fail-fast texture upload,
explicit consumed diffusion noise, per-episode logs and repeat checks.
Does not optimize, rank, revise or select masks using target observations.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def main(repo,source,out,suite_name,protocol_path,tasks=None,resume_file=None):
    import attack_pi0_cloth as APC
    import attack_pi0 as BASE
    from libero.libero import benchmark
    protocol=json.loads(protocol_path.read_text())
    torch.set_num_threads(2)
    torch.manual_seed(7);np.random.seed(7)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    assert APC.patch_status()=='patched', 'Controller must own target asset patch'
    assert os.environ.get('REFCLOTH_SURFACE_MODE','legacy')!='object_fixed_floor'
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'texture_manifest.json').read_text())
    definitions={x['name']:x for x in manifest}
    textures={}
    for name in protocol['conditions']:
        path=source/'textures'/(name+'.png')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==definitions[name]['file_sha256']
        textures[name]=np.array(Image.open(path).convert('RGB'))
    base=np.array(Image.open(APC.CLOTH_T0).convert('RGB'))
    assert np.array_equal(base,textures['plain']), 'Source and target plain cloth differ'
    real_inject=APC.inject_mujoco_texture
    injection_audit=[]
    def checked_inject(sim,tid,png):
        m=getattr(sim.model,'_model',sim.model)
        assert png.shape[:2]==(int(m.tex_height[tid]),int(m.tex_width[tid]))
        handle=real_inject(sim,tid,png)
        assert BASE._upload_mujoco_texture(sim,tid), 'Native target upload failed'
        injection_audit.append(True)
        return handle
    APC.inject_mujoco_texture=checked_inject
    policy=APC.load_policy(protocol['checkpoint'],model_config=protocol['model_config'],num_denoise=protocol['num_denoise'])
    cfg=APC.training_config.get_config(protocol['model_config']).model
    shape=(cfg.action_horizon,cfg.action_dim)
    suite=benchmark.get_benchmark_dict()[suite_name]()
    suite_index=protocol['suites'].index(suite_name)
    previous=[]
    if resume_file is not None and resume_file.exists():
        old=json.loads(resume_file.read_text())
        assert old['frozen_manifest_sha256']==hashlib.sha256((source/'texture_manifest.json').read_bytes()).hexdigest()
        assert old['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        previous=[r for r in old['records'] if tasks is None or r['task'] in tasks]
    report=dict(suite=suite_name,protocol=protocol,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        frozen_manifest_sha256=hashlib.sha256((source/'texture_manifest.json').read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        records=previous,completed=False,noise_shape=list(shape),target_driven_selection=False,
        resumed_records=len(previous),resume_file=str(resume_file),
        limitations='Exploratory 6 paired episodes per suite; no noninferiority/physical hardware claim')
    (out/'registered.json').write_text(json.dumps(report,indent=2))
    class PairedPolicy:
        def reset(self,task,initial):
            self.task,self.initial,self.nq=task,initial,0
            self.hashes=[];self.repeat_floor=None
        def infer(self,raw):
            seed=710000+suite_index*10000000+self.task*10000+self.initial*100+self.nq
            noise=np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
            noise_hash=hashlib.sha256(noise.tobytes()).hexdigest()
            with torch.no_grad():
                result=policy.infer(raw,noise=noise.copy())
                if self.nq==0:
                    repeat=policy.infer(raw,noise=noise.copy())
                    self.repeat_floor=float(np.max(np.abs(np.asarray(result['actions'])-np.asarray(repeat['actions']))))
                    assert self.repeat_floor<=1e-6,self.repeat_floor
            actions=np.asarray(result['actions'])
            assert actions.ndim==2 and actions.shape[1]==7 and np.isfinite(actions).all()
            self.hashes.append(noise_hash);self.nq+=1
            return result
    paired=PairedPolicy()
    # Same episode keys/noises across conditions. Order is fixed in advance.
    done_keys={(r['task'],r['initial'],r['condition']) for r in previous}
    for task_id in (protocol['tasks'] if tasks is None else tasks):
        task=suite.get_task(task_id)
        inits=suite.get_task_init_states(task_id)
        for initial in protocol['initials']:
            common_hashes=next((r['noise_sha256'] for r in previous if r['task']==task_id and r['initial']==initial),None)
            for condition in protocol['conditions']:
                if (task_id,initial,condition) in done_keys:
                    continue
                paired.reset(task_id,initial)
                t0=time.monotonic()
                injected_before=len(injection_audit)
                ok=APC.evaluate_native(paired,task,inits[initial],textures[condition],
                    max_steps=protocol['max_steps'],num_steps_wait=protocol['settle_steps'],
                    replan_steps=protocol['replan_steps'],floor_mode=suite_name=='libero_object')
                assert len(injection_audit)==injected_before+1
                assert paired.nq>0
                if common_hashes is None:common_hashes=paired.hashes
                overlap=min(len(common_hashes),len(paired.hashes))
                assert common_hashes[:overlap]==paired.hashes[:overlap]
                row=dict(task=task_id,initial=initial,condition=condition,robot_success=bool(ok),
                    attack_success=int(not ok),queries=paired.nq,aa_action_maxdiff=paired.repeat_floor,
                    noise_sha256=paired.hashes,elapsed_seconds=time.monotonic()-t0,
                    png_file_sha256=definitions[condition]['file_sha256'],native_upload_ok=True)
                report['records'].append(row)
                (out/'results.partial.json').write_text(json.dumps(report,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k!='noise_sha256'}),flush=True)
    report['completed']=True
    report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    (out/'results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--suite',required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--tasks',default=None)
    ap.add_argument('--resume-file',type=Path,default=None)
    a=ap.parse_args();main(a.repo,a.source,a.out,a.suite,a.protocol,
        None if a.tasks is None else [int(x) for x in a.tasks.split(',')],a.resume_file)
