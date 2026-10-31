"""Read cached source features through the unchanged production action head.

Only the action head is loaded, never the 7B backbone. No new target-policy
queries. This examines whether a feature-transport improvement survives the
source decoder, separately from a closed-loop success-rate evaluation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from local_transport import transport


def load_head(source, checkpoint, device):
    # Execute exactly the upstream three class definitions without importing
    # unrelated diffusion and vision stacks. Strict checkpoint loading verifies
    # all parameter names and dimensions. Definitions are not reimplemented.
    module=ast.parse(source.read_text())
    names={'MLPResNetBlock','MLPResNet','L1RegressionActionHead'}
    selected=[node for node in module.body if isinstance(node,ast.ClassDef) and node.name in names]
    assert {node.name for node in selected}==names
    namespace=dict(torch=torch,nn=nn,ACTION_DIM=7,NUM_ACTIONS_CHUNK=8)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    head=namespace['L1RegressionActionHead']()
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    weights={key.removeprefix('module.'):value for key,value in weights.items()}
    head.load_state_dict(weights,strict=True)
    head.eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
    return head


def error_summary(predicted,truth,baseline):
    error=(predicted-truth).double()
    base_error=(baseline-truth).double()
    result=dict(relative_action_mse=float(error.square().sum()/base_error.square().sum()),
                action_rmse=float(error.square().mean().sqrt()),baseline_rmse=float(base_error.square().mean().sqrt()))
    for name,sl in [('executed_translation',(...,slice(0,5),slice(0,3))),
                    ('executed_rotation',(...,slice(0,5),slice(3,6))),
                    ('executed_gripper',(...,slice(0,5),slice(6,7)))]:
        result[name]=float(error[sl].square().sum()/base_error[sl].square().sum().clamp_min(1e-20))
    return result


def run(root,head_source,checkpoint,output,device):
    torch.set_num_threads(2)
    head=load_head(head_source,checkpoint,device)
    report=dict(device=device,head_source_sha256=hashlib.sha256(head_source.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                full_backbone_loaded=False,locked_test_consumed=False,
                precision='production head weights and activations bf16',splits={})
    def predict(features):
        with torch.no_grad():
            return head.predict_action(features.reshape(-1,56,4096).to(device=device,dtype=torch.bfloat16)).float().cpu()
    for split in ['fit','validation']:
        predictions,baselines,truths,states=[],[],[],[]
        for path in sorted((root/'pairs'/split).glob('*.pt')):
            rec=torch.load(path,map_location='cpu',weights_only=False)
            tids=sorted([k[:-len('__fR_float')] for k in rec if k.endswith('__fR_float')],
                        key=lambda tid:int(tid.rsplit('s',1)[-1]))
            r=torch.stack([rec[tid+'__fR_float']-rec['fR0'] for tid in tids]).reshape(-1,56,4096)
            n=torch.stack([rec[tid+'__fN_Q']-rec['fN0'] for tid in tids]).reshape(-1,56,4096)
            calibration=list(range(0,len(tids),2));tests=list(range(1,len(tids),2))
            corrected=transport(r[tests],r[calibration],n[calibration])
            native0=rec['fN0'].reshape(1,56,4096)
            target=predict(native0+n[tests])
            base=predict(native0+r[tests])
            pred=predict(native0+corrected)
            # Repeat the exact same inference to expose a numerical floor.
            repeat=predict(native0+n[tests])
            states.append(dict(state=rec['record_id'],task=int(rec['source_task_id']),
                repeat_maxdiff=float((target-repeat).abs().max()),**error_summary(pred,target,base)))
            predictions.append(pred);baselines.append(base);truths.append(target)
            print('completed',split,rec['record_id'],flush=True)
        report['splits'][split]=dict(metrics=error_summary(torch.cat(predictions),torch.cat(truths),torch.cat(baselines)),states=states)
    output.mkdir(parents=True,exist_ok=True)
    (output/'action_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--head-source',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();run(args.root,args.head_source,args.checkpoint,args.out,args.device)
