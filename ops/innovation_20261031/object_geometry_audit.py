"""Inspect geometry behind the preselected Object view interventions; no policy."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def run(output,all_keys=False):
    import attack_pi0_cloth as APC
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_object']()
    report=dict(scope='geometry inspection only; no policy inference',
                surface_mode=os.environ.get('REFCLOTH_SURFACE_MODE','legacy'),records=[])
    clean=APC.patch_status()=='clean';APC.ensure_patch(True)
    try:
        for task_id in (range(10) if all_keys else [0,4,8]):
            task=suite.get_task(task_id);states=suite.get_task_init_states(task_id)
            for initial in (range(10,20) if all_keys else [10,11]):
                env,_=APC._get_libero_env(task,resolution=256)
                try:
                    env.reset();obs=env.set_init_state(states[initial]);sim=env.env.sim;sim.forward()
                    _,bid,_=APC.get_target_model_matrix(env,APC.CLOTH_SEARCH)
                    def snapshot():
                        sites=[]
                        for sid in range(sim.model.nsite):
                            if sim.model.site_id2name(sid)=='table_top':sites.append(sim.data.site_xpos[sid].tolist())
                        return dict(cloth_local=sim.model.body_pos[bid].tolist(),cloth_world=sim.data.body_xpos[bid].tolist(),table_top=sites)
                    before=snapshot();APC.align_cloth_to_table(env);after=snapshot()
                    for _ in range(10):env.step(APC.get_libero_dummy_action())
                    end=snapshot()
                    rec=dict(task=task_id,initial=initial,before=before,after=after,after_wait=end)
                    report['records'].append(rec);print(json.dumps(rec),flush=True)
                finally:APC._close_libero_env(env)
    finally:
        if clean:APC.ensure_patch(False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--all-keys',action='store_true')
    args=ap.parse_args();run(args.out,args.all_keys)
