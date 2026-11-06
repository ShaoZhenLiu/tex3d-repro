"""Resume the fixed all-seed queue with the established native inference path."""
import argparse
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from common import HERE, REPO, NODES, jobs, suite_root
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard
import refcloth_eval_scheduler as scheduler


def main(suite):
    node=NODES[suite];root=suite_root(suite)
    assert os.uname().nodename.endswith('-41') if node=='4090' else os.uname().nodename=='zgc-33-9'
    assert guard.read(root/'evaluation/jobs.json')==jobs(suite)
    try:
        scheduler.main('reference_coverage_20261106_'+suite,lambda:None,
            SimpleNamespace(gpus=[0,1,2,3],share_gpu3=False,resume=True),root=root,node=node,
            allow_stochastic_calibration=node=='09',evaluation_protocol=HERE/(suite+'.json'),
            reused_episode_count=700,memory_budget_mib=18000 if node=='09' else None)
        assert len(list((root/'evaluation/episodes').glob('*.json')))==3700
        guard.write(root/'complete.json',dict(completed=True,episodes=3700,new_episodes=3000,reused_episodes=700,suite=suite))
    except Exception as error:
        guard.write(root/'controller_error.json',dict(type=type(error).__name__,detail=str(error),time=time.time(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['goal','long'],required=True)
    main(p.parse_args().suite)
