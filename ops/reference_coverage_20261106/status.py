"""Report real records and model progress for the fixed coverage extension."""
import argparse
import fcntl
import json
import sys
import time
from common import REPO, ROOT, NODES, jobs, suite_root
sys.path.insert(0,str(REPO/'ops'))
import refcloth_resource as guard


def snapshot(node):
    suite=next(s for s,n in NODES.items() if n==node);root=suite_root(suite)
    with (ROOT/('supervisor_'+node+'.lock')).open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);supervisor=False
        except BlockingIOError:supervisor=True
    active=[dict(pid=e['pid'],gpu=e['gpu'],phase=e['phase'],label=e['label'],
                 progress_age_seconds=guard.progress_age(e,time.time()))
            for e in guard.active() if str(ROOT) in e['root']]
    episodes=len(list((root/'evaluation/episodes').glob('*.json')))
    return dict(time=time.time(),node=node,suite=suite,supervisor=supervisor,
        episodes=episodes,new_episodes=episodes-700,reused_episodes=700,planned_episodes=3700,
        completed_bundles=sum((root/'evaluation/completed'/(j['id']+'.json')).exists() for j in jobs(suite)),
        total_bundles=190,active=active,complete=guard.read(root/'complete.json',{}).get('completed',False),
        accepted=guard.read(ROOT/'accepted.json',{}).get('completed',False),
        controller=guard.read(root/'eval_controller_status.json',{}),
        controller_error=guard.read(root/'controller_error.json',{}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True)
    print(json.dumps(snapshot(p.parse_args().node)))
