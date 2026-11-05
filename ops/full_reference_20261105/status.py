"""Read real completion counters and a lock-protected supervisor state."""
import argparse
import fcntl
import json
import time
import sys

from common import REPO, ROOT, NODES
sys.path.insert(0, str(REPO/'ops'))
import refcloth_resource as guard


def snapshot(node):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/('supervisor_'+node+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            supervisor = False
        except BlockingIOError:
            supervisor = True
    target = next(t for t, n in NODES.items() if n == node)
    folder = ROOT/'targets'/target
    count = len(list((folder/'evaluation/episodes').glob('*.json')))
    jobs = guard.read(folder/'evaluation/jobs.json', [])
    done = sum((folder/'evaluation/completed'/(j['id']+'.json')).exists() for j in jobs)
    active = [dict(pid=e['pid'], gpu=e['gpu'], phase=e['phase'], label=e['label'])
              for e in guard.active() if str(ROOT) in e['root']]
    return dict(time=time.time(), node=node, target=target, supervisor=supervisor,
        episodes=count, planned_episodes=2800, completed_bundles=done, total_bundles=len(jobs),
        complete=guard.read(folder/'complete.json', {}).get('completed', False), active=active,
        accepted=guard.read(ROOT/'accepted.json', {}).get('completed', False),
        controller=guard.read(folder/'eval_controller_status.json', {}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--node', required=True)
    print(json.dumps(snapshot(parser.parse_args().node)))
