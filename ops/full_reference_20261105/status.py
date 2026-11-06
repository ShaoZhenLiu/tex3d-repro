"""Read real completion counters and a lock-protected supervisor state."""
import argparse
import fcntl
import json
import time
import sys

from common import REPO, ROOT, LONG_ROOT, NODES, evaluation_sources, long_jobs, migration_enabled
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
    migrated = migration_enabled()
    count = sum(len(list((source/'evaluation/episodes').glob(suite+'_*.json')))
                for source, suites, _ in evaluation_sources(target) for suite in suites)
    jobs = guard.read(folder/'evaluation/jobs.json', [])
    if migrated and target == 'pi05':
        done = sum((folder/'evaluation/completed'/(j['id']+'.json')).exists()
                   for j in jobs if j['suite'] != 'long')
        done += sum((LONG_ROOT/'evaluation/completed'/(j['id']+'.json')).exists() for j in long_jobs())
    else:
        done = sum((folder/'evaluation/completed'/(j['id']+'.json')).exists() for j in jobs)
    active = [dict(pid=e['pid'], gpu=e['gpu'], phase=e['phase'], label=e['label'],
                   progress_age_seconds=guard.progress_age(e, time.time()))
              for e in guard.active() if str(ROOT) in e['root']]
    rerun = dict(enabled=migrated, episodes=len(list((LONG_ROOT/'evaluation/episodes').glob('long_*.json'))),
                 planned_episodes=700, complete=guard.read(LONG_ROOT/'complete.json', {}).get('completed', False),
                 completed_bundles=sum((LONG_ROOT/'evaluation/completed'/(j['id']+'.json')).exists() for j in long_jobs()),
                 total_bundles=40, controller_error=guard.read(LONG_ROOT/'controller_error.json', {}))
    complete = guard.read(folder/'complete.json', {}).get('completed', False)
    controller = guard.read(folder/'eval_controller_status.json', {})
    if migrated:
        controller = (guard.read(LONG_ROOT/'eval_controller_status.json', {}) if node == '09'
                      else dict(retired_from_gpu_evaluation=True, collecting_long_from_09=True))
    return dict(time=time.time(), node=node, target=target, supervisor=supervisor,
        episodes=count, planned_episodes=2800, completed_bundles=done, total_bundles=len(jobs),
        complete=complete, active=active, long_rerun=rerun, migrated=migrated,
        work_complete=complete and (rerun['complete'] if migrated and node == '09' else True),
        gpu_evaluation_retired=(migrated and node == '4090'),
        accepted=guard.read(ROOT/'accepted.json', {}).get('completed', False),
        controller=controller)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--node', required=True)
    print(json.dumps(snapshot(parser.parse_args().node)))
