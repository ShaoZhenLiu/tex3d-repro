"""Resource-guarded target controller using the established native evaluator."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from common import HERE, NODES, REPO, ROOT
sys.path.insert(0, str(REPO/'ops'))
import refcloth_eval_scheduler as scheduler
import refcloth_resource as guard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['pi0', 'pi05'], required=True)
    args = parser.parse_args()
    node = NODES[args.target]
    assert (os.uname().nodename.endswith('-41') if node == '4090' else os.uname().nodename == 'zgc-33-9')
    root = ROOT/'targets'/args.target
    reused = len(guard.read(root/'evaluation/reused_manifest.json'))
    scheduler.main('full_reference_%s_20261105' % args.target, lambda: None,
        SimpleNamespace(gpus=[0, 1, 2, 3], share_gpu3=False, resume=True),
        root=root, node=node, allow_stochastic_calibration=(node == '09'),
        evaluation_protocol=HERE/(args.target+'.json'), reused_episode_count=reused)
    assert len(list((root/'evaluation/episodes').glob('*.json'))) == 2800
    guard.write(root/'complete.json', dict(completed=True, episodes=2800,
                reused_episodes=reused, new_episodes=2800-reused, target=args.target))
