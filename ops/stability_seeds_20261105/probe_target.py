"""Run the current evaluator's calibration while retaining both Plain traces."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import attack_pi0_cloth as cloth
import refcloth_resource as guard
from stability_20261102 import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--worker', default='a100_current')
parser.add_argument('--memory-fraction', type=float, default=.31)
args = parser.parse_args()
args.smoke = True
args.compare_existing_smoke = True
args.root.mkdir(parents=True, exist_ok=True)

original = cloth.evaluate_native
traces = []

def capture(policy, *positional, **keywords):
    started = time.monotonic()
    success = original(policy, *positional, **keywords)
    trace = dict(robot_success=bool(success), queries=len(policy.rows),
                 elapsed_seconds=time.monotonic()-started,
                 trajectory=copy.deepcopy(policy.rows))
    traces.append(trace)
    evaluate.write(args.root / ('trace_%d.json' % (len(traces)-1)), trace)
    return success

cloth.evaluate_native = capture
result = dict(completed=False, passed=False, formal_episodes=0,
              evaluator_sha256=hashlib.sha256(Path(evaluate.__file__).read_bytes()).hexdigest(),
              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started_at=time.time())
try:
    evaluate.main(args)
except Exception as error:
    result.update(error_type=type(error).__name__, error=str(error))
    raise
else:
    result['passed'] = True
finally:
    result.update(completed=True, finished_at=time.time(),
                  observed_queries=[x['queries'] for x in traces],
                  observed_successes=[x['robot_success'] for x in traces])
    if len(traces) >= 2:
        fields = ('noise_sha256', 'main_sha256', 'wrist_sha256', 'proprio_sha256',
                  'action_sha256', 'actions')
        result['within_host_trajectory_exact'] = (
            traces[0]['robot_success'] == traces[1]['robot_success']
            and traces[0]['trajectory'] == traces[1]['trajectory'])
        result['mismatching_queries'] = {
            field: [i for i, (left, right) in
                    enumerate(zip(traces[0]['trajectory'], traces[1]['trajectory']))
                    if left.get(field) != right.get(field)]
            for field in fields
        }
        baseline = args.root / 'evaluation/episodes/spatial_t0_i10_plain.json'
        if baseline.exists():
            canonical = json.loads(baseline.read_text())
            result['canonical_trajectory_exact'] = (
                traces[0]['robot_success'] == canonical['robot_success']
                and traces[0]['trajectory'] == canonical['trajectory'])
    evaluate.write(args.root / 'probe_result.json', result)

