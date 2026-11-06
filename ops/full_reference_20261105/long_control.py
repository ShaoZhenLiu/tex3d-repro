"""Run the whole pi05 Long block on A100 with the established fixed recipe."""
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from common import HERE, LONG_ROOT, REPO, long_jobs
sys.path.insert(0, str(REPO/'ops'))
import refcloth_resource as guard
import refcloth_eval_scheduler as scheduler


def main():
    assert os.uname().nodename == 'zgc-33-9'
    assert guard.read(LONG_ROOT/'prepared09.json')['passed']
    assert guard.read(LONG_ROOT/'evaluation/jobs.json') == long_jobs()
    try:
        scheduler.main('full_reference_pi05_long09_20261106', lambda: None,
            SimpleNamespace(gpus=[0, 1, 2, 3], share_gpu3=False, resume=True),
            root=LONG_ROOT, node='09', allow_stochastic_calibration=True,
            evaluation_protocol=HERE/'pi05_long09.json', reused_episode_count=0,
            memory_budget_mib=18000)
        assert len(list((LONG_ROOT/'evaluation/episodes').glob('*.json'))) == 700
        guard.write(LONG_ROOT/'complete.json', dict(completed=True, episodes=700,
                    new_episodes=700, reused_episodes=0, target='pi05', suites=['long'], node='09'))
    except Exception as error:
        guard.write(LONG_ROOT/'controller_error.json', dict(type=type(error).__name__, detail=str(error),
            commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
