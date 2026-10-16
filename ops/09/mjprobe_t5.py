import sys
sys.path.insert(0, '/data/shaozhen.liu/code/tex3d/pi')
from attack_pi0 import _get_libero_env, get_libero_dummy_action  # noqa: E402
from libero.libero import benchmark  # noqa: E402
import mujoco  # noqa: E402

print('mujoco', mujoco.__version__)
b = benchmark.get_benchmark_dict()['libero_spatial']()
task = b.get_task(5)
inits = b.get_task_init_states(5)

env, desc = _get_libero_env(task, 256)
dummy = get_libero_dummy_action()

KEYS = ['akita_black_bowl_1', 'akita_black_bowl_2',
        'glazed_rim_porcelain_ramekin_1', 'plate_1', 'cookies_1']

for ep in range(2):
    env.reset()
    env.set_init_state(inits[ep])
    sim = env.env.sim
    sim.forward()
    ids = {k: sim.model.body_name2id(k + '_main') for k in KEYS}

    def track():
        return {k: tuple(round(float(v), 3) for v in sim.data.xpos[i])
                for k, i in ids.items()}

    print(f'ep{ep} t=0    {track()}')
    step = 0
    for t in (10, 25, 50, 100, 200, 400):
        while step < t:
            env.step(dummy)
            step += 1
        r = track()
        b1, rk = r['akita_black_bowl_1'], r['glazed_rim_porcelain_ramekin_1']
        dx, dy = b1[0] - rk[0], b1[1] - rk[1]
        dz = b1[2] - rk[2]
        print(f'ep{ep} t={t:<4} bowl1_on_ramekin_offset=({dx:+.3f},{dy:+.3f},{dz:+.3f}) '
              f'succ={env.env._check_success()} {r}')

from attack_pi0 import _close_libero_env
_close_libero_env(env)
