"""Freeze the existing exports and construct the quantized amplitude controls."""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from common import INITIALS, PRIOR, PROTOCOL, ROOT, SEEDS, TASKS, controls, guard, local_node, sha


def freeze(source, destination, digest):
    source, destination = Path(source), Path(destination)
    assert sha(source) == digest, source
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    assert sha(destination) == digest, destination


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def rms(delta):
    return float(np.sqrt(np.mean(np.square(delta))))


def match_lower_rms(base, main, wrist):
    assert base.shape == main.shape == wrist.shape
    delta = main - base
    target = rms(wrist - base)
    assert 0 < target < rms(delta) and np.abs(delta).max() <= 64

    def render(scale):
        return np.rint(base + scale * delta).astype(np.uint8)

    lower, upper = 0., 1.
    for _ in range(40):
        middle = (lower + upper) / 2
        if rms(render(middle) - base) < target:
            lower = middle
        else:
            upper = middle
    scale = min((lower, upper), key=lambda value: abs(rms(render(value) - base) - target))
    image = render(scale)
    measured = rms(image - base)
    assert abs(measured - target) < .025
    assert np.abs(image - base).max() <= np.abs(delta).max()
    return image, dict(scale=scale, full_main_rms=rms(delta), target_wrist_rms=target,
                       scaled_main_rms=measured, rms_error=abs(measured - target),
                       full_main_linf=float(np.abs(delta).max()),
                       scaled_main_linf=float(np.abs(image - base).max()),
                       wrist_linf=float(np.abs(wrist - base).max()))


def main(node):
    assert node == local_node()
    seeds = SEEDS[node]
    assert guard.read(PRIOR / 'accepted.json')['completed']
    previous = guard.read(PRIOR / 'evaluation_plan.json')
    assert tuple(previous['seeds']) == seeds and previous['node'] == node
    manifest = {}
    prior_manifest = guard.read(PRIOR / 'texture_manifest.json')
    for name in controls(seeds):
        item = prior_manifest[name]
        destination = ROOT / 'frozen' / (name + '.png')
        freeze(PRIOR / item['path'], destination, item['sha256'])
        manifest[name] = dict(item, path=str(destination.relative_to(ROOT)))

    base = rgb(ROOT / manifest['plain']['path'])
    amplitudes = []
    for seed in seeds:
        name = f's{seed}_main'
        main_image = rgb(ROOT / manifest[name]['path'])
        wrist_image = rgb(ROOT / manifest[f's{seed}_wrist']['path'])
        image, record = match_lower_rms(base, main_image, wrist_image)
        low = f's{seed}_main_low'
        destination = ROOT / 'frozen' / (low + '.png')
        if destination.is_file():
            assert np.array_equal(rgb(destination), image)
        else:
            Image.fromarray(image).save(destination)
        manifest[low] = dict(path=str(destination.relative_to(ROOT)), sha256=sha(destination),
                             seed=seed, reference='main_rms_matched_to_wrist')
        amplitudes.append(dict(record, seed=seed, texture_sha256=manifest[low]['sha256']))
        for suffix, view in [('ap', 'main'), ('pa', 'wrist')]:
            manifest[f's{seed}_{suffix}'] = dict(manifest[name], camera_mode=view)

    source_protocol = PRIOR / 'evaluation_protocol.json'
    protocol = guard.read(source_protocol)
    evaluation = protocol['evaluation']
    assert evaluation['suites'] == ['spatial', 'object']
    assert evaluation['tasks'] == list(TASKS) and evaluation['initials'] == list(INITIALS)
    canonical = PRIOR / evaluation['calibration_reference']
    freeze(canonical, ROOT / evaluation['calibration_reference'], sha(canonical))
    evaluation['camera_diagnostic'] = dict(prior=str(PRIOR), seeds=list(seeds),
        controls=controls(seeds), scope='Object only, paired states and query-index noise from the previous study')
    guard.write(PROTOCOL, protocol)
    guard.write(ROOT / 'texture_manifest.json', manifest)
    guard.write(ROOT / 'texture_amplitudes.json', amplitudes)
    guard.write(ROOT / 'prepared.json', dict(passed=True, node=node, seeds=seeds,
        previous_acceptance_sha256=sha(PRIOR / 'accepted.json'),
        previous_protocol_sha256=sha(source_protocol), protocol_sha256=sha(PROTOCOL)))
    print('FROZEN_INPUTS_READY', node, amplitudes, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=SEEDS, required=True)
    main(parser.parse_args().node)
