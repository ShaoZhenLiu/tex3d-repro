"""Scientific static export; bars are state medians over three explicit noises."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    view = json.loads((folder/'multiview_summary.json').read_text())
    lang = json.loads((folder/'instruction_summary.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    keys = [str(r['task'])+'/'+str(r['initial']) for r in view['by_state']]
    xs = np.arange(6)
    colors = {'MR': '#176a83', 'random101': '#c38331', 'random102': '#6d7780'}
    for j, texture in enumerate(colors):
        x = xs+(j-1)*.22
        vals = [r['textures'][texture]['median_amplification'] for r in view['by_state']]
        axes[0].bar(x, vals, width=.21, color=colors[texture], label=texture)
        for pos, record in zip(x, view['by_state']):
            points = record['textures'][texture]['amplification_per_noise']
            axes[0].scatter([pos]*len(points), points, s=11, color='#202b32', alpha=.65, zorder=3)
        vals = [r['textures'][texture]['median_language_texture_interaction_rms'] for r in lang['by_state']]
        axes[1].bar(x, vals, width=.21, color=colors[texture], label=texture)
    axes[0].axhline(1.1, color='#a44b43', linestyle='--', linewidth=1)
    axes[0].set(title='Cross-view amplification is state dependent', ylabel='Joint / (main + wrist) translation norm')
    axes[1].set(title='Same-scene language-texture interaction', ylabel='Interaction RMS: first 5 translation actions')
    for ax in axes:
        ax.set_xticks(xs, keys)
        ax.set_xlabel('Frozen scene: original task / initial index')
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False)
    fig.text(.5, -.01, '6 exploratory target scenes; 3 shared-noise repeats. Action diagnostics, not closed-loop success.', ha='center', fontsize=9)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(folder/('intervention_evidence.'+ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', type=Path)
    main(ap.parse_args().folder)
