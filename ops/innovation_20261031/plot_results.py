"""Standalone research figures; no policy or GPU use."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    d=json.loads((folder/'diagnostics.json').read_text())
    local=json.loads((folder/'local_transport.json').read_text())
    action=json.loads((folder/'action_audit.json').read_text())
    gradient=json.loads((folder/'gradient_probe.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    groups=['new_texture','new_state','both_new'];x=np.arange(3)
    ax.bar(x-.17,[d['crossed'][g]['projection']['64']['q_err'] for g in groups],.34,label='Residual energy retained',color='#dc684b')
    ax.bar(x+.17,[d['crossed'][g]['projection']['64']['q_signal'] for g in groups],.34,label='Native signal retained',color='#3588a3')
    ax.set_xticks(x,['New texture\nseen early states','Late state\nseen texture','Both held out'])
    ax.set_ylim(0,1.08);ax.set_ylabel('Energy ratio');ax.set_title('A  State shift dominates this crossed diagnostic')
    ax.legend(fontsize=8,loc='lower right');ax.grid(axis='y',alpha=.15)
    ax=axes[0,1]
    vals=[1,local['splits']['validation']['metrics']['relative_error_vs_rendered'],
          action['splits']['validation']['metrics']['relative_action_mse'],
          action['splits']['validation']['metrics']['executed_translation']]
    bars=ax.bar(np.arange(4),vals,color=['#a8afb3','#3588a3','#247e60','#b9984b'])
    ax.set_xticks(np.arange(4),['Identity','Feature\ntransport','All action\ncoordinates','Executed\ntranslation'])
    ax.set_ylim(0,1.14);ax.set_ylabel('Prediction error / paired baseline')
    ax.set_title('B  Value fidelity improves with state-specific probes')
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.025,'%.3f'%v,ha='center')
    ax.text(.02,.02,'16 validation states; 2 native source probes per state',transform=ax.transAxes,fontsize=8)
    ax.grid(axis='y',alpha=.15)
    ax=axes[1,0]
    starts=['init','moderate','frozen_mr'];ratios=[]
    for start in starts:
        rows=[r for r in gradient['rows'] if r['start']==start]
        sums={method:sum(r['native_gain'] for r in rows if r['method']==method) for method in ['matched','transport']}
        ratios.append(sums['transport']/sums['matched'])
    bars=ax.bar(np.arange(3),ratios,color=['#ad6660','#3588a3','#ad6660'])
    ax.axhline(1,color='black',lw=1,label='Matched baseline')
    ax.axhline(1.1,color='#777',ls='--',lw=1,label='Predeclared gain gate')
    ax.set_xticks(np.arange(3),['Initialization','Moderate texture','Frozen trained MR'])
    ax.set_ylim(0,1.3);ax.set_ylabel('Native one-step gain / matched gain')
    ax.set_title('C  Better feature values do not ensure better updates')
    for b,v in zip(bars,ratios):ax.text(b.get_x()+b.get_width()/2,v+.025,'%.3f'%v,ha='center')
    ax.legend(fontsize=8,loc='lower right');ax.grid(axis='y',alpha=.15)
    ax=axes[1,1];ax.axis('off')
    text=('EXPLORATORY EVIDENCE\n\n'
          'Source: OpenVLA-OFT / LIBERO Spatial\n'
          'Value test: held-out probe textures at 16 states\n'
          'Update test: 4 fresh source initial states,\n'
          '3 texture starts, 2 step sizes\n\n'
          'Fixed zero-anchored transport:\n'
          '  11 / 24 paired update wins\n'
          '  Aggregate gain ratio 0.997\n'
          '  Native-update gate: FAIL\n\n'
          'These are not target success-rate results.\n'
          'Original paper results remain unchanged.')
    ax.text(.02,.98,text,va='top',fontsize=11,linespacing=1.4)
    fig.savefig(folder/'exploration_evidence.png',dpi=180)
    fig.savefig(folder/'exploration_evidence.pdf')
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);main(ap.parse_args().folder)
