"""Static scientific exports for the complete mathematical screening grid."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def main(root,out):
    d=json.loads((root/'math_summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.bbox':'tight'})
    out.mkdir(parents=True,exist_ok=True)
    methods=['mean_sign','pcgrad_sign','pcgrad_vector','maximin']
    colors=['#777f87','#c18b3d','#8c729c','#176f89']
    fig,axes=plt.subplots(2,3,figsize=(12.5,7))
    for row,norm in enumerate(['raw','l1']):
        for col,start in enumerate(['init','moderate','frozen_mr']):
            ax=axes[row,col]
            records=sorted([r for r in d['geometry'] if r['normalization']==norm and r['start']==start],key=lambda r:r['eta'])
            for method,color in zip(methods,colors):
                ax.plot([.01,.05],[r['methods'][method]['minimum'] for r in records],
                        marker='o',markersize=4,label=method,color=color)
            ax.axhline(0,color='#333',linewidth=.7)
            ax.set_title(norm+' / '+start)
            ax.set_xticks([.01,.05])
            ax.set_xlabel('Parameter step bound')
            ax.set_ylabel('Worst heldout native J gain'+(' / fit L1' if row else ''))
            ax.grid(alpha=.2)
    axes[0,1].legend(loc='upper left',fontsize=8,frameon=False)
    fig.suptitle('Source initial5 fit -> independent initial6: all starts and step bounds',fontsize=13)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('box_native_screen.'+ext),dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.7,4.2))
    ax=axes[0]
    ax.add_patch(Polygon([[-1,-1],[1,-1],[1,1],[-1,1]],closed=True,fill=False,color='#68727b'))
    ax.add_patch(Polygon([[0,0],[0,1],[.5,1]],closed=True,color='#d8e9df',alpha=.9))
    for name,(x,y),color in [('mean+sign',(-1,1),'#777f87'),('PCGrad+sign',(1,1),'#c18b3d'),
                             ('box max-min',(1/3,1),'#176f89')]:
        ax.annotate('',xy=(x,y),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=2))
        ax.text(x,y+.08,name,ha='center',fontsize=8,color=color)
    ax.set(xlim=(-1.25,1.3),ylim=(-1.2,1.4),xlabel='Update coordinate 1',ylabel='Update coordinate 2',
           title='Strict sign counterexample')
    ax.set_aspect('equal')
    ax.text(-1.15,-.95,'g1=(1,0), g2=(-2,1)\nLP gains=(1/3, 1/3)',fontsize=8)
    xs=np.arange(3)
    labels=['init','moderate','frozen MR']
    a=[d['fidelity'][s]['action_AA_relative_error'] for s in ['init','moderate','frozen_mr']]
    b=[d['fidelity'][s]['action_interaction_relative_error'] for s in ['init','moderate','frozen_mr']]
    axes[1].bar(xs-.18,a,.35,label='AA displacement',color='#8d9da6')
    axes[1].bar(xs+.18,b,.35,label='Four-corner interaction',color='#176f89')
    axes[1].set_xticks(xs,labels)
    axes[1].set(title='Interaction did not improve native fidelity',ylabel='Error / native-signal norm')
    axes[1].legend(frameon=False,fontsize=8)
    positive=np.array([5/64,0.])
    axes[2].bar([0,1],positive,color='#176f89',label='Positive robust score')
    axes[2].bar([0,1],1-positive,bottom=positive,color='#dce2e5',label='Zero robust score')
    axes[2].set_xticks([0,1],['Absolute ellipse','Relative ellipse'])
    axes[2].set(ylim=(0,1.15),ylabel='Fraction of 64 heldout texture-state pairs',title='Conditional lower bounds became vacuous')
    axes[2].text(0,1.025,'coverage 96.9%',ha='center',fontsize=8)
    axes[2].text(1,1.025,'coverage 92.2%',ha='center',fontsize=8)
    axes[2].legend(loc='center',frameon=False,fontsize=8)
    fig.text(.5,-.015,'Mathematical identities and source action diagnostics do not establish closed-loop attack success.',ha='center',fontsize=9)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('mathematical_diagnostics.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('out',type=Path)
    a=ap.parse_args();main(a.root,a.out)
