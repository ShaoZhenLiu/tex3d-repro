"""Scientific figures from the accepted same-state and closed-loop records."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import REPO,ROOT,read


def main():
    font=REPO/'experiments/refcloth_uv_gradient/20261109/prepared/times.ttf'
    assert font.exists()
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fresh=read(ROOT/'fresh_feature_direction/results.json')
    closed=read(ROOT/'analysis/closed_loop_summary.json')
    if fresh is None or closed is None:return
    rows=[r for r in fresh['rows'] if r['route']=='png']
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8))
    colors=['#b4472a','#2877a6']
    for ax,fields,title,ylabel in [(axes[0],['matched_cosine','native_cosine'],'(a) Displacement direction','Cosine with native displacement'),
                                  (axes[1],['relative_matched_error','relative_native_error'],'(b) Displacement residual','Error / native displacement norm')]:
        for index,row in enumerate(rows):
            jitter=(index%8-3.5)*.018
            ax.plot([jitter,1+jitter],[row[f] for f in fields],color='#b6b6b6',alpha=.38,lw=.6,zorder=1)
        for j,field in enumerate(fields):
            values=[r[field] for r in rows]
            xx=[j+(i%8-3.5)*.018 for i in range(len(rows))]
            ax.scatter(xx,values,s=20,color=colors[j],alpha=.7,edgecolors='none',zorder=2)
            mean=float(np.mean(values));ax.plot([j-.2,j+.2],[mean,mean],color=colors[j],lw=3,zorder=3)
            ax.annotate('%.3f'%mean,(j,mean),xytext=(0,8),textcoords='offset points',ha='center',fontweight='bold',color=colors[j])
        ax.set_xticks([0,1],['Matched','Native ref.']);ax.set_xlim(-.4,1.4)
        ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold',pad=16)
        ax.grid(axis='y',alpha=.15);ax.text(.5,-.24,'32 fixed source states',transform=ax.transAxes,ha='center')
    axes[0].set_ylim(0,1.06);axes[1].set_ylim(bottom=0)
    names=['plain','main_only','wrist_only','both'];labels=['Plain','Main only','Wrist only','Both']
    values=[closed['successes'][c] for c in names]
    ax=axes[2];ax.bar(range(4),values,color=['#8c969d','#648fb2','#8970ac','#b4472a'],width=.64)
    for j,v in enumerate(values):ax.text(j,v+1.5,str(v),ha='center',fontweight='bold')
    ax.set_xticks(range(4),labels,rotation=22,ha='right');ax.set_ylim(0,110);ax.set_yticks(range(0,101,20))
    ax.set_ylabel('Task success (%)');ax.set_title('(c) Target closed-loop intervention',loc='left',fontweight='bold',pad=16)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.text(.5,-.28,r'$\pi_{0.5}$-Object, 100 paired episodes / condition',transform=ax.transAxes,ha='center')
    fig.subplots_adjust(left=.06,right=.99,bottom=.28,top=.82,wspace=.45)
    for suffix in ['png','svg']:fig.savefig(ROOT/'analysis'/('mechanism_results.'+suffix),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
