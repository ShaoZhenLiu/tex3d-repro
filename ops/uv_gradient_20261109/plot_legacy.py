"""Visualize actual historical endpoints; no synthetic or pending training values."""
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from common import ROOT,read

font=ROOT/'prepared/times.ttf'
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']='Times New Roman'
plt.rcParams['svg.fonttype']='none'
figure,axes=plt.subplots(1,2,figsize=(7.5,3.1),layout='constrained')
colors=['#87949e','#2277ae','#d97835']
for axis,policy,title in zip(axes,['identity','augmented'],['(a) Identity probe','(b) Common augmented probe']):
    for index,state in enumerate(['init','on','off']):
        record=read(ROOT/'legacy'/(state+'.json'))
        assert record and record['historical']
        values=[row['cosine'] for row in record['policies'][policy]['pooled_pair_rows'] if row['objective']=='feature']
        xs=[index+offset for offset in [-.10,-.06,-.02,.02,.06,.10]]
        axis.scatter(xs,values,s=25,color=colors[index],alpha=.8)
        mean=statistics.mean(values)
        axis.plot([index-.19,index+.19],[mean]*2,color=colors[index],linewidth=2.8)
        axis.text(index,.78,'%.3f'%mean,ha='center',va='center',fontsize=11,color=colors[index])
    axis.axhline(0,color='.6',linewidth=.7,linestyle='--')
    axis.set(xticks=[0,1,2],xticklabels=['Initial','UV-On','UV-Off'],ylim=(-.5,.85),title=title)
    axis.set_ylabel('Pairwise task-gradient cosine')
    axis.spines[['top','right']].set_visible(False)
figure.savefig(ROOT/'legacy/legacy_gradient_control.png',dpi=200)
figure.savefig(ROOT/'legacy/legacy_gradient_control.svg')
print('HISTORICAL_PLOT_SAVED',flush=True)
