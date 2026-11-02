"""Fixed task-0/init-10 placement panel from saved native measurements."""
from pathlib import Path
import numpy as np
from PIL import Image


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (root/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    modes=[('legacy_ray','main'),('production_floor','main'),('legacy_ray','wrist'),('production_floor','wrist')]
    for i,(label,view) in enumerate(modes):
        folder=root/'raw/placement/t0_i10'/label
        images={k:np.array(Image.open(str(folder/(k+'_'+view+'.png')))).astype(float)
                for k in ['plain','hidden','marker_a','refcloth']}
        data=[images['plain'].astype(np.uint8),images['marker_a'].astype(np.uint8),
              np.max(np.abs(images['plain']-images['hidden']),axis=-1),
              np.max(np.abs(images['refcloth']-images['plain']),axis=-1)]
        for j,image in enumerate(data):
            ax=axes[i,j]
            if j<2:ax.imshow(image)
            else:ax.imshow(image,cmap='magma',vmin=0,vmax=100)
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(['Plain cloth','Magenta marker','|Plain - hidden|','|RefCloth - Plain|'][j])
            if j==0:ax.set_ylabel(('Old placement' if label=='legacy_ray' else 'Correct placement')+'\n'+view,fontsize=11)
    fig.suptitle('Same Object initial state, different cloth placement: task 0 / init 10\nNo policy actions; native 256 x 256; difference heatmaps share RGB scale [0,100]')
    fig.tight_layout();fig.savefig(str(root/'figures/geometry_initial_state_comparison.png'),dpi=150);plt.close(fig)


if __name__=='__main__':main(Path(__file__).resolve().parents[1])
