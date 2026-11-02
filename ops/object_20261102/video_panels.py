"""Fixed-time contact sheets of all eight predeclared two-camera videos."""
import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=root/'raw'
    if not raw.exists():raw=root
    out=root/'video_frames';out.mkdir(exist_ok=True)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    metadata=[]
    for target in ['pi0','pi05']:
        fig,axes=plt.subplots(4,3,figsize=(12,8))
        for i,(task,condition) in enumerate([(0,'plain'),(0,'refcloth'),(4,'plain'),(4,'refcloth')]):
            key='%s_t%d_i10_%s'%(target,task,condition)
            folder=raw/(target+'_main')
            video=folder/'videos'/(key+'.mp4')
            record=json.loads((folder/(key+'.json')).read_text())
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(video)],text=True))
            duration=float(info['format']['duration'])
            stream=next(x for x in info['streams'] if x['codec_type']=='video')
            fps=float(Fraction(stream['avg_frame_rate']))
            last_frame_time=(int(stream['nb_frames'])-1)/fps
            row=dict(key=key,robot_success=record['robot_success'],duration=duration,last_frame_time=last_frame_time,frames=[])
            for j,fraction in enumerate([0.,.5,.97]):
                stamp=last_frame_time*fraction
                path=out/('%s_f%d.png'%(key,j))
                subprocess.run(['ffmpeg','-v','error','-y','-ss','%.6f'%stamp,'-i',str(video),'-frames:v','1','-threads','1',str(path)],check=True)
                assert path.exists()
                axes[i,j].imshow(np.array(Image.open(path)))
                axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                axes[i,j].set_title('%.2f s / %.2f s'%(stamp,duration),fontsize=9)
                if j==0:axes[i,j].set_ylabel('Task %d / %s\n%s'%(task,condition,'SUCCESS' if record['robot_success'] else 'FAILURE'),fontsize=10)
                row['frames'].append(dict(time_seconds=stamp,fraction=fraction,path=str(path.relative_to(root))))
            metadata.append(row)
        fig.suptitle(target+': all predeclared qualitative cases (init 10)\nEach frame: main camera (left), wrist camera (right); fixed 0%, 50%, 97% of available frame-time span',fontsize=12)
        fig.tight_layout();fig.savefig(str(figures/(target+'_video_contact_sheet.png')),dpi=160);plt.close(fig)
    (root/'video_frame_manifest.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)
