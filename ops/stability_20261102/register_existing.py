"""Freeze the six existing textures and enumerate all paired evaluation cases."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main(a):
    stock=a.repo/'experiments/pi0_cloth/oft_cloth'
    files={
        'plain':a.repo.parent/'tex3d-deps/libero-pi/libero/libero/assets/tablecloth/cloth_base.png',
        's7_native':stock/'train_20261021_153634_tvfix-bs8/Universal_Cloth_OV.png',
        's7_mr':stock/'train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png',
        's8_native':stock/'train_20261027_175626_e05_s8_native/Universal_Cloth_OV.png',
        's8_mr':stock/'train_20261028_115945_e05_s8_mr/Universal_Cloth_OV.png',
        's9_native':stock/'train_20261027_145929_e05_s9_native/Universal_Cloth_OV.png',
        's9_mr':stock/'train_20261028_115356_e05_s9_mr/Universal_Cloth_OV.png'}
    # Repo is /data/.../code/tex3d; dependencies are its sibling tex3d-deps.
    plain=files['plain']
    if not plain.exists():
        files['plain']=a.repo.parent/'tex3d-deps/libero/libero/libero/assets/tablecloth/cloth_base.png'
    assert not (a.root/'texture_manifest.json').exists()
    frozen=a.root/'frozen';frozen.mkdir(exist_ok=True)
    manifest={}
    base=np.array([204,199,194])
    for name,source in files.items():
        assert source.exists(),source
        dest=frozen/(name+'.png');shutil.copy2(source,dest)
        image=np.array(Image.open(dest).convert('RGB'));assert image.shape==(256,256,3)
        delta=image.astype(float)-base
        assert abs(delta).max()<=64
        item=dict(path=str(dest.relative_to(a.root)),source=str(source),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            rms=float(np.sqrt(np.mean(delta**2))),linf=float(abs(delta).max()),frozen_existing=True)
        if name!='plain':
            seed,reference=name.split('_');item.update(seed=int(seed[1:]),reference=reference)
        manifest[name]=item
    # This exact pixel identity ties the seed-7 pair to the previously delivered paper.
    assert manifest['s7_native']['pixel_sha256']=='d20acd0f6423b7703916240abc7075ca9d7927499aa7dc143c565089f898be94'
    assert manifest['s7_mr']['pixel_sha256']=='8a87319e2cea903c110a7bfdb4607b03bfe28d095b93334d986d54b6f6d30684'
    (a.root/'texture_manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[]
    for suite in ['spatial','object']:
        for task in range(10):
            jobs.append(dict(id='%s_t%d_plain'%(suite,task),suite=suite,task=task,conditions=['plain']))
            for seed in [7,8,9,10,11,12]:
                jobs.append(dict(id='%s_t%d_s%d'%(suite,task,seed),suite=suite,task=task,
                                 conditions=['s%d_native'%seed,'s%d_mr'%seed]))
    assert sum(10*len(job['conditions']) for job in jobs)==2600
    (a.root/'evaluation').mkdir(exist_ok=True)
    (a.root/'evaluation/jobs.json').write_text(json.dumps(jobs,indent=2))
    print('Frozen seven existing PNGs; registered 140 bundles / 2600 episodes.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
