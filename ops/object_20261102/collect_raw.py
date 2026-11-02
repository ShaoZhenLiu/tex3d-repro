"""Collect only completed, bounded Object artifacts; exclude models and caches."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def main(root,out):
    assert (root/'controller_done.json').exists()
    names=['pi0_main','pi05_main','pi0_tail','pi05_tail','smoke_pi0','smoke_pi05','placement','diagnostics','evidence','video_frames']
    cases=[]
    for name in names[:4]:
        x=json.loads((root/name/'results.json').read_text());assert x['completed']
        cases.extend(json.loads(p.read_text()) for p in (root/name).glob('pi*_t*_i*_*.json'))
    assert len(cases)==120
    assert len({(x['target'],x['task'],x['initial'],x['condition']) for x in cases})==120
    scenes=json.loads((root/'evidence/scene_xml_snapshot_midrun.json').read_text())
    changed=[f for f,h in scenes.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    report=dict(completed=True,episodes=len(cases),scene_files_checked=len(scenes),scene_files_changed=changed,
        live_object_inference_processes=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
                                         if 'ops/object_20261102/run_object.py' in s])
    assert not report['live_object_inference_processes']
    (root/'evidence/collection_checks.json').write_text(json.dumps(report,indent=2))
    # out must be outside root; prevent recursively archiving the archive itself.
    assert root.resolve() not in out.resolve().parents
    sources=[]
    for name in names:
        folder=root/name
        if folder.exists():sources.extend(p for p in folder.rglob('*') if p.is_file())
    sources.extend(p for p in root.iterdir() if p.is_file() and (p.suffix=='.log' or p.name in
                   ['controller_done.json','controller_status.json','video_frame_manifest.json']))
    if (root/'figures').exists():sources.extend((root/'figures').glob('*video_contact_sheet.png'))
    assert not any(p.suffix in ['.pt','.pth','.ckpt','.safetensors'] for p in sources)
    sources=sorted(set(p for p in sources if p.suffix!='.tmp' and '__pycache__' not in p.parts))
    with tarfile.open(out,'w:gz',compresslevel=4) as tar:
        for path in sources:tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    receipt=dict(file=str(out),bytes=out.stat().st_size,sha256=digest,files=len(sources),episodes=120)
    out.with_suffix(out.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();main(a.root,a.out)
