"""Create a checksummed, self-contained Object result bundle."""
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    assert json.loads((root/'final_integrity_checks.json').read_text(encoding='utf-8'))['passed']
    assert (root/'Object完整测试报告.html').exists()
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'
                 and '__pycache__' not in p.parts and p.suffix!='.tmp')
    manifest=dict(files=len(files),total_bytes=sum(p.stat().st_size for p in files),
                  sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (root/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    out=root.parent/(root.name+'.zip')
    with zipfile.ZipFile(str(out),'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in files+[root/'artifact_manifest.json']:
            z.write(str(p),arcname=(Path(root.name)/p.relative_to(root)).as_posix())
    receipt=dict(file=str(out),files=len(files)+1,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (root.parent/(root.name+'_archive_receipt.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
