"""Separate marker exposure, plain-hidden contrast and actual texture changes."""
import numpy as np


def measure_three(images,thresholds=(1,5,10,20)):
    a={k:np.asarray(v,dtype=np.float64) for k,v in images.items()}
    assert len({x.shape for x in a.values()})==1
    assert all(x.ndim==3 and x.shape[-1]==3 and np.isfinite(x).all() for x in a.values())
    diffs=dict(marker=np.abs(a['marker_a']-a['marker_b']),
               plain_hidden=np.abs(a['plain']-a['hidden']),
               ref_plain=np.abs(a['refcloth']-a['plain']))
    result=dict(shape=list(a['plain'].shape),thresholds={},
        rms_rgb255={k:float(np.sqrt(np.mean(d*d))) for k,d in diffs.items()},
        maxabs_rgb255={k:float(d.max()) for k,d in diffs.items()})
    for t in thresholds:
        masks={k:d.max(axis=-1)>float(t)+1e-4 for k,d in diffs.items()}
        g,p,r=masks['marker'],masks['plain_hidden'],masks['ref_plain']
        result['thresholds'][str(t)]=dict(marker_fraction=float(g.mean()),plain_hidden_fraction=float(p.mean()),
            ref_plain_fraction=float(r.mean()),
            hidden_missed_marker_fraction=(float((g&~p).sum()/g.sum()) if g.any() else None),
            ref_change_within_marker_fraction=(float((g&r).sum()/g.sum()) if g.any() else None),
            hidden_outside_marker_fraction=float((p&~g).mean()),
            marker_hidden_iou=(float((g&p).sum()/(g|p).sum()) if (g|p).any() else None))
    return result
