"""CPU-only exact comparison of all immutable input leaves, with bounded copies."""
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

import refcloth_resource as guard


def compare(a,b,counts):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
        counts['tensors']+=1;counts['bytes']+=a.numel()*a.element_size()
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b);counts['numpy_arrays']+=1
    elif isinstance(a,Mapping):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k],counts)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y,counts)
    else:
        assert a==b,(type(a),type(b))


def main():
    torch.set_num_threads(1)
    s=guard.snapshot();assert s['available']>=50*guard.GIB
    assert s['headroom'] is None or s['headroom']>=24*guard.GIB
    root=Path('/data/shaozhen.liu/code/tex3d');results=[]
    for p in [root/'experiments/pi0_cloth/oft_cloth/r0f_pool.pt',
              root/'experiments/pi0_cloth/oft_cloth/mr_plain_cache.pt',
              root/'experiments/refcloth_coupling/20261102/plain_image_cache.pt']:
        started=time.time()
        normal=torch.load(str(p),map_location='cpu',weights_only=False)
        mapped=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
        counts=dict(tensors=0,numpy_arrays=0,bytes=0)
        compare(normal,mapped,counts)
        results.append(dict(path=str(p),exact_all_leaves=True,counts=counts,seconds=time.time()-started))
        del normal,mapped;gc.collect()
        print('MMAP_EXACT',str(p),counts,flush=True)
    out=root/'experiments/refcloth_stability/20261102/mmap_validation.json'
    guard.write(out,dict(passed=True,host=os.uname().nodename,pytorch=torch.__version__,results=results))


if __name__=='__main__':main()
