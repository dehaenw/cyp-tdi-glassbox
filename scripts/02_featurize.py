#!/usr/bin/env python
"""Build the feature blocks.

    python scripts/02_featurize.py                # morgan + standard  (~1 min)
    python scripts/02_featurize.py --mordred      # + Mordred 2D      (~30 min)
    python scripts/02_featurize.py --casper       # + CASPER 3D       (~55 min)

The CASPER matrices are SHIPPED in features/ because the CASPER package is not
on PyPI yet; --casper is only for recomputing them. Mordred IS pip-installable,
so its full matrix is a build step -- but the columns the deployed models
actually use are shipped too (features/mordred_slim_*), which is all that
`explain()` and `predict` need.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import features as F

ap = argparse.ArgumentParser()
ap.add_argument('--mordred', action='store_true')
ap.add_argument('--casper', action='store_true')
ap.add_argument('--data', default='data'); ap.add_argument('--out', default='features')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tr = pd.read_csv(f'{a.data}/tdi_train.csv'); te = pd.read_csv(f'{a.data}/tdi_test.csv')

print('unfolded Morgan (radius 2, support >= 10)...', flush=True)
X, bits, prov = F.build_morgan(tr.SMILES.tolist(), cache=f'{a.out}/morgan_train.pkl')
print(f'  {X.shape[1]} distinct environments kept out of '
      f'{len(prov)} seen; density {(X > 0).mean():.3%}')
np.save(f'{a.out}/standard_train.npy', F.standard_block(tr.SMILES.tolist()))
np.save(f'{a.out}/standard_test.npy', F.standard_block(te.SMILES.tolist()))
json.dump(F.STANDARD_NAMES, open(f'{a.out}/standard_names.json', 'w'))
print(f'  standard block: {len(F.STANDARD_NAMES)} columns '
      f'(10 physchem + 30 a-priori MBI motifs)')

if a.mordred:
    for tag, d in [('train', tr), ('test', te)]:
        print(f'Mordred 2D, {tag}...', flush=True)
        M, names = F.compute_mordred(d.SMILES.tolist())
        np.save(f'{a.out}/mordred_full_{tag}.npy', M)
        json.dump(names, open(f'{a.out}/mordred_full_names.json', 'w'))
        print(f'  {M.shape}')
if a.casper:
    for tag, d in [('train', tr), ('test', te)]:
        print(f'CASPER 3D, {tag} (slow)...', flush=True)
        C, names = F.compute_casper(d.SMILES.tolist())
        np.save(f'{a.out}/casper_{tag}.npy', C)
        json.dump(names, open(f'{a.out}/casper_names.json', 'w'))
        print(f'  {C.shape}  failed={int((~np.isfinite(C)).all(1).sum())}')
print('done')
