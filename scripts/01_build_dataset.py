#!/usr/bin/env python
"""Fetch the challenge data and derive labels, harvested negatives and CV groups.

    python scripts/01_build_dataset.py [--local ../data]

Writes data/tdi_train.csv, data/tdi_harvested.csv, data/tdi_test.csv,
data/groups.npy. Nothing here is hand-curated; re-running it from the
HuggingFace source reproduces the files byte for byte.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import data as D

ap = argparse.ArgumentParser()
ap.add_argument('--local', default=None, help='directory with the challenge CSVs')
ap.add_argument('--out', default='data')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

raw = D.load_raw(a.local)
df, test = D.build(raw)
harv = D.harvested_negatives(df)

print(f'assembled {df.shape}   blinded test {test.shape}')
for iso in D.ISOS:
    rel, mine = df[f'{iso}_y_released'], df[f'{iso}_y']
    ok = rel.notna() & mine.notna()
    print(f'  {iso}: derived labels agree with the released column on '
          f'{(rel[ok] == mine[ok]).mean():.4f} of {int(ok.sum())} rows   '
          f'suspect={int(df[f"{iso}_suspect"].sum())}')
print()
for iso in D.TARGETS:
    p, y = D.pool_for(iso, df)
    ph, yh = D.pool_for(iso, df, harv)
    print(f'  {iso}: measured pool n={len(y)} prevalence={y.mean():.3f}   '
          f'+harvested n={len(yh)} prevalence={yh.mean():.3f}')

print('\nButina grouping for grouped CV (this is the slow step)...', flush=True)
S = D.tanimoto_matrix(df.SMILES.tolist())
g = D.butina_groups(S, 0.45)
print(f'  {len(np.unique(g))} clusters over {len(g)} compounds')

df.to_csv(f'{a.out}/tdi_train.csv', index=False)
harv.to_csv(f'{a.out}/tdi_harvested.csv', index=False)
test.to_csv(f'{a.out}/tdi_test.csv', index=False)
np.save(f'{a.out}/groups.npy', g)
print(f'\nwrote {a.out}/tdi_train.csv, tdi_harvested.csv, tdi_test.csv, groups.npy')
