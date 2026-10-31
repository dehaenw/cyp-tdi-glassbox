#!/usr/bin/env python
"""Predict on the blinded test set, in the challenge's submission format.

    python scripts/05_predict.py

Writes predictions/submission.csv/.parquet and predictions/manifest.json.

The boolean call uses the CV-chosen POSITIVE RATE applied as a quantile of the
750 test predictions, which is what was submitted. `prob_cut` from the model
file is the single-molecule equivalent; the two disagree on a handful of
compounds whenever the test distribution differs from the training one, and the
manifest reports how many.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import FeatureSet, GlassBox, features as F

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data'); ap.add_argument('--feat', default='features')
ap.add_argument('--out', default='predictions')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

te = pd.read_csv(f'{a.data}/tdi_test.csv')
tr = pd.read_csv(f'{a.data}/tdi_train.csv')
_, bits, prov = F.build_morgan(tr.SMILES.tolist(), cache=f'{a.feat}/morgan_train.pkl')
fs = FeatureSet(te.SMILES.tolist(), F.project_morgan(te.SMILES.tolist(), bits), bits, prov,
                np.load(f'{a.feat}/standard_test.npy'),
                np.load(f'{a.feat}/casper_test.npy'),
                json.load(open(f'{a.feat}/casper_names.json')),
                np.load(f'{a.feat}/mordred_slim_test.npy'),
                json.load(open(f'{a.feat}/mordred_slim_names.json')))

sub = pd.DataFrame({'SMILES': te.SMILES, 'Molecule_Name': te.Molecule_Name})
man = {}
for iso in ['CYP2D6', 'CYP3A4']:
    m = GlassBox.load(f'models/{iso}.json')
    op = m.operating_point
    p = m.predict_proba(fs)
    by_rate = p >= np.quantile(p, 1 - op['positive_rate'])
    by_cut = p >= op['prob_cut']
    sub[f'{iso}_is_TDI'] = by_rate.astype(bool)
    man[iso] = dict(n_terms=len(m.terms), n_train=m.n_train, **op,
                    calls_by_rate=int(by_rate.sum()), calls_by_prob_cut=int(by_cut.sum()),
                    disagreements=int((by_rate != by_cut).sum()))
    print(f'{iso}: {len(m.terms)} terms  rate={op["positive_rate"]:.3f} -> '
          f'{int(by_rate.sum())}/{len(te)} calls   (absolute prob_cut would call '
          f'{int(by_cut.sum())}; {int((by_rate != by_cut).sum())} differ)', flush=True)

errs = []
if len(sub) != 750:
    errs.append(f'row count {len(sub)}')
for c in ['SMILES', 'Molecule_Name', 'CYP2D6_is_TDI', 'CYP3A4_is_TDI']:
    if c not in sub:
        errs.append(f'missing {c}')
    elif sub[c].isna().any():
        errs.append(f'NaN in {c}')
for c in ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']:
    if sub[c].dtype != bool:
        errs.append(f'{c} not bool')
if list(sub.Molecule_Name) != list(te.Molecule_Name):
    errs.append('row order changed')
man['validation'] = 'PASS' if not errs else 'FAIL: ' + '; '.join(errs)
print('VALIDATION:', man['validation'])
sub.to_csv(f'{a.out}/submission.csv', index=False)
sub.to_parquet(f'{a.out}/submission.parquet', index=False)
json.dump(man, open(f'{a.out}/manifest.json', 'w'), indent=2)
print(f'wrote {a.out}/submission.csv/.parquet')
