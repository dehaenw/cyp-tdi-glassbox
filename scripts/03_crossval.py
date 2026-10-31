#!/usr/bin/env python
"""Reproduce the headline cross-validation numbers.

    python scripts/03_crossval.py                      # verified seeds 24-35
    python scripts/03_crossval.py --seeds 0 12         # the tuning block
    python scripts/03_crossval.py --baseline           # also morgan-only

Expected (verified block, seeds 24-35):
    CYP3A4  0.3253     CYP2D6  0.1907     macro  0.2580
and the matched morgan-only baseline on those same seeds:
    CYP3A4  0.3009     CYP2D6  0.1864     macro  0.2436
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import FeatureSet, data as D, features as F, validate as V

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', nargs=2, type=int, default=[24, 36])
ap.add_argument('--baseline', action='store_true', help='also run morgan-only')
ap.add_argument('--jobs', type=int, default=6)
ap.add_argument('--data', default='data'); ap.add_argument('--feat', default='features')
a = ap.parse_args()
seeds = range(*a.seeds)

CFG = json.load(open('models/config.json'))
tr = pd.read_csv(f'{a.data}/tdi_train.csv')
harv = pd.read_csv(f'{a.data}/tdi_harvested.csv')
groups = np.load(f'{a.data}/groups.npy')
X, bits, prov = F.build_morgan(tr.SMILES.tolist(), cache=f'{a.feat}/morgan_train.pkl')
fs = FeatureSet(tr.SMILES.tolist(), X, bits, prov,
                np.load(f'{a.feat}/standard_train.npy'),
                np.load(f'{a.feat}/casper_train.npy'),
                json.load(open(f'{a.feat}/casper_names.json')),
                np.load(f'{a.feat}/mordred_full_train.npy'),
                json.load(open(f'{a.feat}/mordred_full_names.json')))

out = {}
for iso in ['CYP3A4', 'CYP2D6']:
    c = dict(CFG[iso], isoform=iso)
    rows, y = D.pool_for(iso, tr, harv if c['harvest'] else None)
    runs = [('glass box', c)]
    if a.baseline:
        runs.append(('morgan only', dict(c, casper=0, mordred=0)))
    print('=' * 92)
    print(f'{iso}  n={len(y)}  prevalence={y.mean():.3f}  seeds {seeds.start}-{seeds.stop-1}',
          flush=True)
    for label, cc in runs:
        t = time.time()
        mcc, cur, _ = V.crossval(fs, rows, y, groups, cc, seeds, n_jobs=a.jobs)
        nt = cc['morgan'] + cc['casper'] + cc['mordred'] + 40
        one_se, argmax = V.rate_one_se(cur, V.RATES)
        print(f'  {label:12s} {nt:3d} terms  MCC(LOSO) = {mcc.mean():.4f} +/- {mcc.std():.4f}'
              f'   rate: argmax {argmax:.3f} / 1-SE {one_se:.3f}   [{time.time()-t:.0f}s]',
              flush=True)
        out[f'{iso}|{label}'] = {'mcc': round(float(mcc.mean()), 4),
                                 'sd': round(float(mcc.std()), 4), 'n_terms': nt,
                                 'rate_argmax': argmax, 'rate_1se': one_se}
macro = np.mean([out[f'{i}|glass box']['mcc'] for i in ['CYP3A4', 'CYP2D6']])
print('=' * 92)
print(f'macro MCC = {macro:.4f}')
if a.baseline:
    mb = np.mean([out[f'{i}|morgan only']['mcc'] for i in ['CYP3A4', 'CYP2D6']])
    print(f'macro morgan-only = {mb:.4f}   ->  3D + Mordred are worth {macro-mb:+.4f}')
out['macro'] = round(float(macro), 4)
os.makedirs('results', exist_ok=True)
json.dump(out, open(f'results/crossval_seeds{a.seeds[0]}_{a.seeds[1]-1}.json', 'w'), indent=1)
