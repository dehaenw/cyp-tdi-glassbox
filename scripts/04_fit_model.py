#!/usr/bin/env python
"""Fit the deployed models on the full pools and save them as readable JSON.

    python scripts/04_fit_model.py [--rule 1se|argmax]

Writes models/CYP3A4.json, models/CYP2D6.json -- the complete model, one entry
per term, and models/mordred_slim_*.npy, the only Mordred columns the fitted
models use (so explaining and predicting never needs the 32 MB full matrix).

TWO OPERATING POINTS, and the difference is not cosmetic. Cross-validation
chose a POSITIVE RATE, because the probability scale drifts between folds while
the rate does not, and the challenge submission applied that rate as a quantile
of the 750 test predictions. That is a population-level call and it cannot be
made for one molecule in isolation. So the model also stores `prob_cut`, the
out-of-fold probability at that same rate, which IS a single-molecule
threshold. They answer different questions and the saved model carries both.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import FeatureSet, GlassBox, data as D, features as F, validate as V

ap = argparse.ArgumentParser()
ap.add_argument('--rule', default='1se', choices=['1se', 'argmax'])
ap.add_argument('--jobs', type=int, default=6)
ap.add_argument('--data', default='data'); ap.add_argument('--feat', default='features')
a = ap.parse_args()

CFG = json.load(open('models/config.json'))
tr = pd.read_csv(f'{a.data}/tdi_train.csv')
harv = pd.read_csv(f'{a.data}/tdi_harvested.csv')
groups = np.load(f'{a.data}/groups.npy')
X, bits, prov = F.build_morgan(tr.SMILES.tolist(), cache=f'{a.feat}/morgan_train.pkl')
MOR = np.load(f'{a.feat}/mordred_full_train.npy')
MORN = json.load(open(f'{a.feat}/mordred_full_names.json'))
fs = FeatureSet(tr.SMILES.tolist(), X, bits, prov,
                np.load(f'{a.feat}/standard_train.npy'),
                np.load(f'{a.feat}/casper_train.npy'),
                json.load(open(f'{a.feat}/casper_names.json')), MOR, MORN)

used = set()
for iso in ['CYP3A4', 'CYP2D6']:
    c = dict(CFG[iso], isoform=iso)
    rows, y = D.pool_for(iso, tr, harv if c['harvest'] else None)
    # The OPERATING POINT is chosen on the tuning seeds, exactly as the
    # challenge submission did, so this reconstructs the deployed model. The
    # PERFORMANCE is reported from the verification seeds, which no grid and no
    # threshold choice ever touched. Mixing those two up is how a threshold
    # chosen on the reported curve inflates an MCC by ~0.02.
    _, cur_t, oofs_t = V.crossval(fs, rows, y, groups, c, V.TUNING_SEEDS, n_jobs=a.jobs)
    one_se, argmax = V.rate_one_se(cur_t, V.RATES)
    rate = one_se if a.rule == '1se' else argmax
    cut = float(np.mean([np.quantile(o, 1 - rate) for o in oofs_t]))
    mcc, _, _ = V.crossval(fs, rows, y, groups, c, V.VERIFY_SEEDS, n_jobs=a.jobs)

    m = GlassBox(iso, c['morgan'], c['casper'], c['mordred'], c['C'], c['rho']).fit(fs, rows, y)
    m.operating_point = {'rule': a.rule, 'positive_rate': round(float(rate), 4),
                         'prob_cut': round(cut, 4),
                         'cv_mcc_loso': round(float(mcc.mean()), 4),
                         'cv_mcc_loso_sd': round(float(mcc.std()), 4),
                         'rate_seeds': f'{V.TUNING_SEEDS.start}-{V.TUNING_SEEDS.stop - 1}',
                         'cv_seeds': f'{V.VERIFY_SEEDS.start}-{V.VERIFY_SEEDS.stop - 1}'}
    m.save(f'models/{iso}.json')
    used.update(t['name'] for t in m.terms if t['block'] == 'mordred')
    byblock = pd.Series([t['block'] for t in m.terms]).value_counts().to_dict()
    print(f'{iso}: {len(m.terms)} terms {byblock}  n_train={m.n_train}  '
          f'MCC={mcc.mean():.4f}+/-{mcc.std():.4f}  rate={rate:.3f} ({a.rule})  '
          f'prob_cut={cut:.3f}  -> models/{iso}.json', flush=True)

# slim Mordred: only the columns the two fitted models reference
keep = [i for i, n in enumerate(MORN) if n in used]
for tag in ['train', 'test']:
    np.save(f'{a.feat}/mordred_slim_{tag}.npy',
            np.load(f'{a.feat}/mordred_full_{tag}.npy')[:, keep])
json.dump([MORN[i] for i in keep], open(f'{a.feat}/mordred_slim_names.json', 'w'))
print(f'wrote mordred_slim_*.npy with {len(keep)} of {len(MORN)} columns '
      f'({os.path.getsize(f"{a.feat}/mordred_slim_train.npy")/1e3:.0f} kB vs '
      f'{os.path.getsize(f"{a.feat}/mordred_full_train.npy")/1e6:.0f} MB)')
