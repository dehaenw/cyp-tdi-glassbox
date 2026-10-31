"""Grouped, repeated cross-validation with an honestly-chosen threshold.

Three things here are not optional, and each was worth a measurable amount of
self-deception when omitted:

  GROUPED folds (Butina clusters). The library is analogue series. A random
  split puts near-identical compounds on both sides of the fold and reports a
  score the leaderboard will not honour.

  MANY SEEDS. One 5-fold split of ~2,500 compounds has an MCC standard error
  around 0.013. Differences smaller than that are not findings. Every number
  in the README is a mean over 12 repeats with its spread quoted.

  LEAVE-ONE-SEED-OUT thresholding. The operating point is chosen on 11 repeats
  and scored on the 12th, so the threshold never sees the data it is scored on.

SEED HYGIENE. Seeds 0-11 tuned the configuration, 24-35 verified it, 36-47 are
untouched. The verified block is the one quoted, because a number measured on
the seeds that chose the config is optimistic by an unknown amount (here it was
0.008 on CYP3A4).
"""
import numpy as np

from . import data as D
from . import select as S
from .model import GlassBox
from .select import rate_one_se          # re-exported: the operating-point rule

RATES = np.linspace(0.02, 0.80, 79)
TUNING_SEEDS, VERIFY_SEEDS, UNTOUCHED_SEEDS = range(12), range(24, 36), range(36, 48)


def oof_one_seed(fs, rows, y, groups, seed, cfg, k=5):
    fo = D.folds(groups, seed)[rows]
    o = np.zeros(len(y))
    for f in range(k):
        va, tr = np.where(fo == f)[0], np.where(fo != f)[0]
        m = GlassBox(cfg['isoform'], cfg['morgan'], cfg['casper'], cfg['mordred'],
                     C=cfg['C'], rho=cfg['rho']).fit(fs, rows[tr], y[tr])
        o[va] = m.predict_proba(fs, rows[va])
    return o


def crossval(fs, rows, y, groups, cfg, seeds, n_jobs=1):
    """-> (per-seed LOSO MCC, MCC-vs-rate curve, out-of-fold probabilities).

    Everything inside the fold: the morgan selection, the Mann-Whitney ranking,
    the winsorisation bounds, the medians, the scaler and the fit.
    """
    seeds = list(seeds)
    args = [(fs, rows, y, groups, s, cfg) for s in seeds]
    if n_jobs > 1:
        from multiprocessing import get_context
        with get_context('fork').Pool(n_jobs) as p:
            oofs = p.starmap(oof_one_seed, args)
    else:
        oofs = [oof_one_seed(*a) for a in args]
    mcc, cur = S.loso_mcc(y, oofs, RATES)
    return mcc, cur, np.array(oofs)
