"""Feature selection: pick a handful of terms from tens of thousands, honestly.

Unfolding an ECFP leaves ~23,000 distinct radius-2 environments, ~3,000 with
support >= 10. A linear model cannot take them all, and L1 on a sparse
collinear basis picks one arbitrary member of every correlated group. So terms
are chosen by a STANDARDISED EFFECT SIZE and then decorrelated greedily.

Binary columns (substructure present/absent) use the Wald z of the log odds
ratio with a Haldane 0.5 correction on all four cells:

    z = log(ad/bc) / sqrt(1/a + 1/b + 1/c + 1/d)

The denominator is the point. A bit present in 3 molecules that are all
positive has a huge odds ratio and almost no evidence; dividing by its own
standard error puts it behind a bit at 90/400. Ranking on the odds ratio alone
fills the model with 3/3 accidents.

Continuous columns (CASPER 3D, Mordred) use the Mann-Whitney U z, the
non-parametric analogue: scale-free, so Mordred's arbitrary units do not
matter, and penalised by n in the same way.

Every step here runs INSIDE the training fold -- the medians, the
winsorisation bounds, the ranks, the selection itself.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import matthews_corrcoef

MIN_SUP, CHUNK = 10, 4096


def wald_z(P, y, cand):
    """Wald z of the log odds ratio per candidate column, in memory-lean chunks."""
    N, npos = len(y), float(y.sum())
    yf = y.astype(np.float32)
    z = np.empty(len(cand), dtype=np.float64)
    for i0 in range(0, len(cand), CHUNK):
        blk = P[:, cand[i0:i0 + CHUNK]].astype(np.float32)
        tp = (blk.T @ yf).astype(np.float64)
        npres = blk.sum(0, dtype=np.float64)
        a = tp + .5                            # present & positive
        b = npres - tp + .5                    # present & negative
        c = npos - tp + .5                     # absent  & positive
        d = N - npres - npos + tp + .5         # absent  & negative
        z[i0:i0 + CHUNK] = np.log(a * d / (b * c)) / np.sqrt(1/a + 1/b + 1/c + 1/d)
    return z


def select_binary(P, y, kmax, rho=0.7, min_sup=MIN_SUP):
    """-> (picked column indices in order, {column: z}).

    Two filters before scoring: support >= min_sup, and DEGENERACY COLLAPSE --
    columns with an identical presence vector are one feature, so only the
    first is kept. That merge is label-free, hence free of selection bias.

    The greedy decorrelation is NESTED in kmax: the order does not depend on
    kmax, so one pass serves a whole grid of model sizes.
    """
    cand = np.where(P.sum(0) >= min_sup)[0]
    seen, uniq = set(), []
    for j in cand:
        k = np.packbits(P[:, j]).tobytes()
        if k not in seen:
            seen.add(k); uniq.append(j)
    cand = np.array(uniq)
    z = wald_z(P, y, cand)
    order = cand[np.argsort(-np.abs(z), kind='stable')]
    n = P.shape[0]
    taken, T = [], np.empty((n, kmax), dtype=np.float32)
    for j in order:
        v = P[:, j].astype(np.float32)
        v -= v.mean(); s = v.std()
        if s < 1e-9:
            continue
        v /= s
        if taken and np.abs(T[:, :len(taken)].T @ v).max() / n > rho:
            continue
        T[:, len(taken)] = v; taken.append(j)
        if len(taken) == kmax:
            break
    return np.array(taken), dict(zip(cand, z))


def mw_z(R, y):
    """Mann-Whitney z per column from pre-computed ranks (ties averaged)."""
    N = len(y); n1 = float(y.sum()); n0 = N - n1
    R1 = R[y == 1].sum(0)
    return (R1 - n1 * (N + 1) / 2.0) / np.sqrt(n1 * n0 * (N + 1) / 12.0)


def select_continuous(X, y, kmax, rho=0.7, lo=1.0, hi=99.0):
    """-> (picked columns in order, (median, low, high) bounds, {column: z}).

    The bounds travel with the selection so held-out rows get the TRAINING
    fold's median imputation and winsorisation: Mordred has columns with
    1e6-scale outliers that would otherwise dominate a standardised linear fit.
    Decorrelation runs on ranks, i.e. Spearman, for the same robustness.
    """
    med = np.nanmedian(X, axis=0)
    Xi = np.where(np.isfinite(X), X, med)
    ql, qh = np.percentile(Xi, [lo, hi], axis=0)
    keep = np.where(Xi.std(0) > 1e-12)[0]
    R = np.apply_along_axis(rankdata, 0, Xi[:, keep])
    z = mw_z(R, y)
    order = keep[np.argsort(-np.abs(z), kind='stable')]
    pos = {j: i for i, j in enumerate(keep)}
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    n = len(y); taken, T = [], np.empty((n, kmax), dtype=np.float64)
    for j in order:
        v = Z[:, pos[j]]
        if taken and np.abs(T[:, :len(taken)].T @ v).max() / n > rho:
            continue
        T[:, len(taken)] = v; taken.append(j)
        if len(taken) == kmax:
            break
    return np.array(taken), (med, ql, qh), dict(zip(keep, z))


def apply_continuous(X, cols, bounds):
    """Impute with the TRAIN median, clip to the TRAIN winsorisation bounds."""
    med, ql, qh = bounds
    V = np.where(np.isfinite(X[:, cols]), X[:, cols], med[cols])
    return np.clip(V, ql[cols], qh[cols])


def loso_mcc(y, oofs, rates):
    """Leave-one-seed-out MCC: pick the operating point on 11 CV repeats, score
    it on the 12th. Picking the threshold on the same curve you report is the
    commonest way to overstate an MCC, and it is worth ~0.02 here.

    The threshold is carried as a POSITIVE RATE, not a probability cut: the
    probability scale shifts between folds and models, the rate does not.
    """
    cur = np.array([[matthews_corrcoef(y, (o >= np.quantile(o, 1 - r)).astype(int))
                     for r in rates] for o in oofs])
    out = []
    for i in range(len(oofs)):
        others = np.delete(np.arange(len(oofs)), i)
        r = rates[int(np.argmax(cur[others].mean(0)))]
        out.append(matthews_corrcoef(y, (oofs[i] >= np.quantile(oofs[i], 1 - r)).astype(int)))
    return np.array(out), cur


def rate_one_se(cur, rates):
    """Smallest rate whose mean MCC is within 1 SE of the peak. The MCC-vs-rate
    curve is flat over ~8 grid steps, so the argmax is a high-variance choice;
    this costs ~0.002-0.004 CV MCC for a less extrapolated operating point."""
    m, se = cur.mean(0), cur.std(0) / np.sqrt(len(cur))
    j = int(np.argmax(m))
    return float(rates[int(np.where(m >= m[j] - se[j])[0].min())]), float(rates[j])
