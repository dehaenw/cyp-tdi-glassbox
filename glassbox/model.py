"""The model, and the reason this repo exists: `explain()` is not a surrogate.

    logit(TDI) = b0 + SUM w_i . z(log1p count of Morgan environment i)
                    + SUM c_l . z(physchem / a-priori MBI motif l)
                    + SUM v_j . z(CASPER 3D surface feature j)
                    + SUM u_k . z(Mordred 2D descriptor k)

One L2 logistic regression per isoform. No trees, no boosting, no embeddings,
no ensembling, no stacking. 152 terms for CYP3A4, 88 for CYP2D6.

WHY THAT MATTERS FOR INTERPRETABILITY. For a linear model the per-term
contribution `w_i . z_i` is not an approximation of what the model did -- it IS
what the model did. The explanation and the computation are the same
arithmetic, so there is no surrogate to be unfaithful, no sampling noise, no
background-distribution choice, and no additivity assumption to violate. Ask
SHAP to explain a gradient-boosted model and you get an estimate of a
counterfactual; ask this model and you get its working.

The saved artifact is readable JSON with one entry per term, carrying the
coefficient, the standardisation and the imputation bound. The whole model fits
on a few screens, which is the point.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import features as F
from . import select as S


class FeatureSet:
    """The four blocks for one set of molecules, with their names."""

    def __init__(self, smiles, morgan, morgan_bits, morgan_prov,
                 standard, casper=None, casper_names=None,
                 mordred=None, mordred_names=None):
        self.smiles = list(smiles)
        self.morgan, self.morgan_bits, self.morgan_prov = morgan, list(morgan_bits), morgan_prov
        self.standard = standard
        self.casper, self.casper_names = casper, list(casper_names or [])
        self.mordred, self.mordred_names = mordred, list(mordred_names or [])

    def block(self, name):
        return {'casper': (self.casper, self.casper_names),
                'mordred': (self.mordred, self.mordred_names)}[name]


class GlassBox:
    """Fit on a pool, predict anywhere, explain exactly."""

    def __init__(self, isoform, morgan_k=64, casper_k=32, mordred_k=16, C=0.1, rho=0.7):
        self.isoform, self.C, self.rho = isoform, C, rho
        self.quota = {'morgan': morgan_k, 'casper': casper_k, 'mordred': mordred_k}
        self.terms, self.intercept, self.operating_point = [], 0.0, {}

    # ------------------------------------------------------------------ fit --
    def fit(self, fs, rows, y):
        """`rows` indexes into `fs`; selection and scaling see only those rows."""
        cols, vals = [], []

        msel, _ = S.select_binary((fs.morgan[rows] > 0), y, self.quota['morgan'], self.rho)
        for j in msel:
            bit = fs.morgan_bits[j]
            frag, rad = F.describe_morgan(bit, fs.morgan_prov, fs.smiles)
            cols.append({'block': 'morgan', 'name': frag, 'bit': int(bit), 'radius': int(rad)})
        vals.append(np.log1p(fs.morgan[np.ix_(rows, msel)]))

        std_med = np.nanmedian(fs.standard[rows], axis=0)
        for n, m in zip(F.STANDARD_NAMES, std_med):
            cols.append({'block': 'standard', 'name': n, 'median': float(m)})
        vals.append(np.where(np.isfinite(fs.standard[rows]), fs.standard[rows], std_med))

        for b in ['casper', 'mordred']:
            X, names = fs.block(b)
            if X is None or not self.quota[b]:
                continue
            sel, bounds, _ = S.select_continuous(X[rows], y, self.quota[b], self.rho)
            med, ql, qh = bounds
            for j in sel:
                cols.append({'block': b, 'name': names[j], 'median': float(med[j]),
                             'lo': float(ql[j]), 'hi': float(qh[j])})
            vals.append(S.apply_continuous(X[rows], sel, bounds))

        A = np.hstack(vals)
        sc = StandardScaler().fit(A)
        lr = LogisticRegression(C=self.C, max_iter=4000).fit(sc.transform(A), y)
        for t, c, m, s in zip(cols, lr.coef_[0], sc.mean_, sc.scale_):
            t.update(coef=float(c), center=float(m), scale=float(s))
        self.terms, self.intercept = cols, float(lr.intercept_[0])
        self.n_train = int(len(y))
        return self

    # ------------------------------------------------------------ transform --
    def _values(self, fs, rows):
        """Raw per-term values (before standardisation), in term order."""
        out = np.empty((len(rows), len(self.terms)))
        want = {}
        for i, t in enumerate(self.terms):
            want.setdefault(t['block'], []).append(i)

        if 'morgan' in want:
            bits = [self.terms[i]['bit'] for i in want['morgan']]
            ix = {b: j for j, b in enumerate(fs.morgan_bits)}
            if all(b in ix for b in bits):
                M = fs.morgan[np.ix_(rows, [ix[b] for b in bits])]
            else:                                     # molecules outside fs
                M = F.to_matrix(F.sparse_counts([fs.smiles[r] for r in rows])[0], bits)
            out[:, want['morgan']] = np.log1p(M)

        if 'standard' in want:
            med = np.array([self.terms[i]['median'] for i in want['standard']])
            V = fs.standard[rows]
            out[:, want['standard']] = np.where(np.isfinite(V), V, med)

        for b in ['casper', 'mordred']:
            if b not in want:
                continue
            X, names = fs.block(b)
            if X is None:
                raise ValueError(
                    f"model needs the '{b}' block but the FeatureSet has none. "
                    f"Use the precomputed matrices in features/, or compute them "
                    f"with glassbox.features.compute_{b}().")
            pos = {n: j for j, n in enumerate(names)}
            idx = [pos[self.terms[i]['name']] for i in want[b]]
            lo = np.array([self.terms[i]['lo'] for i in want[b]])
            hi = np.array([self.terms[i]['hi'] for i in want[b]])
            med = np.array([self.terms[i]['median'] for i in want[b]])
            V = X[np.ix_(rows, idx)]
            out[:, want[b]] = np.clip(np.where(np.isfinite(V), V, med), lo, hi)
        return out

    def _z(self, fs, rows):
        V = self._values(fs, rows)
        c = np.array([t['center'] for t in self.terms])
        s = np.array([t['scale'] for t in self.terms])
        return V, (V - c) / np.where(s > 0, s, 1.0)

    def decision(self, fs, rows=None):
        rows = np.arange(len(fs.smiles)) if rows is None else np.asarray(rows)
        _, Z = self._z(fs, rows)
        w = np.array([t['coef'] for t in self.terms])
        return self.intercept + Z @ w

    def predict_proba(self, fs, rows=None):
        return 1.0 / (1.0 + np.exp(-self.decision(fs, rows)))

    def call(self, fs, rows=None, rate=None):
        """Boolean calls at a positive RATE (the convention the CV used)."""
        p = self.predict_proba(fs, rows)
        r = self.operating_point.get('positive_rate') if rate is None else rate
        return p >= np.quantile(p, 1 - r)

    # -------------------------------------------------------------- explain --
    def explain(self, fs, row):
        """Per-term contributions in LOG-ODDS for one molecule, exact and additive.

        contribution = coef . (value - center) / scale, and
        logit = intercept + sum(contributions) holds to floating point. The
        returned frame is sorted by |contribution|; `readable` renders CASPER's
        encoded names as sentences.
        """
        row = int(row)
        V, Z = self._z(fs, [row])
        w = np.array([t['coef'] for t in self.terms])
        contrib = w * Z[0]
        df = pd.DataFrame({
            'block': [t['block'] for t in self.terms],
            'term': [t['name'] for t in self.terms],
            'value': V[0], 'z': Z[0], 'coef': w, 'contribution': contrib,
            'readable': [F.casper_names_pretty(t['name']) if t['block'] == 'casper'
                         else t['name'] for t in self.terms],
            # carried through so a Morgan term can be drawn on the molecule
            'bit': [t.get('bit', -1) for t in self.terms],
            'radius': [t.get('radius', -1) for t in self.terms],
        })
        df = df.reindex(df.contribution.abs().sort_values(ascending=False).index)
        logit = self.intercept + contrib.sum()
        df.attrs.update(isoform=self.isoform, smiles=fs.smiles[row],
                        intercept=self.intercept, logit=float(logit),
                        probability=float(1 / (1 + np.exp(-logit))),
                        baseline_probability=float(1 / (1 + np.exp(-self.intercept))))
        return df.reset_index(drop=True)

    @staticmethod
    def summary(df, top=10):
        """One-screen text rendering of an explain() frame."""
        a = df.attrs
        out = [f"{a['isoform']}  p(TDI) = {a['probability']:.3f}"
               f"   (baseline from the intercept alone {a['baseline_probability']:.3f})",
               f"logit = {a['logit']:+.3f} = intercept {a['intercept']:+.3f} "
               f"+ {len(df)} term contributions", '',
               f"{'contrib':>9}  {'block':9} term", '-' * 78]
        for _, r in df.head(top).iterrows():
            out.append(f"{r.contribution:+9.3f}  {r.block:9} {r.readable[:58]}")
        pos = df[df.contribution > 0].contribution.sum()
        neg = df[df.contribution < 0].contribution.sum()
        out += ['-' * 78,
                f"{'':9}  pushing TOWARD TDI {pos:+.3f} / AWAY {neg:+.3f}",
                f"{'':9}  by block: " + '  '.join(
                    f'{b}={g.contribution.sum():+.2f}'
                    for b, g in df.groupby('block', sort=False))]
        return '\n'.join(out)

    # ------------------------------------------------------------- IO -------
    def save(self, path):
        json.dump({'isoform': self.isoform, 'C': self.C, 'rho': self.rho,
                   'quota': self.quota, 'n_train': getattr(self, 'n_train', None),
                   'intercept': self.intercept,
                   'operating_point': self.operating_point,
                   'n_terms': len(self.terms), 'terms': self.terms},
                  open(path, 'w'), indent=1)
        return path

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        m = cls(d['isoform'], **{f'{k}_k': v for k, v in d['quota'].items()},
                C=d['C'], rho=d['rho'])
        m.terms, m.intercept = d['terms'], d['intercept']
        m.operating_point, m.n_train = d.get('operating_point', {}), d.get('n_train')
        return m
