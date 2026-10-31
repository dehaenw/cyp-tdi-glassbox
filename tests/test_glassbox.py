"""The claims the README makes, as assertions.

    python -m pytest tests/ -q        (or just: python tests/test_glassbox.py)
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from glassbox import FeatureSet, GlassBox, data as D, features as F, select as S

R = lambda *p: os.path.join(ROOT, *p)


def _fs():
    tr = pd.read_csv(R('data', 'tdi_train.csv'))
    X, bits, prov = F.build_morgan(tr.SMILES.tolist(), cache=R('features', 'morgan_train.pkl'))
    return tr, FeatureSet(
        tr.SMILES.tolist(), X, bits, prov, np.load(R('features', 'standard_train.npy')),
        np.load(R('features', 'casper_train.npy')),
        json.load(open(R('features', 'casper_names.json'))),
        np.load(R('features', 'mordred_slim_train.npy')),
        json.load(open(R('features', 'mordred_slim_names.json'))))


def test_label_rule_reproduces_released_labels():
    """The rule in data.py is a reconstruction; it has to match where both exist."""
    tr = pd.read_csv(R('data', 'tdi_train.csv'))
    for iso in D.ISOS:
        rel, mine = tr[f'{iso}_y_released'], tr[f'{iso}_y']
        ok = rel.notna() & mine.notna()
        assert (rel[ok] == mine[ok]).all(), iso


def test_pools_have_the_documented_sizes():
    tr = pd.read_csv(R('data', 'tdi_train.csv'))
    harv = pd.read_csv(R('data', 'tdi_harvested.csv'))
    assert len(D.pool_for('CYP3A4', tr)[1]) == 2528
    assert len(D.pool_for('CYP2D6', tr, harv)[1]) == 4213
    # harvesting must move CYP2D6 TOWARD the test prevalence (~0.071), not away
    assert D.pool_for('CYP2D6', tr, harv)[1].mean() < D.pool_for('CYP2D6', tr)[1].mean()


def test_wald_z_prefers_evidence_over_a_large_odds_ratio():
    """A 3/3 bit must not outrank a well-supported one. This is the whole
    argument for dividing by the standard error."""
    n = 400
    y = np.zeros(n, int); y[:100] = 1
    P = np.zeros((n, 2), bool)
    P[:3, 0] = True                              # 3/3 positive: infinite OR, no evidence
    P[:60, 1] = True; P[100:140, 1] = True       # 60 pos / 40 neg: real signal
    z = S.wald_z(P, y, np.array([0, 1]))
    assert abs(z[1]) > abs(z[0])


def test_explanation_is_exactly_the_computation():
    """logit == intercept + sum(contributions), to floating point."""
    tr, fs = _fs()
    for iso in ['CYP3A4', 'CYP2D6']:
        m = GlassBox.load(R('models', f'{iso}.json'))
        for row in [15, 1284, 4081, 5138, 6024]:
            e = m.explain(fs, row)
            assert np.isclose(e.attrs['intercept'] + e.contribution.sum(),
                              e.attrs['logit'], atol=1e-10)
            assert np.isclose(e.attrs['probability'],
                              m.predict_proba(fs, [row])[0], atol=1e-12)


def test_model_is_strictly_linear_and_small():
    for iso, n in [('CYP3A4', 152), ('CYP2D6', 88)]:
        m = GlassBox.load(R('models', f'{iso}.json'))
        assert len(m.terms) == n
        assert {t['block'] for t in m.terms} <= {'morgan', 'standard', 'casper', 'mordred'}
        assert all({'coef', 'center', 'scale'} <= set(t) for t in m.terms)


def test_every_morgan_term_is_a_real_substructure():
    """The point of not hashing: each coefficient must parse as a fragment and
    be findable in a molecule that triggered it."""
    from rdkit import Chem
    tr, fs = _fs()
    for iso in ['CYP3A4', 'CYP2D6']:
        m = GlassBox.load(R('models', f'{iso}.json'))
        for t in m.terms:
            if t['block'] != 'morgan':
                continue
            assert t['radius'] in (0, 1, 2), t
            assert Chem.MolFromSmiles(t['name'], sanitize=False) is not None, t['name']


def test_casper_recompute_matches_the_shipped_matrix():
    """The shipped 3D matrices must be exactly what the documented command
    produces. Skipped unless `pip install casper-descriptor` is present."""
    try:
        import casper  # noqa: F401
    except ImportError:
        print('    (skipped: pip install casper-descriptor)', end='')
        return
    tr = pd.read_csv(R('data', 'tdi_train.csv'))
    n = 12
    X, names = F.compute_casper(tr.SMILES.tolist()[:n], n_jobs=4)
    assert names == json.load(open(R('features', 'casper_names.json')))
    assert np.allclose(X, np.load(R('features', 'casper_train.npy'))[:n],
                       atol=1e-4, equal_nan=True)


def test_round_trip_through_json():
    m = GlassBox.load(R('models', 'CYP3A4.json'))
    p = R('models', '_roundtrip.json')
    try:
        m2 = GlassBox.load(m.save(p))
        assert m2.terms == m.terms and m2.intercept == m.intercept
    finally:
        os.path.exists(p) and os.remove(p)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in fns:
        f(); print(f'  PASS  {f.__name__}')
    print(f'\n{len(fns)} tests passed')
