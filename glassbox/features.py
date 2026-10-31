"""The four feature blocks. Every column has a name a chemist can act on.

  morgan    UNFOLDED radius-2 environments -- the idea this project turns on
  standard  10 physchem + 30 a-priori MBI motifs, fixed before any fitting
  casper    3D surface-mapped atomic properties, pooled over conformers
  mordred   2D descriptors

WHY UNFOLDED. A conventional ECFP hashes ~23,000 distinct environments into
2048 bits, so a coefficient belongs to a collision bucket and means nothing on
its own. Keeping the fingerprint sparse and unfolded costs nothing and makes
every surviving coefficient one drawable substructure. On CYP3A4 this alone took
the linear model from 0.2573 to 0.3090 -- the single biggest win in the project.

Atom pairs and RDKit path fingerprints were built the same way and REJECTED:
20/20 configurations worse, with a mechanism (the Wald z rewards support, and
pairs are 5.4x denser, so they crowd out sparse informative environments; with
per-family quotas correcting that they still did not help). The signature was
consistent -- they raise AUC and lower MCC. Only `morgan` is kept here.
"""
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2)

# ---------------------------------------------------------------- physchem ---
PHYS = {'MolLogP': Descriptors.MolLogP, 'TPSA': Descriptors.TPSA,
        'MolWt': Descriptors.MolWt, 'HBD': Descriptors.NumHDonors,
        'HBA': Descriptors.NumHAcceptors, 'RotB': Descriptors.NumRotatableBonds,
        'FracCSP3': Descriptors.FractionCSP3, 'ArRings': Descriptors.NumAromaticRings,
        'Rings': Descriptors.RingCount, 'HeavyAtoms': Descriptors.HeavyAtomCount}

# --------------------------------------------------------- MBI motif panel ---
# Literature-precedented mechanism-based-inactivation / reactive-metabolite
# substructures. FIXED BEFORE ANY MODEL WAS FITTED -- nothing here was chosen
# by looking at a score, which is what makes them a fair baseline to beat.
ALERTS = {
    'methylenedioxyphenyl': 'c1cc2OCOc2cc1',
    'terminal_alkyne':      '[CX2;H1]#[CX2]',
    'internal_alkyne':      '[CX2]#[CX2]',
    'furan':                'c1ccoc1',
    'thiophene':            'c1ccsc1',
    'benzofuran':           'c1ccc2occc2c1',
    'thiazole':             'c1scnc1',
    'pyrrole':              '[nX3;H1]1cccc1',
    'aniline_prim':         '[NX3;H2][c]',
    'aniline_sec_tert':     '[NX3;H0,H1;!$(N[C,S]=[O,S,N])][c]',
    'phenol':               '[OX2H][c]',
    'para_aminophenol':     '[OX2H]c1ccc([NX3])cc1',
    'hydrazine':            '[NX3][NX3]',
    'thiourea_thioamide':   '[CX3]=[SX1]',
    'cyclopropylamine':     '[NX3]C1CC1',
    'N_alkyl_piperazine':   '[NX3]1CCN([CX4])CC1',
    'tert_alkylamine':      '[NX3;H0;!$(N[c]);!$(N[C,S]=[O,S,N])]([CX4])([CX4])[CX4]',
    'michael_acceptor':     '[CX3]=[CX3][CX3]=[OX1]',
    'quinone':              'O=C1C=CC(=O)C=C1',
    'epoxide':              'C1OC1',
    'nitroaromatic':        '[c][NX3](=O)=O',
    'alkyl_halide':         '[CX4][Cl,Br,I]',
    'isothiocyanate':       'N=C=S',
    'halophenol':           '[OX2H]c1ccc([F,Cl,Br,I])cc1',
    'indole':               'c1ccc2[nH]ccc2c1',
    'imidazole':            'c1cnc[nH]1',
    'N_aryl_imidazole':     '[c]n1ccnc1',
    'pyridine':             'c1ccncc1',
    'sulfur_divalent':      '[SX2;!$(S(=O));!$(S=O)]',
    'benzylic_CH2_N':       '[NX3][CH2]c1ccccc1',
}
PAT = {k: Chem.MolFromSmarts(v) for k, v in ALERTS.items()}
STANDARD_NAMES = list(PHYS) + list(ALERTS)


def standard_block(smiles):
    """10 physchem + 30 motif flags -> (n, 40). NaN where RDKit fails."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    phys = np.array([[f(m) if m else np.nan for f in PHYS.values()] for m in mols], float)
    alert = np.array([[bool(m is not None and m.HasSubstructMatch(p)) for p in PAT.values()]
                      for m in mols], float)
    return np.hstack([phys, alert])


# ------------------------------------------------- unfolded Morgan counts ----
def sparse_counts(smiles):
    """-> (list of {bit: count}, {bit: (row, (atom, radius))} provenance).

    The provenance map is what makes a coefficient drawable later: it records
    one molecule and one atom environment that produced each bit.
    """
    rows, prov = [], {}
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append({}); continue
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        fp = MORGAN.GetSparseCountFingerprint(m, additionalOutput=ao)
        rows.append(dict(fp.GetNonzeroElements()))
        for b, envs in ao.GetBitInfoMap().items():
            if b not in prov and envs:
                prov[b] = (i, envs[0])
    return rows, prov


def to_matrix(rows, bits):
    ix = {b: j for j, b in enumerate(bits)}
    X = np.zeros((len(rows), len(bits)), dtype=np.float32)
    for i, d in enumerate(rows):
        for b, c in d.items():
            j = ix.get(b)
            if j is not None:
                X[i, j] = c
    return X


def build_morgan(smiles, cache=None, min_count=10):
    """-> (counts matrix, bit ids, provenance). `min_count` is support, not a
    frequency cut: a term the model never sees ten times cannot be estimated."""
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    rows, prov = sparse_counts(smiles)
    sup = Counter()
    for d in rows:
        sup.update(d.keys())
    bits = sorted(b for b, c in sup.items() if c >= min_count)
    out = (to_matrix(rows, bits), bits, prov)
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        with open(cache, 'wb') as fh:
            pickle.dump(out, fh)
    return out


def project_morgan(smiles, bits):
    """New molecules onto an existing bit list (nothing is re-fitted)."""
    rows, _ = sparse_counts(smiles)
    return to_matrix(rows, bits)


def describe_morgan(bit, prov, smiles):
    """-> (fragment SMILES, radius). The radius matters: two environments of
    different radius can serialise to the same fragment string, and reporting
    only the string hides that they are different features."""
    if bit not in prov:
        return f'morgan:{bit}', -1
    i, e = prov[bit]
    m = Chem.MolFromSmiles(smiles[i])
    at, rad = int(e[0]), int(e[1])
    try:
        if rad == 0:
            return Chem.MolFragmentToSmiles(m, [at], canonical=True), 0
        env = Chem.FindAtomEnvironmentOfRadiusN(m, rad, at)
        return Chem.MolToSmiles(Chem.PathToSubmol(m, env), canonical=True), rad
    except Exception:
        return f'morgan:{bit}', rad


# ------------------------------------------------------ CASPER and Mordred ---
def casper_names_pretty(name):
    """'mean:aromatic|ac[10.5,12.0)A' -> a sentence a chemist can picture."""
    try:
        pool, rest = name.split(':', 1)
        prop, enc = rest.split('|', 1)
        pool_txt = {'mean': 'average over conformers',
                    'std': 'variation across conformers'}.get(pool, pool)
        if enc.startswith('ac'):
            rng = enc[2:].rstrip('A')
            return (f'{prop} surface autocorrelation at {rng} A '
                    f'(how often two {prop}-like surface patches sit that far apart) '
                    f'[{pool_txt}]')
        rng = enc[len('hist'):]
        return f'fraction of surface with {prop} in {rng} [{pool_txt}]'
    except Exception:
        return name


_CASPER_CFG = None


def _casper_init(cfg):
    """Pool initializer. The worker must be a module-level function and the
    config must arrive through initargs, or multiprocessing cannot pickle
    either -- a nested closure fails with "Can't get local object"."""
    global _CASPER_CFG
    _CASPER_CFG = cfg


def _casper_one(arg):
    i, s = arg
    try:
        import casper
        return i, np.asarray(casper.featurize(s, _CASPER_CFG), dtype=np.float32)
    except Exception:
        return i, None        # one NaN row, not a dead run


def compute_casper(smiles, n_confs=10, seed=61453, n_jobs=8):
    """Optional: recompute the 3D block.

    Needs `pip install casper-descriptor` (the module is `casper`; `pip install
    casper` is an unrelated project). https://github.com/dehaenw/casper

    The shipped matrices in features/ make this unnecessary for replication --
    it is a ~55 minute conformer run. The knobs below are CASPER's defaults, and
    a 45-configuration sweep confirmed they are already optimal: see
    results/casper_sweep_report.log.
    """
    import casper
    from multiprocessing import Pool
    cfg = casper.CasperConfig(n_confs=n_confs, seed=seed,
                              properties=tuple(casper.PROPERTIES),
                              encoding=('hist', 'autocorr'), conf_pool=('mean', 'std'))
    names = list(casper.featurize('CCOc1ccccc1', cfg, return_names=True)[1])
    rows = [None] * len(smiles)
    # Longest-first. Conformer cost varies ~13x across a set like this, and
    # feeding the expensive molecules first removes what was otherwise a
    # 12-minute straggler tail at the end of the run.
    cost = [(Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else 0)
            for s in smiles]
    order = [(int(i), smiles[int(i)]) for i in np.argsort(-np.asarray(cost))]
    with Pool(n_jobs, initializer=_casper_init, initargs=(cfg,)) as pool:
        for i, v in pool.imap_unordered(_casper_one, order, chunksize=1):
            rows[i] = v
    X = np.array([v if v is not None else np.full(len(names), np.nan, np.float32)
                  for v in rows], dtype=np.float32)
    return X, names


def compute_mordred(smiles):
    """1,613 2D descriptors; keeps the ones finite on >98% of rows and
    non-constant. Mordred IS pip-installable, so this is fully reproducible."""
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    df = calc.pandas([m for m in mols if m is not None], quiet=True)
    X = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    keep = (np.isfinite(X).mean(0) > 0.98) & (np.nanstd(X, axis=0) > 0)
    return X[:, keep], [n for n, k in zip(df.columns, keep) if k]


def require(path, how):
    """Fail with an actionable message instead of a bare FileNotFoundError.

    Large intermediates are not committed, so a fresh clone legitimately lacks
    some of them; the fix is always one command and the user should be told it.
    """
    if not os.path.exists(path):
        raise SystemExit(f'\nmissing: {path}\n  build it with:  {how}\n')
    return path
