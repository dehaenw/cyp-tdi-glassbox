"""Dataset assembly: labels, the suspect flag, harvested negatives, CV groups.

The challenge data is Apache-2.0 and lives on HuggingFace, so nothing is
redistributed here -- `load_raw()` fetches it. Everything downstream is derived
by code you can read.

THE LABEL RULE. The challenge releases `is_TDI` booleans, but a model is only
as trustworthy as its labels, so the rule was reconstructed from the two pIC50
arms and checked against the released column:

    D = direct-inhibition pIC50, T = TDI-condition pIC50
    D >  4  ->  positive iff (T - D) > log10(2) = 0.301
    D <= 4  ->  positive iff T > 4.301              ("inferred positive")
    both arms below LOQ        ->  negative         ("assigned negative")
    T measured alone, T <= 4.301 -> negative        (determined without D)

It reproduces the released labels exactly where both are defined. The residue
is interesting: rows the released file calls NEGATIVE although only the TDI arm
was measured and that arm is active (T > 4.301) are not derivable from the
stated rule. Those are flagged `_suspect` and dropped from the CYP3A4 pool
rather than silently trusted.
"""
import numpy as np
import pandas as pd

HF = 'hf://datasets/openadmet/cyp-challenge-train-test'
FILES = {'tdi': 'cyp-challenge-TRAIN_TDI.csv',
         'emax': 'cyp-challenge-TRAIN_Emax.csv',
         'sc': 'cyp-challenge-single-concentration-TRAIN.csv',
         'test': 'cyp-challenge-TEST-BLINDED.csv'}

SHIFT = np.log10(2.0)          # 0.30103, the challenge's 2-fold shift criterion
LOQ = 4.0
ISOS = ['CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']
TARGETS = ['CYP2D6', 'CYP3A4']   # the two isoforms scored in the TDI track
HARVEST_CUT = -0.3               # log2 fold-change above which a screen read is
                                 # a confident non-hit (see harvested_negatives)


def load_raw(root=None):
    """Fetch the four challenge tables. `root` reads a local directory instead."""
    base = root.rstrip('/') if root else HF
    return {k: pd.read_csv(f'{base}/{v}') for k, v in FILES.items()}


def label_from_arms(D, T):
    """Vectorised is_TDI from the two pIC50 arms. NaN where undeterminable."""
    D, T = np.asarray(D, float), np.asarray(T, float)
    y = np.full(len(D), np.nan)
    both = ~np.isnan(D) & ~np.isnan(T)
    y[both] = np.where(D[both] > LOQ, (T[both] - D[both]) > SHIFT, T[both] > LOQ + SHIFT)
    tonly = np.isnan(D) & ~np.isnan(T)
    y[tonly & (T <= LOQ + SHIFT)] = 0.0
    return y


def build(raw):
    """-> (train table with labels/suspect/screen columns, blinded test table)."""
    tdi, emax, sc = raw['tdi'], raw['emax'], raw['sc']
    df = tdi[['Molecule_Name', 'SMILES']].copy()
    for iso in ISOS:
        D, T = tdi[f'{iso}_pIC50_direct_inhibition'], tdi[f'{iso}_pIC50_TDI_condition']
        released = tdi[f'{iso}_is_TDI'] if f'{iso}_is_TDI' in tdi else emax[f'{iso}_is_TDI']
        df[f'{iso}_D'], df[f'{iso}_T'] = D, T
        df[f'{iso}_y_released'] = released.map({True: 1.0, False: 0.0})
        df[f'{iso}_y'] = label_from_arms(D, T)
        df[f'{iso}_suspect'] = (D.isna() & T.notna() & (T > LOQ + SHIFT)
                                & released.eq(False)).astype(int)
    for iso in ISOS:                      # single-concentration screen, TDI arm
        s = sc[sc.enzyme == iso].set_index('Molecule_Name')
        df[f'{iso}_sc_log2fc'] = df.Molecule_Name.map(s['log2fc_estimate'])
    return df, raw['test']


def harvested_negatives(df, cut=HARVEST_CUT):
    """CYP2D6 has only ~1,900 measured labels at 15% prevalence, while the test
    set turned out to be ~7% positive -- the measured pool is enriched for hits
    and the model is fitted to the wrong population. The single-concentration
    screen covers thousands more compounds; a read above `cut` (little or no
    inhibition) is a CONFIDENT NEGATIVE even though no pIC50 was ever fitted.

    Harvesting those more than doubles the CYP2D6 pool and moves its prevalence
    toward the test set's. This was the single largest CYP2D6 win in the project
    and the leaderboard confirmed the prevalence it implied.
    """
    out = df[['Molecule_Name', 'SMILES']].copy()
    for iso in ISOS:
        y = df[f'{iso}_y'].copy()
        y[df[f'{iso}_suspect'] == 1] = np.nan
        take = y.isna() & df[f'{iso}_sc_log2fc'].notna() & (df[f'{iso}_sc_log2fc'] > cut)
        y2 = y.copy(); y2[take] = 0.0
        out[f'{iso}_y'] = y2
        out[f'{iso}_harvested'] = take.astype(int)
    return out


def tanimoto_matrix(smiles, radius=2, nbits=2048):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    n = len(fps)
    S = np.eye(n, dtype=np.float32)
    for i in range(1, n):
        s = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]), dtype=np.float32)
        S[i, :i] = s; S[:i, i] = s
    return S


def butina_groups(S, cutoff=0.45):
    """Cluster for GROUPED CV. The library is analogue series; a random split
    puts near-identical compounds on both sides and reports a fantasy score."""
    from rdkit.ML.Cluster import Butina
    n = len(S)
    dists = []
    for i in range(1, n):
        dists.extend((1.0 - S[i, :i]).tolist())
    cl = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    g = np.empty(n, int)
    for ci, c in enumerate(cl):
        for j in c:
            g[j] = ci
    return g


def folds(groups, seed, k=5):
    """Group-disjoint k folds, shuffled by `seed`. Repeating CV over many seeds
    is how the noise band on every number in the README was measured."""
    u = np.unique(groups)
    rng = np.random.default_rng(seed); rng.shuffle(u)
    a = {gg: i % k for i, gg in enumerate(u)}
    return np.array([a[x] for x in groups])


def pool_for(iso, df, harvested=None):
    """Row indices and labels for one isoform. CYP3A4 uses measured labels with
    suspect rows dropped; CYP2D6 adds the harvested negatives."""
    if harvested is not None:
        keep = harvested[f'{iso}_y'].notna().values
        return np.where(keep)[0], harvested[f'{iso}_y'].values[keep].astype(int)
    keep = df[f'{iso}_y'].notna().values & (df[f'{iso}_suspect'] == 0).values
    return np.where(keep)[0], df[f'{iso}_y'].values[keep].astype(int)
