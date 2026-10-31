# Precomputed feature matrices

| file | shape | why it is here |
|---|---|---|
| `casper_train.npy`, `casper_test.npy` | (6145, 480), (750, 480) | Shipped so a clean clone skips a ~55 min conformer run. `float32`, 12 MB + 1.4 MB. Recompute with `pip install casper-descriptor` and `02_featurize.py --casper`. |
| `casper_names.json` | 480 names | e.g. `mean:aromatic|ac[10.5,12.0)A` — pooling : property \| encoding. |
| `mordred_slim_{train,test}.npy` | (n, 29) | Only the Mordred columns the two fitted models reference. 713 kB instead of 32 MB; enough for `explain()` and `predict`. |
| `standard_{train,test}.npy` | (n, 40) | 10 physchem + 30 a-priori MBI motifs. Cheap to rebuild; committed so the notebook runs on a clean clone. |

Not committed, because they rebuild quickly:
`morgan_train.pkl` (~1 min, `02_featurize.py`) and `mordred_full_*.npy`
(~30 min, `02_featurize.py --mordred`, needed only to re-run selection in CV).

Row order matches `data/tdi_train.csv` and `data/tdi_test.csv` exactly.

These are not opaque blobs: `tests/test_glassbox.py` recomputes the first rows
of `casper_train.npy` with `compute_casper()` and asserts they match **exactly**
(max abs difference 0.0), so the shipped matrices are verifiably what the
documented command produces. That test skips if `casper-descriptor` is absent.
