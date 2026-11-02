# A glass-box model for CYP time-dependent inhibition

A **strictly linear** classifier for time-dependent inhibition (TDI) of CYP3A4
and CYP2D6, built for the [OpenADMET CYP Inhibition Blind
Challenge](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/).
152 terms for CYP3A4, 88 for CYP2D6. No trees, no boosting, no neural
embeddings, no ensembling — and every term is a substructure you can draw, a
physical property, or a surface feature with a length scale.

It reaches **93% of the macro MCC of the black-box ensemble** it was built to be
an honest alternative to, and **106% on CYP2D6**.

```
logit p(TDI) = b0 + Σ wᵢ·z(log1p count of Morgan environment i)   [selected]
                  + Σ c_l·z(physchem / a-priori MBI motif l)      [fixed panel]
                  + Σ vⱼ·z(CASPER 3D surface feature j)           [selected]
                  + Σ u_k·z(Mordred 2D descriptor k)              [selected]
```

## Why "natively" interpretable

The usual move is to fit a black box and explain it afterwards with SHAP or
LIME. That gives you an *estimate of a counterfactual* under a surrogate model,
a sampled background distribution, and an additivity assumption that the true
model does not satisfy.

Here the per-term contribution `wᵢ·zᵢ` is not an approximation of what the model
did — **it is what the model did**. `logit = b0 + Σ contributions` holds to
floating-point exactness, which the notebook verifies rather than asserts. There
is no surrogate to be unfaithful and no background set to choose.

Two consequences worth the trouble:

- **Every Morgan coefficient is a drawable substructure.** The fingerprint is
  kept *unfolded* (sparse, never hashed), so a coefficient belongs to one
  specific atom environment instead of a 2048-bit collision bucket.
- **The saved model is readable JSON** — 1,400 lines, one entry per term with
  its coefficient, standardisation and imputation bound. You can read the whole
  model, which is the only real test of an interpretable one.

## Quickstart

```python
from glassbox import GlassBox, FeatureSet
import json, numpy as np, pandas as pd

m  = GlassBox.load('models/CYP3A4.json')
fs = FeatureSet(...)                      # see notebooks/why_is_this_a_TDI.ipynb
e  = m.explain(fs, row=5138)
print(GlassBox.summary(e))
```

```
CYP3A4  p(TDI) = 0.980   (baseline from the intercept alone 0.224)
logit = +3.917 = intercept -1.241 + 152 term contributions

  contrib  block     term
------------------------------------------------------------------------------
   +3.530  standard  terminal_alkyne
   +2.528  standard  internal_alkyne
   +0.679  mordred   fragCpx
   -0.507  morgan    N
   -0.467  casper    fraction of surface with mr in [3.50,4.00)
   ...
           pushing TOWARD TDI +9.530 / AWAY -4.372
           by block: standard=+5.66  mordred=+0.02  morgan=-0.29  casper=-0.23
```

A terminal alkyne on a pyridine, called at p=0.98, because of the terminal
alkyne — the textbook mechanism-based inactivator, oxidised to a ketene that
acylates the apoprotein. The experimental label agrees.

**[`notebooks/why_is_this_a_TDI.ipynb`](notebooks/why_is_this_a_TDI.ipynb)** walks
through that case, a second one where *learned* environments carry the call with
no a-priori alert firing, the exactness check, and what the 3D block sees.

## Results

Grouped (Butina) 5-fold CV, 12 repeats, threshold chosen leave-one-seed-out.
Measured on **seeds 24–35, which no grid and no threshold choice ever touched**,
against a morgan-only baseline on those same seeds:

| | morgan only | + 3D & Mordred | terms | vs black-box ensemble |
|---|---|---|---|---|
| CYP3A4 | 0.3009 | **0.3253** | 152 | 86% |
| CYP2D6 | 0.1864 | **0.1907** | 88 | **106%** |
| macro | 0.2437 | **0.2580** | | **93%** |

`python scripts/03_crossval.py --baseline` reproduces this table exactly.

The last column is the CV ratio to a black-box ensemble measured under the
same protocol. The leaderboard revised it to 94% on CYP3A4 and 108% on
CYP2D6 — CV had the two isoforms the wrong way round, which is the next
section.

### On the blind leaderboard

Three versions of this model were scored on the challenge's held-out set over
three days, together with the black-box ensemble it is measured against:

| | CYP3A4 | CYP2D6 | macro |
|---|---|---|---|
| morgan + fixed panel only (104 / 56 terms), rate 0.14 | 0.3712 | 0.2467 | 0.3089 |
| this model, + 3D & Mordred (152 / 88 terms), rate 0.20 | **0.3835** | 0.2013 | 0.2924 |
| same two rankings, CYP2D6 rate cut to 0.112 | **0.3835** | **0.3011** | **0.3423** |
| black-box ensemble (LightGBM / TabICL / learned embeddings) | 0.4094 | 0.2796 | 0.3445 |

**The 3D block did what it was added to do.** CYP3A4 rose +0.012, from 91% to
**94% of the ensemble**, landing within 0.008 of a projection registered before
the file was submitted. CV had under-predicted it by +0.058, consistent with
every CYP3A4 measurement we have.

**CYP2D6's swing of 0.10 was entirely the threshold.** All three submissions
report a recall of 0.5017 — they found the *same* true inactivators — while the
number of positive calls ran 105, 150, 84. Inverting the published metrics
(identifiable up to the unknown size of the scored subset, which cancels out of
MCC) recovers the same 22.3 true positives every time, against 83, 107 and 60
false ones. The extra calls bought nothing; removing them was worth +0.055.

The positive rate had moved because it is chosen by cross-validation, and that
choice is biased:

> **Grouped CV over-estimates the MCC-optimal positive rate, and the bias is
> worse the rarer the class.** Grouped folds delete whole Butina clusters, so a
> held-out compound's nearest training neighbour is at Tanimoto 0.426; a test
> compound's is at 0.587. CV is therefore scoring a *weaker ranker* than the
> real task presents, and for a weak ranker over-calling is MCC-optimal. At
> CYP3A4's 29% prevalence that costs almost nothing. At CYP2D6's 7.1% it is
> brutal.

The third submission was built to measure that bias rather than argue about it.
Holding CYP3A4 byte-identical makes the macro arithmetic exact, so cutting only
the CYP2D6 rate reads the isoform off directly. Aligning the cross-validated
cumulative-recall curve to the test curve *by recall* then quantifies the bias
per isoform:

| | CV depth at matched recall | test depth | overshoot |
|---|---|---|---|
| CYP3A4 (29% prevalence) | 377 of 706 | 343 of 706 | **1.10×** |
| CYP2D6 (7.1% prevalence) | 156 of 628 | 83 of 628 | **1.89×** |

At 29% prevalence cross-validation picks the operating point correctly, and the
deployed CYP3A4 threshold turns out to be bracketed at its optimum — the cut
bands convert above breakeven and the add bands below it, so the right move was
to change nothing. At 7% it overshoots by nearly a factor of two.

So the honest leaderboard reading is **94% of the ensemble on CYP3A4, 108% on
CYP2D6, and 99% of its macro** — from 152 and 88 readable terms against learned
embeddings, gradient boosting and in-context tabular models. The operating point
deserved as much care as the model, which is the opposite of where the effort
went.

## Replication

Nothing is hand-curated and no data is redistributed. The challenge data is
Apache-2.0 on HuggingFace and is fetched by the first script.

```bash
conda env create -f environment.yml && conda activate cyp-glassbox
pip install -e .            # or  pip install -e ".[all]"  to recompute features

python scripts/01_build_dataset.py      # fetch + labels + Butina CV groups  (~2 min)
python scripts/02_featurize.py          # unfolded Morgan + fixed panel      (~1 min)
python scripts/02_featurize.py --mordred   # Mordred 2D                      (~30 min)
python scripts/03_crossval.py --baseline   # reproduces the table above      (~1 min)
python scripts/04_fit_model.py          # fit + save models/*.json           (~1 min)
python scripts/05_predict.py            # blinded-test predictions
```

Verified end to end: the pipeline above reproduces the research code's Butina
partition, labels and harvested set **byte-identically**, the CV numbers to four
decimals, and the submitted test predictions **bit-identically on all 750 rows
for both isoforms**.

### On the two feature blocks that need extra packages

Both are pip-installable, and both matrices are shipped anyway so a clean clone
can explain and predict immediately:

- **CASPER** (the 3D block) is `pip install casper-descriptor` — the
  distribution is named `casper-descriptor` while the module is `casper`, so
  `pip show casper` reports nothing even when it is installed, and `pip install
  casper` fetches an unrelated project. Source:
  [github.com/dehaenw/casper](https://github.com/dehaenw/casper) (MIT).
  The matrices are committed (13 MB) so you can skip a 55-minute conformer run;
  you need the package to recompute them (`02_featurize.py --casper`) or to
  render surface patches.
- **Mordred** (the 2D block) is `pip install mordred`. Only the ~29 columns the
  fitted models reference are committed (713 kB instead of 32 MB), which is
  enough for `explain()` and `predict`. The full matrix is a build step because
  feature *selection* re-runs inside every CV fold.

## The ideas that did the work

**Unfolding the fingerprint.** A radius-2 ECFP on this library has ~23,000
distinct environments, 2,963 with support ≥10. Hashing them into 2,048 bits
destroys exactly the thing that makes a linear model worth having. Keeping them
unfolded took CYP3A4 from 0.2573 to 0.3090 — the single largest win in the
project — and made every coefficient drawable.

**Selecting on a standardised effect size, not an odds ratio.** Terms are ranked
by the Wald z of the log odds ratio with a Haldane ½ correction on all four
cells, `z = log(ad/bc) / √(1/a+1/b+1/c+1/d)`. The denominator is the point: a
bit present in 3 molecules that happen all to be positive has a huge odds ratio
and no evidence, and ranking on the ratio alone fills the model with 3/3
accidents. Continuous blocks use the Mann-Whitney U z, the non-parametric
analogue — scale-free, so Mordred's arbitrary units do not matter.

**Harvesting negatives from the screening data.** CYP2D6 has only ~1,500
measured labels at 22% prevalence, while the test set turned out to be ~7%
positive: the measured pool is enriched for hits and the model was being fitted
to the wrong population. A single-concentration screen read showing no
inhibition is a *confident negative* even with no pIC50 ever fitted. Harvesting
those took the pool to 4,213 at 7.7% prevalence. The leaderboard later confirmed
the prevalence this implied.

**Validation that does not lie.** The library is analogue series, so folds are
grouped by Butina cluster; one split has an MCC standard error around 0.013, so
every number is a mean over 12 repeats; and the operating point is chosen on 11
repeats and scored on the 12th, because choosing a threshold on the curve you
report is worth about +0.02 of self-deception. Seeds 0–11 tuned, 24–35 verified,
36–47 remain untouched.

## What did not work

Negative results, with the mechanism where we found one — these cost as much
effort as the positives and are the part most likely to save someone else time.

- **Atom pairs and RDKit path fingerprints**: 20/20 configurations worse. The
  Wald z rewards support and atom pairs are 5.4× denser, so they crowd out
  sparse informative environments; per-family quotas fixed that bias and they
  *still* did not help. Consistent signature — they raise AUC and lower MCC.
- **Tuning CASPER's knobs**: a 45-configuration sweep (surface probe, dot
  density, autocorrelation reach and resolution, conformer count, pooling
  statistics) produced **nothing** that beat the defaults; the best cell was
  +0.0005 on CYP3A4 and −0.0014 on CYP2D6. The water-accessible surface was
  actively *harmful* (−0.021 on CYP2D6) and `density=16` really is the knee.
  The sweep also measured its own floor: refitting with a different ETKDG
  conformer seed and otherwise identical settings moves CYP3A4 CV MCC by 0.007
  (10/12 seeds, sign test p=0.039), which is larger than every knob effect.
- **Hand-written mechanistic rules** beyond the fixed motif panel: three
  validated rules, zero CV gain each — ECFP had already subsumed them.
- **Per-class probability calibration**: the deficit is within-class
  discrimination, not calibration, and an oracle bound said so before we built
  the estimator.

## Honest limitations

- **Individual coefficients in correlated groups are not individually
  reproducible.** Bootstrap-refit stability across these terms runs from 13% to
  100%. A term at 25% stability is a placeholder for a correlated family, not
  *the* chosen feature. Predictions are stable; credit assignment within a
  correlated group is not, and no interpretability method can fix that.
- **It is still 0.2580 macro MCC in CV, 0.2924 on the leaderboard.** This is a
  genuinely hard endpoint with noisy labels, and the model is interpretable, not
  accurate in absolute terms.
- **The 3D block's absolute values carry conformer-seed noise** of about 0.007
  CV MCC. The seed was fixed arbitrarily and never selected on, so the reported
  numbers are unbiased — but a different seed gives a slightly different model.
- **Two different operating points.** CV chose a positive *rate*, which the
  submission applied as a quantile over the 750 test compounds. That is a
  population-level call and cannot be made for a single molecule, so the model
  also stores `prob_cut`, the out-of-fold probability at that rate. They
  disagree on 35–39 of 750 compounds; `05_predict.py` reports how many.
  This is not a detail. The single largest swing in anything we measured —
  larger than the 3D block, larger than every feature family we tested — came
  from the positive rate on CYP2D6 moving by 0.06, with the model unchanged.
- **The CYP2D6 harvested negatives are screen-derived**, not pIC50-confirmed. It
  was the right call here and the leaderboard supported it, but they are a
  weaker label than a measured one.

## Layout

```
glassbox/
  data.py       labels (rule reconstructed and checked), harvested negatives, Butina groups
  features.py   unfolded Morgan + provenance, the fixed 40-term panel, CASPER/Mordred
  select.py     the Wald-z and Mann-Whitney-z pickers, LOSO thresholding
  model.py      GlassBox: fit / predict / explain / readable-JSON save
  validate.py   grouped repeated CV with seed hygiene
  draw.py       substructure highlighting and contribution waterfalls
scripts/        01 data → 02 features → 03 crossval → 04 fit → 05 predict
notebooks/      why_is_this_a_TDI.ipynb
models/         CYP3A4.json, CYP2D6.json — the complete fitted models
results/        logs and JSON behind every number quoted above
```

## Data and license

Code: MIT (see `LICENSE`). Data: the [OpenADMET CYP challenge
dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test),
Apache-2.0, fetched not redistributed. CASPER is a separate MIT project by the same
author: [github.com/dehaenw/casper](https://github.com/dehaenw/casper),
`pip install casper-descriptor`.

`predictions/` is **git-ignored while the challenge is open** so this repo does
not hand out test-set answers; `python scripts/05_predict.py` regenerates them.
