# A glass-box model for CYP time-dependent inhibition

A linear classifier for time-dependent inhibition (TDI) of CYP3A4 and CYP2D6,
built for the [OpenADMET CYP Inhibition Blind
Challenge](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/).
One L2 logistic regression per isoform: 152 terms for CYP3A4, 88 for CYP2D6.
No trees, no boosting, no learned embeddings, no ensembling. The model is a
list of features and weights.

```
logit p(TDI) = b0 + Σ wᵢ · z(featureᵢ)
```

The features come from four blocks:

- **Unfolded Morgan (ECFP)**: radius-2 atom environments, kept sparse and never
  hashed, so each weight belongs to one specific substructure.
- **Standard panel**: 10 physicochemical properties and 30 mechanism-based
  inhibition motifs, fixed in advance.
- **CASPER**: 3D molecular surface features, each with a surface property and a
  length scale.
- **Mordred**: 2D descriptors.

Because the model is linear, `logit = b0 + Σ contributions` holds to
floating-point exactness. A term's contribution is what the model did, not an
estimate of it, so there is no SHAP or LIME surrogate and no background set to
choose. The fitted models are readable JSON in `models/`, one entry per term.

[`notebooks/why_is_this_a_TDI.ipynb`](notebooks/why_is_this_a_TDI.ipynb) walks
through a worked prediction. **[MORE_INFO.md](MORE_INFO.md)** has the
cross-validation and blind-leaderboard results, the comparison against an
internal black-box ensemble, replication instructions, what did not work, and
the limitations.

Code: MIT (see `LICENSE`). Data: the [OpenADMET CYP challenge
dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test),
Apache-2.0, fetched not redistributed.
