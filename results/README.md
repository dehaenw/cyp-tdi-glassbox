# Evidence behind every number in the README

| file | what it shows |
|---|---|
| `01_build_dataset.log` | the label rule agreeing with the released column on 1.0000 of rows, all four isoforms; pool sizes and prevalences |
| `03_crossval_verify.log` | the headline table: CYP3A4 0.3253, CYP2D6 0.1907, macro 0.2580, and the matched morgan-only baseline on the same seeds |
| `crossval_seeds24_35.json` | the same, machine-readable, with the argmax and 1-SE rates |
| `04_fit_model.log` | the final fits, term counts, and the chosen operating points |
| `05_predict.log` | blinded-test call counts and the rate-vs-absolute-threshold disagreement |
| `casper_sweep_report.log` | the 45-configuration CASPER knob sweep read as a factorial: every knob's marginal effect |
| `casper_sweep.json` | all 90 measured cells (45 configs x 2 isoforms) from that sweep |

The sweep is the longest negative result here and the most reusable: it says
CASPER's defaults are already optimal, that the water-accessible surface is
actively harmful for this endpoint, and that the conformer-seed floor (0.007 CV
MCC) is larger than any knob effect — so there is nothing to tune.
