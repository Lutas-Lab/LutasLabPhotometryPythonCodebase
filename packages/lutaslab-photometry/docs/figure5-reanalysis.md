# Figure 5 smooth-basis reanalysis

This workflow reanalyzes the processed 20-second trials deposited for Figure
5. It complements, rather than replaces, the exact archived-MATLAB prediction
check exposed by `lutaslab-validate-figure5-glm`.

## Model

For each mouse, the primary Gaussian ridge model contains separate temporal
terms for:

- licks, represented by 12 raised-cosine functions from -5 to +8 seconds;
- visual-cue onset, represented by 8 functions from 0 to +8 seconds; and
- Ensure onset, represented by 8 functions from 0 to +7 seconds.

Convolutions reset at every trial boundary. Five contiguous blocks of complete
trials provide the outer test folds. Within each outer-training set, the ridge
penalty is selected with another grouped cross-validation across the remaining
trial blocks. Thus, every reported prediction is made for trials excluded from
both coefficient estimation and penalty selection.

The workflow compares the full model with term-omission models, a cue-only
model, and an ingestive model containing licks plus Ensure. It also repeats the
full model with 6, 12, and 24 lick basis functions. The paper's full-resolution
raw-lag results are retained in the output table as descriptive in-sample
references and are not presented as held-out scores.

## Current results

Across 15 mice, mean nested held-out R-squared was 0.296 for the full model,
0.292 for licks plus Ensure, and 0.020 for cue alone. The paired ingestive-minus-
cue advantage was 0.272 R-squared (two-sided paired Wilcoxon p = 0.00012).

Omitting each term from the full model gave mean unique contributions of 0.024
for licking, 0.004 for cue, and 0.017 for Ensure. Unique licking was positive
in 11 of 15 mice (two-sided Wilcoxon p = 0.030); unique cue was not consistent
(p = 0.890), and unique Ensure did not reach the same threshold (p = 0.107).
These three term-wise p-values are exploratory and unadjusted for multiple
comparisons; the licking result would not remain below 0.05 after Holm
correction across the three terms.

Mean held-out R-squared was 0.297, 0.296, and 0.299 with 6, 12, and 24 lick
basis functions, respectively. The primary conclusion is therefore insensitive
to basis size over this range.

These are predictive decompositions, not causal effects. Cue and Ensure occur
at nearly fixed trial times, licking is correlated with Ensure delivery, and
the deposited file contains already processed, trial-aligned fluorescence.
The ablations quantify additional held-out predictive information conditional
on the other modeled events.

## Run

From the repository root:

```powershell
lutaslab-run-figure5-reanalysis `
  "C:\path\to\Depository Data" `
  "outputs\figure5-reanalysis" `
  --basis-counts 6 12 24 `
  --primary-basis-count 12
```

The output directory contains `mouse_results.csv`, `basis_sensitivity.csv`,
`summary.json`, and `reanalysis_summary.png`.
