# Statistical status of the deposited-data reanalyses

These analyses were developed after publication and after inspection of the
deposited data. They are therefore exploratory reanalyses, not preregistered
confirmatory tests. The labels "primary" and "secondary" below define a stable
reporting order for future reruns; they do not retroactively make a test
confirmatory.

All model comparisons use out-of-fold predictions from the same contiguous
trial-block splits. Ridge strength is selected inside each outer training
partition. The mouse is the biological unit for group inference.

## Figure 5 reanalysis

Primary descriptive comparison:

- paired mouse-level held-out R² difference between the ingestive model
  (lick + Ensure delivery) and the cue-only model.

Secondary/exploratory comparisons:

- unique held-out ΔR² for lick, cue, and Ensure terms;
- cue-epoch versus consumption-epoch performance;
- temporal-basis-size sensitivity;
- comparison with the published in-sample raw-lag R² values.

The three unique-term Wilcoxon tests are a family. Unadjusted p-values remain
in the current output for transparency; a confirmatory report should add a
Holm correction and state that choice before rerunning.

## Bout-structure analysis

The fixed reference bout definition is a maximum 1.0-second interlick gap and
at least three licks.

Primary descriptive comparison:

- bout terms beyond the cue + lick model at the 1.0-second definition.

Key complementary comparison:

- Ensure delivery beyond cue + lick + bout terms.

The 0.5- and 1.5-second definitions are sensitivity analyses. All inferential
tests in this analysis are exploratory and currently unadjusted.

## Multitastant analysis

For each prespecified tastant pair, the primary comparison is the mouse-level
held-out ΔR² from allowing a condition-specific delivery kernel rather than a
shared delivery kernel, conditional on licking and bout predictors.

The five tastant pairs form one family. Current two-sided Wilcoxon p-values are
reported unadjusted because the analysis is exploratory. A future confirmatory
analysis should prespecify either one contrast or a multiplicity correction
(Holm is the default recommendation). Delivery-kernel difference AUC and
cross-tastant transfer scores are secondary diagnostics.

The flagged DK33 water-versus-quinine fit remains in the exported table. Group
summaries use median and interquartile range where that unstable fit would
otherwise dominate the mean.

## Interpretation

Held-out R² and ΔR² quantify predictive information, not biological causality.
Event kernels are conditional model components and should not be described as
isolated sensory, motor, or causal responses without an experimental
manipulation that separates those variables.
