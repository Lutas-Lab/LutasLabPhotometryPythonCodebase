# Multitastant delivery-kernel analysis

This analysis asks whether delivery-aligned photometry depends on tastant
identity after accounting for individual licks and lick-bout structure.

## Design

Delivery-centered, trial-level data are compared within the same mice for:

- Ensure versus sucralose;
- Ensure versus sucrose;
- sucrose versus sucralose;
- Ensure versus quinine; and
- water versus quinine.

The paired model contains condition-specific lick and bout terms, a shared
delivery kernel, and an optional tastant-by-delivery interaction. The held-out
gain from that interaction is the primary tastant-identity statistic. Ridge
strength is selected within each outer training set, and complete contiguous
trial blocks are held out.

## Result

Tastant-specific delivery kernels provided small, directionally consistent
held-out improvements for several comparisons:

- Ensure versus sucralose: median delta R-squared 0.0045, positive in 5/5
  mice, two-sided Wilcoxon p = 0.0625;
- Ensure versus sucrose: median 0.0076, positive in 4/5 mice, p = 0.125;
- sucrose versus sucralose: median 0.0003, positive in 3/5 mice, p = 1.0;
- Ensure versus quinine: median 0.0035, positive in 6/8 mice, p = 0.078;
- water versus quinine: median 0.0020, positive in 8/9 mice, p = 0.055.

One water-versus-quinine mouse had a catastrophically poor shared-kernel fold
(held-out R-squared below -1). It is retained and flagged in the output rather
than silently excluded; median and interquartile summaries prevent it from
controlling the group estimate.

No individual two-sided comparison crosses p = 0.05, and the tests are not
adjusted across pairs. The data therefore provide suggestive but not definitive
evidence for tastant-dependent delivery responses. The strongest patterns are
the consistent directions for Ensure versus sucralose and water versus
quinine. Cross-tastant transfer is generally poor, but that comparison is also
sensitive to session differences and cannot by itself establish sensory coding.

## Run

```powershell
python scripts/run_multitastant_analysis.py `
  "C:\path\to\Depository Data" `
  "outputs\multitastant-analysis"
```

Outputs include mouse-level scores, conditional delivery kernels, a JSON
summary, and editable SVG plus PNG figures.
