# Figure 5 lick-bout structure analysis

This analysis tests whether the post-Ensure photometry response can be
explained by the structure of consummatory lick bouts rather than by individual
lick timing or the Ensure TTL.

## Bout representation

The primary definition groups at least three licks when adjacent licks are no
more than 1 second apart. Each bout contributes four types of predictors:

- a smooth temporal kernel at bout onset;
- a lick-count-modulated bout-onset kernel;
- a binary occupancy signal from the first through the last lick; and
- the within-bout lick rate during that occupancy period.

The complete analysis is repeated with maximum interlick gaps of 0.5, 1.0,
and 1.5 seconds. Every model also contains the visual-cue term as a nuisance
predictor. Ridge strength is selected inside each outer training set, and all
reported R-squared values come from held-out contiguous blocks of complete
trials.

## Result

At the primary 1-second bout definition, mean held-out R-squared across 15 mice
was:

- individual licks: 0.279;
- bout structure: 0.200;
- individual licks plus bout structure: 0.281;
- individual licks plus Ensure: 0.296; and
- individual licks plus bout structure plus Ensure: 0.301.

Adding bout structure beyond individual licks contributed only 0.002 held-out
R-squared (two-sided Wilcoxon p = 0.847). Adding bout structure beyond licks
and Ensure contributed 0.005 (p = 0.277). These effects were close to zero for
all three bout-gap definitions.

Conversely, individual lick timing added 0.013 R-squared beyond bout structure
and Ensure (11 of 15 mice; p = 0.022). Ensure added 0.019 beyond individual
licks and bout structure (10 of 15 mice; p = 0.073).

The term-wise p-values are exploratory and unadjusted for multiple
comparisons. The analysis does not support the hypothesis that conventional
bout onset, size, occupancy, or rate explains the large post-Ensure signal.
Individual lick timing retains information not captured by those bout
summaries. The Ensure contribution is positive on average but remains difficult
to isolate because delivery and consummatory licking are tightly correlated.

## Run

```powershell
python scripts/run_figure5_bout_analysis.py `
  "C:\path\to\Depository Data" `
  "outputs\figure5-bout-analysis" `
  --bout-gaps 0.5 1.0 1.5 `
  --primary-bout-gap 1.0 `
  --min-licks 3
```

Outputs are `mouse_results.csv`, `summary.json`, and editable SVG plus PNG
summary figures.
