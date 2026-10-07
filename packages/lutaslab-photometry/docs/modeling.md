# Modeling and forecasting

## Supported GLM

The supported default for continuous photometry is the NumPy/SciPy ridge GLM
in `lutaslab_core.glm`. The synthetic
[`event_glm_and_transfer.py`](../../lutaslab-core/examples/event_glm_and_transfer.py) example
builds a trial-reset raised-cosine design, performs grouped ridge selection,
and fits a causal gamma transfer function without paper-specific data.

NeMoS remains available through `lutaslab_photometry.nemos_analysis` for older
notebooks and analyses that explicitly require NeMoS/JAX objects. This optional
backend requires Python 3.12 and uses the tested JAX 0.11 release series.
Install it with:

```powershell
uv sync --frozen --all-packages --extra nemos
```

## Predictors and signal representations

Current behavioral predictors include locomotion, licking, visual cue, and
solenoid/reward. Models can use raw 465 fractional fluorescence while
estimating a broad session-scale fluorescence component separately. That broad
component may contain biology as well as bleaching and should not automatically
be interpreted as a pure nuisance signal.

Behavioral predictors can be represented with temporal basis functions.
Non-positive lag windows use only present and past behavior and can support a
predictive interpretation. Two-sided windows characterize temporal association
but are not causal because they include future behavior.

```text
negative lag = predictor before the response
zero lag     = simultaneous predictor and response
positive lag = predictor after the response
```

Correlated variables should be fit jointly when estimating unique predictive
contributions. Full-versus-reduced model comparisons can distinguish, for
example, the contribution of licking after accounting for cue and reward.

## Validation and regularization

Photometry samples are strongly autocorrelated, so evaluation uses temporally
blocked cross-validation rather than randomly shuffled samples. Temporal gaps
around held-out blocks reduce leakage from lagged predictor windows. Learned
preprocessing, broad-fluorescence estimation, and predictor scaling are fitted
on training samples and applied to held-out data. Ridge regularization
stabilizes correlated multi-predictor designs. Behavioral GLMs select ridge
strength with blocked inner folds contained entirely inside each outer training
partition, then refit on that partition before reporting performance on its
untouched outer test block.

## Forecasting

The optional forecasting workflow predicts future photometry, locomotion, lick
occurrence, or lick counts from past-only features:

```powershell
uv sync --frozen --all-packages --extra forecasting

uv run lutaslab-run-forecasting `
  --manifest analysis/sessions.csv `
  --data-root "Z:\Photometry" `
  --output-dir analysis/forecasts/licks `
  --target lick_binary `
  --horizons 0.5 1 2 5 `
  --history 5 `
  --target-window 1
```

Each target and feature row has an explicit prediction time. Expanding-window
evaluation keeps training observations before test observations and inserts a
gap covering predictor history, forecast horizon, and any future counting
window. By default, each outer training prefix uses its own gapped,
forward-chaining inner validation to select regularization strength. The model
is then refit on the complete outer training prefix and evaluated once on the
future outer block. Use `--alphas` to change the candidate grid. `--alpha`
bypasses selection only when a strength was prespecified independently of the
reported results.

Each run compares a fold-local constant baseline, target-history, cross-modal,
and combined models. The constant prediction is estimated using only the outer
training fold: its mean for continuous and count targets, or lick prevalence
for binary targets.

Continuous targets use ridge regression, lick occurrence uses logistic
regression, and lick counts use Poisson regression. Raw 465 is the default
photometry representation; `--photometry-source dff` is a secondary
comparison. Outputs include session metrics, mouse means, group mean and SEM,
performance-versus-horizon figures, and a JSON settings record. Binary outputs
report average precision as the primary metric, AUROC, log loss, test-set
prevalence, and average-precision gain over prevalence. The history-only model
remains the stronger scientific baseline for incremental cross-modal value.
Predictive performance does not by itself establish biological causality.
