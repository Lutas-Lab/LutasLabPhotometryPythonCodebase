from collections import defaultdict

import numpy as np

from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel


CONTINUOUS_TARGETS = {"photometry", "locomotion"}
LICK_TARGETS = {"lick_binary", "lick_count"}
VALID_TARGETS = CONTINUOUS_TARGETS | LICK_TARGETS
MODEL_ORDER = ("history_only", "cross_modal", "combined")


def _event_signal(event_times, time, *, binary=False):
    event_times = np.asarray(event_times, dtype=float)
    time = np.asarray(time, dtype=float)
    dt = float(np.median(np.diff(time)))
    edges = np.concatenate(([time[0] - dt / 2], time + dt / 2))
    counts, _ = np.histogram(event_times[np.isfinite(event_times)], bins=edges)
    counts = counts.astype(float)
    return (counts > 0).astype(float) if binary else counts


def prepare_forecast_signals(session, *, dt=0.1, channel=1, photometry_source="raw465"):
    """Resample photometry, locomotion, and task events onto one uniform grid."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if photometry_source not in ("raw465", "dff"):
        raise ValueError("photometry_source must be 'raw465' or 'dff'.")

    time_key = f"photo_time_465_ch{int(channel)}"
    signal_key = (
        f"photometry_465_ch{int(channel)}"
        if photometry_source == "raw465"
        else f"dff_ch{int(channel)}"
    )
    required = {
        time_key,
        signal_key,
        "locomotion_time",
        "processed_locomotion",
        "lick_times",
        "cue_onset",
        "solenoid_onset",
    }
    missing = required.difference(session)
    if missing:
        raise ValueError(f"Processed session is missing forecast keys: {sorted(missing)}")

    photometry_time = np.asarray(session[time_key], dtype=float)
    photometry = np.asarray(session[signal_key], dtype=float)
    locomotion_time = np.asarray(session["locomotion_time"], dtype=float)
    locomotion = np.asarray(session["processed_locomotion"], dtype=float)
    if len(photometry_time) != len(photometry) or len(locomotion_time) != len(locomotion):
        raise ValueError("Forecast signal time and value arrays must have equal lengths.")
    if np.any(np.diff(photometry_time) <= 0) or np.any(np.diff(locomotion_time) <= 0):
        raise ValueError("Forecast signal timestamps must be strictly increasing.")

    start = max(float(photometry_time[0]), float(locomotion_time[0]))
    end = min(float(photometry_time[-1]), float(locomotion_time[-1]))
    if end - start < 2 * dt:
        raise ValueError("Photometry and locomotion do not have enough overlapping time.")
    time = start + np.arange(int(np.floor((end - start) / dt)) + 1) * dt

    return {
        "time": time,
        "photometry": np.interp(time, photometry_time, photometry),
        "locomotion": np.interp(time, locomotion_time, locomotion),
        "licking": _event_signal(session["lick_times"], time),
        "cue": _event_signal(session["cue_onset"], time, binary=True),
        "solenoid": _event_signal(session["solenoid_onset"], time, binary=True),
        "photometry_source": photometry_source,
    }


def _lag_samples(dt, history, lag_step):
    if history < 0 or lag_step <= 0:
        raise ValueError("history must be nonnegative and lag_step must be positive.")
    lag_seconds = np.arange(0.0, history + 0.5 * lag_step, lag_step)
    samples = np.unique(np.rint(lag_seconds / dt).astype(int))
    return samples, samples.astype(float) * dt


def build_forecast_dataset(
    signals,
    *,
    target="photometry",
    horizon=1.0,
    history=5.0,
    lag_step=0.5,
    target_window=1.0,
):
    """Build past-only features at time t and a target beginning at t + horizon."""
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {sorted(VALID_TARGETS)}.")
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and nonnegative.")
    if target in LICK_TARGETS and (not np.isfinite(target_window) or target_window <= 0):
        raise ValueError("target_window must be positive for lick targets.")

    time = np.asarray(signals["time"], dtype=float)
    if time.ndim != 1 or len(time) < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("signals['time'] must be a strictly increasing vector.")
    dt = float(np.median(np.diff(time)))
    names = ("photometry", "locomotion", "licking", "cue", "solenoid")
    arrays = {name: np.asarray(signals[name], dtype=float) for name in names}
    if any(array.shape != time.shape for array in arrays.values()):
        raise ValueError("Every forecast signal must match signals['time'].")

    lag_samples, lag_seconds = _lag_samples(dt, history, lag_step)
    horizon_samples = int(np.ceil(horizon / dt))
    window_samples = max(1, int(np.ceil(target_window / dt)))
    target_extent = horizon_samples + (window_samples if target in LICK_TARGETS else 1)
    anchors = np.arange(int(lag_samples.max()), len(time) - target_extent + 1)
    if len(anchors) < 10:
        raise ValueError("Too few samples remain after applying history and forecast windows.")

    columns = []
    feature_names = []
    feature_groups = {}
    for name in names:
        group_indices = []
        for lag_index, lag in enumerate(lag_samples):
            group_indices.append(len(columns))
            columns.append(arrays[name][anchors - lag])
            feature_names.append(f"{name}[t-{lag_seconds[lag_index]:g}s]")
        feature_groups[name] = np.asarray(group_indices, dtype=int)
    X = np.column_stack(columns)

    target_start = anchors + horizon_samples
    if target in CONTINUOUS_TARGETS:
        y = arrays[target][target_start]
    else:
        counts = np.asarray(
            [np.sum(arrays["licking"][start : start + window_samples]) for start in target_start],
            dtype=float,
        )
        y = (counts > 0).astype(float) if target == "lick_binary" else counts

    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return {
        "X": X[finite],
        "y": y[finite],
        "anchor_time": time[anchors][finite],
        "target_time": time[target_start][finite],
        "feature_names": tuple(feature_names),
        "feature_groups": feature_groups,
        "target": target,
        "horizon": float(horizon),
        "history": float(history),
        "target_window": float(target_window),
        "dt": dt,
    }


def get_model_columns(dataset):
    """Return history-only, cross-modal, and combined feature indices."""
    target = dataset["target"]
    groups = dataset["feature_groups"]
    target_group = {
        "photometry": "photometry",
        "locomotion": "locomotion",
        "lick_binary": "licking",
        "lick_count": "licking",
    }[target]
    cross_groups = [name for name in groups if name != target_group]
    history_columns = groups[target_group]
    cross_columns = np.concatenate([groups[name] for name in cross_groups])
    return {
        "history_only": history_columns,
        "cross_modal": cross_columns,
        "combined": np.concatenate((history_columns, cross_columns)),
    }


def make_forward_folds(time, *, n_folds=5, gap_seconds=0.0, initial_train_fraction=0.5):
    """Create expanding-window folds whose training samples precede testing."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 4 or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing vector with four samples.")
    if n_folds < 1:
        raise ValueError("n_folds must be positive.")
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be nonnegative.")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must lie between zero and one.")

    first_test = int(np.ceil(len(time) * initial_train_fraction))
    test_candidates = np.arange(first_test, len(time))
    if len(test_candidates) < n_folds:
        raise ValueError("Not enough samples for the requested forward folds.")

    folds = []
    for test_indices in np.array_split(test_candidates, n_folds):
        cutoff = time[test_indices[0]] - gap_seconds
        train_indices = np.flatnonzero(time < cutoff)
        if len(train_indices) < 2:
            raise ValueError("The forecast gap leaves too few initial training samples.")
        folds.append((train_indices, test_indices))
    return folds


def _fit_estimator(target, alpha, X_train, y_train):
    try:
        from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
    except ModuleNotFoundError as error:
        raise ImportError(
            "Forecasting requires scikit-learn. Install with: "
            "pip install -e '.[forecasting]'"
        ) from error

    if target in CONTINUOUS_TARGETS:
        return Ridge(alpha=alpha).fit(X_train, y_train)
    if target == "lick_binary":
        if len(np.unique(y_train)) < 2:
            return None
        strength = 1.0 / max(float(alpha), np.finfo(float).eps)
        return LogisticRegression(C=strength, max_iter=2000).fit(X_train, y_train)
    return PoissonRegressor(alpha=alpha, max_iter=1000).fit(X_train, y_train)


def _predict_estimator(model, target, X_test):
    if target == "lick_binary":
        return model.predict_proba(X_test)[:, 1]
    return model.predict(X_test)


def score_forecast(y_true, prediction, target):
    """Calculate target-appropriate held-out forecast metrics."""
    from sklearn.metrics import (
        average_precision_score,
        log_loss,
        mean_absolute_error,
        mean_poisson_deviance,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(prediction)
    y_true = y_true[valid]
    prediction = prediction[valid]
    if len(y_true) == 0:
        raise ValueError("No finite held-out predictions were produced.")

    if target in CONTINUOUS_TARGETS:
        return {
            "r2": float(r2_score(y_true, prediction)),
            "mse": float(mean_squared_error(y_true, prediction)),
            "mae": float(mean_absolute_error(y_true, prediction)),
        }
    if target == "lick_binary":
        clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        two_classes = len(np.unique(y_true)) == 2
        return {
            "average_precision": (
                float(average_precision_score(y_true, clipped)) if two_classes else np.nan
            ),
            "roc_auc": float(roc_auc_score(y_true, clipped)) if two_classes else np.nan,
            "log_loss": float(log_loss(y_true, clipped, labels=[0.0, 1.0])),
        }

    clipped = np.clip(prediction, 1e-7, None)
    deviance = float(mean_poisson_deviance(y_true, clipped))
    return {
        "poisson_deviance": deviance,
        "negative_poisson_deviance": -deviance,
        "mae": float(mean_absolute_error(y_true, clipped)),
    }


def fit_forecast_models(
    dataset,
    *,
    n_folds=5,
    alpha=1.0,
    gap_seconds=None,
    initial_train_fraction=0.5,
):
    """Fit three model comparisons with train-only scaling and forward CV."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as error:
        raise ImportError(
            "Forecasting requires scikit-learn. Install with: "
            "pip install -e '.[forecasting]'"
        ) from error

    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")
    if gap_seconds is None:
        gap_seconds = dataset["history"] + dataset["horizon"]
        if dataset["target"] in LICK_TARGETS:
            gap_seconds += dataset["target_window"]
    folds = make_forward_folds(
        dataset["anchor_time"],
        n_folds=n_folds,
        gap_seconds=gap_seconds,
        initial_train_fraction=initial_train_fraction,
    )
    model_columns = get_model_columns(dataset)
    results = {}
    for model_name in MODEL_ORDER:
        columns = model_columns[model_name]
        X = dataset["X"][:, columns]
        prediction = np.full(len(dataset["y"]), np.nan, dtype=float)
        coefficients = []
        completed_folds = 0
        for train_indices, test_indices in folds:
            scaler = StandardScaler().fit(X[train_indices])
            X_train = scaler.transform(X[train_indices])
            X_test = scaler.transform(X[test_indices])
            model = _fit_estimator(
                dataset["target"], alpha, X_train, dataset["y"][train_indices]
            )
            if model is None:
                continue
            prediction[test_indices] = _predict_estimator(
                model, dataset["target"], X_test
            )
            coefficients.append(np.asarray(model.coef_, dtype=float).reshape(-1))
            completed_folds += 1
        valid = np.isfinite(prediction)
        if not np.any(valid):
            raise ValueError(f"No forecast folds could fit model {model_name!r}.")
        results[model_name] = {
            "prediction": prediction,
            "metrics": score_forecast(
                dataset["y"][valid], prediction[valid], dataset["target"]
            ),
            "feature_names": tuple(dataset["feature_names"][index] for index in columns),
            "coefficients": coefficients,
            "n_test": int(np.sum(valid)),
            "n_folds": completed_folds,
        }
    return {
        "models": results,
        "gap_seconds": float(gap_seconds),
        "folds": folds,
    }


def primary_metric(target):
    return {
        "photometry": "r2",
        "locomotion": "r2",
        "lick_binary": "average_precision",
        "lick_count": "negative_poisson_deviance",
    }[target]


def forecast_manifest(
    sessions,
    data_root,
    *,
    target,
    horizons,
    dt=0.1,
    history=5.0,
    lag_step=0.5,
    target_window=1.0,
    channel="manifest",
    photometry_source="raw465",
    n_folds=5,
    alpha=1.0,
    gap_seconds=None,
    initial_train_fraction=0.5,
):
    """Run leakage-safe within-session forecasts for every manifest session."""
    rows = []
    for info in sessions:
        selected_channel = resolve_session_channel(info, channel)
        path = processed_session_path(data_root, info)
        session = load_session(path)
        signals = prepare_forecast_signals(
            session,
            dt=dt,
            channel=selected_channel,
            photometry_source=photometry_source,
        )
        for horizon in horizons:
            dataset = build_forecast_dataset(
                signals,
                target=target,
                horizon=horizon,
                history=history,
                lag_step=lag_step,
                target_window=target_window,
            )
            fitted = fit_forecast_models(
                dataset,
                n_folds=n_folds,
                alpha=alpha,
                gap_seconds=gap_seconds,
                initial_train_fraction=initial_train_fraction,
            )
            for model_name, model_result in fitted["models"].items():
                rows.append(
                    {
                        **info,
                        "path": str(path),
                        "target": target,
                        "channel": selected_channel,
                        "horizon": float(horizon),
                        "model": model_name,
                        "n_samples": len(dataset["y"]),
                        "n_test": model_result["n_test"],
                        "n_folds": model_result["n_folds"],
                        "gap_seconds": fitted["gap_seconds"],
                        **model_result["metrics"],
                    }
                )
    return rows


def summarize_forecasts(rows, metric):
    """Average session metrics within mouse, then summarize across mice."""
    mouse_groups = defaultdict(list)
    for row in rows:
        mouse_groups[(row["mouse"], row["horizon"], row["model"])].append(row)

    mouse_rows = []
    for (mouse, horizon, model), session_rows in mouse_groups.items():
        values = np.asarray([row.get(metric, np.nan) for row in session_rows], dtype=float)
        mouse_rows.append(
            {
                "mouse": mouse,
                "horizon": horizon,
                "model": model,
                "metric": metric,
                "value": float(np.nanmean(values)),
                "n_sessions": len(session_rows),
            }
        )

    group_groups = defaultdict(list)
    for row in mouse_rows:
        group_groups[(row["horizon"], row["model"])].append(row["value"])
    group_rows = []
    for (horizon, model), values in group_groups.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        sem = np.nan
        if len(finite) > 1:
            sem = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
        group_rows.append(
            {
                "horizon": horizon,
                "model": model,
                "metric": metric,
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "sem": sem,
                "n_mice": len(finite),
            }
        )
    return mouse_rows, group_rows
