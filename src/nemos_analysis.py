import numpy as np
import pynapple as nap
import nemos as nmo

from scipy.signal import convolve


# ============================================================
# General utilities
# ============================================================

def _validate_window(window):
    """
    Validate a temporal lag window.

    Negative lag:
        predictor occurs before response.

    Zero:
        predictor and response are simultaneous.

    Positive lag:
        predictor occurs after response.
    """

    if len(window) != 2:
        raise ValueError(
            "window must contain exactly two values."
        )

    start, end = map(
        float,
        window
    )

    if start >= end:
        raise ValueError(
            "window start must be less than window end."
        )

    return start, end


def _get_dt(time):
    """
    Estimate the median sampling interval.
    """

    time = np.asarray(
        time,
        dtype=float
    )

    if len(time) < 2:
        raise ValueError(
            "At least two timestamps are required."
        )

    dt = np.median(
        np.diff(time)
    )

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            "Invalid sampling interval."
        )

    return float(dt)


# ============================================================
# Photometry preparation
# ============================================================

def prepare_raw465_response(
    session,
    data,
    channel=1
):
    """
    Put raw demultiplexed 465 on the locomotion timebase.

    The response is:

        (F - median(F)) / median(F)

    No IRLS correction or high-pass filtering is applied.
    """

    glm_time = np.asarray(
        data["locomotion"].t,
        dtype=float
    )

    glm_ts = nap.Ts(
        t=glm_time,
        time_units="s"
    )

    raw_465 = nap.Tsd(
        t=session[
            f"photo_time_465_ch{channel}"
        ],
        d=session[
            f"photometry_465_ch{channel}"
        ],
        time_units="s"
    )

    raw_465_glm = raw_465.interpolate(
        glm_ts
    )

    f465 = np.asarray(
        raw_465_glm.d,
        dtype=float
    )

    f0 = np.nanmedian(
        f465
    )

    if not np.isfinite(f0) or f0 == 0:
        raise ValueError(
            "Invalid 465 reference value."
        )

    y = (
        f465 - f0
    ) / f0

    return (
        glm_time,
        y,
        f0
    )


def create_slow_time_basis(
    glm_time,
    n_basis_funcs=10
):
    """
    Create broad spline functions spanning the entire session.
    """

    glm_time = np.asarray(
        glm_time,
        dtype=float
    )

    duration = (
        glm_time[-1]
        - glm_time[0]
    )

    if duration <= 0:
        raise ValueError(
            "Time vector must span a positive duration."
        )

    normalized_time = (
        glm_time - glm_time[0]
    ) / duration

    basis = nmo.basis.BSplineEval(
        n_basis_funcs=n_basis_funcs,
        order=3,
        bounds=(0.0, 1.0),
        label="slow_time"
    )

    X = basis.compute_features(
        normalized_time
    )

    return (
        np.asarray(X, dtype=float),
        basis
    )


def estimate_slow_component(
    glm_time,
    y,
    n_basis_funcs=10
):
    """
    Estimate broad session-scale fluorescence.

    The slow component is retained separately rather than
    permanently subtracted from the original session.
    """

    X_slow, basis = create_slow_time_basis(
        glm_time,
        n_basis_funcs
    )

    valid = (
        np.all(
            np.isfinite(X_slow),
            axis=1
        )
        &
        np.isfinite(y)
    )

    model = nmo.glm.GLM(
        observation_model="Gaussian"
    )

    model.fit(
        X_slow[valid],
        y[valid]
    )

    slow = np.full(
        len(y),
        np.nan
    )

    slow[valid] = np.asarray(
        model.predict(
            X_slow[valid]
        )
    )

    residual = (
        y
        - slow
    )

    return {
        "slow": slow,
        "residual": residual,
        "model": model,
        "basis": basis,
        "X": X_slow,
        "valid_mask": valid
    }


def prepare_behavioral_response(
    session,
    data,
    channel=1,
    n_slow_basis=10
):
    """
    Prepare raw 465, slow component, and residual response.
    """

    (
        glm_time,
        y,
        f0
    ) = prepare_raw465_response(
        session,
        data,
        channel
    )

    decomposition = estimate_slow_component(
        glm_time,
        y,
        n_slow_basis
    )

    return {
        "glm_time": glm_time,
        "y": y,
        "slow_prediction": decomposition["slow"],
        "residual": decomposition["residual"],
        "slow_model": decomposition["model"],
        "slow_basis": decomposition["basis"],
        "X_slow": decomposition["X"],
        "valid_mask": decomposition["valid_mask"],
        "f0": f0,
        "channel": channel,
        "n_slow_basis": n_slow_basis
    }


# ============================================================
# Temporal predictor basis
# ============================================================

def create_temporal_basis(
    window,
    n_basis_funcs=6,
    label="predictor"
):
    """
    Create a raised-cosine evaluation basis over an explicit
    lag window.
    """

    start, end = _validate_window(
        window
    )

    basis = nmo.basis.RaisedCosineLinearEval(
        n_basis_funcs=n_basis_funcs,
        bounds=(start, end),
        label=label
    )

    return basis


def create_two_sided_temporal_design(
    signal,
    time,
    window=(-7.5, 7.5),
    n_basis_funcs=6,
    label="predictor"
):
    """
    Create a temporal design matrix for arbitrary lag windows.

    Negative lag:
        predictor precedes response.

    Positive lag:
        predictor follows response.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    time = np.asarray(
        time,
        dtype=float
    )

    if signal.ndim != 1:
        raise ValueError(
            "signal must be one-dimensional."
        )

    if len(signal) != len(time):
        raise ValueError(
            "signal and time must have the same length."
        )

    dt = _get_dt(
        time
    )

    start, end = _validate_window(
        window
    )

    start_samples = int(
        np.ceil(
            start / dt
        )
    )

    end_samples = int(
        np.floor(
            end / dt
        )
    )

    lag_samples = np.arange(
        start_samples,
        end_samples + 1
    )

    lag_times = (
        lag_samples * dt
    )

    basis = create_temporal_basis(
        window=window,
        n_basis_funcs=n_basis_funcs,
        label=label
    )

    basis_values = np.asarray(
        basis.evaluate(
            lag_times
        ),
        dtype=float
    )

    n_samples = len(signal)
    n_basis = basis_values.shape[1]

    X = np.full(
        (
            n_samples,
            n_basis
        ),
        np.nan
    )

    for basis_index in range(
        n_basis
    ):

        weights = basis_values[
            :,
            basis_index
        ]

        X[:, basis_index] = convolve(
            signal,
            weights[::-1],
            mode="same"
        )

    # --------------------------------------------------------
    # Invalidate samples whose requested lag window extends
    # outside the recording.
    # --------------------------------------------------------

    left_edge = max(
        0,
        -start_samples
    )

    right_edge = max(
        0,
        end_samples
    )

    if left_edge > 0:
        X[:left_edge, :] = np.nan

    if right_edge > 0:
        X[-right_edge:, :] = np.nan

    return {
        "X": X,
        "basis": basis,
        "lag_times": lag_times,
        "dt": dt,
        "window": window,
        "n_basis_funcs": n_basis_funcs
    }


# ============================================================
# Event timestamps -> GLM signal
# ============================================================

def event_times_to_signal(
    event_times,
    glm_time,
    mode="count"
):
    """
    Convert event timestamps to a common-timebase signal.

    mode="count":
        event count per GLM bin.

    mode="binary":
        1 if any event occurred in the bin.
    """

    glm_time = np.asarray(
        glm_time,
        dtype=float
    )

    if hasattr(
        event_times,
        "t"
    ):
        event_times = np.asarray(
            event_times.t,
            dtype=float
        )
    else:
        event_times = np.asarray(
            event_times,
            dtype=float
        )

    dt = _get_dt(
        glm_time
    )

    edges = np.concatenate(
        [
            [glm_time[0] - dt / 2],
            glm_time + dt / 2
        ]
    )

    counts, _ = np.histogram(
        event_times,
        bins=edges
    )

    counts = counts.astype(
        float
    )

    if mode == "count":
        return counts

    if mode == "binary":
        return (
            counts > 0
        ).astype(float)

    raise ValueError(
        "mode must be 'count' or 'binary'."
    )


# ============================================================
# Standard behavioral predictors
# ============================================================

def prepare_standard_predictors(
    session,
    data,
    glm_time
):
    """
    Prepare locomotion, licking, cue, and solenoid on the
    common GLM timebase.
    """

    glm_time = np.asarray(
        glm_time,
        dtype=float
    )

    locomotion = np.asarray(
        data["locomotion"].d,
        dtype=float
    )

    if len(locomotion) != len(glm_time):
        raise ValueError(
            "Locomotion and GLM timebase do not match."
        )

    licking = event_times_to_signal(
        session["lick_times"],
        glm_time,
        mode="count"
    )

    cue = event_times_to_signal(
        session["cue_onset"],
        glm_time,
        mode="binary"
    )

    solenoid = event_times_to_signal(
        session["solenoid_onset"],
        glm_time,
        mode="binary"
    )

    return {
        "locomotion": locomotion,
        "licking": licking,
        "cue": cue,
        "solenoid": solenoid
    }


def get_default_association_windows():
    """
    Default two-sided temporal-association windows.
    """

    return {
        "locomotion": (-7.5, 7.5),
        "licking": (-7.5, 7.5),
        "cue": (-5.0, 10.0),
        "solenoid": (-5.0, 10.0)
    }


def get_default_causal_windows():
    """
    Default causal/predictive windows.
    """

    return {
        "locomotion": (-15.0, 0.0),
        "licking": (-15.0, 0.0),
        "cue": (-15.0, 0.0),
        "solenoid": (-15.0, 0.0)
    }


# ============================================================
# Design matrix utilities
# ============================================================

def combine_design_matrices(
    *matrices
):
    """
    Horizontally concatenate design matrices.
    """

    if len(matrices) == 0:
        raise ValueError(
            "At least one matrix is required."
        )

    arrays = [
        np.asarray(
            matrix,
            dtype=float
        )
        for matrix in matrices
    ]

    n_samples = arrays[0].shape[0]

    for array in arrays:

        if array.ndim != 2:
            raise ValueError(
                "All matrices must be 2D."
            )

        if array.shape[0] != n_samples:
            raise ValueError(
                "All matrices must have the same "
                "number of rows."
            )

    return np.column_stack(
        arrays
    )


def get_valid_samples(
    X,
    y
):
    """
    Return rows with finite design matrix and response values.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.all(
            np.isfinite(X),
            axis=1
        )
        &
        np.isfinite(y)
    )

    return (
        X[valid],
        y[valid],
        valid
    )


# ============================================================
# Metrics
# ============================================================

def mse_score(
    y_true,
    y_pred
):
    """
    Mean squared error.
    """

    return np.mean(
        (
            np.asarray(y_true)
            - np.asarray(y_pred)
        ) ** 2
    )


def r2_score(
    y_true,
    y_pred
):
    """
    Coefficient of determination.
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - np.mean(y_true)
        ) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return (
        1
        - ss_res / ss_tot
    )


# ============================================================
# Ridge Gaussian GLM
# ============================================================

def fit_gaussian_glm(
    X,
    y,
    regularizer="Ridge",
    regularizer_strength=1.0
):
    """
    Fit a Gaussian GLM.

    By default uses Ridge regularization.
    """

    model = nmo.glm.GLM(
        observation_model="Gaussian",
        regularizer=regularizer,
        regularizer_strength=regularizer_strength
    )

    model.fit(
        np.asarray(
            X,
            dtype=float
        ),
        np.asarray(
            y,
            dtype=float
        )
    )

    return model


# ============================================================
# Blocked folds with temporal gap
# ============================================================

def make_blocked_folds(
    n_samples,
    n_folds=5
):
    """
    Create contiguous test blocks.
    """

    indices = np.arange(
        n_samples
    )

    blocks = np.array_split(
        indices,
        n_folds
    )

    return [
        (
            np.setdiff1d(
                indices,
                test_indices
            ),
            test_indices
        )
        for test_indices in blocks
    ]


def make_gapped_folds(
    n_samples,
    n_folds=5,
    gap_samples=0
):
    """
    Create contiguous test blocks and remove a temporal buffer
    around each test block from training.

    Parameters
    ----------
    gap_samples : int
        Number of samples excluded on each side of the test
        block.
    """

    if gap_samples < 0:
        raise ValueError(
            "gap_samples must be nonnegative."
        )

    indices = np.arange(
        n_samples
    )

    test_blocks = np.array_split(
        indices,
        n_folds
    )

    folds = []

    for test_indices in test_blocks:

        test_start = test_indices[0]
        test_end = test_indices[-1]

        gap_start = max(
            0,
            test_start - gap_samples
        )

        gap_end = min(
            n_samples - 1,
            test_end + gap_samples
        )

        train_mask = np.ones(
            n_samples,
            dtype=bool
        )

        train_mask[
            gap_start:gap_end + 1
        ] = False

        train_indices = indices[
            train_mask
        ]

        folds.append(
            (
                train_indices,
                test_indices
            )
        )

    return folds


def make_gapped_folds_from_indices(
    indices,
    n_folds=3,
    gap_samples=0
):
    """
    Create blocked folds when the available training samples
    are already represented by arbitrary global indices.

    This is used for nested CV.
    """

    indices = np.asarray(
        indices,
        dtype=int
    )

    blocks = np.array_split(
        indices,
        n_folds
    )

    folds = []

    for test_indices in blocks:

        train_mask = np.ones(
            len(indices),
            dtype=bool
        )

        # Exclude test block
        test_set = set(
            test_indices.tolist()
        )

        # Exclude samples within the temporal gap
        for position, global_index in enumerate(
            indices
        ):

            if (
                global_index
                in test_set
            ):
                train_mask[position] = False
                continue

            distance = np.min(
                np.abs(
                    test_indices
                    - global_index
                )
            )

            if distance <= gap_samples:

                train_mask[
                    position
                ] = False

        train_indices = indices[
            train_mask
        ]

        folds.append(
            (
                train_indices,
                test_indices
            )
        )

    return folds


# ============================================================
# Nested CV for ridge strength
# ============================================================

def select_ridge_strength(
    X,
    y,
    candidate_strengths,
    inner_folds=3,
    gap_samples=0
):
    """
    Select ridge strength using only the supplied training data.

    Selection is performed with blocked inner CV.

    Returns
    -------
    best_strength
    scores
    """

    n_samples = len(y)

    folds = make_gapped_folds(
        n_samples=n_samples,
        n_folds=inner_folds,
        gap_samples=gap_samples
    )

    scores = {}

    for strength in candidate_strengths:

        fold_mse = []

        for train_indices, test_indices in folds:

            if len(train_indices) == 0:
                continue

            model = fit_gaussian_glm(
                X[
                    train_indices
                ],
                y[
                    train_indices
                ],
                regularizer="Ridge",
                regularizer_strength=strength
            )

            prediction = np.asarray(
                model.predict(
                    X[
                        test_indices
                    ]
                )
            )

            fold_mse.append(
                mse_score(
                    y[
                        test_indices
                    ],
                    prediction
                )
            )

        if len(fold_mse) == 0:
            scores[strength] = np.nan
        else:
            scores[strength] = np.mean(
                fold_mse
            )

    valid_scores = {
        strength: score
        for strength, score in scores.items()
        if np.isfinite(score)
    }

    if not valid_scores:
        raise RuntimeError(
            "Unable to select a ridge strength."
        )

    best_strength = min(
        valid_scores,
        key=valid_scores.get
    )

    return (
        best_strength,
        scores
    )


# ============================================================
# Build behavioral design matrix
# ============================================================

def build_behavioral_design(
    glm_time,
    predictors,
    windows,
    n_basis_funcs=6
):
    """
    Build one combined temporal design matrix.

    Returns
    -------
    X
    bases
    """

    pieces = []
    bases = {}

    for name, signal in predictors.items():

        result = create_two_sided_temporal_design(
            signal=signal,
            time=glm_time,
            window=windows[name],
            n_basis_funcs=n_basis_funcs,
            label=name
        )

        pieces.append(
            result["X"]
        )

        bases[name] = result

    X = combine_design_matrices(
        *pieces
    )

    return (
        X,
        bases
    )


# ============================================================
# Nested-CV behavioral -> photometry
# ============================================================

def fit_behavior_to_photometry(
    response,
    predictor_signals,
    predictor_windows,
    predictor_basis_funcs=6,
    n_outer_folds=5,
    n_inner_folds=3,
    candidate_strengths=None,
    gap_seconds=10.0
):
    """
    Fit behavior -> residual photometry using nested blocked CV.

    Outer CV estimates held-out performance.

    Inner CV selects Ridge regularization strength.

    A temporal gap is applied around each outer test block.

    Parameters
    ----------
    response : dict
        Output from prepare_behavioral_response().

    predictor_signals : dict
        Common-timebase predictors.

    predictor_windows : dict
        Temporal windows for each predictor.

    predictor_basis_funcs : int
        Basis functions per predictor.

    candidate_strengths : sequence
        Ridge strengths.

    gap_seconds : float
        Exclusion buffer around each test block.

        This should be at least as large as the largest absolute
        lag in the model.
    """

    if candidate_strengths is None:

        candidate_strengths = [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0
        ]

    glm_time = np.asarray(
        response["glm_time"],
        dtype=float
    )

    y = np.asarray(
        response["residual"],
        dtype=float
    )

    X, bases = build_behavioral_design(
        glm_time=glm_time,
        predictors=predictor_signals,
        windows=predictor_windows,
        n_basis_funcs=predictor_basis_funcs
    )

    (
        X_valid,
        y_valid,
        valid_mask
    ) = get_valid_samples(
        X,
        y
    )

    valid_time = glm_time[
        valid_mask
    ]

    dt = _get_dt(
        valid_time
    )

    gap_samples = int(
        np.ceil(
            gap_seconds
            / dt
        )
    )

    outer_folds = make_gapped_folds(
        n_samples=len(y_valid),
        n_folds=n_outer_folds,
        gap_samples=gap_samples
    )

    fold_results = []

    oof_prediction = np.full(
        len(y_valid),
        np.nan
    )

    # --------------------------------------------------------
    # Outer CV
    # --------------------------------------------------------

    for fold_number, (
        outer_train,
        outer_test
    ) in enumerate(
        outer_folds,
        start=1
    ):

        X_train = X_valid[
            outer_train
        ]

        y_train = y_valid[
            outer_train
        ]

        X_test = X_valid[
            outer_test
        ]

        y_test = y_valid[
            outer_test
        ]

        # ----------------------------------------------------
        # Inner selection of ridge strength
        # ----------------------------------------------------

        (
            best_strength,
            inner_scores
        ) = select_ridge_strength(
            X=X_train,
            y=y_train,
            candidate_strengths=candidate_strengths,
            inner_folds=n_inner_folds,
            gap_samples=gap_samples
        )

        # ----------------------------------------------------
        # Refit on ALL outer-training data
        # with selected strength
        # ----------------------------------------------------

        model = fit_gaussian_glm(
            X_train,
            y_train,
            regularizer="Ridge",
            regularizer_strength=best_strength
        )

        prediction = np.asarray(
            model.predict(
                X_test
            )
        )

        oof_prediction[
            outer_test
        ] = prediction

        fold_results.append(
            {
                "fold": fold_number,
                "start_time": valid_time[
                    outer_test[0]
                ],
                "end_time": valid_time[
                    outer_test[-1]
                ],
                "n_test": len(
                    outer_test
                ),
                "r2": r2_score(
                    y_test,
                    prediction
                ),
                "mse": mse_score(
                    y_test,
                    prediction
                ),
                "ridge_strength": best_strength,
                "inner_scores": inner_scores,
                "coefficients": np.asarray(
                    model.coef_
                ).squeeze()
            }
        )

    return {
        "model": None,
        "X": X,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "valid_time": valid_time,
        "valid_mask": valid_mask,
        "oof_prediction": oof_prediction,
        "bases": bases,
        "predictors": tuple(
            predictor_signals.keys()
        ),
        "windows": predictor_windows,
        "fold_results": fold_results,
        "candidate_strengths": (
            candidate_strengths
        ),
        "gap_seconds": gap_seconds,
        "gap_samples": gap_samples,
        "n_outer_folds": n_outer_folds,
        "n_inner_folds": n_inner_folds,
        "mean_r2": np.mean(
            [
                result["r2"]
                for result in fold_results
            ]
        ),
        "median_r2": np.median(
            [
                result["r2"]
                for result in fold_results
            ]
        ),
        "mean_mse": np.mean(
            [
                result["mse"]
                for result in fold_results
            ]
        ),
        "median_mse": np.median(
            [
                result["mse"]
                for result in fold_results
            ]
        )
    }


# ============================================================
# Single-predictor convenience function
# ============================================================

def fit_single_behavior_predictor(
    response,
    predictors,
    predictor_name,
    window,
    n_basis_funcs=6,
    n_outer_folds=5,
    n_inner_folds=3,
    candidate_strengths=None,
    gap_seconds=None
):
    """
    Fit a regularized nested-CV model for one predictor.
    """

    if predictor_name not in predictors:
        raise ValueError(
            f"Unknown predictor: {predictor_name}"
        )

    if gap_seconds is None:
        gap_seconds = max(
            abs(window[0]),
            abs(window[1])
        )

    return fit_behavior_to_photometry(
        response=response,
        predictor_signals={
            predictor_name:
                predictors[predictor_name]
        },
        predictor_windows={
            predictor_name:
                window
        },
        predictor_basis_funcs=n_basis_funcs,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        candidate_strengths=candidate_strengths,
        gap_seconds=gap_seconds
    )


# ============================================================
# Fit all single predictors
# ============================================================

def fit_all_single_predictors(
    response,
    predictors,
    windows,
    n_basis_funcs=6,
    n_outer_folds=5,
    n_inner_folds=3,
    candidate_strengths=None
):
    """
    Fit each predictor separately using nested CV.
    """

    results = {}

    for name in predictors:

        results[name] = (
            fit_single_behavior_predictor(
                response=response,
                predictors=predictors,
                predictor_name=name,
                window=windows[name],
                n_basis_funcs=n_basis_funcs,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                candidate_strengths=(
                    candidate_strengths
                )
            )
        )

    return results


# ============================================================
# Full behavioral model
# ============================================================

def fit_full_behavior_model(
    response,
    predictors,
    windows,
    n_basis_funcs=6,
    n_outer_folds=5,
    n_inner_folds=3,
    candidate_strengths=None
):
    """
    Fit the complete regularized behavioral model.
    """

    gap_seconds = max(
        max(
            abs(window[0]),
            abs(window[1])
        )
        for window in windows.values()
    )

    return fit_behavior_to_photometry(
        response=response,
        predictor_signals=predictors,
        predictor_windows=windows,
        predictor_basis_funcs=n_basis_funcs,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        candidate_strengths=candidate_strengths,
        gap_seconds=gap_seconds
    )


# ============================================================
# Reduced behavioral models
# ============================================================

def fit_reduced_behavior_model(
    response,
    predictors,
    windows,
    exclude,
    n_basis_funcs=6,
    n_outer_folds=5,
    n_inner_folds=3,
    candidate_strengths=None
):
    """
    Fit the full model while excluding one predictor.
    """

    if exclude not in predictors:
        raise ValueError(
            f"Unknown predictor: {exclude}"
        )

    reduced_predictors = {
        name: signal
        for name, signal in predictors.items()
        if name != exclude
    }

    reduced_windows = {
        name: windows[name]
        for name in reduced_predictors
    }

    return fit_full_behavior_model(
        response=response,
        predictors=reduced_predictors,
        windows=reduced_windows,
        n_basis_funcs=n_basis_funcs,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        candidate_strengths=candidate_strengths
    )


def fit_all_reduced_models(
    response,
    predictors,
    windows,
    n_basis_funcs=6,
    n_outer_folds=5,
    n_inner_folds=3,
    candidate_strengths=None
):
    """
    Fit all one-predictor-excluded models.
    """

    results = {}

    for name in predictors:

        results[name] = (
            fit_reduced_behavior_model(
                response=response,
                predictors=predictors,
                windows=windows,
                exclude=name,
                n_basis_funcs=n_basis_funcs,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                candidate_strengths=(
                    candidate_strengths
                )
            )
        )

    return results


# ============================================================
# Full vs reduced comparison
# ============================================================

def compare_full_and_reduced(
    full_model,
    reduced_models
):
    """
    Compare the full model with each reduced model.

    Positive delta R2:
        removing the predictor hurts predictive performance.

    Positive delta MSE:
        removing the predictor increases prediction error.
    """

    comparison = {}

    full_r2 = full_model[
        "mean_r2"
    ]

    full_mse = full_model[
        "mean_mse"
    ]

    for name, reduced in reduced_models.items():

        comparison[name] = {
            "full_r2": full_r2,
            "reduced_r2": reduced[
                "mean_r2"
            ],
            "delta_r2": (
                full_r2
                - reduced["mean_r2"]
            ),
            "full_mse": full_mse,
            "reduced_mse": reduced[
                "mean_mse"
            ],
            "delta_mse": (
                reduced["mean_mse"]
                - full_mse
            )
        }

    return comparison


# ============================================================
# Reconstruct temporal kernel
# ============================================================

def reconstruct_temporal_kernel(
    basis,
    coefficients,
    lag_times
):
    """
    Reconstruct a temporal kernel from basis coefficients.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=float
    ).squeeze()

    basis_values = np.asarray(
        basis.evaluate(
            lag_times
        ),
        dtype=float
    )

    kernel = (
        basis_values
        @ coefficients
    )

    return (
        np.asarray(
            lag_times,
            dtype=float
        ),
        kernel
    )


# ============================================================
# Split coefficients by predictor
# ============================================================

def split_model_coefficients(
    coefficients,
    predictor_names,
    n_basis_funcs
):
    """
    Split concatenated model coefficients by predictor.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=float
    ).squeeze()

    expected = (
        len(predictor_names)
        * n_basis_funcs
    )

    if len(coefficients) != expected:
        raise ValueError(
            "Coefficient count does not match "
            "predictor count × basis functions."
        )

    output = {}

    start = 0

    for name in predictor_names:

        end = (
            start
            + n_basis_funcs
        )

        output[name] = (
            coefficients[
                start:end
            ]
        )

        start = end

    return output