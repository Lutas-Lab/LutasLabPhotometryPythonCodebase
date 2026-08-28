import numpy as np

def find_ttl_pulses(ttl,threshold=1.5,min_width=None, max_width=None):
    """
    find complete TTL-high pulses.
    Returns
    -------
    rising : np.ndarray
        Sample indices of valid rising edges
    falling : npndarray
        Sample indices of valid falling edges
    """
    
    ttl_high = ttl>threshold
    rising = np.where(np.diff(ttl_high.astype(int))==1)[0]+1
    falling = np.where(np.diff(ttl_high.astype(int))==-1)[0]+1

    pairs=[]
    j=0
    for r in rising:
        while j<len(falling) and falling[j]<r:
            j+= 1
        if j < len(falling):
            f=falling[j]     
            pairs.append((r,f))
            j+=1

    rising_valid=np.array([p[0] for p in pairs])
    falling_valid=np.array([p[1] for p in pairs])

    widths = falling_valid-rising_valid
    good = np.ones(len(widths),dtype=bool)
    if min_width is not None:
        good &= widths >=min_width

    if max_width is not None:
        good &= widths <= max_width

    return rising_valid[good],falling_valid[good]
    
def preprocess_photometry(
    raw_photo,
    ttl_465,
    ttl_405,
    timestamps,
    edge=3
):
    """
    Separate interleaved 465 nm and 405 nm photometry signals
    from one photoreceiver channel.

    Parameters
    ----------
    raw_photo : array-like
        Raw photoreceiver signal.

    ttl_465 : array-like
        TTL signal indicating when the 465 nm LED is on.

    ttl_405 : array-like
        TTL signal indicating when the 405 nm LED is on.

    timestamps : array-like
        Timestamp for each raw photometry sample.

    edge : int
        Number of samples to exclude from the beginning and end
        of each LED pulse.

    Returns
    -------
    photo_time_465 : np.ndarray
        Timestamp of each 465 nm measurement.

    photo_465 : np.ndarray
        Extracted 465 nm photometry values.

    photo_time_405 : np.ndarray
        Timestamp of each 405 nm measurement.

    photo_405 : np.ndarray
        Extracted 405 nm photometry values.
    """

    rising_465, falling_465 = find_ttl_pulses(ttl_465)
    rising_405, falling_405 = find_ttl_pulses(ttl_405)

    photo_465 = []
    photo_time_465 = []

    photo_405 = []
    photo_time_405 = []

    # Extract 465 nm measurements
    for r, f in zip(rising_465, falling_465):

        start = r + edge
        stop = f - edge

        photo_465.append(
            np.median(raw_photo[start:stop])
        )

        photo_time_465.append(
            np.mean(timestamps[start:stop])
        )

    # Extract 405 nm measurements
    for r, f in zip(rising_405, falling_405):

        start = r + edge
        stop = f - edge

        photo_405.append(
            np.median(raw_photo[start:stop])
        )

        photo_time_405.append(
            np.mean(timestamps[start:stop])
        )

    return (
        np.array(photo_time_465),
        np.array(photo_465),
        np.array(photo_time_405),
        np.array(photo_405)
    )

def align_reference_to_experimental(
    reference_time,
    reference_signal,
    experimental_time
):
    """
    Interpolate the reference photometry signal onto the
    timestamps of the experimental photometry signal.
    """

    reference_time = np.asarray(reference_time)
    reference_signal = np.asarray(reference_signal)
    experimental_time = np.asarray(experimental_time)

    aligned_reference = np.interp(
        experimental_time,
        reference_time,
        reference_signal
    )

    return aligned_reference

def irls_dff(
    exp_signal,
    iso_signal,
    irls_constant=1.4,
    max_iter=50,
    tolerance=1e-8
):
    """
    Compute artifact-corrected dF/F using robust IRLS regression.

    Python implementation of the approach used by IRLS_dFF.m:
    robust bisquare regression of the isosbestic signal onto
    the experimental signal.

    Parameters
    ----------
    exp_signal : array-like
        Experimental signal (e.g. 465 nm).

    iso_signal : array-like
        Isosbestic/reference signal (e.g. aligned 405 nm).

    irls_constant : float
        Bisquare tuning constant. Suggested value = 1.4.

    max_iter : int
        Maximum number of IRLS iterations.

    tolerance : float
        Convergence criterion.

    Returns
    -------
    dff : np.ndarray
        Artifact-corrected dF/F.

    fitted_iso_signal : np.ndarray
        Robust regression fit of the isosbestic signal.
    """

    exp_signal = np.asarray(
        exp_signal,
        dtype=float
    ).squeeze()

    iso_signal = np.asarray(
        iso_signal,
        dtype=float
    ).squeeze()

    # Check input lengths
    if len(exp_signal) != len(iso_signal):
        raise ValueError(
            "exp_signal and iso_signal must have the same length."
        )

    # Check for invalid values
    if (
        not np.all(np.isfinite(exp_signal))
        or not np.all(np.isfinite(iso_signal))
    ):
        raise ValueError(
            "Signals contain NaN or infinite values."
        )

    if irls_constant <= 0:
        raise ValueError(
            "irls_constant must be greater than zero."
        )

    # Design matrix:
    #
    # experimental = intercept + slope * isosbestic
    X = np.column_stack([
        np.ones(len(iso_signal)),
        iso_signal
    ])

    y = exp_signal

    # Initial ordinary least-squares fit
    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )[0]

    # Leverage values
    X_pinv = np.linalg.pinv(X)

    leverage = np.sum(
        X * X_pinv.T,
        axis=1
    )

    # Prevent numerical problems when leverage approaches 1
    leverage = np.minimum(
        leverage,
        0.9999
    )

    # Number of fitted coefficients
    p = X.shape[1]

    # -------------------------
    # IRLS iterations
    # -------------------------

    for _ in range(max_iter):

        residuals = y - X @ beta

        # Absolute deviation from median residual
        residual_median = np.median(residuals)

        abs_deviation = np.abs(
            residuals - residual_median
        )

        # MATLAB robustfit excludes the smallest p
        # absolute deviations when estimating MAD.
        sorted_deviation = np.sort(abs_deviation)

        mad = np.median(
            sorted_deviation[p:]
        )

        scale = mad / 0.6745

        # Avoid division by zero
        if scale <= np.finfo(float).eps:
            break

        # Adjust residuals for leverage
        adjusted_residuals = (
            residuals /
            np.sqrt(1 - leverage)
        )

        # Standardized residuals
        u = (
            adjusted_residuals /
            (irls_constant * scale)
        )

        # Tukey bisquare weights
        weights = np.zeros_like(u)

        inside = np.abs(u) < 1

        weights[inside] = (
            1 - u[inside] ** 2
        ) ** 2

        # Weighted least squares
        sqrt_weights = np.sqrt(weights)

        X_weighted = (
            X * sqrt_weights[:, None]
        )

        y_weighted = (
            y * sqrt_weights
        )

        beta_new = np.linalg.lstsq(
            X_weighted,
            y_weighted,
            rcond=None
        )[0]

        # Test convergence
        change = np.linalg.norm(
            beta_new - beta
        )

        scale_beta = max(
            np.linalg.norm(beta),
            1.0
        )

        beta = beta_new

        if change <= tolerance * scale_beta:
            break

    # Robust fitted isosbestic signal
    fitted_iso_signal = X @ beta

    # dF/F
    dff = (
        exp_signal - fitted_iso_signal
    ) / fitted_iso_signal

    return dff, fitted_iso_signal
    
def preprocess_locomotion(
    locomotion,
    ttl,
    timestamps,
    threshold=1.5,
    min_width=4,
    max_width=6,
    invert=True
):
    """
    Preprocess locomotion and align it to locomotion TTL pulses.

    Parameters
    ----------
    locomotion : array-like
        Raw locomotion signal.

    ttl : array-like
        TTL pulses corresponding to locomotion samples.

    timestamps : array-like
        Acquisition timestamps.

    threshold : float
        TTL detection threshold.

    min_width : int
        Minimum TTL pulse width in samples.

    max_width : int
        Maximum TTL pulse width in samples.

    invert : bool
        If True, invert the locomotion signal polarity.
        Default is True.

    Returns
    -------
    loco_time : np.ndarray
        Timestamp of each locomotion sample.

    locomotion : np.ndarray
        Processed locomotion signal.
    """

    rising, falling = find_ttl_pulses(
        ttl,
        threshold=threshold,
        min_width=min_width,
        max_width=max_width
    )

    # Get time of each locomotion sample from TTL rising edges
    loco_time = timestamps[rising]

    locomotion = np.asarray(
        locomotion
    ).squeeze()

    n_loco = len(locomotion)
    n_ttl = len(loco_time)

    difference = n_loco - n_ttl

    # Correct small mismatch between locomotion samples
    # and TTL pulses
    if difference != 0:

        print(
            f"Locomotion/TTL mismatch: "
            f"{n_loco} locomotion samples vs "
            f"{n_ttl} TTL pulses "
            f"(difference = {difference})"
        )

        old_x = np.linspace(
            0,
            1,
            n_loco
        )

        new_x = np.linspace(
            0,
            1,
            n_ttl
        )

        locomotion = np.interp(
            new_x,
            old_x,
            locomotion
        )

    else:
        print(
            "locomotion and TTL counts match."
        )

    # Invert locomotion polarity to match
    # behavioral convention
    if invert:
        locomotion = -locomotion

    return loco_time, locomotion

def preprocess_visual_cue(
    session,
    threshold=1.5,
    max_pulse_gap=0.5
):
    """
    Group individual visual-cue TTL pulses into cue presentations.

    Pulses separated by less than max_pulse_gap seconds
    are considered part of the same cue.
    """

    cue_signal = session["visual_cue"]
    timestamps = session["timestamps"]

    # Use existing TTL pulse detector
    rising, falling = find_ttl_pulses(
        cue_signal,
        threshold=threshold
    )

    # Return empty arrays if no pulses were detected
    if len(rising) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert individual pulse edges to timestamps
    pulse_onsets = timestamps[rising]
    pulse_offsets = timestamps[falling]

    # Find the gaps between consecutive pulses
    pulse_gaps = pulse_onsets[1:] - pulse_offsets[:-1]

    # A large gap indicates the start of a new cue
    new_cue = np.concatenate([
        [True],
        pulse_gaps > max_pulse_gap
    ])

    cue_id = np.cumsum(new_cue) - 1

    cue_onset = []
    cue_offset = []

    # Collapse each group of pulses into one cue
    for i in range(cue_id[-1] + 1):
        this_cue = cue_id == i

        cue_onset.append(pulse_onsets[this_cue][0])
        cue_offset.append(pulse_offsets[this_cue][-1])

    cue_onset = np.array(cue_onset)
    cue_offset = np.array(cue_offset)
    cue_duration = cue_offset - cue_onset

    return cue_onset, cue_offset, cue_duration

def preprocess_solenoid_opening(
    session,
    threshold=1.5
):
    """
    Detect solenoid opening events.

    Each TTL pulse is treated as one solenoid opening event.

    Parameters
    ----------
    session : dict
        Loaded session containing:
            session["solenoid_opening"]
            session["timestamps"]

    threshold : float
        Voltage threshold used to detect TTL pulses.

    Returns
    -------
    solenoid_onset : np.ndarray
        Timestamp of each solenoid opening.

    solenoid_offset : np.ndarray
        Timestamp of each solenoid closing.

    solenoid_duration : np.ndarray
        Duration of each solenoid opening.
    """

    solenoid_signal = session["solenoid_opening"]
    timestamps = session["timestamps"]

    # Reuse the existing TTL detector
    rising, falling = find_ttl_pulses(
        solenoid_signal,
        threshold=threshold
    )

    # Convert sample indices to timestamps
    solenoid_onset = timestamps[rising]
    solenoid_offset = timestamps[falling]

    # Calculate opening duration
    solenoid_duration = solenoid_offset - solenoid_onset

    return (
        solenoid_onset,
        solenoid_offset,
        solenoid_duration
    )

def preprocess_licking(
    session,
    threshold=1.5
):
    """
    Detect individual lick events.

    Each TTL pulse is treated as one lick.
    Pulse duration is ignored.

    Parameters
    ----------
    session : dict
        Loaded session containing:
            session["licking"]
            session["timestamps"]

    threshold : float
        Voltage threshold used to detect TTL pulses.

    Returns
    -------
    lick_times : np.ndarray
        Timestamp of each individual lick.
    """

    licking_signal = session["licking"]
    timestamps = session["timestamps"]

    # Reuse the existing TTL detector
    rising, falling = find_ttl_pulses(
        licking_signal,
        threshold=threshold
    )

    # We only care about the rising edge of each lick
    lick_times = timestamps[rising]

    return lick_times

def find_lick_bouts(
    lick_times,
    max_interlick_interval=1.0,
    min_licks=3
):
    """
    Group individual licks into lick bouts.

    Consecutive licks belong to the same bout when the
    interval between them is <= max_interlick_interval.

    Parameters
    ----------
    lick_times : array-like
        Timestamp of each individual lick.

    max_interlick_interval : float
        Maximum interval between consecutive licks for them
        to belong to the same bout, in seconds.

    min_licks : int
        Minimum number of licks required for a bout.

    Returns
    -------
    bout_onset : np.ndarray
        Timestamp of the first lick in each bout.

    bout_offset : np.ndarray
        Timestamp of the last lick in each bout.

    bout_duration : np.ndarray
        Duration of each bout.

    bout_lick_count : np.ndarray
        Number of licks in each bout.
    """

    lick_times = np.asarray(lick_times).squeeze()

    if len(lick_times) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([], dtype=int)
        )

    # Time between consecutive licks
    interlick_intervals = np.diff(lick_times)

    # A gap larger than the threshold starts a new bout
    new_bout = np.concatenate([
        [True],
        interlick_intervals > max_interlick_interval
    ])

    bout_id = np.cumsum(new_bout) - 1

    bout_onset = []
    bout_offset = []
    bout_lick_count = []

    for i in range(bout_id[-1] + 1):

        this_bout = bout_id == i
        this_bout_licks = lick_times[this_bout]

        # Ignore isolated/small groups
        if len(this_bout_licks) < min_licks:
            continue

        bout_onset.append(this_bout_licks[0])
        bout_offset.append(this_bout_licks[-1])
        bout_lick_count.append(len(this_bout_licks))

    bout_onset = np.array(bout_onset)
    bout_offset = np.array(bout_offset)
    bout_lick_count = np.array(bout_lick_count, dtype=int)

    bout_duration = bout_offset - bout_onset

    return (
        bout_onset,
        bout_offset,
        bout_duration,
        bout_lick_count
    )
    
def classify_cue_licking(
    cue_onset,
    cue_offset,
    lick_times,
    post_cue_window=2.0
):
    """
    Classify cue trials based on licking during the cue
    and during the period immediately following cue offset.

    Cue duration may vary across trials. The actual cue onset
    and offset are used for each trial.

    Parameters
    ----------
    cue_onset : array-like
        Cue onset timestamps.

    cue_offset : array-like
        Cue offset timestamps.

    lick_times : array-like
        Individual lick timestamps.

    post_cue_window : float
        Length of the post-cue response window in seconds.
        Default = 2 seconds.

    Returns
    -------
    cue_lick : np.ndarray of bool
        At least one lick occurred during the cue.

    post_cue_lick : np.ndarray of bool
        At least one lick occurred after cue offset and within
        the post-cue response window.

    cue_only : np.ndarray of bool
        Licking during cue but not post-cue.

    post_only : np.ndarray of bool
        Licking post-cue but not during cue.

    cue_and_post : np.ndarray of bool
        Licking occurred during both periods.

    cue_miss : np.ndarray of bool
        No licking during either period.
    """

    cue_onset = np.asarray(cue_onset)
    cue_offset = np.asarray(cue_offset)
    lick_times = np.asarray(lick_times)

    if len(cue_onset) != len(cue_offset):
        raise ValueError(
            "cue_onset and cue_offset must have the same length."
        )

    cue_lick = np.zeros(
        len(cue_onset),
        dtype=bool
    )

    post_cue_lick = np.zeros(
        len(cue_onset),
        dtype=bool
    )

    for i, (onset, offset) in enumerate(
        zip(cue_onset, cue_offset)
    ):

        # Licking during the actual cue period
        cue_lick[i] = np.any(
            (lick_times >= onset)
            &
            (lick_times <= offset)
        )

        # Licking during the 2 s after cue offset
        post_cue_lick[i] = np.any(
            (lick_times > offset)
            &
            (lick_times <= offset + post_cue_window)
        )

    # ----------------------------------------
    # Combined trial classifications
    # ----------------------------------------

    cue_only = (
        cue_lick
        &
        ~post_cue_lick
    )

    post_only = (
        ~cue_lick
        &
        post_cue_lick
    )

    cue_and_post = (
        cue_lick
        &
        post_cue_lick
    )

    cue_miss = (
        ~cue_lick
        &
        ~post_cue_lick
    )

    return (
        cue_lick,
        post_cue_lick,
        cue_only,
        post_only,
        cue_and_post,
        cue_miss
    )
    
def preprocess_session(session):
    """
    Run all preprocessing steps and add processed data
    to the session dictionary.
    """

    # -------------------------
    # Photometry channel 1
    # -------------------------

    (
        photo_time_465_ch1,
        photometry_465_ch1,
        photo_time_405_ch1,
        photometry_405_ch1
    ) = preprocess_photometry(
        session["raw_photometry_ch1"],
        session["ttl_465"],
        session["ttl_405"],
        session["timestamps"]
    )

    session["photo_time_465_ch1"] = photo_time_465_ch1
    session["photometry_465_ch1"] = photometry_465_ch1

    session["photo_time_405_ch1"] = photo_time_405_ch1
    session["photometry_405_ch1"] = photometry_405_ch1
    
    # Align 405 to 465 timebase
    photometry_405_aligned_ch1 = align_reference_to_experimental(
        photo_time_405_ch1,
        photometry_405_ch1,
        photo_time_465_ch1
    )
    
    session["photometry_405_aligned_ch1"] = photometry_405_aligned_ch1
    
    
    # IRLS correction
    dff_ch1, photometry_405_fitted_ch1 = irls_dff(
        photometry_465_ch1,
        photometry_405_aligned_ch1,
        irls_constant=1.4
    )
    
    session["photometry_405_fitted_ch1"] = photometry_405_fitted_ch1
    session["dff_ch1"] = dff_ch1

    # -------------------------
    # Photometry channel 2
    # -------------------------

    (
        photo_time_465_ch2,
        photometry_465_ch2,
        photo_time_405_ch2,
        photometry_405_ch2
    ) = preprocess_photometry(
        session["raw_photometry_ch2"],
        session["ttl_465"],
        session["ttl_405"],
        session["timestamps"]
    )

    session["photo_time_465_ch2"] = photo_time_465_ch2
    session["photometry_465_ch2"] = photometry_465_ch2

    session["photo_time_405_ch2"] = photo_time_405_ch2
    session["photometry_405_ch2"] = photometry_405_ch2

    # Align 405 to 465 timebase
    photometry_405_aligned_ch2 = align_reference_to_experimental(
        photo_time_405_ch2,
        photometry_405_ch2,
        photo_time_465_ch2
    )
    
    session["photometry_405_aligned_ch2"] = photometry_405_aligned_ch2
    
    
    # IRLS correction
    dff_ch2, photometry_405_fitted_ch2 = irls_dff(
        photometry_465_ch2,
        photometry_405_aligned_ch2,
        irls_constant=1.4
    )
    
    session["photometry_405_fitted_ch2"] = photometry_405_fitted_ch2
    session["dff_ch2"] = dff_ch2

    # -------------------------
    # Locomotion
    # -------------------------

    loco_time, locomotion = preprocess_locomotion(
        session["locomotion"],
        session["locomotion_ttlpulses"],
        session["timestamps"]
    )

    session["locomotion_time"] = loco_time
    session["processed_locomotion"] = locomotion


    # -------------------------
    # Visual cue
    # -------------------------

    cue_onset, cue_offset, cue_duration = preprocess_visual_cue(
        session
    )

    session["cue_onset"] = cue_onset
    session["cue_offset"] = cue_offset
    session["cue_duration"] = cue_duration


    # -------------------------
    # Solenoid opening
    # -------------------------

    (
        solenoid_onset,
        solenoid_offset,
        solenoid_duration
    ) = preprocess_solenoid_opening(
        session
    )

    session["solenoid_onset"] = solenoid_onset
    session["solenoid_offset"] = solenoid_offset
    session["solenoid_duration"] = solenoid_duration


    # -------------------------
    # Individual licks
    # -------------------------

    lick_times = preprocess_licking(
        session
    )

    session["lick_times"] = lick_times


    # -------------------------
    # Lick bouts
    # -------------------------

    (
        lick_bout_onset,
        lick_bout_offset,
        lick_bout_duration,
        lick_bout_lick_count
    ) = find_lick_bouts(
        lick_times,
        max_interlick_interval=1.0,
        min_licks=3
    )

    session["lick_bout_onset"] = lick_bout_onset
    session["lick_bout_offset"] = lick_bout_offset
    session["lick_bout_duration"] = lick_bout_duration
    session["lick_bout_lick_count"] = lick_bout_lick_count

    # -------------------------
    # Cue licking classification
    # -------------------------

    (
        cue_lick,
        post_cue_lick,
        cue_only,
        post_only,
        cue_and_post,
        cue_miss
    ) = classify_cue_licking(
        cue_onset,
        cue_offset,
        lick_times,
        post_cue_window=2.0
    )

    session["cue_lick"] = cue_lick
    session["post_cue_lick"] = post_cue_lick

    session["cue_only"] = cue_only
    session["post_only"] = post_only
    session["cue_and_post"] = cue_and_post
    session["cue_miss"] = cue_miss
    
    return session