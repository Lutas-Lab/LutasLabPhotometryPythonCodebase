"""Reprocess the 18-mouse Figure 4 CeA cohort into a PKA-ready dopamine input."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from lutaslab_core.perievent import extract_perievent_trials, normalize_trials
from src.load_data import get_session_paths, load_session_data
from src.preprocess import preprocess_session
from src.save_sessiondata import save_session


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _select_trials(
    session: dict,
    signal: np.ndarray,
    channel: int,
    *,
    window: tuple[float, float],
    dt: float,
    postcue_window: tuple[float, float],
    boundary_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cue_onsets = np.asarray(session["cue_onset"], dtype=float)
    lick_times = np.asarray(session["lick_times"], dtype=float)
    peri_time, trials, valid = extract_perievent_trials(
        np.asarray(session[f"photo_time_465_ch{channel}"], dtype=float),
        signal,
        cue_onsets,
        window=window,
        dt=dt,
    )
    trials = normalize_trials(peri_time, trials, "zscore", (window[0], 0.0))
    valid_cues = cue_onsets[valid]
    selected = np.asarray(
        [
            np.any(
                (lick_times - cue > postcue_window[0])
                & (lick_times - cue <= postcue_window[1] + boundary_tolerance)
            )
            for cue in valid_cues
        ],
        dtype=bool,
    )
    return peri_time, trials[selected], valid_cues[selected]


def _event_rate(
    event_times: np.ndarray,
    alignments: np.ndarray,
    peri_time: np.ndarray,
    dt: float,
) -> np.ndarray:
    edges = np.r_[peri_time - dt / 2, peri_time[-1] + dt / 2]
    return np.vstack(
        [np.histogram(event_times - alignment, bins=edges)[0] / dt for alignment in alignments]
    )


def _bootstrap_interval(
    traces: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, traces.shape[0], size=(n_bootstrap, traces.shape[0]))
    bootstrap_means = traces[indices].mean(axis=1)
    return tuple(np.percentile(bootstrap_means, (2.5, 97.5), axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deposited_mat", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--save-processed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processed_root = args.output / "processed"
    if args.save_processed:
        processed_root.mkdir(exist_ok=True)

    deposited = loadmat(args.deposited_mat, simplify_cells=True)["ConcatData"]
    mice = np.asarray(deposited["mice"], dtype=str)
    dates = np.asarray(deposited["exptdates"], dtype=str)
    runs_by_mouse = [np.atleast_1d(value).astype(int) for value in deposited["runs"]]
    legacy_names = np.asarray(deposited["nidaqfilename"], dtype=str)
    deposited_mouse_ids = np.asarray(deposited["mouseidnumlist"], dtype=int)
    deposited_runs = np.asarray(deposited["concatmicerunnum"], dtype=int)
    deposited_traces = np.asarray(deposited["concatmicephotom"], dtype=float)

    window = (-5.0, 15.0)
    input_params = deposited["InputParams"]
    trial_filter = str(input_params["trialstouse"])
    selection_length = float(input_params["cuelength"])
    if trial_filter == "CueLickingHits":
        selection_window = (0.0, selection_length)
    elif trial_filter == "PostCueLickingHits":
        selection_window = (selection_length, 2.0 * selection_length)
    else:
        raise ValueError(f"Unsupported deposited trial filter: {trial_filter!r}")
    manifest_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    mouse_raw_trials: dict[str, list[np.ndarray]] = defaultdict(list)
    mouse_legacy_trials: dict[str, list[np.ndarray]] = defaultdict(list)
    mouse_dff_trials: dict[str, list[np.ndarray]] = defaultdict(list)
    mouse_lick_rates: dict[str, list[np.ndarray]] = defaultdict(list)
    mouse_ensure_rates: dict[str, list[np.ndarray]] = defaultdict(list)
    peri_time_reference: np.ndarray | None = None

    for mouse_id, (mouse, date, runs, legacy_name) in enumerate(
        zip(mice, dates, runs_by_mouse, legacy_names, strict=True), start=1
    ):
        for run in runs:
            deposited_rows = np.flatnonzero(
                (deposited_mouse_ids == mouse_id) & (deposited_runs == run)
            )
            if deposited_rows.size == 0:
                audit_rows.append(
                    {
                        "mouse": mouse,
                        "date": date,
                        "run": int(run),
                        "deposited_trials": 0,
                        "raw_selected_trials": 0,
                        "channel_1_correlation": "",
                        "channel_2_correlation": "",
                        "selected_channel": "",
                        "status": "excluded_zero_deposited_trials",
                    }
                )
                continue

            paths = get_session_paths(mouse, date, int(run), args.raw_root)
            session = preprocess_session(load_session_data(paths))
            channel_trials: dict[
                int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ] = {}
            channel_scores: dict[int, float] = {}
            deposited_mean = deposited_traces[deposited_rows].mean(axis=0)
            for channel in (1, 2):
                peri_time, trials, selected_cues = _select_trials(
                    session,
                    np.asarray(session[f"photometry_465_ch{channel}"], dtype=float),
                    channel,
                    window=window,
                    dt=args.dt,
                    postcue_window=selection_window,
                    boundary_tolerance=0.0,
                )
                _, reference_trials, reference_cues = _select_trials(
                    session,
                    np.asarray(
                        session[f"photometry_405_aligned_ch{channel}"], dtype=float
                    ),
                    channel,
                    window=window,
                    dt=args.dt,
                    postcue_window=selection_window,
                    boundary_tolerance=0.0,
                )
                if not np.array_equal(selected_cues, reference_cues):
                    raise ValueError(
                        f"{mouse} {date} run {run}: 465 and 405 trials differ"
                    )
                legacy_trials = trials - reference_trials
                channel_trials[channel] = (
                    peri_time,
                    trials,
                    legacy_trials,
                    selected_cues,
                )
                if trials.shape[0] == deposited_rows.size:
                    channel_scores[channel] = _correlation(
                        deposited_mean, legacy_trials.mean(axis=0)
                    )
                else:
                    channel_scores[channel] = float("nan")

            finite_scores = {
                channel: score
                for channel, score in channel_scores.items()
                if np.isfinite(score)
            }
            if not finite_scores:
                raise ValueError(
                    f"{mouse} {date} run {run}: raw trial count does not match "
                    f"the {deposited_rows.size} deposited trials"
                )
            selected_channel = max(finite_scores, key=finite_scores.get)
            peri_time, raw_trials, legacy_trials, selected_cues = channel_trials[
                selected_channel
            ]
            _, dff_trials, dff_cues = _select_trials(
                session,
                np.asarray(session[f"dff_ch{selected_channel}"], dtype=float),
                selected_channel,
                window=window,
                dt=args.dt,
                postcue_window=selection_window,
                boundary_tolerance=0.0,
            )
            if not np.array_equal(selected_cues, dff_cues):
                raise ValueError(f"{mouse} {date} run {run}: raw and dF/F trials differ")

            lick_rate = _event_rate(
                np.asarray(session["lick_times"], dtype=float),
                selected_cues,
                peri_time,
                args.dt,
            )
            ensure_rate = _event_rate(
                np.asarray(session["solenoid_onset"], dtype=float),
                selected_cues,
                peri_time,
                args.dt,
            )
            mouse_raw_trials[mouse].append(raw_trials)
            mouse_legacy_trials[mouse].append(legacy_trials)
            mouse_dff_trials[mouse].append(dff_trials)
            mouse_lick_rates[mouse].append(lick_rate)
            mouse_ensure_rates[mouse].append(ensure_rate)
            peri_time_reference = peri_time

            legacy_path = (
                args.raw_root
                / mouse
                / f"{mouse}_{date}"
                / f"{mouse}-{date}-{int(run):03d}-{legacy_name}"
            )
            manifest_rows.append(
                {
                    "mouse": mouse,
                    "date": date,
                    "run": int(run),
                    "group": "CeA",
                    "condition": "trained_2s_cue",
                    "channel": selected_channel,
                }
            )
            audit_rows.append(
                {
                    "mouse": mouse,
                    "date": date,
                    "run": int(run),
                    "deposited_trials": int(deposited_rows.size),
                    "raw_selected_trials": int(raw_trials.shape[0]),
                    "channel_1_correlation": channel_scores[1],
                    "channel_2_correlation": channel_scores[2],
                    "selected_channel": selected_channel,
                    "status": "matched",
                    "raw_nidaq": paths["photometry_path"],
                    "running": paths["locomotion_path"],
                    "legacy_processed": legacy_path,
                }
            )
            if args.save_processed:
                session["channel"] = selected_channel
                save_session(session, processed_root)
            print(
                f"{mouse} {date} run {int(run)}: n={raw_trials.shape[0]}, "
                f"channel={selected_channel}, r1={channel_scores[1]:.3f}, "
                f"r2={channel_scores[2]:.3f}"
            )

    _write_csv(args.output / "sessions.csv", manifest_rows)
    _write_csv(args.output / "channel_audit.csv", audit_rows)
    if peri_time_reference is None:
        raise ValueError("No contributing sessions were found")

    raw_mouse_traces = np.vstack(
        [np.vstack(mouse_raw_trials[mouse]).mean(axis=0) for mouse in mice]
    )
    legacy_mouse_traces = np.vstack(
        [np.vstack(mouse_legacy_trials[mouse]).mean(axis=0) for mouse in mice]
    )
    dff_mouse_traces = np.vstack(
        [np.vstack(mouse_dff_trials[mouse]).mean(axis=0) for mouse in mice]
    )
    lick_mouse_rates = np.vstack(
        [np.vstack(mouse_lick_rates[mouse]).mean(axis=0) for mouse in mice]
    )
    ensure_mouse_rates = np.vstack(
        [np.vstack(mouse_ensure_rates[mouse]).mean(axis=0) for mouse in mice]
    )
    raw_low, raw_high = _bootstrap_interval(
        raw_mouse_traces, n_bootstrap=args.bootstrap, seed=args.seed
    )
    legacy_low, legacy_high = _bootstrap_interval(
        legacy_mouse_traces, n_bootstrap=args.bootstrap, seed=args.seed + 1
    )
    dff_low, dff_high = _bootstrap_interval(
        dff_mouse_traces, n_bootstrap=args.bootstrap, seed=args.seed + 2
    )
    np.savez_compressed(
        args.output / "figure4_dopamine_input.npz",
        time=peri_time_reference,
        mouse_names=mice,
        legacy_mouse_traces=legacy_mouse_traces,
        legacy_population_mean=legacy_mouse_traces.mean(axis=0),
        legacy_population_sem=legacy_mouse_traces.std(axis=0, ddof=1)
        / np.sqrt(len(mice)),
        legacy_population_ci_low=legacy_low,
        legacy_population_ci_high=legacy_high,
        raw465_mouse_traces=raw_mouse_traces,
        raw465_population_mean=raw_mouse_traces.mean(axis=0),
        raw465_population_sem=raw_mouse_traces.std(axis=0, ddof=1) / np.sqrt(len(mice)),
        raw465_population_ci_low=raw_low,
        raw465_population_ci_high=raw_high,
        dff_mouse_traces=dff_mouse_traces,
        dff_population_mean=dff_mouse_traces.mean(axis=0),
        dff_population_sem=dff_mouse_traces.std(axis=0, ddof=1) / np.sqrt(len(mice)),
        dff_population_ci_low=dff_low,
        dff_population_ci_high=dff_high,
        lick_rate_mouse_traces=lick_mouse_rates,
        ensure_rate_mouse_traces=ensure_mouse_rates,
        trial_counts=np.asarray(
            [sum(array.shape[0] for array in mouse_raw_trials[mouse]) for mouse in mice]
        ),
    )

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    for axis, traces, low, high, title, color in (
        (
            axes[0],
            legacy_mouse_traces,
            legacy_low,
            legacy_high,
            "Published-method dopamine input (Z465 - Z405)",
            "#6C3483",
        ),
        (
            axes[1],
            raw_mouse_traces,
            raw_low,
            raw_high,
            "Raw 465 sensitivity",
            "#2F73D5",
        ),
        (
            axes[2],
            dff_mouse_traces,
            dff_low,
            dff_high,
            "IRLS dF/F sensitivity",
            "#E67E22",
        ),
    ):
        mean = traces.mean(axis=0)
        axis.plot(peri_time_reference, mean, color=color, linewidth=1.6)
        axis.fill_between(peri_time_reference, low, high, color=color, alpha=0.2)
        axis.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
        axis.set_ylabel("Photometry\nZ-score")
        axis.set_title(title)
    axes[3].plot(peri_time_reference, lick_mouse_rates.mean(axis=0), color="black", label="Licking")
    axes[3].plot(
        peri_time_reference,
        ensure_mouse_rates.mean(axis=0),
        color="#9B59B6",
        label="Ensure TTL",
    )
    axes[3].set_ylabel("Events/s")
    axes[3].set_xlabel("Time from cue onset (s)")
    axes[3].legend(frameon=False)
    for axis in axes:
        axis.axvspan(0, 2, color="#CFE8F7", alpha=0.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "figure4_dopamine_input.png", dpi=250)
    plt.close(fig)

    summary = {
        "paper_figure": "Figure 4a-e (not Figure 2)",
        "mouse_count": int(len(mice)),
        "contributing_session_count": len(manifest_rows),
        "excluded_zero_trial_sessions": int(
            sum(row["status"] == "excluded_zero_deposited_trials" for row in audit_rows)
        ),
        "trial_count": int(sum(row["deposited_trials"] for row in audit_rows)),
        "deposited_trial_filter": trial_filter,
        "selection_rule": (
            f"lick >{selection_window[0]:g} and "
            f"<={selection_window[1]:g} seconds after cue onset"
        ),
        "window_seconds": list(window),
        "sample_interval_seconds": args.dt,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "notes": [
            "Each mouse contributes one mean trace, regardless of its trial count.",
            "Receiver channels were selected session-by-session by matching published-method Z465-Z405 traces to deposited session means.",
            "The NPZ preserves mouse traces and uncertainty for downstream PKA modeling.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
