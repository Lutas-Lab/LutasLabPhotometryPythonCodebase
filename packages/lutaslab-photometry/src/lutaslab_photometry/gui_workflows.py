"""Pure helpers shared by the Streamlit prototype and its tests."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

MANIFEST_COLUMNS = ("mouse", "date", "run", "group", "condition", "channel")


def normalize_manifest_rows(rows):
    """Validate editor rows and return CSV-ready dictionaries."""
    normalized = []
    seen = set()
    for index, raw_row in enumerate(rows, start=1):
        row = {key: raw_row.get(key, "") for key in MANIFEST_COLUMNS}
        if not any(str(value).strip() for value in row.values() if value is not None):
            continue

        mouse = str(row["mouse"] or "").strip()
        date = str(row["date"] or "").strip()
        run_text = str(row["run"] or "").strip()
        if not mouse or not date or not run_text:
            raise ValueError(f"Row {index} has an empty mouse, date, or run value.")
        if len(date) != 6 or not date.isdigit():
            raise ValueError(f"Row {index} date must contain six digits (YYMMDD).")
        try:
            run = int(run_text)
        except ValueError as error:
            raise ValueError(f"Row {index} run must be an integer.") from error
        if run < 0:
            raise ValueError(f"Row {index} run cannot be negative.")

        channel_text = str(row["channel"] or "1").strip()
        try:
            channel = int(channel_text)
        except ValueError as error:
            raise ValueError(f"Row {index} channel must be 1 or 2.") from error
        if channel not in (1, 2):
            raise ValueError(f"Row {index} channel must be 1 or 2.")

        identifier = (mouse, date, run)
        if identifier in seen:
            raise ValueError(
                f"Row {index} duplicates {mouse} {date} run {run}."
            )
        seen.add(identifier)
        normalized.append(
            {
                "mouse": mouse,
                "date": date,
                "run": run,
                "group": str(row["group"] or "").strip(),
                "condition": str(row["condition"] or "").strip(),
                "channel": channel,
            }
        )

    if not normalized:
        raise ValueError("Add at least one session to the manifest.")
    return normalized


def manifest_csv_text(rows):
    """Serialize validated manifest rows to CSV text."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalize_manifest_rows(rows))
    return output.getvalue()


def write_manifest(path, rows):
    """Atomically write validated rows and return the resolved manifest path."""
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = manifest_csv_text(rows)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(csv_text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


def _python_command(project_root, script_name):
    del project_root  # Retained for compatibility with callers that also set cwd.
    return [
        sys.executable,
        "-m",
        f"lutaslab_photometry.cli.{Path(script_name).stem}",
    ]


def build_preprocess_command(
    project_root,
    manifest,
    data_root,
    *,
    overwrite=False,
    continue_on_error=True,
    post_cue_window=2.0,
):
    """Build the maintained batch-preprocessing command without executing it."""
    command = _python_command(project_root, "run_preprocess_batch.py")
    command.extend(
        [
            "--manifest",
            str(Path(manifest)),
            "--data-root",
            str(Path(data_root)),
            "--post-cue-window",
            str(float(post_cue_window)),
        ]
    )
    if overwrite:
        command.append("--overwrite")
    if continue_on_error:
        command.append("--continue-on-error")
    return command


def build_psth_command(
    project_root,
    manifest,
    data_root,
    output_dir,
    *,
    event_key="cue_onset",
    signal="photometry",
    photometry_signal="dff",
    channel="manifest",
    window=(-5.0, 20.0),
    dt=0.02,
    normalization="zscore",
    baseline=(-5.0, 0.0),
    stratify=True,
    null_method="none",
    n_shuffles=500,
    seed=0,
    null_exclusion=0.0,
    trial_class="all",
    post_cue_window=2.0,
    heatmaps=False,
    heatmap_sort="event_order",
    heatmap_sort_window=None,
    heatmap_sort_direction="auto",
    heatmap_unmatched="bottom",
    heatmap_cmap="coolwarm",
    first_event_only=False,
    allow_partial_windows=False,
    group_filter=None,
    condition_filter=None,
):
    """Build the maintained event-aligned analysis command."""
    command = _python_command(project_root, "run_psth.py")
    command.extend(
        [
            "--manifest",
            str(Path(manifest)),
            "--data-root",
            str(Path(data_root)),
            "--output-dir",
            str(Path(output_dir)),
            "--event-key",
            str(event_key),
            "--signal",
            str(signal),
            "--photometry-signal",
            str(photometry_signal),
            "--channel",
            str(channel),
            "--window",
            str(float(window[0])),
            str(float(window[1])),
            "--dt",
            str(float(dt)),
            "--normalization",
            str(normalization),
            "--baseline",
            str(float(baseline[0])),
            str(float(baseline[1])),
            "--null-method",
            str(null_method),
            "--n-shuffles",
            str(int(n_shuffles)),
            "--seed",
            str(int(seed)),
            "--null-exclusion",
            str(float(null_exclusion)),
            "--trial-class",
            str(trial_class),
            "--post-cue-window",
            str(float(post_cue_window)),
        ]
    )
    command.append("--stratify" if stratify else "--no-stratify")
    if group_filter is not None:
        command.extend(["--group", str(group_filter)])
    if condition_filter is not None:
        command.extend(["--condition", str(condition_filter)])
    if first_event_only:
        command.append("--first-event-only")
    if allow_partial_windows:
        command.append("--allow-partial-windows")
    if heatmaps:
        command.extend(
            [
                "--heatmaps",
                "--heatmap-sort",
                str(heatmap_sort),
                "--heatmap-sort-direction",
                str(heatmap_sort_direction),
                "--heatmap-unmatched",
                str(heatmap_unmatched),
                "--heatmap-cmap",
                str(heatmap_cmap),
            ]
        )
        if heatmap_sort_window is not None:
            command.extend(
                [
                    "--heatmap-sort-window",
                    str(float(heatmap_sort_window[0])),
                    str(float(heatmap_sort_window[1])),
                ]
            )
    return command


def build_behavior_glm_command(
    project_root,
    manifest,
    data_root,
    output_dir,
    *,
    channel="manifest",
    photometry_source="raw465",
    history=5.0,
    lag_step=0.5,
    dt=0.1,
    folds=5,
    inner_folds=3,
    group_filter=None,
    condition_filter=None,
):
    """Build a leakage-safe contemporaneous behavioral photometry GLM."""
    command = _python_command(project_root, "run_forecasting.py")
    command.extend(
        [
            "--manifest",
            str(Path(manifest)),
            "--data-root",
            str(Path(data_root)),
            "--output-dir",
            str(Path(output_dir)),
            "--target",
            "photometry",
            "--horizons",
            "0.0",
            "--history",
            str(float(history)),
            "--lag-step",
            str(float(lag_step)),
            "--dt",
            str(float(dt)),
            "--channel",
            str(channel),
            "--photometry-source",
            str(photometry_source),
            "--folds",
            str(int(folds)),
            "--inner-folds",
            str(int(inner_folds)),
        ]
    )
    if group_filter is not None:
        command.extend(["--group", str(group_filter)])
    if condition_filter is not None:
        command.extend(["--condition", str(condition_filter)])
    return command


def build_lifetime_command(
    project_root,
    action,
    workflow,
    manifest,
    data_root,
    output_dir=None,
    *,
    signal=None,
    event=None,
    window=(-5.0, 20.0),
    dt=0.1,
    normalization="zscore",
    baseline=(-5.0, 0.0),
    heatmaps=True,
    heatmap_sort="event_order",
    heatmap_sort_window=None,
    heatmap_sort_direction="auto",
    heatmap_unmatched="bottom",
    heatmap_cmap="coolwarm",
    lick_kernel_seconds=10.0,
    ensure_kernel_seconds=20.0,
    first_event_only=False,
    allow_partial_windows=False,
    overwrite=False,
    group_filter=None,
    condition_filter=None,
    iflip3_fit_settings=None,
):
    """Build a FluoPulse or iFLiP3 workflow command."""

    if action not in {"preprocess", "psth", "glm"}:
        raise ValueError("action must be preprocess, psth, or glm")
    if workflow not in {"fluopulse", "iflip3"}:
        raise ValueError("workflow must be fluopulse or iflip3")
    command = _python_command(project_root, "run_lifetime_workflow.py")
    command.extend(
        [
            action,
            "--workflow",
            workflow,
            "--manifest",
            str(Path(manifest)),
            "--data-root",
            str(Path(data_root)),
        ]
    )
    if output_dir is not None:
        command.extend(["--output-dir", str(Path(output_dir))])
    if group_filter is not None:
        command.extend(["--group", str(group_filter)])
    if condition_filter is not None:
        command.extend(["--condition", str(condition_filter)])
    if action == "preprocess" and overwrite:
        command.append("--overwrite")
    if action == "preprocess" and workflow == "iflip3" and iflip3_fit_settings is not None:
        command.extend(
            [
                "--iflip3-fit-settings",
                json.dumps(iflip3_fit_settings, separators=(",", ":"), sort_keys=True),
            ]
        )
    if signal is not None:
        command.extend(["--signal", str(signal)])
    if action == "psth":
        command.extend(
            [
                "--event",
                str(event),
                "--window",
                str(float(window[0])),
                str(float(window[1])),
                "--dt",
                str(float(dt)),
                "--normalization",
                str(normalization),
                "--baseline",
                str(float(baseline[0])),
                str(float(baseline[1])),
                "--heatmap-sort",
                str(heatmap_sort),
                "--heatmap-sort-direction",
                str(heatmap_sort_direction),
                "--heatmap-unmatched",
                str(heatmap_unmatched),
                "--heatmap-cmap",
                str(heatmap_cmap),
            ]
        )
        command.append("--heatmaps" if heatmaps else "--no-heatmaps")
        if first_event_only:
            command.append("--first-event-only")
        if allow_partial_windows:
            command.append("--allow-partial-windows")
        if heatmap_sort_window is not None:
            command.extend(
                [
                    "--heatmap-sort-window",
                    str(float(heatmap_sort_window[0])),
                    str(float(heatmap_sort_window[1])),
                ]
            )
    if action == "glm":
        command.extend(
            [
                "--lick-kernel-seconds",
                str(float(lick_kernel_seconds)),
                "--ensure-kernel-seconds",
                str(float(ensure_kernel_seconds)),
            ]
        )
    return command


def display_command(command):
    """Render an argument list as a copyable platform-appropriate command."""
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(
    command,
    project_root,
    on_output: Callable[[str], None] | None = None,
):
    """Run one maintained workflow, streaming and capturing combined output."""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    output_parts = []
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=Path(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
        raise RuntimeError("Workflow output could not be captured.")
    with process.stdout:
        for line in process.stdout:
            output_parts.append(line)
            if on_output is not None:
                on_output(line)
    return process.wait(), "".join(output_parts)
