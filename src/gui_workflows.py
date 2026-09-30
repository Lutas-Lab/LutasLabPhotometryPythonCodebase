"""Pure helpers shared by the Streamlit prototype and its tests."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
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
    """Write validated rows to a manifest and return its resolved path."""
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_csv_text(rows), encoding="utf-8")
    return manifest_path


def _python_command(project_root, script_name):
    return [
        sys.executable,
        str(Path(project_root).resolve() / "scripts" / script_name),
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
    channel="manifest",
    window=(-5.0, 20.0),
    dt=0.02,
    normalization="zscore",
    baseline=(-5.0, 0.0),
    stratify=True,
    null_method="none",
    n_shuffles=500,
    seed=0,
    trial_class="all",
    post_cue_window=2.0,
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
            "--trial-class",
            str(trial_class),
            "--post-cue-window",
            str(float(post_cue_window)),
        ]
    )
    command.append("--stratify" if stratify else "--no-stratify")
    return command


def display_command(command):
    """Render an argument list as a copyable platform-appropriate command."""
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command, project_root):
    """Run one maintained workflow and capture combined output."""
    result = subprocess.run(
        [str(part) for part in command],
        cwd=Path(project_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout
