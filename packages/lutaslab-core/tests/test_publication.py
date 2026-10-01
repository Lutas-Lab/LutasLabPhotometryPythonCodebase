import csv
import gzip
import json

import pytest

from lutaslab_core.publication import PublicationBundle, validate_publication_bundle


def test_build_and_validate_portable_bundle(tmp_path):
    root = tmp_path / "deposit"
    bundle = PublicationBundle(
        root,
        title="Example paper figure data",
        description="Values needed to regenerate the example figure.",
        creators=["Lutas Lab"],
        code_url="https://example.org/code",
        code_version="abc123",
    )
    table = bundle.add_table(
        "traces",
        [
            {"mouse": "m1", "time_s": 0.0, "value": 1.0},
            {"mouse": "m1", "time_s": 0.02, "value": 1.5},
        ],
        columns=["mouse", "time_s", "value"],
        description="Individual-mouse trace, not population summary only.",
        units={"time_s": "s", "value": "z-score"},
        primary_key=["mouse", "time_s"],
        panel_ids=["2A"],
    )
    model = bundle.add_json(
        "model_spec",
        {"normalization": "baseline z-score", "window_s": [-5, 15]},
        description="Parameters required to interpret the trace.",
        panel_ids=["2A"],
    )
    bundle.add_figure(
        "2A",
        title="Cue-aligned dopamine",
        sources=[table.relative_to(root), model.relative_to(root)],
        command="python reproduce.py --panel 2A",
    )
    bundle.finalize(readme_notes="Raw acquisition data are deposited separately.")

    result = validate_publication_bundle(root)
    assert result.valid, result.errors
    assert result.file_count == 2
    with gzip.open(table, "rt", encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream))[1]["value"] == "1.5"
    manifest = json.loads((root / "figure_manifest.json").read_text())
    assert manifest["figures"]["2A"]["sources"] == [
        "tables/traces.csv.gz",
        "metadata/model_spec.json",
    ]


def test_validation_detects_changed_data(tmp_path):
    root = tmp_path / "deposit"
    bundle = PublicationBundle(root, title="Test")
    table = bundle.add_table(
        "values",
        [{"id": 1, "value": 2.0}],
        columns=["id", "value"],
        description="Source values.",
    )
    bundle.finalize()
    table.write_bytes(table.read_bytes() + b"changed")

    result = validate_publication_bundle(root)
    assert not result.valid
    assert any("checksum mismatch" in error for error in result.errors)


def test_bundle_rejects_unsafe_and_duplicate_paths(tmp_path):
    bundle = PublicationBundle(tmp_path / "deposit", title="Test")
    bundle.add_json("spec", {}, description="Model specification.")
    with pytest.raises(ValueError, match="already exists"):
        bundle.add_json("spec", {}, description="Duplicate.")
    with pytest.raises(ValueError, match="contained"):
        bundle.add_figure(
            "1A",
            title="Unsafe",
            sources=["../outside.csv"],
            command="none",
        )


def test_empty_table_keeps_declared_schema(tmp_path):
    root = tmp_path / "deposit"
    bundle = PublicationBundle(root, title="Empty table")
    table = bundle.add_table(
        "excluded_sessions",
        [],
        columns=["mouse", "reason"],
        description="No exclusions in this dataset.",
    )
    bundle.finalize()

    with gzip.open(table, "rt", encoding="utf-8") as stream:
        assert stream.read().strip() == "mouse,reason"
    assert validate_publication_bundle(root).valid


def test_table_rejects_undeclared_columns(tmp_path):
    bundle = PublicationBundle(tmp_path / "deposit", title="Test")
    with pytest.raises(ValueError, match="undeclared columns"):
        bundle.add_table(
            "values",
            [{"id": 1, "undocumented": 2}],
            columns=["id"],
            description="Source values.",
        )
