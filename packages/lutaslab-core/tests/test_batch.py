import shutil
from pathlib import Path

import pytest

from lutaslab_core.batch import run_batch


def test_batch_records_processed_skipped_and_failed():
    folder = Path("packages/lutaslab-core/tests/_batch_tmp")
    existing = folder / "existing.out"
    folder.mkdir(parents=True, exist_ok=True)
    existing.write_text("keep", encoding="utf-8")
    items = [{"id": "skip"}, {"id": "ok"}, {"id": "bad"}]

    def destination(item):
        return existing if item["id"] == "skip" else folder / f"{item['id']}.out"

    def process(item):
        if item["id"] == "bad":
            raise ValueError("broken")
        output = destination(item)
        output.write_text("done", encoding="utf-8")
        return output

    try:
        results = run_batch(
            items,
            process,
            destination_for=destination,
            identifier_for=lambda item: item["id"],
            continue_on_error=True,
        )
        assert [row["status"] for row in results] == ["skipped", "processed", "failed"]
        assert results[-1]["error"] == "broken"
        assert existing.read_text(encoding="utf-8") == "keep"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_batch_raises_contextual_error_by_default():
    with pytest.raises(RuntimeError, match="Failed to process bad"):
        run_batch(
            [{"id": "bad"}],
            lambda item: (_ for _ in ()).throw(ValueError("broken")),
            identifier_for=lambda item: item["id"],
        )
