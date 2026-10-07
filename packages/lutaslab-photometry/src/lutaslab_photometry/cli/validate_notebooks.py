"""Validate maintained notebooks without executing experimental analyses."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

DEFAULT_FOLDERS = (
    Path("notebooks"),
    Path("packages/fluopulse-analysis/examples/notebooks"),
    Path("packages/iflip3-analysis/examples/notebooks"),
)
FORBIDDEN_SOURCE_MARKERS = (
    "C:\\Users\\",
    "outputs\\fluopulse-analysis",
    "outputs\\iflip3-analysis",
)


def notebook_paths(paths: list[Path]) -> list[Path]:
    """Expand notebook files and folders into a stable unique file list."""

    expanded: set[Path] = set()
    for path in paths:
        if path.is_dir():
            expanded.update(path.glob("*.ipynb"))
        elif path.suffix == ".ipynb":
            expanded.add(path)
    return sorted(expanded)


def validate_notebook(path: Path) -> list[str]:
    """Return validation errors for one source-controlled notebook."""

    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read notebook JSON: {error}"]
    if notebook.get("nbformat") != 4:
        errors.append("notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return [*errors, "notebook cells must be a list"]
    for index, cell in enumerate(cells):
        source_value = cell.get("source", "")
        source = "".join(source_value) if isinstance(source_value, list) else source_value
        if not isinstance(source, str):
            errors.append(f"cell {index}: source is not text")
            continue
        for marker in FORBIDDEN_SOURCE_MARKERS:
            if marker.lower() in source.lower():
                errors.append(f"cell {index}: contains machine-specific path {marker!r}")
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"cell {index}: execution_count is not cleared")
        if cell.get("outputs"):
            errors.append(f"cell {index}: outputs are not cleared")
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as error:
            errors.append(f"cell {index}: Python syntax error: {error.msg}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_FOLDERS))
    args = parser.parse_args()
    paths = notebook_paths(args.paths)
    if not paths:
        print("No notebooks found.", file=sys.stderr)
        return 1
    failure_count = 0
    for path in paths:
        errors = validate_notebook(path)
        if errors:
            failure_count += 1
            for error in errors:
                print(f"{path}: {error}", file=sys.stderr)
    if failure_count:
        print(f"Notebook validation failed for {failure_count}/{len(paths)} files.")
        return 1
    print(f"Validated {len(paths)} output-free notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
