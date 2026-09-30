"""Validate Python raw-lag GLM predictions against published Figure 5 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.depository_glm import validate_saved_figure5_glm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Path to the Depository Data folder")
    parser.add_argument("--mouse", type=int, default=1, help="Published numeric mouse ID")
    parser.add_argument(
        "--epoch",
        choices=("dry", "total", "both"),
        default="both",
        help="Figure 5 epoch to validate",
    )
    args = parser.parse_args()
    epochs = ("dry", "total") if args.epoch == "both" else (args.epoch,)
    results = [
        validate_saved_figure5_glm(args.data_root, epoch, mouse_id=args.mouse).to_dict()
        for epoch in epochs
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
