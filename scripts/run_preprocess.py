import argparse
import sys
from pathlib import Path


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from src.load_data import (
    get_session_paths,
    load_session_data,
)

from src.preprocess import (
    preprocess_session,
)

from src.save_sessiondata import (
    save_session,
)


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load and preprocess a photometry session, "
            "then save the processed session."
        )
    )

    parser.add_argument(
        "--mouse",
        required=True,
        help="Mouse identifier, e.g. DK21"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Session date, e.g. 230704"
    )

    parser.add_argument(
        "--run",
        required=True,
        type=int,
        help="Run number, e.g. 3"
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    """
    Load, preprocess, and save one session.
    """

    args = parse_arguments()

    mouse = args.mouse
    date = args.date
    run = args.run

    print()
    print("=" * 60)
    print("Photometry preprocessing")
    print("=" * 60)

    print(
        f"Mouse: {mouse}"
    )

    print(
        f"Date:  {date}"
    )

    print(
        f"Run:   {run}"
    )

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Find raw session files
    # --------------------------------------------------------

    print(
        "Finding session files..."
    )

    session_paths = get_session_paths(
        mouse_name=mouse,
        date=date,
        run=run
    )

    print(
        "Session paths:"
    )

    print(
        session_paths
    )

    print()

    # --------------------------------------------------------
    # Load raw session
    # --------------------------------------------------------

    print(
        "Loading raw session..."
    )

    session = load_session_data(
        session_paths
    )

    print(
        "Raw session loaded."
    )

    print()

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print(
        "Preprocessing session..."
    )

    processed_session = preprocess_session(
        session
    )

    print(
        "Preprocessing complete."
    )

    print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "Saving processed session..."
    )

    save_path = save_session(
        processed_session
    )

    print()

    print(
        "=" * 60
    )

    print(
        "Done."
    )

    print(
        f"Processed session:\n{save_path}"
    )

    print(
        "=" * 60
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()