from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
FILENAME = "diabetes_large_dataset.csv"
DATASET_HANDLE = "datascikhan/diabetes"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Group 6 Kaggle diabetes dataset.")
    parser.add_argument(
        "--source", type=Path,
        help="Use an existing local CSV instead of downloading it through kagglehub.",
    )
    args = parser.parse_args()
    destination = DATA_DIR / FILENAME
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Dataset already present: {destination}")
        return

    if args.source:
        source_file = args.source.expanduser().resolve()
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise SystemExit("Install the project first with `uv sync`.") from exc
        downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE))
        if downloaded.is_file():
            source_file = downloaded
        else:
            matches = list(downloaded.rglob(FILENAME))
            if not matches:
                raise SystemExit(f"Could not find {FILENAME} under Kaggle download path {downloaded}")
            source_file = matches[0]

    if not source_file.is_file():
        raise SystemExit(f"Dataset source does not exist: {source_file}")
    if source_file.suffix.lower() != ".csv":
        raise SystemExit(f"Expected a CSV file, received: {source_file}")
    if source_file.resolve() != destination.resolve():
        shutil.copy2(source_file, destination)
    print(f"Dataset ready: {destination}")


if __name__ == "__main__":
    main()
