import argparse
from pathlib import Path

from src.preprocessing import run_cleaning_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean a raw CSV using the same pipeline as notebook 01."
    )
    parser.add_argument("input_csv", help="Path to the raw input CSV file")
    parser.add_argument("output_csv", help="Path for the cleaned output CSV file")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for multiprocessing",
    )
    parser.add_argument(
        "--split-output-dir",
        default=None,
        help="Directory where chronological train/val/test split files will be written",
    )
    parser.add_argument(
        "--split-prefix",
        default=None,
        help="Filename prefix for the chronological split CSV files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    run_cleaning_pipeline(
        input_path=str(input_path),
        output_path=args.output_csv,
        n_cores=args.cores,
        split_output_dir=args.split_output_dir,
        split_prefix=args.split_prefix,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
