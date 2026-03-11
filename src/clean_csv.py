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
        print_summary=True,
    )


if __name__ == "__main__":
    main()
