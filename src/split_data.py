import argparse
from pathlib import Path

from src.preprocessing import run_chronological_split, run_random_split, run_stratified_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a CSV into train/val/test files using random, chronological, or stratified logic."
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    random_parser = subparsers.add_parser("random", help="Random 80/10/10 split with random_state=42")
    random_parser.add_argument("input_csv", help="Path to the input CSV")
    random_parser.add_argument("output_dir", help="Directory where split CSV files will be written")
    random_parser.add_argument(
        "--prefix",
        default="random_split",
        help="Filename prefix for the generated CSV files",
    )

    chrono_parser = subparsers.add_parser("chronological", help="Chronological 80/10/10 split by scraped_at")
    chrono_parser.add_argument("input_csv", help="Path to the input CSV")
    chrono_parser.add_argument("output_dir", help="Directory where split CSV files will be written")
    chrono_parser.add_argument(
        "--prefix",
        default="chronological_split",
        help="Filename prefix for the generated CSV files",
    )
    chrono_parser.add_argument(
        "--date-column",
        default="scraped_at",
        help="Datetime column used for the chronological split",
    )

    stratified_parser = subparsers.add_parser("stratified", help="Stratified 80/10/10 split by label column")
    stratified_parser.add_argument("input_csv", help="Path to the input CSV")
    stratified_parser.add_argument("output_dir", help="Directory where split CSV files will be written")
    stratified_parser.add_argument(
        "--prefix",
        default="stratified_split",
        help="Filename prefix for the generated CSV files",
    )
    stratified_parser.add_argument(
        "--label-column",
        default="type",
        help="Label column used for the stratified split",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.method == "random":
        train_path, val_path, test_path = run_random_split(
            input_path=str(input_path),
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
    elif args.method == "chronological":
        train_path, val_path, test_path = run_chronological_split(
            input_path=str(input_path),
            output_dir=args.output_dir,
            prefix=args.prefix,
            date_column=args.date_column,
        )
    else:
        train_path, val_path, test_path = run_stratified_split(
            input_path=str(input_path),
            output_dir=args.output_dir,
            prefix=args.prefix,
            label_column=args.label_column,
        )

    print(f"Train split saved to {train_path}")
    print(f"Validation split saved to {val_path}")
    print(f"Test split saved to {test_path}")


if __name__ == "__main__":
    main()
