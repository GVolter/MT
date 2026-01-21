import argparse
from pathlib import Path

import pandas as pd


def normalize_split(name: str) -> str:
    name = name.lower().strip()
    if name in {"dev", "valid", "validation", "val"}:
        return "valid"
    if name in {"train", "training"}:
        return "train"
    if name in {"test", "testing"}:
        return "test"
    return name


def preprocess_corpus(input_csv: Path, output_dir: Path, target_variant: str = "rup_diaro") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = {"ron", target_variant, "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Drop rows with missing source/target
    df = df.dropna(subset=["ron", target_variant, "split"])

    # Normalize split labels once
    df["split"] = df["split"].apply(normalize_split)

    for split_name, split_df in df.groupby("split"):
        # split_name is already normalized here (e.g. "train", "valid", "test")
        src_path = output_dir / f"{split_name}.ro"
        tgt_path = output_dir / f"{split_name}.rup"

        src_path.write_text("\n".join(split_df["ron"].astype(str)), encoding="utf-8")
        tgt_path.write_text("\n".join(split_df[target_variant].astype(str)), encoding="utf-8")

        print(f"Wrote {len(split_df)} pairs to {src_path} and {tgt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Romanian–Aromanian corpus CSV into parallel text files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/rup-ro/aromanian-romanian-MT-corpus-limited.csv"),
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/ro-rup"),
        help="Directory to write train/valid/test .ro/.rup files",
    )
    parser.add_argument(
        "--target-variant",
        type=str,
        default="rup_diaro",
        choices=["rup_diaro", "rup_cunia"],
        help="Which Aromanian variant column to use as target",
    )

    args = parser.parse_args()
    preprocess_corpus(args.input, args.output_dir, args.target_variant)


if __name__ == "__main__":
    main()
