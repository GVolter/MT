import argparse
from pathlib import Path

import pandas as pd


def preprocess_dictionary(input_xls: Path, output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(input_xls, sheet_name=0, header=None)

    if df.shape[1] < 3:
        raise ValueError("Expected at least 3 columns (type, Aromanian, Romanian)")

    df = df[[0, 1, 2]]
    df.columns = ["pos", "rup", "ron"]

    df = df.dropna(subset=["rup", "ron"])

    df["pos"] = df["pos"].astype(str).str.strip()
    df["rup"] = df["rup"].astype(str).str.strip()
    df["ron"] = df["ron"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["rup", "ron", "pos"])

    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Wrote {len(df)} entries to {output_tsv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Papahagi Aromanian–Romanian dictionary into TSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("terminology/resources/Papahagi Dictsionar-Armanesc-Romanesc.xls"),
        help="Path to the input Excel dictionary (.xls)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("terminology/resources/ro-rup/papahagi_dict.tsv"),
        help="Path to the output TSV file",
    )

    args = parser.parse_args()
    preprocess_dictionary(args.input, args.output)


if __name__ == "__main__":
    main()
