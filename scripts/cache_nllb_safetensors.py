import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache NLLB-200 600M locally using safetensors only.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID to cache",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/nllb_600m/hf_nllb_200_600m_sft"),
        help="Local directory to save the model and tokenizer",
    )

    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.model_id}' from hub (this may take a while)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Saving model and tokenizer to '{out_dir}' with safetensors...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
