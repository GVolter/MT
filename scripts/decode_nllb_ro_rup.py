import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def save_lines(lines, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Decode Romanian→Aromanian with a fine-tuned NLLB model")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--src", type=Path, required=True, help="Source file with Romanian sentences")
    parser.add_argument("--out", type=Path, required=True, help="Output file for Aromanian hypotheses")
    parser.add_argument("--src-lang", type=str, default="ron_Latn", help="NLLB source language code")
    parser.add_argument("--tgt-lang", type=str, default="rup_Latn", help="NLLB target language code")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for decoding")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of sentences to decode")

    args = parser.parse_args()

    # Try recommended regex fix; fall back if incompatible with this tokenizer backend
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), fix_mistral_regex=True)
    except TypeError:
        print("Warning: fix_mistral_regex=True not supported for this tokenizer backend; falling back.")
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))

    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_dir))

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Decoding on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Decoding on CPU.")

    model.to(device)
    model.eval()

    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang

    src_lines = load_lines(args.src)
    if args.limit is not None:
        src_lines = src_lines[: args.limit]
    outputs = []

    batch_size = max(1, args.batch_size)

    with torch.no_grad():
        for i in range(0, len(src_lines), batch_size):
            batch = src_lines[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_tokens = model.generate(
                **inputs,
                max_length=args.max_length,
            )
            decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            outputs.extend(decoded_batch)

    save_lines(outputs, args.out)


if __name__ == "__main__":
    main()
