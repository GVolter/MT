import argparse
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_parallel_data(data_dir: Path, src_name: str, tgt_name: str):
    src_path = data_dir / src_name
    tgt_path = data_dir / tgt_name

    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()

    assert len(src_lines) == len(tgt_lines), "Source and target files must have the same number of lines"

    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})


def tokenize_function(example, tokenizer, src_lang, tgt_lang, max_length):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    model_inputs = tokenizer(
        example["src"],
        max_length=max_length,
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["tgt"],
            max_length=max_length,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 600M for Romanian→Aromanian (ro→rup)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/nllb600m_ro-rup.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for source and target",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force training on CPU even if a CUDA GPU is available",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from config for quick/debug runs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_cfg = config["model"]
    train_cfg = config["training"]

    data_dir = Path(train_cfg["data_dir"])
    save_dir = Path(train_cfg["save_dir"])

    # Try recommended regex fix; fall back if incompatible
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"], fix_mistral_regex=True)
    except TypeError:
        print("Warning: fix_mistral_regex=True not supported for this tokenizer backend; falling back.")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["pretrained_name"])

    # Device selection / logging
    if torch.cuda.is_available() and not args.no_gpu:
        n_gpus = torch.cuda.device_count()
        print(f"Using CUDA with {n_gpus} GPU(s); primary device: {torch.cuda.get_device_name(0)}")
    else:
        if not torch.cuda.is_available():
            print("CUDA not available; training will run on CPU.")
        else:
            print("GPU available but '--no-gpu' was set; training will run on CPU.")

    src_lang = model_cfg["src_lang"]
    tgt_lang = model_cfg["tgt_lang"]

    train_dataset = load_parallel_data(
        data_dir,
        train_cfg["train_src"],
        train_cfg["train_tgt"],
    )
    valid_dataset = load_parallel_data(
        data_dir,
        train_cfg["valid_src"],
        train_cfg["valid_tgt"],
    )

    def _tok_fn(batch):
        return tokenize_function(
            batch,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(_tok_fn, batched=True, remove_columns=["src", "tgt"])
    valid_dataset = valid_dataset.map(_tok_fn, batched=True, remove_columns=["src", "tgt"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Allow simple CLI overrides for experimentation
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4)
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 100000)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(save_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=float(train_cfg.get("lr", 3e-5)),
        max_steps=max_steps,
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        predict_with_generate=True,
        fp16=train_cfg.get("fp16", False),
        no_cuda=args.no_gpu,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))


if __name__ == "__main__":
    main()
