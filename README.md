# Integrating Dictionaries into Neural Machine Translation

This project explores terminology-aware neural machine translation by integrating dictionary/terminology resources into neural MT systems.

Current focus:
- Language pairs: Romanian–Aromanian and English–Aromanian (exact dataset and dictionaries will be provided separately and are kept private).
- Models: fine-tuning NLLB-200 600M and experimenting with open Romanian LLMs for terminology-aware translation.
- Evaluation: BLEU, chrF++, BERTScore, COMET, plus terminology-focused metrics.

Main components:
- `data/`: parallel data and processed versions (kept private, not redistributed)
- `terminology/`: terminology extraction, alignment, and dictionary resources
- `models/`: baseline and terminology-aware NMT/LLM models
- `experiments/`: experiment configurations and run metadata
- `evaluation/`: standard MT and terminology-specific metrics and analysis
- `scripts/`: end-to-end pipelines (preprocess, train, evaluate)
- `docs/`: project documentation
