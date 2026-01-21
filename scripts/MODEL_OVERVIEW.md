# Model Overview and Entry Points

This document summarizes the main models and expected scripts.

## NLLB-200 600M

- Romanian→Aromanian (ro→rup)
  - Config: `experiments/configs/nllb600m_ro-rup.yaml`
  - Suggested scripts: `scripts/train_nllb_ro_rup.py`, `scripts/decode_nllb_ro_rup.py`
- English→Aromanian (en→rup)
  - Config: `experiments/configs/nllb600m_en-rup.yaml`
  - Suggested scripts: `scripts/train_nllb_en_rup.py`, `scripts/decode_nllb_en_rup.py`

## Open Romanian LLMs

- Romanian→Aromanian (ro→rup)
  - Config: `experiments/configs/openllm_ro_ro-rup.yaml`
  - Suggested scripts: `scripts/finetune_openllm_ro_ro_rup.py`, `scripts/decode_openllm_ro_ro_rup.py`
- English→Aromanian (en→rup)
  - Config: `experiments/configs/openllm_ro_en-rup.yaml`
  - Suggested scripts: `scripts/finetune_openllm_ro_en_rup.py`, `scripts/decode_openllm_ro_en_rup.py`

Scripts are not yet implemented; they will be added once the dataset and baseline translator are available. All data paths in configs are placeholders and refer to private resources.
