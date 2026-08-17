"""Model loading for the KATHE 2026 English->Kashmiri submission.

Base model: facebook/nllb-200-distilled-1.3B (NLLB-200, distilled 1.3B variant)
Adapter:    LoRA fine-tune (PEFT), r=32/alpha=64/dropout=0.05, targeting
            q_proj,k_proj,v_proj,out_proj,fc1,fc2.

The adapter is hosted on the Hugging Face Hub; this module downloads it
automatically via `peft`/`huggingface_hub` on first use (cached locally
afterward). No local file path is required.
"""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
ADAPTER_REPO = "aymanmakroo/kav"

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"


def load_model(adapter_repo: str = ADAPTER_REPO, device: str | None = None):
    """Load the base NLLB model, apply the LoRA adapter, and return
    (model, tokenizer) ready for generation.

    `device` defaults to "cuda" if available, otherwise "cpu".
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_repo)
    model.to(device)
    model.eval()

    # Force Kashmiri as the generation target language.
    kashmiri_bos_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model.generation_config.forced_bos_token_id = kashmiri_bos_id

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model()
    print(f"Loaded {BASE_MODEL} + adapter from {ADAPTER_REPO} on "
          f"{next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")
