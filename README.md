# kāv: English to Kashmiri Machine Translation
kāv is a LoRA finetuned `facebook/nllb-200-distilled-1.3B` plus a four-step, corpus-driven post-processor for English to Kashmiri machine translation

This repository contains the inference code for the inference pipeline. Adapter weights live on the Hugging Face Hub at [`aymanmakroo/kav`](https://huggingface.co/aymanmakroo/kav) and download automatically on first run.

---

## Score progression

Official Kaggle submissions, oldest first:

| File                            |     Score | Configuration                                                                |
| ------------------------------- | --------: | ---------------------------------------------------------------------------- |
| `submission_baseline_modal.csv` |  **7.67** | Zero-shot IndicTrans2 with default decoding                                  |
| `improved-baseline.csv`         | **14.67** | IndicTrans2 with beam-8 decoding, repetition control, and KashmiriNormalizer |
| `submission_nozero.csv`         | **11.01** | IndicTrans2 trained on an oversampled dataset, resulting in overfitting      |
| `submission_v5_final.csv`       | **15.39** | IndicTrans2 LoRA with FFN targets and matched training/inference decoding    |
| `submission_nllb_cleaned.csv`   | **21.20** | First NLLB-200 LoRA submission                                               |
| `submission_nllb_v4.csv`        | **21.41** | NLLB-200 LoRA with the final post-processing pipeline                        |


---

## Approach

We initially used `ai4bharat/indictrans2-en-indic-1B`, but later switched to a LoRA finetuned `facebook/nllb-200-distilled-1.3B model`. Each hypothesis is then passed through a validated post-processing pipeline. The base-model choice was empirical: under matched data and training conditions, NLLB-200 achieved substantially higher leaderboard performance than `ai4bharat/indictrans2-en-indic-1B`.

---

## Data

Training data is **`ai4bharat/BPCC`** only (`bpcc-seed-latest`, `kas_Arab` split). Raw BPCC is long and news/Wikipedia-like; the competition sentences are short (mean ~7 English tokens). We cleaned and filtered BPCC down to short, single-sentence pairs, then oversampled the shortest slice so the training distribution matches that register.

Oversampling that short slice further, or training on less-filtered longer BPCC, both scored worse on our held-out split. The 45,226-row short mix was the local optimum among the BPCC filters we tried.

---

## Training

- **Method:** LoRA (PEFT), `r=32`, `alpha=64`, `dropout=0.05`
- **Target modules:** `q_proj, k_proj, v_proj, out_proj, fc1, fc2` — attention and feed-forward. Broadening past attention-only was necessary to give the adapter enough trainable capacity to shift Kashmiri output.
- **Schedule:** 3 epochs, batch size 4, gradient accumulation 8 (effective batch 32), learning rate `8e-5`, 100 warmup steps
- **Checkpoint selection:** `load_best_model_at_end=True` on the geometric mean of BLEU and chrF++, computed with `predict_with_generate` under the *same* decoding configuration as real inference. An earlier pipeline selected checkpoints with greedy decode / `max_length=128` while submissions used beam-8 / `max_length=256`; the trainer then preferred a checkpoint that was not actually best at submission time. Aligning those settings was one of the most consequential process changes we made.
- **Early stopping:** patience 8 evaluations, threshold `1e-3`

---

## Decoding

`num_beams=8`, `repetition_penalty=1.3`, `no_repeat_ngram_size=3`, `max_length=256`, forced BOS token `kas_Arab`.

A beam-width × length-penalty sweep confirmed this as a local optimum among the values tested. The repetition penalty and n-gram block fix a real failure mode: naive greedy or low-beam decoding can loop a token or short phrase dozens of times. After official normalization those hypotheses can collapse to empty strings, and empty hypotheses are hard-rejected by the scorer.

---

## Post-processing (`src/postprocess.py`)

Every raw translation goes through four steps, in order:

1. **`KashmiriNormalizer`** (official package). Folds Perso-Arabic letter-form variants (`ي→ی`, `ك→ک`, `ه→ہ`, …) that are orthographically equivalent but use different Unicode codepoints. The official scorer applies the same normalization to the reference, so this step is always safe.

2. **Model-specific diacritic normalizer** (`nllb_diacritic_rules.json`, 102 rules). Mined by comparing this model's own output vocabulary against word frequencies on the Kashmiri side of held-out BPCC. A substitution is kept only when the corrected spelling is attested at least 10× in that BPCC text, and at least 5× more often than the model's original spelling. The table is specific to this adapter. A table mined from a different model *degraded* quality when applied here; if the base model or adapter changes, the table must be re-mined.

3. **Terminal punctuation.** Appends a Kashmiri full stop (`۔`) when a translation has no terminal punctuation.

4. **Teen-number correction** (`teen_number_fix.json`). A systematic error: the model truncates English teen numbers (13–19) to their single-digit base.

---

## Repository layout

```
src/
  load_model.py              # facebook/nllb-200-distilled-1.3B + LoRA adapter
  postprocess.py             # four-step post-processor
  nllb_diacritic_rules.json  # 102-rule model-specific spelling table
  teen_number_fix.json       # 13–19 correction table + known-wrong forms
  translate_single.py        # one sentence (CLI)
  translate_batch.py         # CSV in, CSV out (CLI)
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Adapter weights download from [`aymanmakroo/kav`](https://huggingface.co/aymanmakroo/kav) on first use via `peft` / `huggingface_hub`. No manual file placement is required.

The Hugging Face adapter (and this GitHub repo) are currently **private**. Authenticate first:

```bash
huggingface-cli login          # or: hf auth login
# equivalently: export HF_TOKEN=...
```

---

## Usage

**Single sentence:**

```bash
python -m src.translate_single "She was a true visionary."
```

**Batch (CSV in, CSV out):**

```bash
python -m src.translate_batch --input englishdev.csv --output submission.csv
```

Input CSV must have `ID,sentence` columns. Output is `ID,kashmiri_text` in the same row order, ready for submission.
