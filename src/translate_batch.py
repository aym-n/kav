"""Translate a CSV of English sentences to Kashmiri (batch inference).

Input CSV must have columns: ID, sentence
Output CSV has columns: ID, kashmiri_text (same ID order as input).

Usage:
    python -m src.translate_batch --input englishdev.csv --output submission.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from tqdm import tqdm

from src.generate import DECODING_LADDER, MAX_LENGTH, NUM_BEAMS, detect_repetition_loop, generate_with_retry
from src.load_model import load_model
from src.postprocess import postprocess

BATCH_SIZE = 8


def translate_batch(model, tokenizer, sentences: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    device = next(model.parameters()).device
    default_cfg = DECODING_LADDER[0]

    # Fast pass: batched generation with our validated-best default config.
    raw_outputs: list[str] = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="translating"):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding="longest", max_length=MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                **default_cfg,
            )
        raw_outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    # Repair pass: batching can't escalate decoding per-sentence, so any
    # output the default config produced as a repetition loop gets retried
    # individually against the remaining, stronger rungs of the ladder.
    n_retried = 0
    for i, (src, raw) in enumerate(zip(sentences, raw_outputs)):
        if detect_repetition_loop(raw):
            raw_outputs[i] = generate_with_retry(model, tokenizer, src, start_index=1)
            n_retried += 1
    if n_retried:
        print(f"Retried {n_retried}/{len(sentences)} sentences after detecting a repetition loop.")

    return [postprocess(src, raw) for src, raw in zip(sentences, raw_outputs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV with ID,sentence columns")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model, tokenizer = load_model()

    hyps = translate_batch(model, tokenizer, df["sentence"].astype(str).tolist(), args.batch_size)

    # Guard against the scorer's hard rejection of empty hypotheses.
    hyps = [h if h.strip() else "۔" for h in hyps]

    out_df = pd.DataFrame({"ID": df["ID"], "kashmiri_text": hyps})
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
