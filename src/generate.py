"""Repetition-loop detection and an adaptive decoding retry ladder.

We independently confirmed during development that beam search on this model
occasionally loops on a repeated word or short phrase. A moderate
`repetition_penalty` + `no_repeat_ngram_size` (our validated-best default
config, first entry below) catches most cases, but not all -- some loops only
kick in partway through generation. Rather than accept a looping hypothesis
outright, we detect the loop and retry with escalating repetition
suppression before falling back to the last attempt regardless.
"""
from __future__ import annotations

import torch

NUM_BEAMS = 8
MAX_LENGTH = 256

# First entry is our validated-best default (beam-width/decoding sweep).
# Later entries escalate repetition suppression and are only used as a
# fallback when the default output is detected as a repetition loop.
DECODING_LADDER = [
    {"repetition_penalty": 1.3, "no_repeat_ngram_size": 3},
    {"repetition_penalty": 1.6, "no_repeat_ngram_size": 3},
    {"repetition_penalty": 2.0, "no_repeat_ngram_size": 4},
]


def detect_repetition_loop(text: str, min_repeats: int = 3, max_ngram: int = 8) -> bool:
    """Detect consecutive repetition of a 1..max_ngram-word phrase anywhere
    in the text -- catches loops from the very start as well as loops that
    only kick in partway through generation.
    """
    words = text.split()
    if len(words) < min_repeats:
        return False
    max_n = min(max_ngram, len(words) // min_repeats)
    for n in range(1, max_n + 1):
        for start in range(0, len(words) - n * min_repeats + 1):
            pattern = words[start:start + n]
            repeats = 1
            pos = start + n
            while pos + n <= len(words) and words[pos:pos + n] == pattern:
                repeats += 1
                pos += n
            if repeats >= min_repeats:
                return True
    return False


def _generate_once(model, tokenizer, inputs, repetition_penalty, no_repeat_ngram_size):
    with torch.no_grad():
        return model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )


def generate_with_retry(model, tokenizer, sentence: str, start_index: int = 0) -> str:
    """Translate one sentence, retrying with a stronger decoding config from
    `DECODING_LADDER` if the current output is a detected repetition loop.
    Returns the first non-looping output, or the last attempt if every
    remaining rung still loops.

    `start_index` skips already-tried rungs -- e.g. batch inference tries
    the default config (index 0) for the whole batch up front, then only
    calls this with `start_index=1` for the sentences that looped.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

    text = ""
    for cfg in DECODING_LADDER[start_index:]:
        generated = _generate_once(model, tokenizer, inputs, **cfg)
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        if not detect_repetition_loop(text):
            return text
    return text
