"""Translate a single English sentence to Kashmiri.

Usage:
    python -m src.translate_single "She was a true visionary."
"""
from __future__ import annotations

import argparse

from src.generate import generate_with_retry
from src.load_model import load_model
from src.postprocess import postprocess


def translate_one(model, tokenizer, sentence: str) -> str:
    raw = generate_with_retry(model, tokenizer, sentence)
    return postprocess(sentence, raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentence", help="English sentence to translate")
    args = parser.parse_args()

    model, tokenizer = load_model()
    print(translate_one(model, tokenizer, args.sentence))


if __name__ == "__main__":
    main()
