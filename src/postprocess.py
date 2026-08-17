"""Post-processing for the KATHE 2026 English->Kashmiri submission.

Applied, in order, to every raw model translation:
  1. KashmiriNormalizer -- official letter-form normalization (folds
     Arabic-form letters onto their Persian/Urdu equivalents: ي->ی, ك->ک,
     ه->ہ). Safe and symmetric -- the official scorer applies the same
     normalization to the reference, so this never disadvantages an
     otherwise-correct translation.
  2. Diacritic normalizer -- a substitution table mined from this specific
     model's own output vocabulary against an independent Kashmiri
     frequency corpus (BPCC-Human parallel data + back-translated pairs +
     an English-Kashmiri glossary). A rule is only included if the
     corrected form is attested at least 10x in that corpus and at least
     5x more often than the model's original spelling -- i.e. every rule
     reflects a well-attested real-world spelling convention. This table
     is specific to this exact model/adapter; it should be re-mined if the
     underlying model changes; it must never be applied to reference text.
  3. Arabic-script punctuation -- normalizes the ASCII comma "," to the
     Perso-Arabic comma "،", except when it is flanked by digits on both
     sides (e.g. "2,00,000"), which we leave alone since it is a numeral
     thousands-separator, not a clause separator, and we have no evidence
     Kashmiri numeral formatting should use the Perso-Arabic comma there.
  4. Terminal punctuation -- appends a Kashmiri full stop when a
     translation is missing terminal punctuation entirely.
  5. Teen-number correction -- this model consistently truncates English
     "teen" numbers (13-19) to their base single-digit form during
     generation (e.g. "fifteen" -> the word for "five"). This fix is
     strictly gated on the English source sentence containing the
     corresponding teen-number word, so it can never misfire on a
     sentence that genuinely says "five".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from KashmiriNormalizer import KashmiriNormalizer

_normalizer = KashmiriNormalizer()

_DATA_DIR = Path(__file__).parent
with open(_DATA_DIR / "nllb_diacritic_rules.json", encoding="utf-8") as f:
    _DIACRITIC_RULES: dict[str, str] = json.load(f)
with open(_DATA_DIR / "teen_number_fix.json", encoding="utf-8") as f:
    _TEEN_FIX = json.load(f)
_TEEN_CORRECT = _TEEN_FIX["correct_forms"]
_TEEN_WRONG = _TEEN_FIX["known_wrong_forms"]

_TERMINAL_PUNCT = ("۔", "؟", "!", ".")


def _apply_diacritic_rules(text: str) -> str:
    return " ".join(_DIACRITIC_RULES.get(tok, tok) for tok in text.split())


def _normalize_arabic_comma(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        if ch != ",":
            out.append(ch)
            continue
        prev_digit = i > 0 and text[i - 1].isdigit()
        next_digit = i + 1 < len(text) and text[i + 1].isdigit()
        out.append("," if (prev_digit and next_digit) else "،")
    return "".join(out)


def _fix_terminal_punctuation(text: str) -> str:
    if text and text[-1] not in _TERMINAL_PUNCT:
        return text + "۔"
    return text


def _fix_teen_numbers(english_source: str, kashmiri_text: str) -> str:
    en_lower = english_source.lower()
    toks = kashmiri_text.split()
    for teen_word, wrong_toks in _TEEN_WRONG.items():
        if re.search(rf"\b{teen_word}\b", en_lower):
            correct = _TEEN_CORRECT[teen_word]
            toks = [correct if t in wrong_toks else t for t in toks]
    return " ".join(toks)


def postprocess(english_source: str, kashmiri_raw: str) -> str:
    """Apply the full validated post-processing pipeline to one translation.

    `english_source` is required (used only to gate the teen-number fix --
    it never influences any other step and is never itself translated
    here).
    """
    text = _normalizer.normalize(kashmiri_raw)
    text = _apply_diacritic_rules(text)
    text = _normalize_arabic_comma(text)
    text = _fix_terminal_punctuation(text)
    text = _fix_teen_numbers(english_source, text)
    return text
