"""
arabic_utils.py
----------------
Arabic text normalization utilities for OCR/LLM evaluation.

The key insight for classical Arabic manuscript evaluation is that there are
multiple LEGITIMATE levels of comparison:

  1. RAW: Compare strings exactly as produced (penalizes everything).
  2. NORMALIZED: Canonicalize visually-equivalent variants (alef forms,
     hamza, ya/alef-maqsura, ta-marbuta vs ha), but KEEP diacritics.
  3. NO-TASHKEEL: Strip diacritics, then normalize. This isolates the
     base-letter recognition skill from diacritic restoration.
  4. TASHKEEL-ONLY: Compare only the diacritic sequences (after aligning
     base letters). This is the "diacritic error rate" (DER).

All four perspectives matter for a paper on classical Arabic OCR.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# ---------------------------------------------------------------------------
# Diacritic / tashkeel handling
# ---------------------------------------------------------------------------

# Arabic combining marks (tashkeel + tatweel + sukun + shadda):
# U+064B FATHATAN
# U+064C DAMMATAN
# U+064D KASRATAN
# U+064E FATHA
# U+064F DAMMA
# U+0650 KASRA
# U+0651 SHADDA
# U+0652 SUKUN
# U+0653 MADDAH ABOVE
# U+0654 HAMZA ABOVE
# U+0655 HAMZA BELOW
# U+0656 SUBSCRIPT ALEF
# U+0657 INVERTED DAMMA
# U+0658 MARK NOON GHUNNA
# U+0670 ALEF SUPERSCRIPT (dagger alef)
# U+06D6..U+06ED Quranic annotation marks
DIACRITICS = (
    "ًٌٍَُِّْٕٓٔ"
    "ٰٖٜٟٗ٘ٙٚٛٝٞ"
)
TATWEEL = "ـ"

_DIAC_RE = re.compile(f"[{re.escape(DIACRITICS)}]")
_TATWEEL_RE = re.compile(TATWEEL)


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritics and tatweel."""
    text = _DIAC_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    return text


def diacritics_only(text: str) -> str:
    """Keep only the diacritic sequence — used for DER computation."""
    return "".join(ch for ch in text if ch in DIACRITICS)


# ---------------------------------------------------------------------------
# Letter-form normalization
# ---------------------------------------------------------------------------

# Canonicalization choices follow common Arabic IR / OCR-eval conventions
# (e.g., MADAR, MADAMIRA evaluation conventions). These are documented in
# the paper's Methodology section.
LETTER_NORM_MAP = {
    # All alef variants -> bare alef
    "آ": "ا",  # ALEF WITH MADDA ABOVE
    "أ": "ا",  # ALEF WITH HAMZA ABOVE
    "إ": "ا",  # ALEF WITH HAMZA BELOW
    "ٱ": "ا",  # ALEF WASLA
    # Alef-maqsura -> ya
    "ى": "ي",  # ALEF MAQSURA -> YA
    # Ta-marbuta -> ha (often used by OCR systems)
    # NOTE: we make this OPTIONAL because it's lossy. Keep separate function.
    # Yeh with hamza above -> ya (less common, but seen in OCR output)
    "ئ": "ي",  # YA WITH HAMZA ABOVE
    # Standalone hamza forms
    "ؤ": "و",  # WAW WITH HAMZA -> WAW
    # Persian/Urdu look-alikes that sometimes appear in OCR
    "ک": "ك",  # PERSIAN KEHEH -> KAF
    "ی": "ي",  # PERSIAN YEH -> YA
}


def normalize_letters(text: str, fold_ta_marbuta: bool = False) -> str:
    """Canonicalize letter forms (alef, ya, hamza variants)."""
    for src, tgt in LETTER_NORM_MAP.items():
        text = text.replace(src, tgt)
    if fold_ta_marbuta:
        text = text.replace("ة", "ه")  # TA MARBUTA -> HA
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace, normalize line endings."""
    # Collapse all whitespace to single space, strip
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_punctuation(text: str) -> str:
    """Remove Arabic and Western punctuation."""
    # Arabic punctuation: comma, semicolon, question mark, full stop, etc.
    arabic_punct = "،؛؟٪٫٬٭۔"
    western_punct = r"!\"#$%&'()*+,./:;<=>?@[\\\]^_`{|}~"
    pattern = f"[{re.escape(arabic_punct)}{western_punct}—–-]"
    return re.sub(pattern, " ", text)


# ---------------------------------------------------------------------------
# Line-number prefix handling
# ---------------------------------------------------------------------------

_LINE_PREFIX_RE = re.compile(r"^\s*\d+\s*[\t\.\)]\s*")


def strip_line_numbers(text: str) -> str:
    """
    Many of our files are formatted with a leading line number + tab
    (e.g., "1\tبسم الله..."). We strip that for fair text comparison.
    """
    cleaned = []
    for line in text.splitlines():
        cleaned.append(_LINE_PREFIX_RE.sub("", line))
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Composite preprocessing modes
# ---------------------------------------------------------------------------

def preprocess(
    text,
    *,
    mode: str = "raw",
    fold_ta_marbuta: bool = False,
) -> str:
    """
    Apply one of four canonical preprocessing modes.

    mode ∈ {"raw", "normalized", "no_tashkeel", "tashkeel_only"}
    """
    # Handle empty / NaN values from pandas (empty system outputs).
    if text is None:
        return ""
    if isinstance(text, float):
        import math
        if math.isnan(text):
            return ""
        text = str(text)
    if not isinstance(text, str):
        text = str(text)
    # Always strip line-number prefixes and normalize unicode form.
    text = unicodedata.normalize("NFC", text)
    text = strip_line_numbers(text)
    text = normalize_whitespace(text)

    if mode == "raw":
        return text

    if mode == "tashkeel_only":
        return diacritics_only(text)

    # For "normalized" and "no_tashkeel" we letter-normalize and strip punct.
    text = normalize_letters(text, fold_ta_marbuta=fold_ta_marbuta)
    text = strip_punctuation(text)
    text = normalize_whitespace(text)

    if mode == "no_tashkeel":
        text = strip_tashkeel(text)
        text = normalize_whitespace(text)
        return text

    if mode == "normalized":
        return text

    raise ValueError(f"Unknown preprocessing mode: {mode}")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def char_tokens(text: str) -> list[str]:
    """List of characters (after dropping single spaces)."""
    return list(text.replace(" ", ""))


def word_tokens(text: str) -> list[str]:
    """Whitespace-split tokens."""
    return text.split()
