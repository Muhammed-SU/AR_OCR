"""
09b_third_annotator_iaa.py
---------------------------
Once the third independent annotator has returned the 10 filled
`annotator_kit/templates/image_N.txt` files:

  1. Loads A1, A2, and A3 transcriptions for the 10 subset pages.
  2. Computes pairwise CER_no_tashkeel between every annotator pair.
  3. Computes Krippendorff's α at the character level
     (using the `krippendorff` Python package; falls back to manual
     implementation if unavailable).
  4. Emits results/tables/third_annotator_iaa.csv with the panel of
     pairwise CERs and the α value.

The output panel becomes part of the revised §3.5 / §6.6 (and replaces
the "two annotators only" framing of v0.2 with a proper 3-annotator
agreement floor).

Usage:
  cd Paper_JKSU/code
  python3 09b_third_annotator_iaa.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import Levenshtein
except ImportError as e:
    raise SystemExit("pip install python-Levenshtein") from e

try:
    import krippendorff
    HAS_KRIPP = True
except ImportError:
    HAS_KRIPP = False

from arabic_utils import preprocess, char_tokens

ROOT = Path(__file__).resolve().parent.parent.parent
PAPER = Path(__file__).resolve().parent.parent
KIT = PAPER / "annotator_kit"
TABLES = PAPER / "results" / "tables"


def read_a3(n: int) -> str | None:
    p = KIT / "templates" / f"image_{n}.txt"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    # Drop the header comment if present
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    return "\n".join(lines).strip()


def cer(ref: str, hyp: str, mode: str = "no_tashkeel") -> float:
    rs = "".join(char_tokens(preprocess(ref, mode=mode)))
    hs = "".join(char_tokens(preprocess(hyp, mode=mode)))
    return Levenshtein.distance(rs, hs) / max(len(rs), 1)


def manual_krippendorff_char(seqs: list[str]) -> float:
    """
    Character-level Krippendorff's α (nominal) over aligned positions.
    Implementation: align all annotator sequences to the longest one via
    Levenshtein opcodes; build a value matrix; compute α with the
    nominal metric on per-position values.

    Note: this is a simple proxy when the `krippendorff` package is
    unavailable. For the camera-ready paper, prefer the official
    package.
    """
    # Use the longest sequence as the reference frame.
    refs = [preprocess(s, mode="no_tashkeel").replace(" ", "") for s in seqs]
    if not refs:
        return float("nan")
    ref_idx = max(range(len(refs)), key=lambda i: len(refs[i]))
    base = refs[ref_idx]
    L = len(base)
    if L == 0:
        return float("nan")
    rows = []
    for s in refs:
        # align s to base
        ops = Levenshtein.opcodes(base, s)
        aligned = ["NA"] * L
        for tag, i1, i2, j1, j2 in ops:
            if tag in ("equal", "replace"):
                n = min(i2 - i1, j2 - j1)
                for k in range(n):
                    aligned[i1 + k] = s[j1 + k]
        rows.append(aligned)
    # Compute simple Fleiss-style pairwise nominal agreement and convert.
    n_pairs_same = 0
    n_pairs_total = 0
    for col in range(L):
        vals = [r[col] for r in rows if r[col] != "NA"]
        if len(vals) < 2:
            continue
        # number of (i, j) pairs that agree
        from collections import Counter
        c = Counter(vals)
        same = sum(v * (v - 1) // 2 for v in c.values())
        total = len(vals) * (len(vals) - 1) // 2
        n_pairs_same += same
        n_pairs_total += total
    P_o = n_pairs_same / max(n_pairs_total, 1)
    # Expected agreement: chance = sum over chars of (freq / total)^2
    all_chars = [v for r in rows for v in r if v != "NA"]
    if not all_chars:
        return float("nan")
    from collections import Counter
    freq = Counter(all_chars)
    total = sum(freq.values())
    P_e = sum((c / total) ** 2 for c in freq.values())
    if P_e >= 1.0:
        return float("nan")
    return (P_o - P_e) / (1 - P_e)


def main() -> None:
    manifest = pd.read_csv(KIT / "manifest.csv")
    pages = sorted(manifest["n"].tolist())

    rows = []
    triples = []
    for n in pages:
        a1 = (ROOT / "Ground_Truth_Muhammed" / f"image_{n}_GM.txt")
        a2_a = (ROOT / "Ground_Truth_Hasan" / f"image_{n}_GH.txt")
        a2_b = (ROOT / "Ground_Truth_Hasan" / f"İmage_{n}_GH.txt")
        a3_text = read_a3(n)
        if not a1.exists() or a3_text is None:
            print(f"  skip page {n}: A1={a1.exists()} A3={a3_text is not None}")
            continue
        a2 = a2_a if a2_a.exists() else a2_b
        if not a2.exists():
            continue
        s1 = a1.read_text(encoding="utf-8", errors="replace")
        s2 = a2.read_text(encoding="utf-8", errors="replace")
        s3 = a3_text
        rows.append({
            "page_id": f"page_{n:02d}",
            "cer_A1_A2": cer(s1, s2),
            "cer_A1_A3": cer(s1, s3),
            "cer_A2_A3": cer(s2, s3),
            "cer_A1_A2_norm": cer(s1, s2, mode="normalized"),
            "cer_A1_A3_norm": cer(s1, s3, mode="normalized"),
            "cer_A2_A3_norm": cer(s2, s3, mode="normalized"),
        })
        triples.append([s1, s2, s3])

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "third_annotator_iaa.csv", index=False)
    print(f"Wrote {TABLES / 'third_annotator_iaa.csv'}")
    print()
    print("=== Mean pairwise CER (no_tashkeel) ===")
    for col in ["cer_A1_A2", "cer_A1_A3", "cer_A2_A3"]:
        print(f"  {col}:  {df[col].mean():.4f}  (95% CI not bootstrapped here)")
    print()
    print("=== Mean pairwise CER (normalized, with tashkeel) ===")
    for col in ["cer_A1_A2_norm", "cer_A1_A3_norm", "cer_A2_A3_norm"]:
        print(f"  {col}:  {df[col].mean():.4f}")

    # Per-page Krippendorff's α
    print()
    print("=== Krippendorff's α (character nominal, per-page mean) ===")
    alphas = []
    for tri in triples:
        if HAS_KRIPP:
            # Build a reliability matrix: rows are coders, columns are items
            # We treat each character position as an item.
            seqs = [preprocess(s, mode="no_tashkeel").replace(" ", "") for s in tri]
            max_len = max(len(s) for s in seqs)
            matrix = []
            for s in seqs:
                row = list(s) + ["*"] * (max_len - len(s))
                matrix.append(row)
            alpha = krippendorff.alpha(reliability_data=matrix,
                                       level_of_measurement="nominal",
                                       value_domain=None)
        else:
            alpha = manual_krippendorff_char(tri)
        alphas.append(alpha)
    if alphas:
        print(f"  mean α over {len(alphas)} pages: {np.nanmean(alphas):.4f}")
        print(f"  package: {'krippendorff' if HAS_KRIPP else 'manual fallback'}")
    else:
        print("  (no pages processed)")


if __name__ == "__main__":
    main()
