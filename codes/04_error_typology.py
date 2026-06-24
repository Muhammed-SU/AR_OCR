"""
04_error_typology.py
--------------------
Per-system error typology under the dual-annotator regime.

We classify edits relative to ANNOTATOR-AGREEMENT zones:

    A. CONSENSUS ZONE — substrings where Muhammad and Hasan agree
       character-by-character at the normalized level. System errors here
       are the cleanest signal.
    B. DISAGREEMENT ZONE — substrings where the two annotators differ.
       System "errors" here may simply align with one annotator and not
       the other; we therefore report system→Muhammad and system→Hasan
       error rates separately for these zones.

Error classes (per single character edit):
    HAMZA_FORM       confusion among أ / إ / آ / ٱ / ا
    YA_FORM          ي ↔ ى
    TA_MARBUTA       ة ↔ ه ↔ ت
    DIACRITIC_DROP   diacritic in ref, missing in hyp
    DIACRITIC_ADD    diacritic in hyp, absent from ref
    DIACRITIC_WRONG  wrong diacritic (fatha ↔ kasra etc.)
    LETTER_SUB       any other single-letter substitution
    INSERTION        non-diacritic char in hyp only (hallucination)
    DELETION         non-diacritic char in ref only (truncation)
    PUNCT            punctuation-only edit

Output:
    results/tables/error_typology_counts.csv         system × class × ref
    results/tables/error_typology_examples.json      sample edits per class
    results/tables/annotator_disagreement_zones.csv  per-page: # consensus
                                                     vs disagreement chars
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    import Levenshtein
except ImportError as e:
    raise SystemExit("pip install python-Levenshtein") from e

from arabic_utils import preprocess, DIACRITICS

PAPER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_ROOT / "data"
TABLES_DIR = PAPER_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    "gemini_3_1_pro",
    "claude_sonnet_4_6",
    "kraken",
    "google_vision",
    "kraken_gemma",
]

HAMZA_FAMILY = set("أإآٱا")
YA_FAMILY = set("يى")
TA_FAMILY = set("ةه")
PUNCT_SET = set("،؛؟.,!:")
ARABIC_PUNCT = set("،؛؟٪٫٬٭۔")


def classify_edit(op: str, a: str, b: str) -> str:
    if op == "equal":
        return "EQUAL"
    if op == "insert":
        if b in DIACRITICS:
            return "DIACRITIC_ADD"
        if b in PUNCT_SET or b in ARABIC_PUNCT:
            return "PUNCT"
        return "INSERTION"
    if op == "delete":
        if a in DIACRITICS:
            return "DIACRITIC_DROP"
        if a in PUNCT_SET or a in ARABIC_PUNCT:
            return "PUNCT"
        return "DELETION"
    if op == "replace":
        if a in DIACRITICS and b in DIACRITICS:
            return "DIACRITIC_WRONG"
        if a in DIACRITICS and b not in DIACRITICS:
            return "DIACRITIC_DROP"
        if a not in DIACRITICS and b in DIACRITICS:
            return "DIACRITIC_ADD"
        if a in HAMZA_FAMILY and b in HAMZA_FAMILY:
            return "HAMZA_FORM"
        if a in YA_FAMILY and b in YA_FAMILY:
            return "YA_FORM"
        if a in TA_FAMILY and b in TA_FAMILY:
            return "TA_MARBUTA"
        return "LETTER_SUB"
    return "OTHER"


def analyze_pair(ref: str, hyp: str) -> list[dict]:
    ops = Levenshtein.opcodes(ref, hyp)
    records: list[dict] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                a, b = ref[i1 + k], hyp[j1 + k]
                records.append({"op": "replace", "a": a, "b": b,
                                "klass": classify_edit("replace", a, b)})
            if (i2 - i1) > (j2 - j1):
                for a in ref[i1 + (j2 - j1):i2]:
                    records.append({"op": "delete", "a": a, "b": "",
                                    "klass": classify_edit("delete", a, "")})
            elif (j2 - j1) > (i2 - i1):
                for b in hyp[j1 + (i2 - i1):j2]:
                    records.append({"op": "insert", "a": "", "b": b,
                                    "klass": classify_edit("insert", "", b)})
        elif tag == "delete":
            for a in ref[i1:i2]:
                records.append({"op": "delete", "a": a, "b": "",
                                "klass": classify_edit("delete", a, "")})
        elif tag == "insert":
            for b in hyp[j1:j2]:
                records.append({"op": "insert", "a": "", "b": b,
                                "klass": classify_edit("insert", "", b)})
    return records


def main() -> None:
    corpus_csv = DATA_DIR / "parallel_corpus.csv"
    if not corpus_csv.exists():
        raise SystemExit("Run 01_build_corpus.py first")
    df = pd.read_csv(corpus_csv)
    gt_m = df[df["system"] == "gt_muhammad"].set_index("page_id")["text"]
    gt_h = df[df["system"] == "gt_hasan"].set_index("page_id")["text"]
    common_pages = sorted(set(gt_m.index) & set(gt_h.index))

    # --------------------------------------------------------------
    # Annotator disagreement zone — coarse measure: # chars on which
    # M and H disagree (normalised, no_tashkeel). Helps contextualise.
    # --------------------------------------------------------------
    zone_rows: list[dict] = []
    for pid in common_pages:
        gm_n = preprocess(gt_m[pid], mode="no_tashkeel")
        gh_n = preprocess(gt_h[pid], mode="no_tashkeel")
        d = Levenshtein.distance(gm_n, gh_n)
        zone_rows.append({"page_id": pid,
                          "m_len_no_tashkeel": len(gm_n),
                          "h_len_no_tashkeel": len(gh_n),
                          "edit_distance_m_h": d,
                          "char_disagreement_rate":
                              d / max(len(gm_n), 1)})
    zones_df = pd.DataFrame(zone_rows)
    zones_csv = TABLES_DIR / "annotator_disagreement_zones.csv"
    zones_df.to_csv(zones_csv, index=False)
    print(f"Wrote annotator disagreement zones → {zones_csv}")

    # --------------------------------------------------------------
    # Per-system error typology, computed against each annotator.
    # --------------------------------------------------------------
    counter: dict[tuple[str, str], Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], dict[str, list[dict]]] = \
        defaultdict(lambda: defaultdict(list))
    rng = random.Random(20260518)

    refs = {"muhammad": gt_m, "hasan": gt_h}
    for sid in SYSTEMS:
        sub = df[df["system"] == sid].set_index("page_id")
        for ref_label, ref_series in refs.items():
            for pid in common_pages:
                if pid not in sub.index:
                    continue
                ref_text = preprocess(ref_series[pid], mode="normalized")
                hyp_text = preprocess(sub.loc[pid, "text"], mode="normalized")
                for rec in analyze_pair(ref_text, hyp_text):
                    counter[(sid, ref_label)][rec["klass"]] += 1
                    bucket = examples[(sid, ref_label)][rec["klass"]]
                    if len(bucket) < 10:
                        bucket.append({"page_id": pid, **rec})
                    else:
                        if rng.random() < 0.05:
                            bucket[rng.randrange(10)] = {"page_id": pid, **rec}

    all_klasses = sorted({k for c in counter.values() for k in c})
    rows: list[dict] = []
    for sid in SYSTEMS:
        for ref_label in ["muhammad", "hasan"]:
            cnt = counter[(sid, ref_label)]
            total = sum(cnt.values()) or 1
            row = {"system": sid, "ref": ref_label, "total_edits": total}
            for k in all_klasses:
                row[f"{k}_count"] = cnt[k]
                row[f"{k}_share"] = cnt[k] / total
            rows.append(row)
    counts_df = pd.DataFrame(rows)
    out = TABLES_DIR / "error_typology_counts.csv"
    counts_df.to_csv(out, index=False)
    print(f"Wrote error counts → {out}")

    examples_out = TABLES_DIR / "error_typology_examples.json"
    ser = {
        f"{sid}|{ref}": {k: list(v) for k, v in examples[(sid, ref)].items()}
        for sid in SYSTEMS for ref in ["muhammad", "hasan"]
    }
    examples_out.write_text(json.dumps(ser, ensure_ascii=False, indent=2))
    print(f"Wrote examples → {examples_out}")


if __name__ == "__main__":
    main()
