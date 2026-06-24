"""
05_revision_analyses.py
-----------------------
Major-revision-round analyses requested by reviewers:

  1.  Public-vs-private stratification (contamination probe, C1).
      Reports CER_no_tashkeel per system separately on:
        - 37 public-source pages (Duke / NLM / HathiTrust / BSB)
        - 13 private Furūsiyya pages (authors' own digital corpus)
      If the Gemini lead is comparable on the private subset, training-
      data contamination on public manuscripts is unlikely to be the
      driver of the headline ranking.

  2.  Consensus-only reference regime (C8).
      Re-scores each system using only those character positions where
      A1 and A2 agree at the no_tashkeel level. Adds a fourth reference
      regime (consensus) alongside muhammad / hasan / best.

  3.  Bounded DER (C4).
      Replaces the unbounded DER with der_bounded := Levenshtein on
      diacritic-only subsequences / max(|ref_diac|, |hyp_diac|), which
      is in [0, 1]. Also reports diacritic precision / recall / F1 with
      position-alignment over base letters.

  4.  Outlier sensitivity (C9).
      Recomputes per-system means on the 48-page corpus that excludes
      pages 24 and 33 (catastrophic Kraken / Kraken+Gemma failures).

Outputs (results/tables/):
  revision_stratified.csv          (system × subset × regime × CER metrics)
  revision_consensus_only.csv      (per-page consensus-only scores)
  revision_summary_consensus.csv   (system-level consensus-only summary)
  revision_der_bounded.csv         (per-page bounded DER + P/R/F1)
  revision_outlier_sensitivity.csv (Table 1 without pages 24/33)
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

from arabic_utils import (
    preprocess,
    char_tokens,
    word_tokens,
    DIACRITICS,
)

PAPER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_ROOT / "data"
TABLES = PAPER_ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    "gemini_3_1_pro",
    "claude_sonnet_4_6",
    "kraken",
    "google_vision",
    "kraken_gemma",
]

CATASTROPHIC_PAGES = {"page_24", "page_33"}

# -- Aux ----------------------------------------------------------------

def cer(ref: str, hyp: str) -> float:
    rs = "".join(char_tokens(ref))
    hs = "".join(char_tokens(hyp))
    return Levenshtein.distance(rs, hs) / max(len(rs), 1)


def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.05):
    vals = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(20260518)
    means = [float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
             for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


# -- Load corpus and provenance ----------------------------------------

corpus = pd.read_csv(DATA_DIR / "parallel_corpus.csv")
prov = pd.read_csv(DATA_DIR / "manuscript_provenance.csv")
prov["subset"] = prov["repository"].apply(
    lambda r: "private" if r == "Private Furūsiyya corpus (authors)" else "public"
)

gt_m = corpus[corpus["system"] == "gt_muhammad"].set_index("page_id")["text"]
gt_h = corpus[corpus["system"] == "gt_hasan"].set_index("page_id")["text"]
common_pages = sorted(set(gt_m.index) & set(gt_h.index))

# Pre-compute normalized text per page per source
def prep_nt(text):
    return preprocess(text, mode="no_tashkeel")

def prep_norm(text):
    return preprocess(text, mode="normalized")

# -- ANALYSIS 1: Public-vs-Private stratification ----------------------
print("=== ANALYSIS 1: Public-vs-private stratification (contamination probe) ===")
existing = pd.read_csv(TABLES / "per_page_metrics.csv")
merged = existing.merge(prov[["page_id", "subset", "manuscript_title"]],
                        on="page_id", how="left")
strat_rows = []
for sid in SYSTEMS:
    for ref in ["muhammad", "hasan", "best"]:
        for subset in ["public", "private", "all"]:
            sub = merged[(merged["system"] == sid) & (merged["ref"] == ref)]
            if subset != "all":
                sub = sub[sub["subset"] == subset]
            for col in ["cer_no_tashkeel", "cer_normalized"]:
                mean, lo, hi = bootstrap_ci(sub[col].dropna().tolist())
                strat_rows.append({
                    "system": sid, "ref": ref, "subset": subset,
                    "metric": col, "n_pages": len(sub),
                    "mean": mean, "ci_lo": lo, "ci_hi": hi,
                })
strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv(TABLES / "revision_stratified.csv", index=False)
print(f"Wrote {TABLES / 'revision_stratified.csv'}")

# Print the contamination-probe headline
print("\n  -- CER (no tashkeel, ref=best) by subset --")
print(f"  {'system':22s} {'public (n=37)':>16s} {'private (n=13)':>16s} {'Δ priv-pub':>14s}")
print("  " + "-" * 70)
pivot = strat_df[(strat_df["ref"] == "best") &
                 (strat_df["metric"] == "cer_no_tashkeel")]
for sid in SYSTEMS:
    pub = pivot[(pivot["system"] == sid) & (pivot["subset"] == "public")]["mean"].iloc[0]
    pri = pivot[(pivot["system"] == sid) & (pivot["subset"] == "private")]["mean"].iloc[0]
    print(f"  {sid:22s}  {pub:>14.4f}    {pri:>14.4f}   {pri-pub:>+12.4f}")

# -- ANALYSIS 2: Consensus-only regime ---------------------------------
print("\n=== ANALYSIS 2: Consensus-only reference regime ===")
print("Computing consensus mask per page (positions where A1 == A2 after no-tashkeel norm)...")

def consensus_score(gt_a: str, gt_b: str, hyp: str) -> tuple[float, int]:
    """
    Compute CER restricted to character positions where A1 and A2 agree.
    Implementation: align A1 and A2 via Levenshtein opcodes; the 'equal'
    segments define consensus zones. For each consensus zone in A1, we
    find the corresponding segment in the hypothesis (via Levenshtein
    alignment of A1 and hyp) and compute edit distance there.
    Returns (cer_consensus, n_consensus_chars).
    """
    a = preprocess(gt_a, mode="no_tashkeel").replace(" ", "")
    b = preprocess(gt_b, mode="no_tashkeel").replace(" ", "")
    h = preprocess(hyp,  mode="no_tashkeel").replace(" ", "")
    if not a or not b:
        return float("nan"), 0
    # Find consensus zones in A1 (a) where it agrees with A2 (b)
    ops_ab = Levenshtein.opcodes(a, b)
    consensus_spans = [(i1, i2) for tag, i1, i2, _, _ in ops_ab if tag == "equal"]
    if not consensus_spans:
        return float("nan"), 0
    # Build the consensus reference string from a
    consensus_ref = "".join(a[i1:i2] for i1, i2 in consensus_spans)
    if not consensus_ref:
        return float("nan"), 0
    # For the hypothesis, we approximate the consensus span by aligning
    # a and h, then projecting the consensus spans through that alignment.
    ops_ah = Levenshtein.opcodes(a, h)
    # Build a position map: for each position in a, which position in h
    # is aligned to it (or -1 if deleted in h).
    pos_map = [-1] * (len(a) + 1)
    for tag, i1, i2, j1, j2 in ops_ah:
        if tag in ("equal", "replace"):
            span_a = i2 - i1
            span_h = j2 - j1
            for k in range(min(span_a, span_h)):
                pos_map[i1 + k] = j1 + k
        elif tag == "delete":
            for k in range(i2 - i1):
                pos_map[i1 + k] = -1
        elif tag == "insert":
            # nothing in a moves
            pass
    # Build hypothesis consensus substring by extracting h positions
    # corresponding to the consensus zones in a.
    h_consensus_parts = []
    for i1, i2 in consensus_spans:
        for k in range(i1, i2):
            j = pos_map[k]
            if j >= 0:
                h_consensus_parts.append(h[j])
    consensus_hyp = "".join(h_consensus_parts)
    d = Levenshtein.distance(consensus_ref, consensus_hyp)
    return d / max(len(consensus_ref), 1), len(consensus_ref)

cons_rows = []
for sid in SYSTEMS:
    sub = corpus[corpus["system"] == sid].set_index("page_id")
    for pid in common_pages:
        if pid not in sub.index:
            continue
        score, n_chars = consensus_score(gt_m[pid], gt_h[pid], sub.loc[pid, "text"])
        cons_rows.append({
            "page_id": pid, "system": sid,
            "n_consensus_chars": n_chars,
            "cer_consensus": score,
        })
cons_df = pd.DataFrame(cons_rows)
cons_df.to_csv(TABLES / "revision_consensus_only.csv", index=False)

cons_summary = []
for sid in SYSTEMS:
    s = cons_df[cons_df["system"] == sid]["cer_consensus"].dropna().tolist()
    mean, lo, hi = bootstrap_ci(s)
    cons_summary.append({
        "system": sid, "n_pages": len(s),
        "cer_consensus_mean": mean,
        "cer_consensus_ci_lo": lo, "cer_consensus_ci_hi": hi,
    })
cons_sum_df = pd.DataFrame(cons_summary)
cons_sum_df.to_csv(TABLES / "revision_summary_consensus.csv", index=False)
print(f"Wrote {TABLES / 'revision_consensus_only.csv'}")
print(f"Wrote {TABLES / 'revision_summary_consensus.csv'}")
print()
print("  -- Consensus-only CER per system --")
print(f"  {'system':22s} {'CER consensus':>16s} {'95% CI':>22s}")
for r in cons_summary:
    print(f"  {r['system']:22s}  {r['cer_consensus_mean']:>12.4f}    "
          f"[{r['cer_consensus_ci_lo']:.4f}, {r['cer_consensus_ci_hi']:.4f}]")

# -- ANALYSIS 3: Bounded DER + diacritic P/R/F1 -----------------------
print("\n=== ANALYSIS 3: Bounded DER + diacritic precision/recall/F1 ===")

def diacritic_only(s: str) -> str:
    return "".join(c for c in s if c in DIACRITICS)

def diacritic_prf(ref: str, hyp: str) -> tuple[float, float, float]:
    """
    Position-aligned diacritic precision/recall/F1.

    Align the raw text (with diacritics) of ref and hyp at the character
    level. For each aligned position that is a diacritic in either, count:
      tp = both ref and hyp have a diacritic AND they match
      fp = hyp has a diacritic that ref does not (or wrong diacritic)
      fn = ref has a diacritic that hyp does not (or wrong diacritic)
    """
    r = preprocess(ref, mode="normalized")
    h = preprocess(hyp, mode="normalized")
    tp = fp = fn = 0
    ops = Levenshtein.opcodes(r, h)
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):
                if r[i1 + k] in DIACRITICS:
                    tp += 1
        elif tag == "replace":
            n_match = min(i2 - i1, j2 - j1)
            for k in range(n_match):
                rc, hc = r[i1 + k], h[j1 + k]
                if rc in DIACRITICS and hc in DIACRITICS:
                    if rc == hc:
                        tp += 1
                    else:
                        fp += 1; fn += 1
                elif rc in DIACRITICS:
                    fn += 1
                elif hc in DIACRITICS:
                    fp += 1
            # Excess on either side
            if (i2 - i1) > n_match:
                for k in range(i1 + n_match, i2):
                    if r[k] in DIACRITICS:
                        fn += 1
            elif (j2 - j1) > n_match:
                for k in range(j1 + n_match, j2):
                    if h[k] in DIACRITICS:
                        fp += 1
        elif tag == "delete":
            for k in range(i2 - i1):
                if r[i1 + k] in DIACRITICS:
                    fn += 1
        elif tag == "insert":
            for k in range(j2 - j1):
                if h[j1 + k] in DIACRITICS:
                    fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec and rec and not math.isnan(prec) and not math.isnan(rec)) else float("nan")
    return prec, rec, f1

def der_bounded(ref: str, hyp: str) -> float:
    r_d = diacritic_only(preprocess(ref, mode="raw"))
    h_d = diacritic_only(preprocess(hyp, mode="raw"))
    if not r_d and not h_d:
        return 0.0
    return Levenshtein.distance(r_d, h_d) / max(len(r_d), len(h_d))

der_rows = []
# vs Muhammad
for sid in SYSTEMS + ["gt_hasan"]:
    sub = corpus[corpus["system"] == sid].set_index("page_id")
    for pid in common_pages:
        if pid not in sub.index:
            continue
        prec, rec, f1 = diacritic_prf(gt_m[pid], sub.loc[pid, "text"])
        db = der_bounded(gt_m[pid], sub.loc[pid, "text"])
        der_rows.append({
            "page_id": pid,
            "system": sid,
            "ref": "muhammad",
            "der_bounded": db,
            "diacritic_precision": prec,
            "diacritic_recall": rec,
            "diacritic_f1": f1,
        })
der_df = pd.DataFrame(der_rows)
der_df.to_csv(TABLES / "revision_der_bounded.csv", index=False)
print(f"Wrote {TABLES / 'revision_der_bounded.csv'}")

der_summary = []
for sid in SYSTEMS + ["gt_hasan"]:
    sub = der_df[(der_df["system"] == sid) & (der_df["ref"] == "muhammad")]
    row = {"system": sid}
    for col in ["der_bounded", "diacritic_precision", "diacritic_recall", "diacritic_f1"]:
        mean, lo, hi = bootstrap_ci(sub[col].dropna().tolist())
        row[f"{col}_mean"] = mean
        row[f"{col}_ci_lo"] = lo
        row[f"{col}_ci_hi"] = hi
    der_summary.append(row)
der_sum_df = pd.DataFrame(der_summary)
der_sum_df.to_csv(TABLES / "revision_der_summary.csv", index=False)
print(f"Wrote {TABLES / 'revision_der_summary.csv'}")
print()
print("  -- Diacritic precision / recall / F1 (vs Muhammad) --")
print(f"  {'system':22s} {'DER_b':>8s} {'P':>8s} {'R':>8s} {'F1':>8s}")
for r in der_summary:
    print(f"  {r['system']:22s} {r['der_bounded_mean']:>8.3f} "
          f"{r['diacritic_precision_mean']:>8.3f} "
          f"{r['diacritic_recall_mean']:>8.3f} "
          f"{r['diacritic_f1_mean']:>8.3f}")

# -- ANALYSIS 4: Outlier sensitivity -----------------------------------
print("\n=== ANALYSIS 4: Outlier sensitivity (excluding pages 24, 33) ===")
ex = existing[(existing["ref"] == "best") &
              (~existing["page_id"].isin(CATASTROPHIC_PAGES))]
out_rows = []
for sid in SYSTEMS:
    sub = ex[ex["system"] == sid]
    for col in ["cer_no_tashkeel", "cer_normalized", "wer_no_tashkeel"]:
        mean, lo, hi = bootstrap_ci(sub[col].dropna().tolist())
        out_rows.append({
            "system": sid, "metric": col,
            "n_pages": len(sub),
            "mean_excl": mean, "ci_lo_excl": lo, "ci_hi_excl": hi,
        })
out_df = pd.DataFrame(out_rows)
out_df.to_csv(TABLES / "revision_outlier_sensitivity.csv", index=False)
print(f"Wrote {TABLES / 'revision_outlier_sensitivity.csv'}")
print()
print("  -- CER (no tashkeel, ref=best) excluding pages 24, 33 --")
print(f"  {'system':22s} {'CER (all)':>10s} {'CER (excl)':>12s} {'Δ':>8s}")
all_means = (existing[(existing["ref"] == "best")]
             .groupby("system")["cer_no_tashkeel"].mean())
for sid in SYSTEMS:
    a = all_means[sid]
    e = out_df[(out_df["system"] == sid) &
               (out_df["metric"] == "cer_no_tashkeel")]["mean_excl"].iloc[0]
    print(f"  {sid:22s} {a:>10.4f}    {e:>10.4f}   {e-a:>+8.4f}")

print("\n=== Done. New tables under results/tables/revision_*.csv ===")
