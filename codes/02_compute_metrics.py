"""
02_compute_metrics.py
---------------------
Computes the paper's full metric panel under the dual-annotator regime.

The corpus has TWO independent ground-truth transcriptions (Muhammad =
`gt_muhammad`, Hasan = `gt_hasan`). For every OCR / LLM system we therefore
emit metric values under THREE reference regimes:

    ref = "muhammad"     score = system vs. Muhammad
    ref = "hasan"        score = system vs. Hasan
    ref = "best"         multi-reference: score is taken w.r.t. whichever
                         annotator's text is closest (smallest edit distance).
                         For chrF / BLEU we take the MAX score across the two.

In addition, we compute INTER-ANNOTATOR AGREEMENT (IAA): the same metric
panel with Hasan as hypothesis and Muhammad as reference (or vice versa).
This IAA is reported as a "human-ceiling" baseline against which system
performance is contextualised.

Output tables (under results/tables/):
    per_page_metrics.csv          long-format: page_id × system × ref ×
                                  metric value
    summary_per_system.csv        mean + 95% bootstrap CI, by system × ref
    summary_per_system_x_difficulty.csv  same but stratified by difficulty
    iaa_summary.csv               IAA values (Hasan vs Muhammad), per page +
                                  aggregate
    wilcoxon_pairwise.csv         paired Wilcoxon on per-page CER (no
                                  tashkeel, ref="best") across systems
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    import Levenshtein
except ImportError as e:
    raise SystemExit("pip install python-Levenshtein") from e

try:
    from sacrebleu.metrics import CHRF, BLEU
except ImportError as e:
    raise SystemExit("pip install sacrebleu") from e

from scipy.stats import wilcoxon

from arabic_utils import (
    preprocess,
    char_tokens,
    word_tokens,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAPER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_ROOT / "data"
RESULTS_DIR = PAPER_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_CSV = DATA_DIR / "parallel_corpus.csv"

ANNOTATORS = ["gt_muhammad", "gt_hasan"]
SYSTEMS = [
    "gemini_3_1_pro",
    "claude_sonnet_4_6",
    "kraken",
    "google_vision",
    "kraken_gemma",
]
REFS = ["muhammad", "hasan", "best"]
BOOTSTRAP_N = 1000


# ---------------------------------------------------------------------------
# Core distances
# ---------------------------------------------------------------------------

def _seq_levenshtein(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def cer(ref: str, hyp: str) -> float:
    rs = "".join(char_tokens(ref))
    hs = "".join(char_tokens(hyp))
    return Levenshtein.distance(rs, hs) / max(len(rs), 1)


def wer(ref: str, hyp: str) -> float:
    rs = word_tokens(ref)
    hs = word_tokens(hyp)
    return _seq_levenshtein(rs, hs) / max(len(rs), 1)


_CHRF = CHRF(word_order=0)
_BLEU = BLEU(effective_order=True)


def metrics_one_pair(gt: str, hp: str) -> dict[str, float]:
    g_raw = preprocess(gt, mode="raw")
    h_raw = preprocess(hp, mode="raw")
    g_norm = preprocess(gt, mode="normalized")
    h_norm = preprocess(hp, mode="normalized")
    g_nt = preprocess(gt, mode="no_tashkeel")
    h_nt = preprocess(hp, mode="no_tashkeel")
    g_dia = preprocess(gt, mode="tashkeel_only")
    h_dia = preprocess(hp, mode="tashkeel_only")

    d: dict[str, float] = {}
    d["cer_raw"]          = cer(g_raw,  h_raw)
    d["cer_normalized"]   = cer(g_norm, h_norm)
    d["cer_no_tashkeel"]  = cer(g_nt,   h_nt)
    d["wer_normalized"]   = wer(g_norm, h_norm)
    d["wer_no_tashkeel"]  = wer(g_nt,   h_nt)
    d["der"] = (Levenshtein.distance(g_dia, h_dia) / len(g_dia)) if len(g_dia) > 0 else float("nan")
    d["chrf_no_tashkeel"] = _CHRF.sentence_score(h_nt, [g_nt]).score / 100.0
    d["bleu_no_tashkeel"] = _BLEU.sentence_score(h_nt, [g_nt]).score / 100.0
    return d


def combine_best(m: dict[str, float], h: dict[str, float]) -> dict[str, float]:
    """
    For error-rate metrics (lower is better) take the min.
    For chrF/BLEU (higher is better) take the max.
    """
    out: dict[str, float] = {}
    for k in m:
        a, b = m[k], h[k]
        if math.isnan(a):
            out[k] = b
        elif math.isnan(b):
            out[k] = a
        elif k.startswith(("cer_", "wer_", "der")):
            out[k] = min(a, b)
        elif k.startswith(("chrf", "bleu")):
            out[k] = max(a, b)
        else:
            out[k] = (a + b) / 2
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], *, n_resamples: int = BOOTSTRAP_N,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    vals = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(20260518)
    means = [float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
             for _ in range(n_resamples)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(vals)), lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CORPUS_CSV.exists():
        raise SystemExit(f"Corpus not found: {CORPUS_CSV}. Run 01_build_corpus.py first.")

    df = pd.read_csv(CORPUS_CSV)

    gt_m = df[df["system"] == "gt_muhammad"].set_index("page_id")["text"]
    gt_h = df[df["system"] == "gt_hasan"].set_index("page_id")["text"]
    common_pages = sorted(set(gt_m.index) & set(gt_h.index))
    print(f"Pages with both annotators: {len(common_pages)}")

    # -----------------------------------------------------------------
    # 1) IAA: Hasan vs Muhammad (treat Hasan as hypothesis, M as ref)
    # -----------------------------------------------------------------
    iaa_rows: list[dict] = []
    for pid in common_pages:
        m = metrics_one_pair(gt_m[pid], gt_h[pid])
        m["page_id"] = pid
        iaa_rows.append(m)
    iaa_df = pd.DataFrame(iaa_rows)
    iaa_csv = RESULTS_DIR / "iaa_per_page.csv"
    iaa_df.to_csv(iaa_csv, index=False)
    print(f"Wrote per-page IAA → {iaa_csv}")

    metric_cols = [c for c in iaa_df.columns if c != "page_id"]
    iaa_summary_rows: list[dict] = []
    row = {"comparison": "hasan_vs_muhammad"}
    for m_col in metric_cols:
        mean, lo, hi = bootstrap_ci(iaa_df[m_col].dropna().tolist())
        row[f"{m_col}_mean"] = mean
        row[f"{m_col}_ci_lo"] = lo
        row[f"{m_col}_ci_hi"] = hi
    iaa_summary_rows.append(row)
    iaa_sum_df = pd.DataFrame(iaa_summary_rows)
    iaa_sum_csv = RESULTS_DIR / "iaa_summary.csv"
    iaa_sum_df.to_csv(iaa_sum_csv, index=False)
    print(f"Wrote IAA summary → {iaa_sum_csv}")

    # -----------------------------------------------------------------
    # 2) Per-page system metrics (× three reference regimes)
    # -----------------------------------------------------------------
    per_page_records: list[dict] = []
    for sid in SYSTEMS:
        sub = df[df["system"] == sid].set_index("page_id")
        for pid in common_pages:
            if pid not in sub.index:
                continue
            hyp = sub.loc[pid, "text"]
            m_metrics = metrics_one_pair(gt_m[pid], hyp)
            h_metrics = metrics_one_pair(gt_h[pid], hyp)
            best_metrics = combine_best(m_metrics, h_metrics)
            for ref_label, mset in [("muhammad", m_metrics),
                                    ("hasan", h_metrics),
                                    ("best", best_metrics)]:
                record = {"page_id": pid, "system": sid, "ref": ref_label}
                record.update(mset)
                per_page_records.append(record)

    per_page = pd.DataFrame(per_page_records)
    out_per_page = RESULTS_DIR / "per_page_metrics.csv"
    per_page.to_csv(out_per_page, index=False)
    print(f"Wrote per-page metrics → {out_per_page} ({len(per_page)} rows)")

    # -----------------------------------------------------------------
    # 3) System × ref summary with bootstrap CI
    # -----------------------------------------------------------------
    summary_rows: list[dict] = []
    for sid in SYSTEMS:
        for ref_label in REFS:
            sub = per_page[(per_page["system"] == sid) &
                           (per_page["ref"] == ref_label)]
            row = {"system": sid, "ref": ref_label, "n_pages": len(sub)}
            for m_col in metric_cols:
                mean, lo, hi = bootstrap_ci(sub[m_col].dropna().tolist())
                row[f"{m_col}_mean"]  = mean
                row[f"{m_col}_ci_lo"] = lo
                row[f"{m_col}_ci_hi"] = hi
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    out_summary = RESULTS_DIR / "summary_per_system.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote summary → {out_summary}")

    # -----------------------------------------------------------------
    # 4) Pairwise Wilcoxon on per-page CER (no tashkeel, ref="best")
    # -----------------------------------------------------------------
    best_only = per_page[per_page["ref"] == "best"]
    pivot = best_only.pivot(index="page_id", columns="system",
                            values="cer_no_tashkeel")
    wilcoxon_rows: list[dict] = []
    for i in range(len(SYSTEMS)):
        for j in range(i + 1, len(SYSTEMS)):
            a, b = SYSTEMS[i], SYSTEMS[j]
            if a not in pivot.columns or b not in pivot.columns:
                continue
            paired = pivot[[a, b]].dropna()
            if len(paired) < 5:
                continue
            stat, p = wilcoxon(paired[a].values, paired[b].values,
                               zero_method="zsplit")
            wilcoxon_rows.append({
                "system_a": a,
                "system_b": b,
                "n_pairs": len(paired),
                "mean_diff": float((paired[a] - paired[b]).mean()),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
            })
    wilcoxon_rows.sort(key=lambda r: r["p_value"])
    m_tests = len(wilcoxon_rows)
    for idx, r in enumerate(wilcoxon_rows):
        r["p_holm"] = min(1.0, r["p_value"] * (m_tests - idx))
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    out_wilcoxon = RESULTS_DIR / "wilcoxon_pairwise.csv"
    wilcoxon_df.to_csv(out_wilcoxon, index=False)
    print(f"Wrote Wilcoxon → {out_wilcoxon} ({len(wilcoxon_df)} pairs)")

    # -----------------------------------------------------------------
    # 5) Stratify by difficulty (ref = best)
    # -----------------------------------------------------------------
    difficulty_csv = DATA_DIR / "difficulty_labels.csv"
    if difficulty_csv.exists():
        labels = pd.read_csv(difficulty_csv)
        merged = per_page.merge(labels[["page_id", "difficulty"]],
                                on="page_id", how="left")
        strat_rows: list[dict] = []
        for sid in SYSTEMS:
            for ref_label in REFS:
                sub = merged[(merged["system"] == sid) &
                             (merged["ref"] == ref_label)]
                for diff, ssub in sub.groupby("difficulty"):
                    row = {"system": sid, "ref": ref_label,
                           "difficulty": diff, "n_pages": len(ssub)}
                    for m_col in metric_cols:
                        mean, lo, hi = bootstrap_ci(ssub[m_col].dropna().tolist())
                        row[f"{m_col}_mean"] = mean
                        row[f"{m_col}_ci_lo"] = lo
                        row[f"{m_col}_ci_hi"] = hi
                    strat_rows.append(row)
        strat = pd.DataFrame(strat_rows)
        out_strat = RESULTS_DIR / "summary_per_system_x_difficulty.csv"
        strat.to_csv(out_strat, index=False)
        print(f"Wrote stratified summary → {out_strat}")
    else:
        print(f"(Skip stratification — no {difficulty_csv})")

    # -----------------------------------------------------------------
    # 6) Console headline
    # -----------------------------------------------------------------
    print("\n=== Headline (CER no-tashkeel, ref=best, mean ± 95% CI) ===")
    headline = summary[summary["ref"] == "best"].sort_values(
        "cer_no_tashkeel_mean"
    )
    for _, r in headline.iterrows():
        print(f"  {r['system']:22s}  "
              f"CER = {r['cer_no_tashkeel_mean']:.4f}  "
              f"[{r['cer_no_tashkeel_ci_lo']:.4f}, {r['cer_no_tashkeel_ci_hi']:.4f}]")
    print()
    print("=== IAA (Hasan vs Muhammad, CER no-tashkeel) ===")
    ir = iaa_sum_df.iloc[0]
    print(f"  hasan_vs_muhammad   "
          f"CER = {ir['cer_no_tashkeel_mean']:.4f}  "
          f"[{ir['cer_no_tashkeel_ci_lo']:.4f}, {ir['cer_no_tashkeel_ci_hi']:.4f}]")
    print(f"  hasan_vs_muhammad   "
          f"DER = {ir['der_mean']:.4f}  "
          f"[{ir['der_ci_lo']:.4f}, {ir['der_ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
