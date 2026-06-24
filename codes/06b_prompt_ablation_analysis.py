"""
06b_prompt_ablation_analysis.py
-------------------------------
Reads results/prompt_ablation/{system}/{variant}/image_*.txt
and computes CER_no_tashkeel per (system, variant, page) against both
Muhammad and Hasan ground truth, then aggregates to (system, variant)
means with bootstrap CIs.

Output: results/tables/prompt_ablation_summary.csv

Use after 06_prompt_ablation.py has produced the transcriptions.
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

from arabic_utils import preprocess, char_tokens

PAPER = Path(__file__).resolve().parent.parent
DATA = PAPER / "data"
ABL = PAPER / "results" / "prompt_ablation"
TABLES = PAPER / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def cer(ref: str, hyp: str) -> float:
    rs = "".join(char_tokens(preprocess(ref, mode="no_tashkeel")))
    hs = "".join(char_tokens(preprocess(hyp, mode="no_tashkeel")))
    return Levenshtein.distance(rs, hs) / max(len(rs), 1)


def bootstrap_ci(vals: list[float], n: int = 1000, alpha: float = 0.05):
    arr = np.array([v for v in vals if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(20260518)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n)]
    return float(np.mean(arr)), \
           float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def main() -> None:
    corpus = pd.read_csv(DATA / "parallel_corpus.csv")
    gt_m = corpus[corpus["system"] == "gt_muhammad"].set_index("page_id")["text"]
    gt_h = corpus[corpus["system"] == "gt_hasan"].set_index("page_id")["text"]

    rows = []
    for sys_dir in sorted(ABL.iterdir()) if ABL.exists() else []:
        if not sys_dir.is_dir():
            continue
        for var_dir in sorted(sys_dir.iterdir()):
            if not var_dir.is_dir():
                continue
            for txt in sorted(var_dir.glob("image_*.txt")):
                n = int(txt.stem.split("_")[1])
                pid = f"page_{n:02d}"
                hyp = txt.read_text(encoding="utf-8", errors="replace")
                if not hyp:
                    continue
                e_m = cer(gt_m.get(pid, ""), hyp) if pid in gt_m.index else float("nan")
                e_h = cer(gt_h.get(pid, ""), hyp) if pid in gt_h.index else float("nan")
                best = min(e_m, e_h) if (not math.isnan(e_m) and not math.isnan(e_h)) \
                       else (e_m if not math.isnan(e_m) else e_h)
                rows.append({
                    "system": sys_dir.name,
                    "variant": var_dir.name,
                    "page_id": pid,
                    "cer_vs_muhammad": e_m,
                    "cer_vs_hasan": e_h,
                    "cer_best_of_two": best,
                })

    per_page = pd.DataFrame(rows)
    per_page.to_csv(TABLES / "prompt_ablation_per_page.csv", index=False)
    print(f"Wrote {TABLES / 'prompt_ablation_per_page.csv'} ({len(per_page)} rows)")

    summary = []
    for (sys_name, var), sub in per_page.groupby(["system", "variant"]):
        row = {"system": sys_name, "variant": var, "n_pages": len(sub)}
        for col in ["cer_vs_muhammad", "cer_vs_hasan", "cer_best_of_two"]:
            mean, lo, hi = bootstrap_ci(sub[col].dropna().tolist())
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_lo"] = lo
            row[f"{col}_ci_hi"] = hi
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TABLES / "prompt_ablation_summary.csv", index=False)
    print(f"Wrote {TABLES / 'prompt_ablation_summary.csv'}")
    print()
    print("=== Prompt-ablation CER (no tashkeel, best-of-two) ===")
    print(f"  {'system':10s} {'variant':10s} {'n':>4s} {'CER':>10s} {'95% CI':>20s}")
    for _, r in summary_df.iterrows():
        print(f"  {r['system']:10s} {r['variant']:10s} {int(r['n_pages']):>4d} "
              f"{r['cer_best_of_two_mean']:>9.4f}  "
              f"[{r['cer_best_of_two_ci_lo']:.4f}, {r['cer_best_of_two_ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
