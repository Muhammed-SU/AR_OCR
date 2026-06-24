"""
03_make_figures.py
------------------
Produces the paper's figures from the dual-annotator metric tables.

Inputs:
    results/tables/per_page_metrics.csv     (long format with `ref` column)
    results/tables/summary_per_system.csv   (system × ref)
    results/tables/iaa_per_page.csv         (Hasan vs Muhammad)
    (optional) data/difficulty_labels.csv

The "best" reference regime (multi-reference: system credited against
the closer annotator) is used for the main figures. Single-reference
plots are produced as supplementary.

Outputs (results/figures/):
    fig01_box_cer_no_tashkeel.png/.pdf            ref=best
    fig02_box_cer_normalized.png/.pdf             ref=best
    fig03_box_der.png/.pdf                        ref=best
    fig04_per_page_heatmap.png/.pdf               ref=best
    fig05_cost_vs_quality.png/.pdf
    fig06_difficulty_stratified.png/.pdf
    fig07_chrf_vs_bleu.png/.pdf
    fig08_iaa_vs_systems.png/.pdf                 NEW — human ceiling
    fig09_per_ref_comparison.png/.pdf             NEW — Muh vs Hasan refs
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PAPER_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PAPER_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    "gemini_3_1_pro",
    "claude_sonnet_4_6",
    "kraken",
    "google_vision",
    "kraken_gemma",
]
DISPLAY_NAMES = {
    "gemini_3_1_pro":    "Gemini 3.1 Pro",
    "claude_sonnet_4_6": "Claude Sonnet 4.6",
    "kraken":            "Kraken OCR",
    "google_vision":     "Google Cloud Vision",
    "kraken_gemma":      "Kraken + Gemma",
    "hasan_vs_muhammad": "IAA (Hasan ↔ Muhammad)",
}
# Per-page cost estimates (USD) — RATE-CARD values used for prospective
# deployment guidance only. The present study incurred zero actual cost:
# Gemini 3.1 Pro was invoked under a no-cost student tier of Google AI
# Studio; Claude Sonnet 4.6 inference fit inside an existing subscription
# allowance; Kraken and Gemma run locally (Ollama); Google Cloud Vision
# was within free-tier monthly quota for 50 pages.
SYSTEM_COST_USD = {
    "gemini_3_1_pro":    0.030,   # rate-card estimate 2026-05
    "claude_sonnet_4_6": 0.045,   # rate-card estimate 2026-05
    "kraken":            0.000,   # local
    "google_vision":     0.0015,  # GCV doc OCR pricing tier
    "kraken_gemma":      0.000,   # local (Ollama)
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  → {out.name}")
    plt.close(fig)


def _ordered(df: pd.DataFrame, metric: str) -> list[str]:
    med = df.groupby("system")[metric].median()
    return [s for s, _ in sorted(med.items(), key=lambda kv: kv[1])]


def fig_box(df_best: pd.DataFrame, metric: str, ylabel: str, fname: str,
            iaa: pd.DataFrame | None = None) -> None:
    order = _ordered(df_best, metric)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    sns.boxplot(data=df_best, x="system", y=metric, order=order, ax=ax,
                width=0.55, showfliers=False, palette="Set2")
    sns.stripplot(data=df_best, x="system", y=metric, order=order, ax=ax,
                  color="black", size=2.5, alpha=0.5, jitter=0.15)
    # Human ceiling (IAA) — horizontal band
    if iaa is not None and metric in iaa.columns:
        iaa_mean = iaa[metric].mean()
        ax.axhline(iaa_mean, color="crimson", lw=1.2, linestyle="--",
                   label=f"IAA mean = {iaa_mean:.3f}")
        ax.legend(loc="upper left", frameon=False)
    ax.set_xticklabels([DISPLAY_NAMES[s] for s in order], rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    _save(fig, fname)


def fig_heatmap(df_best: pd.DataFrame, metric: str, fname: str) -> None:
    pivot = df_best.pivot(index="page_id", columns="system", values=metric)
    pivot = pivot[[s for s in SYSTEMS if s in pivot.columns]]
    fig, ax = plt.subplots(figsize=(6.5, 11))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", vmin=0,
                cbar_kws={"label": metric},
                linewidths=0.3, linecolor="white")
    ax.set_xticklabels([DISPLAY_NAMES[s] for s in pivot.columns],
                       rotation=30, ha="right")
    ax.set_ylabel("Page")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, fname)


def fig_cost_vs_quality(summary: pd.DataFrame, fname: str,
                        iaa: pd.DataFrame | None = None) -> None:
    sub = summary[summary["ref"] == "best"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, row in sub.iterrows():
        sid = row["system"]
        cost = SYSTEM_COST_USD.get(sid, 0)
        cer_v = row["cer_no_tashkeel_mean"]
        ax.scatter(cost, cer_v, s=120, zorder=3)
        ax.annotate(DISPLAY_NAMES.get(sid, sid), (cost, cer_v),
                    xytext=(8, 4), textcoords="offset points", fontsize=10)
    if iaa is not None:
        iaa_v = iaa["cer_no_tashkeel"].mean()
        ax.axhline(iaa_v, color="crimson", lw=1.2, linestyle="--",
                   label=f"IAA ceiling = {iaa_v:.3f}")
        ax.legend(frameon=False)
    ax.set_xlabel("Estimated cost (USD per page)")
    ax.set_ylabel("Mean CER (no tashkeel)  ↓ better")
    ax.set_xscale("symlog", linthresh=1e-3)
    fig.tight_layout()
    _save(fig, fname)


def fig_difficulty(df_best: pd.DataFrame, labels: pd.DataFrame, fname: str,
                   iaa_by_diff: pd.DataFrame | None = None) -> None:
    merged = df_best.merge(labels[["page_id", "difficulty"]],
                           on="page_id", how="left")
    agg = (merged.groupby(["system", "difficulty"], dropna=False)
                 ["cer_no_tashkeel"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=agg, x="difficulty", y="cer_no_tashkeel",
                hue="system", ax=ax, palette="Set2")
    if iaa_by_diff is not None:
        for _, row in iaa_by_diff.iterrows():
            ax.scatter(row["difficulty"], row["cer_no_tashkeel"],
                       marker="*", color="crimson", s=140, zorder=5)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, [DISPLAY_NAMES.get(l, l) for l in lbls],
              frameon=False, fontsize=9)
    ax.set_xlabel("Manuscript difficulty class")
    ax.set_ylabel("Mean CER (no tashkeel)")
    fig.tight_layout()
    _save(fig, fname)


def fig_chrf_vs_bleu(df_best: pd.DataFrame, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.scatterplot(data=df_best, x="bleu_no_tashkeel",
                    y="chrf_no_tashkeel", hue="system", style="system",
                    s=60, ax=ax)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, [DISPLAY_NAMES.get(l, l) for l in lbls],
              frameon=False)
    ax.set_xlabel("BLEU (no tashkeel)")
    ax.set_ylabel("chrF (no tashkeel)")
    fig.tight_layout()
    _save(fig, fname)


def fig_iaa_vs_systems(summary: pd.DataFrame, iaa_summary: pd.DataFrame,
                       fname: str) -> None:
    """Bar chart: each system's CER (ref=best) + IAA reference line."""
    sub = summary[summary["ref"] == "best"].copy()
    sub["label"] = sub["system"].map(DISPLAY_NAMES)
    sub = sub.sort_values("cer_no_tashkeel_mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(sub["label"], sub["cer_no_tashkeel_mean"],
           yerr=[sub["cer_no_tashkeel_mean"] - sub["cer_no_tashkeel_ci_lo"],
                 sub["cer_no_tashkeel_ci_hi"] - sub["cer_no_tashkeel_mean"]],
           color=sns.color_palette("Set2", len(sub)), capsize=4)
    iaa_v = float(iaa_summary["cer_no_tashkeel_mean"].iloc[0])
    ax.axhline(iaa_v, color="crimson", lw=1.4, linestyle="--",
               label=f"IAA (Hasan vs Muhammad) = {iaa_v:.3f}")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xticklabels(sub["label"], rotation=20, ha="right")
    ax.set_ylabel("CER (no tashkeel), ref = best")
    fig.tight_layout()
    _save(fig, fname)


def fig_per_ref_comparison(per_page: pd.DataFrame, fname: str) -> None:
    """For each system, compare CER against Muhammad vs Hasan refs."""
    sub = per_page[per_page["ref"].isin(["muhammad", "hasan"])].copy()
    sub["label"] = sub["system"].map(DISPLAY_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=sub, x="label", y="cer_no_tashkeel", hue="ref",
                ax=ax, palette={"muhammad": "#4C72B0", "hasan": "#DD8452"},
                showfliers=False, width=0.6)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, ["vs Muhammad", "vs Hasan"], frameon=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("CER (no tashkeel)")
    fig.tight_layout()
    _save(fig, fname)


def main() -> None:
    per_page = pd.read_csv(TABLES_DIR / "per_page_metrics.csv")
    summary = pd.read_csv(TABLES_DIR / "summary_per_system.csv")
    iaa = pd.read_csv(TABLES_DIR / "iaa_per_page.csv")
    iaa_sum = pd.read_csv(TABLES_DIR / "iaa_summary.csv")

    df_best = per_page[per_page["ref"] == "best"]

    print("Figures:")
    fig_box(df_best, "cer_no_tashkeel",
            "CER (no tashkeel), ref = best",
            "fig01_box_cer_no_tashkeel", iaa=iaa)
    fig_box(df_best, "cer_normalized",
            "CER (with tashkeel), ref = best",
            "fig02_box_cer_normalized", iaa=iaa)
    fig_box(df_best, "der",
            "Diacritic Error Rate, ref = best",
            "fig03_box_der", iaa=iaa)
    fig_heatmap(df_best, "cer_no_tashkeel", "fig04_per_page_heatmap")
    fig_cost_vs_quality(summary, "fig05_cost_vs_quality", iaa=iaa)

    labels_csv = PAPER_ROOT / "data" / "difficulty_labels.csv"
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv)
        # Compute IAA per difficulty class
        iaa_with_diff = iaa.merge(labels[["page_id", "difficulty"]],
                                  on="page_id", how="left")
        iaa_by_diff = iaa_with_diff.groupby("difficulty")["cer_no_tashkeel"].mean().reset_index()
        fig_difficulty(df_best, labels, "fig06_difficulty_stratified",
                       iaa_by_diff=iaa_by_diff)
    else:
        print("(Skipping fig06 — no difficulty labels)")

    fig_chrf_vs_bleu(df_best, "fig07_chrf_vs_bleu")
    fig_iaa_vs_systems(summary, iaa_sum, "fig08_iaa_vs_systems")
    fig_per_ref_comparison(per_page, "fig09_per_ref_comparison")


if __name__ == "__main__":
    main()
