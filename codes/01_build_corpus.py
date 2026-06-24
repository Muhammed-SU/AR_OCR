"""
01_build_corpus.py
------------------
Reads ground truth (from BOTH annotators) + 5 system outputs for all 50 pages
and produces a single "long-format" CSV: one row per (page_id, system).

Two annotator-aware "systems" are added:
    gt_muhammad   (file naming: image_NN_GM.txt)
    gt_hasan      (file naming: image_NN_GH.txt, or İmage_NN_GH.txt on a
                   subset of pages — Turkish-locale uppercase İ)

Output:
    data/parallel_corpus.csv
        Columns: page_id, n, system, text, source_path
    data/manifest.csv
        Columns: page_id, n, image_path, gt_muhammad_path, gt_hasan_path,
                 has_<system_id>...
"""
from __future__ import annotations

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent  # /Volumes/LaCie/AR_OCR
PAPER_ROOT = Path(__file__).resolve().parent.parent   # /Volumes/LaCie/AR_OCR/Paper_JKSU
DATA_DIR = PAPER_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = ROOT / "images"
GT_M_DIR = ROOT / "Ground_Truth_Muhammed"
GT_H_DIR = ROOT / "Ground_Truth_Hasan"
RESULTS_ROOT = ROOT / "OCR_s_and_LLM_s_Results"

# OCR/LLM system_id -> (folder name, filename template).
# {n_variants} expands to padded forms (1, 01, 001); {n} to bare numeric.
SYSTEMS: dict[str, tuple[str, str]] = {
    "gemini_3_1_pro":    ("Google_Gemini_3.1_Pro_Results",   "gmn_{n_variants}.txt"),
    "claude_sonnet_4_6": ("Claude_Sonnet_4.6_Results",       "cld_{n_variants}.txt"),
    "kraken":            ("Kraken_OCR_Results",              "image_{n}_krkn.txt"),
    "google_vision":     ("Google_Cloud_Vision_API_Results", "image_{n}_G.V.txt"),
    "kraken_gemma":      ("Kraken_+_Gemma",                  "image_{n}_KG.txt"),
}

# Ground-truth annotators. Filename templates are tried in order; on a
# Turkish-locale macOS, some Hasan files were saved with capital İ.
ANNOTATORS: dict[str, tuple[Path, list[str]]] = {
    "gt_muhammad": (GT_M_DIR, ["image_{n}_GM.txt"]),
    "gt_hasan":    (GT_H_DIR, [
        "image_{n}_GH.txt",
        "İmage_{n}_GH.txt",   # Turkish-locale uppercase İ (U+0130)
        "Image_{n}_GH.txt",   # ASCII fallback
    ]),
}

N_PAGES = 50


def _variant_names(template: str, n: int) -> list[str]:
    """Expand a template into all plausible filename candidates for page n."""
    candidates: list[str] = []
    seen: set[str] = set()
    variants = [str(n), f"{n:02d}", f"{n:03d}"]
    for v in variants:
        name = template.replace("{n}", v).replace("{n_variants}", v)
        if name not in seen:
            seen.add(name)
            candidates.append(name)
    return candidates


def find_existing(folder: Path, n: int, templates: list[str]) -> Path | None:
    """Try every (template × numeric variant) combination for page n."""
    for tmpl in templates:
        for name in _variant_names(tmpl, n):
            candidate = folder / name
            if candidate.exists() and candidate.name != "." + candidate.stem:
                # Skip macOS resource-fork hidden files (._*.txt)
                if name.startswith("._"):
                    continue
                return candidate
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    manifest_rows: list[dict] = []
    parallel_rows: list[dict] = []
    missing: list[tuple[int, str]] = []

    for n in range(1, N_PAGES + 1):
        page_id = f"page_{n:02d}"
        image_path = IMAGES_DIR / f"image_{n}.jpg"
        row: dict = {
            "page_id": page_id,
            "n": n,
            "image_path": str(image_path) if image_path.exists() else "",
        }

        # Ground truths -----------------------------------------------------
        for ann_id, (folder, templates) in ANNOTATORS.items():
            found = find_existing(folder, n, templates)
            row[f"{ann_id}_path"] = str(found) if found else ""
            if found:
                parallel_rows.append({
                    "page_id": page_id,
                    "n": n,
                    "system": ann_id,
                    "text": read_text(found),
                    "source_path": str(found),
                })
            else:
                missing.append((n, ann_id))

        # Each OCR / LLM system --------------------------------------------
        for system_id, (folder_name, template) in SYSTEMS.items():
            folder = RESULTS_ROOT / folder_name
            found = find_existing(folder, n, [template])
            row[f"has_{system_id}"] = bool(found)
            if found:
                parallel_rows.append({
                    "page_id": page_id,
                    "n": n,
                    "system": system_id,
                    "text": read_text(found),
                    "source_path": str(found),
                })
            else:
                missing.append((n, system_id))

        manifest_rows.append(row)

    # --- Write manifest --------------------------------------------------
    manifest_csv = DATA_DIR / "manifest.csv"
    base_fields = ["page_id", "n", "image_path",
                   "gt_muhammad_path", "gt_hasan_path"]
    sys_fields = [f"has_{sid}" for sid in SYSTEMS]
    fieldnames = base_fields + sys_fields
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # --- Write parallel corpus ------------------------------------------
    corpus_csv = DATA_DIR / "parallel_corpus.csv"
    with corpus_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["page_id", "n", "system", "text", "source_path"]
        )
        writer.writeheader()
        writer.writerows(parallel_rows)

    # --- Report ----------------------------------------------------------
    print(f"Wrote manifest  → {manifest_csv}")
    print(f"Wrote corpus    → {corpus_csv}")
    print(f"  Pages: {len(manifest_rows)}")
    print(f"  Total (page, system) rows: {len(parallel_rows)}")

    # Count rows per system
    counts: dict[str, int] = {}
    for r in parallel_rows:
        counts[r["system"]] = counts.get(r["system"], 0) + 1
    print("  Coverage per system:")
    for sid in list(ANNOTATORS.keys()) + list(SYSTEMS.keys()):
        print(f"    {sid:22s}  {counts.get(sid, 0):>3d}/{N_PAGES} pages")

    if missing:
        print(f"  ⚠ Missing files: {len(missing)}")
        for n, sid in missing[:20]:
            print(f"     page {n:02d} / {sid}")
        if len(missing) > 20:
            print(f"     ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
