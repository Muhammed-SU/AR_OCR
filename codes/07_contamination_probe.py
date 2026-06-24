"""
07_contamination_probe.py
-------------------------
Memorisation-completion probe for training-data contamination.

For each tested page:
  1. Crop the TOP HALF of the page image.
  2. Prompt the multimodal LLM to transcribe what is visible AND to
     predict what comes immediately after the crop.
  3. Compare the predicted continuation to the actual bottom-half
     ground truth using character n-gram overlap.

If a model produces a high-fidelity continuation of a page it never
visibly saw, that is strong evidence of memorisation. Memorisation
rates should be substantially higher on public-source pages than on
private Furūsiyya pages (which the authors maintain were never
public-indexed prior to model training).

Usage:
  export GOOGLE_API_KEY=...
  export ANTHROPIC_API_KEY=...
  python3 07_contamination_probe.py \
      --systems gemini claude \
      --pages 1-50 \
      --crop-fraction 0.5

Output:
  results/contamination/{system}/image_{N}_completion.txt
  results/tables/contamination_probe_summary.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import math
import os
import time
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("pip install pillow") from e

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pip install pandas") from e

try:
    from google import genai
    from google.genai import types as gtypes
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from arabic_utils import preprocess

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT / "images"
PAPER = Path(__file__).resolve().parent.parent
OUT_ROOT = PAPER / "results" / "contamination"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
TABLES = PAPER / "results" / "tables"

GEMINI_MODEL_ID = "gemini-3.1-pro-preview"
CLAUDE_MODEL_ID = "claude-sonnet-4-6"

PROMPT = (
    "You are shown the TOP half of a classical Arabic manuscript page. "
    "Two tasks:\n\n"
    "1) Transcribe the Arabic text you can see in the image.\n"
    "2) Predict the text that you believe follows on the BOTTOM half of "
    "the page, based on your knowledge of this manuscript or similar "
    "texts. Mark this clearly with a heading `--- PREDICTED CONTINUATION ---`.\n\n"
    "If you do not recognise the manuscript or have no confident "
    "prediction, write `--- PREDICTED CONTINUATION ---` followed by the "
    "single word `UNKNOWN` and nothing else.\n\n"
    "Do not add explanations or commentary outside the two sections."
)


def crop_top(image_path: Path, frac: float = 0.5) -> tuple[bytes, str]:
    img = Image.open(image_path)
    w, h = img.size
    top = img.crop((0, 0, w, int(h * frac)))
    buf = BytesIO()
    top.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def transcribe_gemini(img_bytes: bytes, mime: str, model: str = GEMINI_MODEL_ID) -> str:
    if not HAS_GOOGLE:
        raise RuntimeError("pip install google-genai")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=[gtypes.Part.from_bytes(data=img_bytes, mime_type=mime), PROMPT],
    )
    return resp.text or ""


def transcribe_claude(img_bytes: bytes, mime: str, model: str = CLAUDE_MODEL_ID) -> str:
    if not HAS_ANTHROPIC:
        raise RuntimeError("pip install anthropic")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64 = base64.b64encode(img_bytes).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": mime, "data": b64
                }},
                {"type": "text", "text": PROMPT},
            ]},
        ],
    )
    return msg.content[0].text if msg.content else ""


SYSTEMS = {"gemini": transcribe_gemini, "claude": transcribe_claude}


def split_response(text: str) -> tuple[str, str]:
    """Split into (transcription_of_top, predicted_continuation)."""
    marker = "--- PREDICTED CONTINUATION ---"
    if marker not in text:
        return text.strip(), ""
    top, bottom = text.split(marker, 1)
    return top.strip(), bottom.strip()


def char_ngram_overlap(a: str, b: str, n: int = 5) -> float:
    """Jaccard-like overlap of char n-grams (after no-tashkeel norm)."""
    a_n = preprocess(a, mode="no_tashkeel").replace(" ", "")
    b_n = preprocess(b, mode="no_tashkeel").replace(" ", "")
    if len(a_n) < n or len(b_n) < n:
        return 0.0
    A = {a_n[i:i + n] for i in range(len(a_n) - n + 1)}
    B = {b_n[i:i + n] for i in range(len(b_n) - n + 1)}
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def parse_pages(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", choices=list(SYSTEMS),
                    default=list(SYSTEMS))
    ap.add_argument("--pages", default="1-50")
    ap.add_argument("--crop-fraction", type=float, default=0.5)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pages = parse_pages(args.pages)
    prov = pd.read_csv(PAPER / "data" / "manuscript_provenance.csv")
    prov_map = prov.set_index("n")["repository"].to_dict()

    rows = []
    for sys_name in args.systems:
        fn = SYSTEMS[sys_name]
        outdir = OUT_ROOT / sys_name
        outdir.mkdir(parents=True, exist_ok=True)
        for n in pages:
            outpath = outdir / f"image_{n}_completion.txt"
            if outpath.exists() and not args.overwrite:
                text = outpath.read_text(encoding="utf-8", errors="replace")
            else:
                img_path = IMAGES_DIR / f"image_{n}.jpg"
                if not img_path.exists():
                    print(f"  skip (no image): {n}")
                    continue
                img_bytes, mime = crop_top(img_path, args.crop_fraction)
                try:
                    text = fn(img_bytes, mime)
                    outpath.write_text(text, encoding="utf-8")
                    print(f"  done: {sys_name}/image_{n}  ({len(text)} chars)")
                except Exception as e:
                    print(f"  FAIL: {sys_name}/image_{n}: {e}")
                    continue
                time.sleep(args.sleep)
            _, prediction = split_response(text)
            # Compare prediction to GT bottom half (approximate by 2nd half of GT)
            gt_path = ROOT / "Ground_Truth_Muhammed" / f"image_{n}_GM.txt"
            if not gt_path.exists():
                continue
            gt = gt_path.read_text(encoding="utf-8", errors="replace")
            cut = len(gt) // 2
            gt_bottom = gt[cut:]
            overlap = char_ngram_overlap(prediction, gt_bottom)
            subset = ("private" if prov_map.get(n) ==
                      "Private Furūsiyya corpus (authors)" else "public")
            rows.append({
                "page_id": f"page_{n:02d}",
                "n": n,
                "system": sys_name,
                "subset": subset,
                "prediction_chars": len(prediction),
                "gt_bottom_chars": len(gt_bottom),
                "char_5gram_jaccard": overlap,
                "predicted_unknown": prediction.strip().upper() == "UNKNOWN",
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "contamination_probe_summary.csv", index=False)
    print(f"\nWrote {TABLES / 'contamination_probe_summary.csv'}")
    print()
    print("=== Contamination probe (char 5-gram Jaccard, prediction vs GT bottom) ===")
    if len(df) == 0:
        print("(no rows collected)")
        return
    agg = (df.groupby(["system", "subset"])
             .agg(n=("char_5gram_jaccard", "count"),
                  jaccard_mean=("char_5gram_jaccard", "mean"),
                  unknown_rate=("predicted_unknown", "mean"))
             .reset_index())
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
