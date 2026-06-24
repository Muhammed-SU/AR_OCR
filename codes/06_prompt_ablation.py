"""
06_prompt_ablation.py
---------------------
Prompt-ablation experiment for Gemini 3.1 Pro and Claude Sonnet 4.6.

Runs each of three prompt variants on each of 50 manuscript pages for each
of the two frontier multimodal LLMs, then compares the resulting CER
distributions to test how much of the headline Gemini lead is prompt-
dependent.

Prompt variants:
    V1 = baseline (the original Appendix A.1 prompt)
    V2 = strict no-restoration ("transcribe only visible diacritics,
         do not infer any tashkīl that is not clearly drawn")
    V3 = layout-aware ("preserve line breaks and column structure;
         transcribe marginalia as separate output blocks")

Usage:
  Set the two API keys as environment variables, then run:
    export GOOGLE_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
    cd Paper_JKSU/code
    python3 06_prompt_ablation.py --systems gemini claude --variants v1 v2 v3 --pages 1-50

Output:
  results/prompt_ablation/{system}/{variant}/image_{N}.txt
  results/tables/prompt_ablation_summary.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# -- Optional dependencies (skip whichever system you do not need) -----
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

# -- Paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT / "images"
PAPER = Path(__file__).resolve().parent.parent
OUT_ROOT = PAPER / "results" / "prompt_ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("ROOT =", ROOT)
print("IMAGES_DIR =", IMAGES_DIR)

# -- Models --------------------------------------------------------------
GEMINI_MODEL_ID = "gemini-3.1-pro-preview"        # adjust to exact identifier used in §4.1
CLAUDE_MODEL_ID = "claude-sonnet-4-6"     # adjust to exact identifier

# -- Prompt variants ----------------------------------------------------
PROMPTS = {
    "v1": (
        "You are an expert OCR system specialized in classical and historical "
        "Arabic manuscripts. Convert the provided manuscript image into plain "
        "Arabic text (.txt).\n\n"
        "Instructions:\n"
        "- Transcribe the text exactly as written, without modernization or "
        "paraphrasing.\n"
        "- Preserve original spelling, orthography, and wording, even if it "
        "differs from modern Arabic.\n"
        "- Maintain line breaks and paragraph structure as they appear in the "
        "manuscript.\n"
        "- Keep right-to-left (RTL) text direction.\n"
        "- Include diacritics (tashkeel) only if they are clearly visible; "
        "otherwise omit them.\n"
        "- Do not correct grammatical or spelling errors.\n"
        "- Ignore decorative elements, stains, page borders, and marginal "
        "ornaments unless they contain readable text.\n"
        "- If a word, letter, or section is unreadable or ambiguous, mark it "
        "as [غير واضح].\n"
        "- Do not add explanations, interpretations, or editorial notes.\n\n"
        "Output only the extracted Arabic text, suitable for direct saving as "
        "a .txt file."
    ),
    "v2": (
        "You are a strict diplomatic-transcription OCR system for classical "
        "Arabic manuscripts. Your job is to record only what is physically "
        "drawn on the page.\n\n"
        "Strict rules:\n"
        "- Transcribe every visible character exactly as drawn.\n"
        "- Transcribe a diacritic (tashkīl mark) ONLY if you can clearly see "
        "it drawn on the page. Do NOT infer or restore any diacritic.\n"
        "- Do not normalize, modernize, or interpret. Preserve the manuscript's "
        "orthography character-for-character.\n"
        "- Maintain line breaks exactly as they appear.\n"
        "- For unreadable spans, write [غير واضح].\n"
        "- Do NOT add commentary, explanation, or correction.\n\n"
        "Output only the transcription."
    ),
    "v3": (
        "You are an expert OCR system for classical Arabic manuscripts. "
        "Transcribe the provided page into Arabic text.\n\n"
        "Layout-aware rules:\n"
        "- Preserve the page's two-dimensional layout. If the page has "
        "multiple columns, output each column in reading order separated by "
        "the line `--- COLUMN BREAK ---`.\n"
        "- Transcribe the main body text first. Then, if present, transcribe "
        "marginal annotations in a separate block prefaced by "
        "`--- MARGINALIA ---`.\n"
        "- For rubricated section headings (typically in red ink), prefix "
        "the line with `[HEADING] `.\n"
        "- Otherwise: transcribe the text exactly as written, preserve "
        "diacritics that are clearly visible, mark unreadable spans as "
        "[غير واضح]. No commentary."
    ),
}

# -- Image loader -------------------------------------------------------
def load_image_b64(n: int) -> tuple[bytes, str]:
    p = IMAGES_DIR / f"image_{n}.jpg"
    if not p.exists():
        raise FileNotFoundError(p)
    data = p.read_bytes()
    return data, "image/jpeg"


# -- Gemini transcription ----------------------------------------------
def transcribe_gemini(image_bytes: bytes, mime: str, prompt: str,
                      model: str = GEMINI_MODEL_ID) -> str:
    if not HAS_GOOGLE:
        raise RuntimeError("pip install google-genai")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=[
            gtypes.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
    )
    return resp.text or ""


# -- Claude transcription -----------------------------------------------
def transcribe_claude(image_bytes: bytes, mime: str, prompt: str,
                      model: str = CLAUDE_MODEL_ID) -> str:
    if not HAS_ANTHROPIC:
        raise RuntimeError("pip install anthropic")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64 = base64.b64encode(image_bytes).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": mime, "data": b64
                }},
                {"type": "text", "text": prompt},
            ]},
        ],
    )
    return msg.content[0].text if msg.content else ""


SYSTEMS = {
    "gemini": transcribe_gemini,
    "claude": transcribe_claude,
}


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
                    default=list(SYSTEMS),
                    help="Which systems to run (gemini, claude).")
    ap.add_argument("--variants", nargs="+", choices=list(PROMPTS),
                    default=list(PROMPTS),
                    help="Which prompt variants to run (v1, v2, v3).")
    ap.add_argument("--pages", default="1-50",
                    help="Page-number spec, e.g. 1-50 or 1,5,10")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between API calls.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run pages whose output already exists.")
    args = ap.parse_args()

    pages = parse_pages(args.pages)
    total = len(pages) * len(args.systems) * len(args.variants)
    print(f"Plan: {len(args.systems)} systems × "
          f"{len(args.variants)} variants × "
          f"{len(pages)} pages = {total} calls")

    done = 0
    for sys_name in args.systems:
        fn = SYSTEMS[sys_name]
        for v in args.variants:
            prompt = PROMPTS[v]
            outdir = OUT_ROOT / sys_name / v
            outdir.mkdir(parents=True, exist_ok=True)
            for n in pages:
                outpath = outdir / f"image_{n}.txt"
                if outpath.exists() and not args.overwrite:
                    print(f"  skip (exists): {sys_name}/{v}/image_{n}")
                    done += 1
                    continue
                try:
                    img, mime = load_image_b64(n)
                    text = fn(img, mime, prompt)
                    outpath.write_text(text, encoding="utf-8")
                    done += 1
                    print(f"  done ({done}/{total}): {sys_name}/{v}/image_{n}  "
                          f"({len(text)} chars)")
                except Exception as e:
                    print(f"  FAIL: {sys_name}/{v}/image_{n}: {e}")
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
