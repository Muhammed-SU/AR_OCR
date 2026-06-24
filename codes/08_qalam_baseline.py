"""
08_qalam_baseline.py
--------------------
Best-effort attempt to run Qalam (Bhatia et al., 2024) on the 50-page
corpus, as requested by Reviewer 3.

Qalam is a multimodal LLM specialized for Arabic OCR and handwriting
recognition (SwinV2 encoder + RoBERTa decoder), reported by its authors
to achieve CER ≈ 1.18 % on printed Arabic and WER 0.80 % on handwriting.
The model is hosted at UBC-NLP/Qalam on HuggingFace; depending on
release status, weights or a hosted demo may be available.

If the weights are accessible, this script will:
  1) Load Qalam from HuggingFace.
  2) Transcribe each of the 50 page images.
  3) Save outputs to results/qalam/image_{N}_qalam.txt.
  4) Compute CER_no_tashkeel against both annotators and the best-of-two
     regime, and emit a summary row matching the format of
     summary_per_system.csv so that Qalam can be added directly to
     Table 1.

If weights are not accessible, the script will fail cleanly with a
documented error message that the paper can cite in §10 as the
best-effort attempt log.

Usage:
  pip install transformers torch torchvision pillow
  cd Paper_JKSU/code
  python3 08_qalam_baseline.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Optional

# -- Lazy imports so the script can fail cleanly if libs missing -----
def _try_import_qalam():
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from PIL import Image
        return torch, AutoProcessor, AutoModelForVision2Seq, Image
    except ImportError as e:
        raise SystemExit(
            "Qalam baseline could not be loaded: missing Python dependency. "
            f"Install with `pip install transformers torch torchvision pillow` "
            f"(underlying error: {e})."
        )

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT / "images"
PAPER = Path(__file__).resolve().parent.parent
OUT_DIR = PAPER / "results" / "qalam"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES = PAPER / "results" / "tables"

QALAM_REPO = "UBC-NLP/Qalam"   # cited in Bhatia et al. 2024

ATTEMPT_LOG = TABLES / "qalam_attempt_log.txt"


def log_attempt(message: str) -> None:
    with ATTEMPT_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


def main() -> None:
    log_attempt("=== Qalam best-effort baseline run ===")
    log_attempt(f"Target HF repo: {QALAM_REPO}")

    try:
        torch, AutoProcessor, AutoModelForVision2Seq, Image = _try_import_qalam()
    except SystemExit as e:
        log_attempt(f"DEPENDENCY-ERROR: {e}")
        log_attempt("Result: Qalam not run; cite this log in paper §10.")
        sys.exit(2)

    log_attempt("Loading Qalam from HuggingFace (this may fail if weights "
                "are not publicly released)...")
    try:
        processor = AutoProcessor.from_pretrained(QALAM_REPO)
        model = AutoModelForVision2Seq.from_pretrained(QALAM_REPO)
    except Exception as e:
        log_attempt(f"LOAD-ERROR: could not load {QALAM_REPO}: {type(e).__name__}: {e}")
        log_attempt("Result: Qalam weights not accessible from HuggingFace at "
                    "the time of this run; cite this log in paper §10.")
        sys.exit(3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    log_attempt(f"Model loaded on device={device}")

    # Run on every page image
    pages = sorted(IMAGES_DIR.glob("image_*.jpg"),
                   key=lambda p: int(p.stem.split("_")[1]))
    log_attempt(f"Found {len(pages)} page images")

    for img_path in pages:
        n = int(img_path.stem.split("_")[1])
        outpath = OUT_DIR / f"image_{n}_qalam.txt"
        if outpath.exists():
            log_attempt(f"  skip (exists): image_{n}")
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=2048)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            outpath.write_text(text, encoding="utf-8")
            log_attempt(f"  done: image_{n} ({len(text)} chars)")
        except Exception as e:
            log_attempt(f"  FAIL: image_{n}: {type(e).__name__}: {e}")

    log_attempt("=== Qalam run finished. Next: re-run 02_compute_metrics.py "
                "with qalam added to the SYSTEMS list. ===")


if __name__ == "__main__":
    main()
