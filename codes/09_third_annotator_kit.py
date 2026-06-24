"""
09_third_annotator_kit.py
-------------------------
Prepares the inputs for a third independent annotator (an Arabic-
philology expert who is NOT a co-author of the paper).

Selects a stratified 10-page subset that covers all four difficulty
classes (C/H/D/S) and writes an annotator kit:
  - annotator_kit/images/image_N.jpg      (10 cropped page images)
  - annotator_kit/templates/image_N.txt   (empty template files for output)
  - annotator_kit/README.md               (annotation instructions in
                                           Arabic + Turkish + English)
  - annotator_kit/manifest.csv            (list of pages with difficulty)

After the third annotator returns the 10 transcription .txt files,
run 09b_third_annotator_iaa.py to compute Krippendorff's α and add a
3rd-annotator row to the IAA panel.
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = ROOT / "images"
PAPER = Path(__file__).resolve().parent.parent
KIT = PAPER / "annotator_kit"
KIT.mkdir(parents=True, exist_ok=True)
(KIT / "images").mkdir(exist_ok=True)
(KIT / "templates").mkdir(exist_ok=True)

# Stratified pick: 1 from C, 4 from H, 3 from D, 2 from S (proportional to 5/22/14/9)
TARGET = {"C": 1, "H": 4, "D": 3, "S": 2}
SEED = 20260518


def main() -> None:
    diff = pd.read_csv(PAPER / "data" / "difficulty_labels.csv")
    rng = random.Random(SEED)
    picked = []
    for cls, count in TARGET.items():
        candidates = diff[diff["difficulty"] == cls]["page_id"].tolist()
        rng.shuffle(candidates)
        picked.extend([(cls, p) for p in candidates[:count]])
    picked.sort(key=lambda x: int(x[1].split("_")[1]))

    print(f"Selected {len(picked)} pages:")
    for cls, pid in picked:
        n = int(pid.split("_")[1])
        src = IMAGES_DIR / f"image_{n}.jpg"
        dst = KIT / "images" / src.name
        if src.exists():
            shutil.copy2(src, dst)
        # Empty template
        (KIT / "templates" / f"image_{n}.txt").write_text(
            f"# Page {n} ({cls}). Transcribe full diacritics. One source-line per output-line.\n",
            encoding="utf-8",
        )
        print(f"  [{cls}] {pid}")

    # Manifest
    with (KIT / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "n", "difficulty", "image_file", "template_file"])
        for cls, pid in picked:
            n = int(pid.split("_")[1])
            w.writerow([pid, n, cls, f"images/image_{n}.jpg",
                        f"templates/image_{n}.txt"])

    # Trilingual instructions
    readme = """\
# Annotator Kit — Third-Annotator Transcription Subset

Thank you for agreeing to transcribe this 10-page subset of our Arabic
manuscript corpus. Your transcriptions will be used to compute
inter-annotator agreement (Krippendorff's α) for a benchmark paper on
classical Arabic OCR.

## What to do (English)

1. Open each image file in `images/image_N.jpg`.
2. For each image, edit the corresponding `templates/image_N.txt`.
3. Transcribe the manuscript text into UTF-8 plain Arabic, line-by-line,
   matching the source page's line breaks.
4. Preserve **all visible diacritics (tashkīl)**. Do not infer or
   restore diacritics that are not drawn on the page.
5. Preserve distinctions among hamza-bearers (أ / إ / آ / ٱ / ء), final
   yā' vs. alif-maqṣūra (ي / ى), and tā'-marbūṭa vs. hā' (ة / ه) **as
   written** in the manuscript.
6. For unreadable or ambiguous spans, write `[غير واضح]`.
7. Do not normalise spelling, do not modernise, do not add commentary.

When done, return the 10 `templates/image_*.txt` files to the
corresponding author.

## Türkçe açıklama

1. `images/image_N.jpg` dosyalarını açın.
2. Her görüntü için ilgili `templates/image_N.txt` dosyasını düzenleyin.
3. UTF-8 düz Arapça metin yazın, satır sırasını manuskripte göre koruyun.
4. **Sayfada açıkça görünen tüm tashkîl işaretlerini** koruyun. Manuskript
   üzerinde olmayan diakritikleri eklemeyin.
5. Hamza varyantları, ya/alif-maksūra farkı, ta-marbūta/ha farkını
   sayfadaki yazılışa göre koruyun.
6. Okunamayan kısımlar için `[غير واضح]` yazın.
7. Yazımı modernleştirmeyin, açıklama eklemeyin.

10 dosya bittiğinde sorumlu yazara geri gönderin.

## شرح بالعربية

1. افتح صور المخطوطات في `images/image_N.jpg`.
2. لكل صورة، عدّل ملف القالب الموافق لها في `templates/image_N.txt`.
3. اكتب النص العربي السطر بالسطر مع المحافظة على فواصل الأسطر كما هي.
4. **احفظ كل التشكيل الظاهر على المخطوطة**؛ لا تُضِف تشكيلًا غير مكتوب.
5. احفظ الهمزات والياء والألف المقصورة والتاء المربوطة والهاء كما رُسمت.
6. إذا تعذّر القراءة، اكتب `[غير واضح]`.
7. لا تصحّح ولا تعدّل ولا تُضِف شرحًا.

اشكرك لمشاركتك.
"""
    (KIT / "README.md").write_text(readme, encoding="utf-8")
    print()
    print(f"Annotator kit written under {KIT}")
    print(f"  - {len(picked)} images in annotator_kit/images/")
    print(f"  - {len(picked)} empty templates in annotator_kit/templates/")
    print(f"  - manifest.csv + README.md (trilingual instructions)")
    print()
    print("Next: send the entire 'annotator_kit' folder to a third annotator")
    print("(NOT a co-author of this paper). When they return the filled .txt")
    print("files, run 09b_third_annotator_iaa.py.")


if __name__ == "__main__":
    main()
