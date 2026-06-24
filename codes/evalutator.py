import os
import re
import string
import unicodedata
import Levenshtein

# =========================
# Flexible Arabic Normalization
# =========================
def normalize_arabic(text):

    # 1️⃣ Normalize Unicode (fix combining characters)
    text = unicodedata.normalize('NFC', text)

    # 2️⃣ remove tashkeel
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)

    # 3️⃣ remove combining hamza above/below
    text = re.sub(r'[\u0654\u0655]', '', text)

    # 4️⃣ unify alef forms
    text = re.sub(r'[إأآٱا]', 'ا', text)

    # 5️⃣ unify ya
    text = re.sub(r'[ىي]', 'ي', text)

    # 6️⃣ unify ta marbuta / ha
    text = re.sub(r'[هة]', 'ه', text)

    # 7️⃣ remove punctuation
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)

    # 8️⃣ normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# CER (Character Level)
# =========================
def cer_analysis(gold, ocr):

    ops = Levenshtein.editops(gold, ocr)

    S = D = I = 0
    S_chars = []
    D_chars = []
    I_chars = []

    for tag, i1, i2 in ops:
        if tag == "replace":
            S += 1
            S_chars.append((gold[i1], ocr[i2]))
        elif tag == "delete":
            D += 1
            D_chars.append(gold[i1])
        elif tag == "insert":
            I += 1
            I_chars.append(ocr[i2])

    N = len(gold)
    CER = (S + D + I) / N if N > 0 else 0

    return S, D, I, S_chars, D_chars, I_chars, N, CER


# =========================
# WER (Word Level)
# =========================
def wer_analysis(ref, hyp):

    m = len(ref)
    n = len(hyp)

    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + 1     # substitution
                )

    S = D = I = 0
    S_words = []
    D_words = []
    I_words = []

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1
            D_words.append(ref[i-1])
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            I += 1
            I_words.append(hyp[j-1])
            j -= 1
        else:
            S += 1
            S_words.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1

    N = len(ref)
    WER = (S + D + I) / N if N > 0 else 0

    return S, D, I, S_words, D_words, I_words, N, WER


# =========================
# MAIN
# =========================
gold_folder = "gold"
ocr_folder = "ocr"
report_folder = "report"

os.makedirs(report_folder, exist_ok=True)

gold_files = [
    f for f in os.listdir(gold_folder)
    if f.endswith(".txt") and "gold" in f
]

for gold_filename in gold_files:

    base_name = gold_filename.replace("_gold", "")
    gold_path = os.path.join(gold_folder, gold_filename)
    ocr_path = os.path.join(ocr_folder, base_name)

    if not os.path.exists(ocr_path):
        print(f"OCR file not found for: {base_name}")
        continue

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_text = normalize_arabic(f.read())

    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_text = normalize_arabic(f.read())

    # ===== CER =====
    cer_S, cer_D, cer_I, cer_S_chars, cer_D_chars, cer_I_chars, cer_N, cer_value = cer_analysis(
    gold_text.replace(" ", ""),
    ocr_text.replace(" ", "")
)

    # ===== WER =====
    gold_words = gold_text.split()
    ocr_words = ocr_text.split()

    wer_S, wer_D, wer_I, wer_S_words, wer_D_words, wer_I_words, wer_N, wer_value = wer_analysis(gold_words, ocr_words)

    # ===== WRITE REPORT =====
    report_path = os.path.join(report_folder, base_name.replace(".txt", ".md"))

    with open(report_path, "w", encoding="utf-8") as report:

        report.write(f"# OCR Evaluation Report : {base_name}\n\n")

        report.write("## CER (Character Error Rate)\n\n")
        report.write(f"Total Characters (from Gold): {cer_N}\n")
        report.write(f"CER: {round(cer_value*100, 2)}%\n\n")
        report.write(f"Substitutions (S) [{cer_S}]:\n{cer_S_chars}\n\n")
        report.write(f"Deletions (D) [{cer_D}]:\n{cer_D_chars}\n\n")
        report.write(f"Insertions (I) [{cer_I}]:\n{cer_I_chars}\n\n")

        report.write("--------------------------------------------------\n\n")

        report.write("## WER (Word Error Rate)\n\n")
        report.write(f"Total Words (from Gold): {wer_N}\n")
        report.write(f"WER: {round(wer_value*100, 2)}%\n\n")
        report.write(f"Substitutions (S) [{wer_S}]:\n{wer_S_words}\n\n")
        report.write(f"Deletions (D) [{wer_D}]:\n{wer_D_words}\n\n")
        report.write(f"Insertions (I) [{wer_I}]:\n{wer_I_words}\n\n")

    print(f"Report generated for: {base_name}")

print("All files processed successfully!")
