import json
import csv
from data.preprocessor import PreProcessor

input_path = "data/fourth_stage/predicted.jsonl"
output_path = "data/fourth_stage/train_discrimination_with_none.csv"
label_map = {
    "none": 0,
    "race": 1,
    "gender": 2,
    "religion": 3,
}
LABELS = ["race", "gender", "religion"]

def select_label(entry):
    """
    Single-label selection with abstain support:
      - If any class is a confident positive (prediction == 1), pick the one with highest confidence.
      - Else if all three are confident negatives (prediction == 0), return 'none'.
      - Else (some null/abstain and no positives) -> return None to SKIP row.
    """
    candidates = []
    preds = {lbl: entry.get(f"{lbl}_prediction", None) for lbl in LABELS}
    confs = {lbl: float(entry.get(f"{lbl}_confidence", 0.0)) for lbl in LABELS}

    for lbl in LABELS:
        if preds[lbl] == 1:
            candidates.append((lbl, confs[lbl]))
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]

    if all(preds.get(lbl) == 0 for lbl in LABELS):
        return "none"

    return None

processor = PreProcessor()
total = skipped_lang = skipped_text = errored = written = skipped_uncertain = 0

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8", newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["text", "label"])
    writer.writeheader()

    for line in infile:
        total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errored += 1
            continue

        if record.get("language") != "en":
            skipped_lang += 1
            continue

        clean_text = (record.get("clean_text") or "").strip()
        if not clean_text:
            content = (record.get("content") or "").strip()
            if not content:
                skipped_text += 1
                continue
            try:
                cleaned = processor.preprocess(content)
                clean_text = (cleaned.get("clean_text") or "").strip()
                if not clean_text:
                    skipped_text += 1
                    continue
            except Exception:
                errored += 1
                continue

        selected = select_label(record)
        if selected is None:
            skipped_uncertain += 1
            continue

        writer.writerow({
            "text": clean_text,
            "label": label_map[selected]
        })
        written += 1

print(f"""Finished.
Total: {total}
Written: {written}
Skipped (non-en): {skipped_lang}
Skipped (empty/clean fail): {skipped_text}
Skipped (uncertain/abstain): {skipped_uncertain}
Errored: {errored}
""")
