import json, csv, os
from collections import Counter
from data.preprocessor import PreProcessor

IN  = "data/fourth_stage/predicted.jsonl"          
OUT = "data/fourth_stage/multilabel_train.csv"   
DROP_UNCERTAIN = True  
proc = PreProcessor()
LABELS = ("race", "gender", "religion")

pos_counts = Counter({lbl: 0 for lbl in LABELS})
cardinality = Counter()         
combos = Counter()            
abstain_counts = Counter({lbl: 0 for lbl in LABELS})

total_read = kept = skipped_lang = skipped_empty = skipped_uncertain = errored = 0

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

with open(IN, "r", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8", newline="") as w:
    wr = csv.DictWriter(w, fieldnames=["text", "race", "gender", "religion"])
    wr.writeheader()

    for line in f:
        total_read += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            errored += 1
            continue

        if rec.get("language") != "en":
            skipped_lang += 1
            continue

        text = (rec.get("clean_text") or "").strip()
        if not text:
            content = (rec.get("content") or "").strip()
            if not content:
                skipped_empty += 1
                continue
            try:
                cleaned = proc.preprocess(content)
                text = (cleaned.get("clean_text") or "").strip()
                if not text:
                    skipped_empty += 1
                    continue
            except Exception:
                errored += 1
                continue

        preds = {lbl: rec.get(f"{lbl}_prediction", None) for lbl in LABELS}

        for lbl in LABELS:
            if preds[lbl] is None:
                abstain_counts[lbl] += 1

        if DROP_UNCERTAIN and any(preds[lbl] is None for lbl in LABELS):
            skipped_uncertain += 1
            continue

        def val(lbl):
            p = preds[lbl]
            if p in (0, 1):
                return int(p)
            return -1   

        row = {
            "text": text,
            "race": val("race"),
            "gender": val("gender"),
            "religion": val("religion"),
        }
        wr.writerow(row)
        kept += 1

        k = sum(int(preds[lbl] == 1) for lbl in LABELS)
        cardinality[k] += 1
        for lbl in LABELS:
            if preds[lbl] == 1:
                pos_counts[lbl] += 1
        combo_labels = [lbl for lbl in LABELS if preds[lbl] == 1]
        combo_key = "+".join(combo_labels) if combo_labels else "none"
        combos[combo_key] += 1

suggested_pos_weight = {}
for lbl in LABELS:
    P = pos_counts[lbl]
    suggested_pos_weight[lbl] = round(((kept - P) / P), 3) if P > 0 else 0.0

stats = {
    "total_read": total_read,
    "total_kept": kept,
    "skipped_non_english": skipped_lang,
    "skipped_empty_or_clean_fail": skipped_empty,
    "skipped_uncertain_rows": skipped_uncertain,
    "errored": errored,
    "positives_per_class": dict(pos_counts),
    "abstains_per_class": dict(abstain_counts),
    "cardinality_counts": dict(cardinality), 
    "combo_counts": dict(combos),
    "suggested_pos_weight": suggested_pos_weight,
    "drop_uncertain": DROP_UNCERTAIN,
    "label_value_for_abstain_when_kept": (-1 if not DROP_UNCERTAIN else None),
}

stats_path = OUT.replace(".csv", "_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"\nWrote multilabel CSV → {OUT}")
print(f"Saved stats → {stats_path}\n")
print(json.dumps(stats, indent=2))
