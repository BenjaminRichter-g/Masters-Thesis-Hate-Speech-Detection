import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss,
    precision_recall_fscore_support, confusion_matrix, classification_report
)

TARGETS = ["race", "gender", "religion"]

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue

def get_multi(rec):
    y = []
    has_any = False
    for k in TARGETS:
        v = rec.get(k, 0)
        try:
            iv = int(v)
        except Exception:
            iv = 0
        y.append(iv)
        has_any = has_any or (iv in (0, 1))
    if not has_any and isinstance(rec.get("gold_labels"), list):
        codes = set()
        for c in rec["gold_labels"]:
            try:
                codes.add(int(c))
            except Exception:
                pass
        y = [int(1 in codes), int(2 in codes), int(3 in codes)]
    return y

class JsonlDataset(Dataset):
    def __init__(self, records):
        self.texts, self.labels, self.ids = [], [], []
        self.skipped = {"no_text": 0, "no_label": 0}
        for r in records:
            t = (r.get("content") or "").strip()
            if not t:
                self.skipped["no_text"] += 1
                continue
            y = get_multi(r)
            if len(y) != 3 or not all(v in (0, 1) for v in y):
                self.skipped["no_label"] += 1
                continue
            self.texts.append(t)
            self.labels.append(y)
            self.ids.append(r.get("id"))
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.labels[i], self.ids[i]

def sigmoid(x): return 1 / (1 + np.exp(-x))

def main():
    ap = argparse.ArgumentParser(description="Evaluate multi-label BERT on JSONL (uses `content`).")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    records = list(iter_jsonl(args.input_jsonl))
    ds = JsonlDataset(records)
    if len(ds) == 0:
        raise SystemExit("No evaluable examples (missing content and/or multilabels).")

    def collate(batch):
        texts, labels, ids = zip(*batch)
        enc = tok(list(texts), truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
        y = torch.tensor(labels, dtype=torch.float32)
        return enc, y, ids, texts

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model.eval()

    all_logits, all_y, all_ids, all_texts = [], [], [], []
    with torch.no_grad():
        for enc, y, ids, texts in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            all_logits.append(out.logits.detach().cpu().numpy())
            all_y.append(y.numpy())
            all_ids.extend(ids)
            all_texts.extend(texts)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    probs = sigmoid(logits)
    y_pred = (probs >= args.threshold).astype(int)

    metrics = {
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro":    recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro":        f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match":     (y_pred == y_true).all(axis=1).mean(),
        "hamming_loss":    hamming_loss(y_true, y_pred),
        "none_accuracy":   ((y_true.sum(axis=1) == 0) == (y_pred.sum(axis=1) == 0)).mean(),
        "threshold":       args.threshold,
        "num_examples":    int(len(ds)),
        "skipped":         ds.skipped,
        "device":          str(device),
        "model_path":      os.path.abspath(args.model_path),
        "input_jsonl":     os.path.abspath(args.input_jsonl),
    }

    report_txt = classification_report(
        y_true, y_pred, target_names=TARGETS, digits=4, zero_division=0
    )
    report_json = classification_report(
        y_true, y_pred, target_names=TARGETS, digits=4, zero_division=0, output_dict=True
    )

    y_true_none = (y_true.sum(axis=1) == 0).astype(int)
    y_pred_none = (y_pred.sum(axis=1) == 0).astype(int)
    p_none, r_none, f1_none, support_none = precision_recall_fscore_support(
        y_true_none, y_pred_none, labels=[1], average=None, zero_division=0
    )
    none_row = {
        "precision": float(p_none[0]),
        "recall": float(r_none[0]),
        "f1-score": float(f1_none[0]),
        "support": int(support_none[0]),
    }
    report_json_with_none = dict(report_json)
    report_json_with_none["none"] = none_row

    y_true_any = 1 - y_true_none
    y_pred_any = 1 - y_pred_none
    cm_any = confusion_matrix(y_true_any, y_pred_any, labels=[0, 1])
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix_any_vs_none.csv"),
               cm_any, fmt="%d", delimiter=",")
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
        f.write("\n\nNone (all-zero) as its own row:\n")
        f.write(f"{'none':<12} prec {none_row['precision']:.4f}  "
                f"rec {none_row['recall']:.4f}  f1 {none_row['f1-score']:.4f}  "
                f"support {none_row['support']}\n")

    with open(os.path.join(args.out_dir, "per_class_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_json_with_none, f, indent=2)

    if args.save_preds:
        with open(os.path.join(args.out_dir, "predictions.jsonl"), "w", encoding="utf-8") as outp:
            for i in range(len(y_pred)):
                outp.write(json.dumps({
                    "id": all_ids[i],
                    "content": all_texts[i],
                    "true_multi": [int(x) for x in y_true[i].tolist()],
                    "pred_multi": [int(x) for x in y_pred[i].tolist()],
                    "probs": probs[i].tolist(),
                    "logits": logits[i].tolist()
                }, ensure_ascii=False) + "\n")

    print("multi-label eval done →", os.path.join(args.out_dir, "metrics.json"))

if __name__ == "__main__":
    main()
