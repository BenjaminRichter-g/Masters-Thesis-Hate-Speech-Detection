import os, json, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)

NAMES = ["none","race","gender","religion"]
NAME2ID = {n:i for i,n in enumerate(NAMES)}

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try: yield json.loads(s)
            except json.JSONDecodeError: continue

def get_label(rec):
    for k in ("gold_primary_label", "label"):
        if k in rec:
            try: return int(rec[k])
            except Exception: pass
    for k in ("gold_primary_label_name","label_name"):
        name = (rec.get(k) or "").lower()
        if name in NAME2ID: return NAME2ID[name]
    return None

class JsonlDataset(Dataset):
    def __init__(self, records):
        self.texts, self.labels, self.ids = [], [], []
        self.skipped = {"no_text":0, "no_label":0}
        for r in records:
            t = (r.get("content") or "").strip()
            if not t:
                self.skipped["no_text"] += 1; continue
            y = get_label(r)
            if y is None:
                self.skipped["no_label"] += 1; continue
            self.texts.append(t); self.labels.append(y); self.ids.append(r.get("id"))
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.labels[i], self.ids[i]

def main():
    ap = argparse.ArgumentParser(description="Evaluate single-label BERT on JSONL (uses `content`).")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    records = list(iter_jsonl(args.input_jsonl))
    ds = JsonlDataset(records)
    if len(ds) == 0:
        raise SystemExit("No evaluable examples (missing content and/or labels).")

    def collate(batch):
        texts, labels, ids = zip(*batch)
        enc = tok(list(texts), truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
        y = torch.tensor(labels, dtype=torch.long)
        return enc, y, ids, texts

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model.eval()

    all_logits, all_y, all_ids, all_texts = [], [], [], []
    with torch.no_grad():
        for enc, y, ids, texts in loader:
            enc = {k:v.to(device) for k,v in enc.items()}
            out = model(**enc)
            all_logits.append(out.logits.detach().cpu().numpy())
            all_y.append(y.numpy())
            all_ids.extend(ids); all_texts.extend(texts)

    logits = np.concatenate(all_logits, 0)
    y_true = np.concatenate(all_y, 0)
    y_pred = logits.argmax(1)

    metrics = {
        "accuracy":        accuracy_score(y_true, y_pred),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "num_examples":    int(len(ds)),
        "skipped":         ds.skipped,
        "device":          str(device),
        "model_path":      os.path.abspath(args.model_path),
        "input_jsonl":     os.path.abspath(args.input_jsonl),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=NAMES, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    if args.save_preds:
        with open(os.path.join(args.out_dir, "predictions.jsonl"), "w", encoding="utf-8") as outp:
            for i in range(len(y_pred)):
                outp.write(json.dumps({
                    "id": all_ids[i],
                    "content": all_texts[i],
                    "true_label": int(y_true[i]),
                    "pred_label": int(y_pred[i]),
                    "logits": logits[i].tolist()
                }, ensure_ascii=False) + "\n")

    print("single-label eval done →", os.path.join(args.out_dir, "metrics.json"))

if __name__ == "__main__":
    main()
