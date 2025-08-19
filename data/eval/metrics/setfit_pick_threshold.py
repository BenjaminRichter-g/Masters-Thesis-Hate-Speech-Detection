import argparse, json, os, csv, numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from setfit import SetFitModel
from data.preprocessor import PreProcessor

def load_dev(jsonl, text_key="clean_text", label_key=None, lang_key="language"):
    if not label_key:
        raise ValueError("--label_key is required (race|gender|religion)")
    X, y = [], []
    pp = None
    with open(jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if lang_key and rec.get(lang_key) != "en":
                continue

            txt = (rec.get(text_key) or "").strip()
            if not txt:
                raw = (rec.get("content") or "").strip()
                if not raw:
                    continue
                if PreProcessor is None:
                    continue
                if pp is None:
                    pp = PreProcessor()
                try:
                    txt = (pp.preprocess(raw).get("clean_text") or "").strip()
                except Exception:
                    continue
            if not txt:
                continue

            lab = rec.get(label_key)
            if lab is None:
                continue
            y.append(int(lab))
            X.append(txt)
    return X, np.array(y, dtype=int)

def batched_probs(model, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        probs = model.predict_proba(texts[i:i+bs])
        out.extend([p[1] for p in probs]) 
    return np.array(out, dtype=float)

def pick_tau_from_pr(y, p, floor=0.95):
    P, R, T = precision_recall_curve(y, p)
    idx = np.where(P >= floor)[0]
    if len(idx) == 0:
        thr_grid = np.unique(np.clip(np.concatenate(([0.0, 1.0], T)), 0.0, 1.0))
        rows = []
        for t in thr_grid:
            y_pred = (p >= t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
            rows.append((t, pr, rc, f1))
        t, pr, rc, f1 = max(rows, key=lambda r: r[3])
        return float(t), {"mode": "best_f1", "P": float(pr), "R": float(rc), "F1": float(f1)}, (P, R, T)
    j = int(idx[0])
    t = float(T[j-1]) if j > 0 and (j-1) < len(T) else 0.5
    return t, {"mode": f"P≥{floor}", "P": float(P[j]), "R": float(R[j])}, (P, R, T)

def save_pr_json(path, kind, auprc, P, R, T, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {
        "kind": kind,  
        "auprc": float(auprc),
        "precision": [float(x) for x in P],
        "recall": [float(x) for x in R],
        "thresholds": [float(x) for x in T],  
        "meta": meta,
        "note": "thresholds array is one shorter than precision/recall by design (sklearn).",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_pr_csv(path, P, R, T):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(["threshold","precision","recall"])
        for i in range(len(T)):
            wr.writerow([float(T[i]), float(P[i]), float(R[i])])
        wr.writerow(["", float(P[-1]), float(R[-1])]) 

def save_raw_csv(path, y, p):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(["label","prob_pos"])
        for yi, pi in zip(y, p):
            wr.writerow([int(yi), float(pi)])

def main():
    ap = argparse.ArgumentParser(
        description="Pick τ_pos/τ_neg for ONE SetFit model; save full PR data for plotting. "
                    "Defaults to your JSONL schema (clean_text + per-class label keys)."
    )
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--dev_jsonl", required=True, help="Gold dev JSONL for THIS class")
    ap.add_argument("--label_key", required=True, help="Use one of: race, gender, religion")
    ap.add_argument("--text_key", default="clean_text", help="Defaults to clean_text; falls back to preprocessing content if missing")
    ap.add_argument("--pos_floor", type=float, default=0.95)
    ap.add_argument("--neg_floor", type=float, default=0.99)
    ap.add_argument("--out", required=True, help="Where to write thresholds summary JSON")
    ap.add_argument("--pr_pos_json", required=True, help="Save positive-class PR arrays JSON")
    ap.add_argument("--pr_neg_json", required=True, help="Save negative-class PR arrays JSON")
    ap.add_argument("--pr_pos_csv",  required=False, help="Optional CSV of (threshold,precision,recall) for positives")
    ap.add_argument("--pr_neg_csv",  required=False, help="Optional CSV of (threshold,precision,recall) for negatives")
    ap.add_argument("--raw_csv",     required=False, help="Optional per-example (label, prob_pos) CSV")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    X, y = load_dev(args.dev_jsonl, text_key=args.text_key, label_key=args.label_key)
    if not len(X):
        raise SystemExit("No examples loaded; check --text_key/--label_key and file content.")

    model = SetFitModel.from_pretrained(args.model_dir)
    p = batched_probs(model, X, bs=args.batch_size)

    auprc_pos = float(average_precision_score(y, p))
    tau_pos, info_pos, (P_pos, R_pos, T_pos) = pick_tau_from_pr(y, p, floor=args.pos_floor)

    y_n, p_n = (1 - y), (1 - p)
    auprc_neg = float(average_precision_score(y_n, p_n))
    tau_neg, info_neg, (P_neg, R_neg, T_neg) = pick_tau_from_pr(y_n, p_n, floor=args.neg_floor)

    summary = {
        "model_dir": args.model_dir,
        "dev_jsonl": args.dev_jsonl,
        "label_key": args.label_key,
        "text_key": args.text_key,
        "pos_precision_floor": args.pos_floor,
        "neg_precision_floor": args.neg_floor,
        "tau_pos": float(tau_pos),  
        "tau_neg": float(tau_neg),
        "auprc_pos": auprc_pos,
        "auprc_neg": auprc_neg,
        "info_pos": info_pos,
        "info_neg": info_neg,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {"model_dir": args.model_dir, "dev_jsonl": args.dev_jsonl, "label_key": args.label_key}
    save_pr_json(args.pr_pos_json, "positive", auprc_pos, P_pos, R_pos, T_pos, meta)
    save_pr_json(args.pr_neg_json, "negative", auprc_neg, P_neg, R_neg, T_neg, meta)
    if args.pr_pos_csv: save_pr_csv(args.pr_pos_csv, P_pos, R_pos, T_pos)
    if args.pr_neg_csv: save_pr_csv(args.pr_neg_csv, P_neg, R_neg, T_neg)
    if args.raw_csv:    save_raw_csv(args.raw_csv, y, p)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved PR (pos) → {args.pr_pos_json}")
    print(f"Saved PR (neg) → {args.pr_neg_json}")
    if args.pr_pos_csv: print(f"Saved PR CSV (pos) → {args.pr_pos_csv}")
    if args.pr_neg_csv: print(f"Saved PR CSV (neg) → {args.pr_neg_csv}")
    if args.raw_csv:    print(f"Saved raw per-example → {args.raw_csv}")

if __name__ == "__main__":
    main()
