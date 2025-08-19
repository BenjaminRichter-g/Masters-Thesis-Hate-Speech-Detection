import os, json, argparse, tempfile, shutil
from typing import List
from transformers import pipeline
from data.preprocessor import PreProcessor

def process(path: str, model: str, batch_size: int = 16, thr: float = 0.80,
            min_len: int = 12, backup: bool = False) -> int:
    clf = pipeline("text-classification", model=model, top_k=None, truncation=True) 
    pp = PreProcessor()

    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path)+".", suffix=".tmp", dir=dirpath, text=True)
    os.close(fd)

    changed = 0
    buf_objs: List[dict] = []
    buf_texts: List[str] = []

    def flush():
        nonlocal changed
        if not buf_objs: return
        out_scores = clf(buf_texts) 
        for obj, scores in zip(buf_objs, out_scores):
            items = scores if isinstance(scores, list) else [scores]
            best = max(items, key=lambda s: s["score"]) if items else None
            if best and best["label"] == "en" and float(best["score"]) >= thr:
                obj["language"] = "en"; changed += 1
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        buf_objs.clear(); buf_texts.clear()

    try:
        with open(path, "r", encoding="utf-8") as inp, \
             open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
            for line in inp:
                try:
                    obj = json.loads(line)
                except Exception:
                    out.write(line); continue

                if obj.get("language") is not None:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n"); continue

                text = obj.get("content") or ""
                try:
                    text = pp.preprocess(text)["clean_text"]
                except Exception:
                    pass

                if len(text) < min_len:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n"); continue

                buf_objs.append(obj); buf_texts.append(text)
                if len(buf_objs) >= batch_size:
                    flush()
            flush()

        if backup: shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)
        return changed
    except BaseException:
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        finally:
            raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In-place: relabel null language rows as 'en' using a Transformer.")
    ap.add_argument("path")
    ap.add_argument("--model", default="papluca/xlm-roberta-base-language-detection")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.80)
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()
    n = process(args.path, args.model, args.batch_size, args.thr, args.min_len, args.backup)
    print(f"updated {n} rows")
