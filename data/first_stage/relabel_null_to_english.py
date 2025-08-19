import json, argparse, os, tempfile, shutil
from data.preprocessor import PreProcessor
import langid 

def process(path, min_len=12, backup=False):  
    pp = PreProcessor()

    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path)+".", suffix=".tmp", dir=dirpath, text=True)
    os.close(fd)

    added = 0
    try:
        with open(path, "r", encoding="utf-8") as inp, \
             open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
            for line in inp:
                try:
                    obj = json.loads(line)
                except Exception:
                    out.write(line); continue 

                if obj.get("language", None) is not None:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n"); continue

                text = obj.get("content") or ""
                try:
                    text = pp.preprocess(text)["clean_text"]
                except Exception:
                    pass

                if len(text) >= min_len:
                    pred, score = langid.classify(text)
                    if pred == "en" and score >= 0.85:
                        obj["language"] = "en"
                        added+=1

                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if backup:
            shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)
        print(f"Lines affected: {added}") 
        return
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In-place: mark null-language rows as 'en' using fastText.")
    ap.add_argument("path", help="JSONL file to modify in-place")
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--backup", action="store_true", help="Save a .bak copy before replacing")
    args = ap.parse_args()
    n = process(args.path, args.min_len, args.backup)
