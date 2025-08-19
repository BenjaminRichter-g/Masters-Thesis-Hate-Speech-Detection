import json
import os
import random
import argparse
from data.preprocessor import PreProcessor

HATE_PER_CLEAN = 10         

def load_labeled_ids(output_file):
    ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    ids.add(json.loads(ln)["id"])
                except Exception:
                    pass
    return ids

def load_pool(path, exclude):
    pool = []
    if not os.path.exists(path):
        print(f"{path} not found – skipping.")
        return pool
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                if obj.get("id") not in exclude:
                    pool.append(obj)
            except json.JSONDecodeError:
                continue
    return pool

def build_worklist(clean_pool, hate_pool, hate_per_clean):
    work, i, j = [], 0, 0
    while i < len(hate_pool) or j < len(clean_pool):
        for _ in range(hate_per_clean):
            if i < len(hate_pool):
                work.append(hate_pool[i]); i += 1
        if j < len(clean_pool):
            work.append(clean_pool[j]); j += 1
    return work

def label_data(clean_file, hate_file, output_file, backup_file, randomize):
    pp = PreProcessor()

    already    = load_labeled_ids(output_file)
    clean_pool = load_pool(clean_file, exclude=already)
    hate_pool  = load_pool(hate_file,  exclude=already)

    if randomize:
        random.shuffle(hate_pool)
    else:
        hate_pool.sort(key=lambda x: x.get("pred_proba_1", 0.0), reverse=True)

    if HATE_PER_CLEAN == 0:
        worklist = hate_pool
    else:
        worklist = build_worklist(clean_pool, hate_pool, HATE_PER_CLEAN)

    if not worklist:
        print("Nothing new to label – pools empty or fully labelled.")
        return

    print(f"🗂  {len(worklist):,} posts queued "
          f"({len(hate_pool):,} discriminatory, {len(clean_pool):,} clean).")
    print(f"    Ratio = {HATE_PER_CLEAN}:1 (hate:clean)\n")

    print("Instructions:")
    print("  [y] = discriminatory")
    print("  [n] = clean / non-discriminatory")
    print("  [Enter] = skip temporarily")
    print("  [b] = backtrack one")
    print("  [q] = quit and save\n")

    new_labels = []
    idx = 0

    while idx < len(worklist):
        data = worklist[idx]
        text = pp.preprocess(data["content"])["clean_text"]
        p1 = data.get("pred_proba_1")
        p0 = data.get("pred_proba_0")
        if not text:
            idx += 1
            continue

        header = f"[{idx+1}/{len(worklist)}]"
        if p1 is not None and p0 is not None:
            header += f"  (p1={p1:.2f}, p0={p0:.2f})"

        print("\n" + "-" * 60)
        print(header, text)
        cmd = input("Label [y/n/Enter/b/q]: ").strip().lower()

        if cmd == "q":
            print("\n[↩] Exiting…")
            break
        if cmd == "b":
            if new_labels:
                print("[⏪] Backtracking one item.")
                idx -= 1
                new_labels.pop()
            else:
                print("[!] Nothing to undo.")
            continue

        if cmd in {"y", "n"}:
            data["label"] = 1 if cmd == "y" else 0
            new_labels.append(data)

            idx += 1

    if new_labels:
        if os.path.exists(output_file) and not os.path.exists(backup_file):
            os.replace(output_file, backup_file)
            print(f"Existing labels backed up → {backup_file}")

        with open(output_file, "a", encoding="utf-8") as out:
            for item in new_labels:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nSession complete.")
    print(f"   Newly labelled → {len(new_labels):,}")
    print(f"   Total file     → {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Interactive labeling (10:1 ratio, optional random order)"
    )
    p.add_argument("--clean",     required=True, help="path to clean posts JSONL")
    p.add_argument("--hate",      required=True, help="path to discriminatory posts JSONL")
    p.add_argument("--output",    required=True, help="where to append your new labels")
    p.add_argument("--backup",    required=True, help="where to store a backup of existing output")
    p.add_argument(
        "--randomize",
        action="store_true",
        help="keep the 10:1 interleave but randomize the hate-pool order instead of sorting by confidence"
    )
    args = p.parse_args()

    label_data(
        clean_file   = args.clean,
        hate_file    = args.hate,
        output_file  = args.output,
        backup_file  = args.backup,
        randomize    = args.randomize,
    )
