import os
import json
import argparse
import hashlib
from typing import Dict, Any, Iterator, List
from data.preprocessor import PreProcessor

LABELS = {
    "0": ("none", 0),
    "1": ("race", 1),
    "2": ("gender", 2),
    "3": ("religion", 3),
}
HELP_TEXT = """\
Keys (space-separated allowed, order = importance):
  0 = none (exclusive)   1 = race   2 = gender   3 = religion
  b = back one item      Enter = skip         q = quit & save        h = help
Examples:
  "1"           → race
  "1 3"         → race > religion
  "2 1 3"       → gender > race > religion
  "0"           → none
"""

def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == "[":
            try:
                arr = json.load(f)
            except json.JSONDecodeError:
                return
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
        else:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue

def load_already_labeled(output_path: str):
    seen_ids, seen_hashes = set(), set()
    if not output_path or not os.path.exists(output_path):
        return seen_ids, seen_hashes
    with open(output_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            if rid is not None:
                seen_ids.add(rid)
            ct = (obj.get("clean_text") or "")
            if ct:
                seen_hashes.add(content_hash(ct))
    return seen_ids, seen_hashes

def build_pool(pp: PreProcessor, input_path: str, seen_ids, seen_hashes):
    pool, total, empty_or_fail, dups = [], 0, 0, 0
    for rec in iter_records(input_path):
        total += 1
        raw = (rec.get("content") or "").strip()
        if not raw and not rec.get("clean_text"):
            empty_or_fail += 1
            continue
        clean = (rec.get("clean_text") or "").strip()
        if not clean:
            try:
                clean = (pp.preprocess(raw).get("clean_text") or "").strip()
            except Exception:
                clean = ""
        if not clean:
            empty_or_fail += 1
            continue
        rid = rec.get("id")
        h = content_hash(clean)
        if (rid is not None and rid in seen_ids) or (h in seen_hashes):
            dups += 1
            continue
        pool.append({
            "id": rid,
            "language": rec.get("language"),
            "content": raw if raw else rec.get("content", ""),
            "clean_text": clean,
        })
    stats = {
        "total_read": total,
        "valid_after_filters": len(pool),
        "skipped_empty_or_clean_fail": empty_or_fail,
        "skipped_duplicates_or_seen": dups,
    }
    return pool, stats

def parse_multi_cmd(cmd: str) -> List[str]:
    """Parse space-separated label keys, dedupe in order, validate."""
    toks = [t for t in cmd.strip().split() if t]
    ordered = []
    for t in toks:
        if t in LABELS and t not in ordered:
            ordered.append(t)
    return ordered

def to_output_record(item: dict, ordered_keys: List[str]) -> dict:
    """Build JSON with ordered labels, primary, and one-hots."""
    if "0" in ordered_keys:
        ordered_keys = ["0"]

    names = [LABELS[k][0] for k in ordered_keys]
    codes = [LABELS[k][1] for k in ordered_keys]

    if ordered_keys:
        primary_key = ordered_keys[0]
        primary_name, primary_code = LABELS[primary_key]
    else:
        primary_name, primary_code = LABELS["0"]  

    flags = {"race": 0, "gender": 0, "religion": 0}
    for k in ordered_keys:
        name = LABELS[k][0]
        if name in flags:  
            flags[name] = 1

    return {
        **item,
        "gold_labels": codes,                 
        "gold_label_names": names,        
        "gold_primary_label": primary_code, 
        "gold_primary_label_name": primary_name,  
        **flags,                             
    }

def label_loop(pool: List[Dict[str, Any]], output_path: str, backup_path: str):
    print("\nInteractive labeling started.")
    print(HELP_TEXT)

    new_labels: List[Dict[str, Any]] = []
    idx = 0
    total = len(pool)

    while idx < total:
        item = pool[idx]
        text = item["clean_text"]
        rid = item.get("id")
        header = f"[{idx+1}/{total}]"
        if rid is not None:
            header += f"  id={rid}"
        print("\n" + "-" * 80)
        print(header)
        print(text)
        cmd = input("Label [space-sep 0/1/2/3, b, Enter=skip, q, h]: ").strip().lower()

        if cmd == "q":
            print("\n[↩] Quitting & saving…")
            break

        if cmd == "h":
            print("\n" + HELP_TEXT)
            continue

        if cmd == "b":
            if new_labels:
                removed = new_labels.pop()
                idx = max(0, idx - 1)
                print(f"[⏪] Backtracked. Removed label for id={removed.get('id')}")
            else:
                print("[!] Nothing to undo.")
            continue

        if cmd == "":
            # Enter (skip)
            idx += 1
            continue

        keys = parse_multi_cmd(cmd)
        if not keys:
            print("[!] No valid keys found. Use 0/1/2/3 (space-separated).")
            continue

        if "0" in keys and len(keys) > 1:
            print("[i] '0' (none) is exclusive — keeping 'none' only.")

        labeled = to_output_record(item, keys)
        new_labels.append(labeled)
        idx += 1

    # persist
    if new_labels:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(backup_path, 'wb') as backup_out:
        with open(output_path, 'rb') as original_in:
            backup_out.write(original_in.read())        

    with open(output_path, "a", encoding="utf-8") as out:
        for obj in new_labels:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n✅ Session complete.")
    print(f"   Newly labeled → {len(new_labels):,}")
    print(f"   Output file   → {output_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Manual labeler (multi-label with ranked order): backtrack, skip, resume."
    )
    ap.add_argument("--input",  required=True, help="Path to JSONL/JSON array (e.g., sample_merged.jsonl)")
    ap.add_argument("--output", required=True, help="Gold labels JSONL (appends; used for resume)")
    ap.add_argument("--backup", default=None, help="Optional backup path for existing output (moved on first write)")
    args = ap.parse_args()

    pp = PreProcessor()

    seen_ids, seen_hashes = load_already_labeled(args.output)

    pool, stats = build_pool(pp, args.input, seen_ids, seen_hashes)

    print("\n=== Pre-labeling stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Items to label this session: {len(pool):,}")

    if not pool:
        print("Nothing to label (all filtered or already labeled).")
        return

    label_loop(pool, args.output, args.backup)

if __name__ == "__main__":
    main()
