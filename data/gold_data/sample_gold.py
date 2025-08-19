import json
import os
import argparse
import random
import hashlib
from typing import Iterator, Dict, Any, List
from data.preprocessor import PreProcessor

def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    """
    Supports JSONL (one object per line) OR a JSON array file.
    Yields dicts. Silently skips malformed lines/entries.
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == "[": 
            try:
                arr = json.load(f)
                for obj in arr:
                    if isinstance(obj, dict):
                        yield obj
            except json.JSONDecodeError:
                return
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

def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser(
        description="Build a small, blind GOLD sample from one JSON/JSONL: English-only, preprocessed, deduped, random-sampled."
    )
    ap.add_argument("--input",  required=True, help="Path to raw JSONL or JSON array")
    ap.add_argument("--output", required=True, help="Output GOLD JSONL path")
    ap.add_argument("--n", type=int, required=True, help="Number of samples to draw (after filtering)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--stats", default=None, help="Optional path to write stats JSON (default: <output>_stats.json)")
    args = ap.parse_args()

    pp = PreProcessor()
    rng = random.Random(args.seed)

    total_read = 0
    non_english = 0
    empty_or_clean_fail = 0
    dedup_skipped = 0

    pool: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_hashes = set()

    for rec in iter_records(args.input):
        total_read += 1

        lang = (rec.get("language") or "").lower()
        if not lang.startswith("en"):
            non_english += 1
            continue

        raw = (rec.get("content") or "").strip()
        if not raw:
            empty_or_clean_fail += 1
            continue

        try:
            clean = (pp.preprocess(raw).get("clean_text") or "").strip()
        except Exception:
            empty_or_clean_fail += 1
            continue
        if not clean:
            empty_or_clean_fail += 1
            continue

        rid = rec.get("id")
        h = content_hash(clean)
        if (rid is not None and rid in seen_ids) or (h in seen_hashes):
            dedup_skipped += 1
            continue
        if rid is not None:
            seen_ids.add(rid)
        seen_hashes.add(h)

        pool.append({
            "id": rid,
            "language": lang,
            "content": raw,        
            "clean_text": clean,  
        })

    valid_count = len(pool)
    if valid_count == 0:
        print("No valid English+cleanable records found.")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as out:
            pass
        stats_path = args.stats or (args.output + "_stats.json")
        with open(stats_path, "w", encoding="utf-8") as jf:
            json.dump({
                "total_read": total_read,
                "valid_after_filters": 0,
                "skipped_non_english": non_english,
                "skipped_empty_or_clean_fail": empty_or_clean_fail,
                "skipped_duplicates": dedup_skipped,
                "sampled": 0
            }, jf, indent=2)
        print(f"Stats → {stats_path}")
        return

    k = min(args.n, valid_count)
    sample = rng.sample(pool, k)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for obj in sample:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    stats = {
        "total_read": total_read,
        "valid_after_filters": valid_count,
        "skipped_non_english": non_english,
        "skipped_empty_or_clean_fail": empty_or_clean_fail,
        "skipped_duplicates": dedup_skipped,
        "sampled": k,
        "input_path": os.path.abspath(args.input),
        "output_path": os.path.abspath(args.output),
        "seed": args.seed,
    }
    stats_path = args.stats or (args.output + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as jf:
        json.dump(stats, jf, indent=2)

    print(f"GOLD sample written → {args.output}")
    print(f"   total_read={total_read:,} | valid={valid_count:,} | sampled={k:,}")
    print(f"   non_english={non_english:,} | empty_or_clean_fail={empty_or_clean_fail:,} | duplicates={dedup_skipped:,}")
    print(f"   stats → {stats_path}")

if __name__ == "__main__":
    main()
