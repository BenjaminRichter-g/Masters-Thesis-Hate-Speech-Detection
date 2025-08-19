import argparse, json, os, random
from collections import defaultdict, Counter

PRIMARY_MAP = {0: "none", 1: "race", 2: "gender", 3: "religion"}
CLASSES = ["race", "gender", "religion", "none"]

def get_primary_name(rec):
    name = rec.get("gold_primary_label_name")
    if isinstance(name, str) and name in CLASSES:
        return name
    pid = rec.get("gold_primary_label")
    if isinstance(pid, int) and pid in PRIMARY_MAP:
        return PRIMARY_MAP[pid]
    gl = rec.get("gold_labels")
    if isinstance(gl, list) and gl == [0]:
        return "none"
    return None   

def main():
    ap = argparse.ArgumentParser(
        description="Sample up to N items per primary class (race/gender/religion/none) from a JSONL."
    )
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--per_class", type=int, default=40, help="Items to sample per class (default: 40)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    buckets = {c: [] for c in CLASSES}
    seen_ids = set()
    counts_total = Counter()
    counts_skipped = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                counts_skipped["bad_json"] += 1
                continue
            rid = rec.get("id")
            if rid is not None:
                if rid in seen_ids:
                    counts_skipped["duplicate_id"] += 1
                    continue
                seen_ids.add(rid)

            primary = get_primary_name(rec)
            if primary is None:
                counts_skipped["no_primary"] += 1
                continue

            if primary not in CLASSES:
                counts_skipped["unknown_primary"] += 1
                continue

            buckets[primary].append(rec)
            counts_total[primary] += 1

    picked = []
    picked_counts = {}
    for c in CLASSES:
        pool = buckets[c]
        rng.shuffle(pool)
        k = min(args.per_class, len(pool))
        picked_counts[c] = k
        picked.extend(pool[:k])

    rng.shuffle(picked)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for obj in picked:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n=== Sampling summary ===")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Requested per class: {args.per_class}\n")
    for c in CLASSES:
        print(f"{c:9s} available = {counts_total[c]:4d} | picked = {picked_counts[c]:4d}")
    print(f"\nTotal written: {len(picked)}")
    if counts_skipped:
        print("\nSkipped counts:", dict(counts_skipped))

if __name__ == "__main__":
    main()
