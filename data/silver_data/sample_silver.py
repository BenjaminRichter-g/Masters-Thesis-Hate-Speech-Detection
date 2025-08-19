#!/usr/bin/env python3
import argparse, os, json, random, hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Any
from tqdm import tqdm

from data.preprocessor import PreProcessor
from setfit import SetFitModel

def get_model_path(models_dir: str) -> Dict[str, str]:
    """
    Returns dict of {model_name: model_path}
    model_name derived from subfolder (e.g. 'race-setfit-model' → 'race').
    """
    model_paths = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if d != "__pycache__" and os.path.isdir(os.path.join(models_dir, d))
    ]
    model_names = [os.path.basename(p).split('-')[0] for p in model_paths]
    return dict(zip(model_names, model_paths))

def bin_of(p: float) -> int:
    """Map [0,1] → bin 0..4 for 0–20, 20–40, 40–60, 60–80, 80–100%."""
    p = max(0.0, min(1.0, float(p)))
    if p < 0.20: return 0
    if p < 0.40: return 1
    if p < 0.60: return 2
    if p < 0.80: return 3
    return 4

def dedup_key(rec: dict) -> str:
    rid = rec.get("id")
    if rid is not None:
        return f"id:{rid}"
    clean = (rec.get("clean_text") or "")
    return "h:" + hashlib.md5(clean.strip().lower().encode("utf-8")).hexdigest()

def score_into_per_model_bins(models: Dict[str, SetFitModel], input_path: str, seed: int = 42):
    rng = random.Random(seed)
    pp = PreProcessor()
    buckets: Dict[str, Dict[int, List[dict]]] = {m: defaultdict(list) for m in models.keys()}
    total_read = skipped_non_en = skipped_empty_or_fail = errored = 0
    per_model_bin_counts = {m: Counter() for m in models.keys()}

    with open(input_path, "r", encoding="utf-8") as f:
        for ln in tqdm(f, desc="Scoring"):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                errored += 1
                continue

            total_read += 1
            lang = (rec.get("language") or "").lower()
            if not lang.startswith("en"):
                skipped_non_en += 1
                continue

            raw = (rec.get("content") or "").strip()
            if not raw:
                skipped_empty_or_fail += 1
                continue
            try:
                clean = (pp.preprocess(raw).get("clean_text") or "").strip()
            except Exception:
                skipped_empty_or_fail += 1
                continue
            if not clean:
                skipped_empty_or_fail += 1
                continue

            blind = {
                "id": rec.get("id"),
                "language": rec.get("language"),
                "content": raw,
                "clean_text": clean,
            }

            try:
                for name, model in models.items():
                    p1 = float(model.predict_proba([clean])[0][1])
                    b = bin_of(p1)
                    buckets[name][b].append(blind)
                    per_model_bin_counts[name][b] += 1
            except Exception:
                errored += 1
                continue

    stats = {
        "seed": seed,
        "total_read": total_read,
        "skipped_non_english": skipped_non_en,
        "skipped_empty_or_clean_fail": skipped_empty_or_fail,
        "errored": errored,
        "per_model_bin_counts": {m: dict(per_model_bin_counts[m]) for m in per_model_bin_counts},
    }
    return buckets, stats

def sample_merge_and_write(buckets: Dict[str, Dict[int, List[dict]]],
                           out_file: str,
                           per_bin: int,
                           seed: int = 42):
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    picked_raw: List[dict] = []
    picked_counts = {}
    available_counts = {}
    for model_name, model_bins in buckets.items():
        picked_counts[model_name] = {}
        available_counts[model_name] = {}
        for b in range(5):
            pool = model_bins.get(b, [])
            available_counts[model_name][b] = len(pool)
            k = min(per_bin, len(pool))
            picked_counts[model_name][b] = k
            if k > 0:
                picked_raw.extend(rng.sample(pool, k))

    seen = set()
    merged = []
    dup_skipped = 0
    for rec in picked_raw:
        k = dedup_key(rec)
        if k in seen:
            dup_skipped += 1
            continue
        seen.add(k)
        merged.append(rec)

    rng.shuffle(merged)

    with open(out_file, "w", encoding="utf-8") as out:
        for obj in merged:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "requested_per_bin_per_model": per_bin,
        "picked_per_bin_per_model": picked_counts,
        "available_per_bin_per_model": available_counts,
        "merged_total_written": len(merged),
        "duplicates_skipped": dup_skipped,
        "output_file": os.path.abspath(out_file),
    }

def main():
    ap = argparse.ArgumentParser(
        description="Score with 3 SetFit models, bin per model (0–20..80–100), sample N per bin per model, "
                    "DEDUP + SHUFFLE and write ONE merged blind JSONL."
    )
    ap.add_argument("--model_dirs", nargs="+", required=True,
                    help="Dirs containing subfolders for each model (e.g., .../race .../gender .../religion)")
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--out_file", required=True, help="Single merged blind JSONL to write")
    ap.add_argument("--stats_out", required=True, help="Stats JSON path")
    ap.add_argument("--per_bin", type=int, default=20, help="N items to sample from each bin per model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models: Dict[str, SetFitModel] = {}
    for model_dir in args.model_dirs:
        for name, path in get_model_path(model_dir).items():
            if name in models:
                continue
            models[name] = SetFitModel.from_pretrained(path)
    if not models:
        raise SystemExit("No models loaded. Check --model_dirs.")
    print(f"[i] Loaded models: {', '.join(sorted(models.keys()))}")

    buckets, stats = score_into_per_model_bins(models, args.input, seed=args.seed)

    sampling_info = sample_merge_and_write(buckets, args.out_file, per_bin=args.per_bin, seed=args.seed)

    stats.update({"sampling": sampling_info})
    os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
    with open(args.stats_out, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)

    print("\n=== Available per-bin per-model ===")
    for m, bins in stats["per_model_bin_counts"].items():
        print(f"{m:9s}: {bins}")
    print("\n=== Sampling summary ===")
    print(json.dumps(sampling_info, indent=2))
    print(f"\nMerged blind sample written → {args.out_file}")
    print(f"Stats → {args.stats_out}")

if __name__ == "__main__":
    main()
