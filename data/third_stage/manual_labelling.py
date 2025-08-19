"""
Interactive labeling with two selection modes:
- interleave (default): same behavior as before (hate:clean = HATE_PER_CLEAN)
- bins: MERGE clean + hate in memory, build bins from the combined distribution,
        sample per-bin, then create the worklist (no artificial interleave).

Resume-safe: skips anything already labeled in --output (and --master if provided),
by **id** OR by **clean_text hash**.

UI/controls:
  [y] = discriminatory
  [n] = clean / non-discriminatory
  [Enter] = skip temporarily
  [b] = backtrack one
  [q] = quit and save
  [h] = help
"""
import os
import json
import random
import argparse
import hashlib
from typing import Dict, Any, Iterator, List, Tuple
import numpy as np
from data.preprocessor import PreProcessor

HATE_PER_CLEAN = 10        

LABEL_HELP = """\
Instructions:
  [y] = discriminatory     [n] = clean / non-discriminatory
  [Enter] = skip temporarily
  [b] = backtrack one      [q] = quit and save       [h] = help
"""

def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    """Supports JSONL (one object per line) OR a JSON array file."""
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
                s = ln.strip()
                if not s:
                    continue
                try:
                    o = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(o, dict):
                    yield o

def load_seen(files: List[str]) -> Tuple[set, set]:
    """
    Given a list of labeled JSONL/JSON paths, return (seen_ids, seen_hashes).
    Hash is based on clean_text if present; else we preprocess content.
    """
    seen_ids, seen_hashes = set(), set()
    pp = PreProcessor()
    for path in files:
        if not path or not os.path.exists(path):
            continue
        for obj in iter_records(path):
            rid = obj.get("id")
            if rid is not None:
                seen_ids.add(rid)
            clean = (obj.get("clean_text") or "").strip()
            if not clean:
                raw = (obj.get("content") or "").strip()
                if not raw:
                    continue
                try:
                    clean = (pp.preprocess(raw).get("clean_text") or "").strip()
                except Exception:
                    continue
            if clean:
                seen_hashes.add(content_hash(clean))
    return seen_ids, seen_hashes

def load_pool(path: str, exclude_ids: set, exclude_hashes: set, pp: PreProcessor) -> List[Dict[str, Any]]:
    """
    Load a pool, computing clean_text when missing, and skip anything already labeled
    by id OR clean_text hash (from --master and --output). Keeps pred_proba_* if present.
    """
    pool = []
    if not os.path.exists(path):
        print(f"[!] {path} not found – skipping.")
        return pool
    for obj in iter_records(path):
        rid = obj.get("id")
        if rid is not None and rid in exclude_ids:
            continue

        clean = (obj.get("clean_text") or "").strip()
        if not clean:
            raw = (obj.get("content") or "").strip()
            if not raw:
                continue
            try:
                clean = (pp.preprocess(raw).get("clean_text") or "").strip()
            except Exception:
                continue
        if not clean:
            continue

        if content_hash(clean) in exclude_hashes:
            continue

        out = {
            "id": obj.get("id"),
            "language": obj.get("language"),
            "content": (obj.get("content") or ""),
            "clean_text": clean,
        }
        if "pred_proba_1" in obj: out["pred_proba_1"] = obj["pred_proba_1"]
        if "pred_proba_0" in obj: out["pred_proba_0"] = obj["pred_proba_0"]
        pool.append(out)
    return pool


def build_worklist_interleave(clean_pool: List[dict], hate_pool: List[dict], hate_per_clean: int) -> List[dict]:
    """Original interleave: take HATE_PER_CLEAN from hate, then 1 from clean, repeat."""
    work, i, j = [], 0, 0
    while i < len(hate_pool) or j < len(clean_pool):
        for _ in range(hate_per_clean):
            if i < len(hate_pool):
                work.append(hate_pool[i]); i += 1
        if j < len(clean_pool):
            work.append(clean_pool[j]); j += 1
    return work

def _fixed_edges() -> List[float]:
    """Fixed edges in [0,1]: 0,0.2,0.4,0.6,0.8,1.0 (epsilon to include 1.0)."""
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]

def _quantile_edges(probs: List[float], num_bins: int) -> List[float]:
    """Quantile edges (ensure strictly increasing with small jitter)."""
    if not probs:
        return _fixed_edges()
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(np.array(probs, dtype=float), qs).tolist()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-7
    edges[-1] = max(edges[-1], edges[-2] + 1e-7)
    return edges

def _assign_bin(p: float, edges: List[float]) -> int:
    """Return bin index i such that edges[i] <= p < edges[i+1]."""
    p = max(0.0, min(1.0, float(p)))
    for i in range(len(edges) - 1):
        if edges[i] <= p < edges[i+1]:
            return i
    return len(edges) - 2

def build_worklist_bins_merged(
    clean_pool: List[dict],
    hate_pool: List[dict],
    per_bin: int = 0,
    within_bin_order: str = "desc",  
    round_robin: bool = True,
    seed: int = 42,
    binning: str = "fixed",         
    num_bins: int = 5,              
) -> Tuple[List[dict], dict]:
    """
    MERGE clean + hate → combined pool; build bins from combined pred_proba_1 distribution;
    sample up to per_bin per bin; form worklist (round-robin high→low or concat).
    """
    rng = random.Random(seed)

    combined = []
    missing = 0
    for rec in (clean_pool + hate_pool):
        p1 = rec.get("pred_proba_1", None)
        if p1 is None:
            missing += 1
            continue
        x = dict(rec)
        x["pred_proba_1"] = float(p1)
        combined.append(x)

    probs = [r["pred_proba_1"] for r in combined]
    if binning == "quantile":
        edges = _quantile_edges(probs, num_bins=num_bins)
    else:
        edges = _fixed_edges()

    bins = {b: [] for b in range(len(edges) - 1)}
    for r in combined:
        b = _assign_bin(r["pred_proba_1"], edges)
        y = dict(r); y["bin"] = b
        bins[b].append(y)

    for b in bins:
        if within_bin_order == "shuffle":
            rng.shuffle(bins[b])
        elif within_bin_order == "asc":
            bins[b].sort(key=lambda x: x["pred_proba_1"])
        else:
            bins[b].sort(key=lambda x: x["pred_proba_1"], reverse=True)

    picked = {b: (bins[b] if per_bin <= 0 else bins[b][:min(per_bin, len(bins[b]))]) for b in bins}

    worklist = []
    ordered_bins = list(range(len(edges) - 1))[::-1]
    if round_robin:
        idx = {b: 0 for b in ordered_bins}
        remaining = sum(len(picked[b]) for b in ordered_bins)
        while remaining > 0:
            for b in ordered_bins:
                i = idx[b]
                if i < len(picked[b]):
                    worklist.append(picked[b][i])
                    idx[b] += 1
                    remaining -= 1
    else:
        for b in ordered_bins:
            worklist.extend(picked[b])

    stats = {
        "mode": "bins_merged",
        "binning": binning,
        "edges": edges,
        "available_per_bin": {int(b): len(bins[b]) for b in bins},
        "picked_per_bin": {int(b): len(picked[b]) for b in picked},
        "missing_pred_proba_1": missing,
        "total_work_items": len(worklist),
    }
    return worklist, stats

def label_data(
    clean_file: str,
    hate_file: str,
    output_file: str,
    backup_file: str,
    master_file: str,
    randomize: bool,
    mode: str = "interleave",       
    per_bin: int = 0,                  
    within_bin_order: str = "desc",    
    round_robin: bool = True,          
    seed: int = 42,                    
    binning: str = "fixed",           
    num_bins: int = 5,                 
):
    pp = PreProcessor()

    seen_ids, seen_hashes = load_seen([master_file, output_file])

    clean_pool = load_pool(clean_file, seen_ids, seen_hashes, pp)
    hate_pool  = load_pool(hate_file,  seen_ids, seen_hashes, pp)

    if randomize:
        random.shuffle(hate_pool)
    else:
        hate_pool.sort(key=lambda x: x.get("pred_proba_1", 0.0), reverse=True)

    bin_stats = None
    if mode == "bins":
        worklist, bin_stats = build_worklist_bins_merged(
            clean_pool=clean_pool,
            hate_pool=hate_pool,
            per_bin=per_bin,
            within_bin_order=within_bin_order,
            round_robin=round_robin,
            seed=seed,
            binning=binning,
            num_bins=num_bins,
        )
    else:
        worklist = build_worklist_interleave(clean_pool, hate_pool, HATE_PER_CLEAN)

    if not worklist:
        print("Nothing new to label – pools empty or fully labelled.")
        return

    print(f"🗂  {len(worklist):,} posts queued "
          f"({len(hate_pool):,} discriminatory, {len(clean_pool):,} clean).")
    if mode == "bins":
        print(f"    Mode=bins (merged)  per_bin={per_bin}  within_bin_order={within_bin_order}  "
              f"round_robin={round_robin}")
        print(f"    Binning={binning}  num_bins={num_bins}")
        print("    Available per bin:", bin_stats["available_per_bin"])
        print("    Picked per bin   :", bin_stats["picked_per_bin"])
        if bin_stats["missing_pred_proba_1"]:
            print(f"    [!] Skipped {bin_stats['missing_pred_proba_1']} items without pred_proba_1")
    else:
        print(f"    Mode=interleave (hate:clean = {HATE_PER_CLEAN}:1; "
              f"{'randomized' if randomize else 'sorted by confidence'})")

    print("\n" + LABEL_HELP)

    new_labels: List[Dict[str, Any]] = []
    idx = 0
    total = len(worklist)

    while idx < total:
        data = worklist[idx]
        text = data.get("clean_text") or ""
        p1 = data.get("pred_proba_1")
        p0 = data.get("pred_proba_0")
        b  = data.get("bin")

        if not text:
            idx += 1
            continue

        header = f"[{idx+1}/{total}]"
        if p1 is not None and p0 is not None:
            header += f"  (p1={float(p1):.2f}, p0={float(p0):.2f})"
        elif p1 is not None:
            header += f"  (p1={float(p1):.2f})"
        if b is not None:
            header += f"  bin={b}"

        print("\n" + "-" * 60)
        print(header)
        print(text)
        cmd = input("Label [y/n/Enter/b/q/h]: ").strip().lower()

        if cmd == "q":
            print("\n[↩] Exiting…")
            break
        if cmd == "h":
            print("\n" + LABEL_HELP)
            continue
        if cmd == "b":
            if new_labels:
                idx = max(0, idx - 1)
                removed = new_labels.pop()
                print(f"Backtracked one item. Removed label for id={removed.get('id')}")
            else:
                print("Nothing to undo.")
            continue
        if cmd in {"y", "n"}:
            data_out = {**data, "label": 1 if cmd == "y" else 0}
            new_labels.append(data_out)
            idx += 1
            continue
        idx += 1

    if new_labels:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        if os.path.exists(output_file) and backup_file and not os.path.exists(backup_file):
            os.replace(output_file, backup_file)
            print(f"Existing labels backed up → {backup_file}")

        with open(output_file, "a", encoding="utf-8") as out:
            for item in new_labels:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nSession complete.")
    print(f"   Newly labelled → {len(new_labels):,}")
    print(f"   Total file     → {output_file}")

def main():
    p = argparse.ArgumentParser(
        description="Interactive labeling (interleave OR merged-bin sampling across clean+hate)."
    )
    p.add_argument("--clean",     required=True, help="path to clean posts JSONL/JSON (should have pred_proba_1 for bins)")
    p.add_argument("--hate",      required=True, help="path to discriminatory posts JSONL/JSON (should have pred_proba_1 for bins)")
    p.add_argument("--output",    required=True, help="where to append your new labels (JSONL)")
    p.add_argument("--backup",    required=False, help="where to store a backup of existing output (moved once)")
    p.add_argument("--master",    required=False, help="(optional) path to master labeled JSONL/JSON to also exclude")
    p.add_argument("--randomize", action="store_true",
                   help="(interleave mode) randomize hate order instead of sorting by confidence")

    p.add_argument("--mode", choices=["interleave", "bins"], default="interleave",
                   help="interleave (default) OR bins (merge clean+hate, then bin & sample)")

    p.add_argument("--binning", choices=["fixed","quantile"], default="fixed",
                   help="fixed: 0–20,20–40,... ; quantile: equal-mass bins on the MERGED pool")
    p.add_argument("--num_bins", type=int, default=5, help="when --binning quantile, number of bins (default 5)")

    p.add_argument("--per_bin", type=int, default=10, help="cap items per bin (0 = take all)")
    p.add_argument("--within_bin_order", choices=["desc","asc","shuffle"], default="desc",
                   help="order within each bin before sampling")
    p.add_argument("--round_robin", action="store_true",
                   help="take one from each bin high→low repeatedly when forming the queue")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    label_data(
        clean_file   = args.clean,
        hate_file    = args.hate,
        output_file  = args.output,
        backup_file  = args.backup,
        master_file  = args.master,
        randomize    = args.randomize,
        mode         = args.mode,
        per_bin      = args.per_bin,
        within_bin_order = args.within_bin_order,
        round_robin  = args.round_robin,
        seed         = args.seed,
        binning      = args.binning,
        num_bins     = args.num_bins,
    )

if __name__ == "__main__":
    main()
