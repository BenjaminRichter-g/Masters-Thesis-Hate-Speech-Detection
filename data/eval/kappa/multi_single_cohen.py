import argparse, json, os, sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

LABELS = ["race", "gender", "religion"]
PRIMARY_MAP = {"none":0, "race":1, "gender":2, "religion":3}


def read_jsonl(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            if rid is None:
                rid = f"__row__:{i}"
            items[rid] = obj
    return items


def map_primary(x):
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        return PRIMARY_MAP.get(x.lower(), None)
    return None

def extract_singlelabel_pairs(A, B):
    a_list, b_list = [], []
    matched = 0
    shared = set(A.keys()) & set(B.keys())
    for rid in shared:
        a = A[rid]; b = B[rid]
        key = None
        for k in ("gold_primary_label", "label", "gold_primary_label_name"):
            if k in a and k in b:
                key = k; break
        if key is None:
            continue
        va = map_primary(a[key]); vb = map_primary(b[key])
        if va is None or vb is None:
            continue
        a_list.append(va); b_list.append(vb)
        matched += 1
    return np.array(a_list, dtype=int), np.array(b_list, dtype=int), matched

def extract_multilabel_pairs(A, B):
    """Return per-class arrays and overlap statistics."""
    pairs = {lbl: ([], []) for lbl in LABELS}
    matched = 0
    any_overlap_flags = []
    both_none_flags = []

    shared = set(A.keys()) & set(B.keys())
    for rid in shared:
        a = A[rid]; b = B[rid]
        ok = True
        for lbl in LABELS:
            if lbl not in a or lbl not in b:
                ok = False; break
            va, vb = a[lbl], b[lbl]
            if va not in (0,1) or vb not in (0,1):
                ok = False; break
        if not ok:
            continue
        matched += 1
        for lbl in LABELS:
            pairs[lbl][0].append(a[lbl])
            pairs[lbl][1].append(b[lbl])
        set_a = {lbl for lbl in LABELS if a[lbl] == 1}
        set_b = {lbl for lbl in LABELS if b[lbl] == 1}
        any_overlap_flags.append(1 if set_a.intersection(set_b) else 0)
        both_none_flags.append(1 if (len(set_a)==0 and len(set_b)==0) else 0)

    for lbl in LABELS:
        pairs[lbl] = (np.array(pairs[lbl][0], dtype=int),
                      np.array(pairs[lbl][1], dtype=int))
    overlap_stats = {
        "any_overlap_rate": float(np.mean(any_overlap_flags)) if any_overlap_flags else None,
        "both_none_rate": float(np.mean(both_none_flags)) if both_none_flags else None,
        "N_overlap_eval": len(any_overlap_flags)
    }
    return pairs, matched, overlap_stats

def gwet_ac1_binary(a, b):
    """Binary Gwet AC1 for two raters; a,b ∈ {0,1}^N."""
    a = np.asarray(a); b = np.asarray(b)
    po = (a == b).mean()
    p1 = (a.mean() + b.mean()) / 2.0
    pe = 2 * p1 * (1 - p1)
    if pe >= 1.0: 
        return 0.0
    return (po - pe) / (1 - pe)

def ac1_macro_ovr(a, b, classes=(0,1,2,3)):
    """Multiclass AC1 via macro average of OvR AC1 across classes."""
    return float(np.mean([gwet_ac1_binary((a==c).astype(int), (b==c).astype(int)) for c in classes]))

def bootstrap_ci(stat_fn, a, b, iters=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(a); idx = np.arange(n)
    vals = []
    for _ in range(iters):
        samp = rng.choice(idx, size=n, replace=True)
        vals.append(stat_fn(a[samp], b[samp]))
    vals = np.sort(np.array(vals, dtype=float))
    lo = vals[int((alpha/2)*iters)]
    hi = vals[int((1-alpha/2)*iters)]
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(
        description="Compute Cohen’s κ & Gwet’s AC1 for BOTH single- and multilabel annotations."
    )
    ap.add_argument("--ann1", required=True, help="Annotator 1 JSONL")
    ap.add_argument("--ann2", required=True, help="Annotator 2 JSONL")
    ap.add_argument("--out_json", help="Optional JSON summary output")
    ap.add_argument("--boot_iters", type=int, default=2000, help="Bootstrap iterations (default 2000)")
    args = ap.parse_args()

    A = read_jsonl(args.ann1)
    B = read_jsonl(args.ann2)
    shared = len(set(A.keys()) & set(B.keys()))
    if shared == 0:
        print("No shared ids between files.", file=sys.stderr)
        sys.exit(1)

    summary = {"n_shared_ids": shared}

    ml_pairs, ml_matched, overlap_stats = extract_multilabel_pairs(A, B)
    summary["multilabel"] = {"matched": ml_matched, **overlap_stats, "per_class": []}

    print(f"\n=== Multilabel reliability (race/gender/religion) ===")
    print(f"Matched items with all flags present: {ml_matched}")
    if overlap_stats["any_overlap_rate"] is not None:
        print(f"Any-overlap rate: {overlap_stats['any_overlap_rate']:.3f}   "
              f"Both-none rate: {overlap_stats['both_none_rate']:.3f}   "
              f"(N={overlap_stats['N_overlap_eval']})")

    kappas, ac1s = [], []
    for lbl in LABELS:
        a, b = ml_pairs[lbl]
        n = len(a)
        if n == 0:
            print(f"{lbl:9s} N=0 (no overlap)")
            summary["multilabel"]["per_class"].append({"label": lbl, "N": 0})
            continue
        k = cohen_kappa_score(a, b)
        ac = gwet_ac1_binary(a, b)
        pa = float((a == b).mean())
        p_yes_a, p_yes_b = float(a.mean()), float(b.mean())
        prev_idx = abs(p_yes_a + p_yes_b - 1.0)
        bias_idx = abs(p_yes_a - p_yes_b)
        k_lo, k_hi = bootstrap_ci(lambda x,y: cohen_kappa_score(x,y), a, b, iters=args.boot_iters)
        ac_lo, ac_hi = bootstrap_ci(gwet_ac1_binary, a, b, iters=args.boot_iters)

        kappas.append(k); ac1s.append(ac)
        summary["multilabel"]["per_class"].append({
            "label": lbl, "N": n,
            "kappa": k, "kappa_CI": [k_lo, k_hi],
            "AC1": ac, "AC1_CI": [ac_lo, ac_hi],
            "percent_agree": pa,
            "prevalence_index": prev_idx,
            "bias_index": bias_idx
        })
        print(f"{lbl:9s} N={n:4d} | κ={k:.3f} [{k_lo:.3f},{k_hi:.3f}]  "
              f"AC1={ac:.3f} [{ac_lo:.3f},{ac_hi:.3f}]  "
              f"%agree={pa:.3f}  prevIdx={prev_idx:.3f}  biasIdx={bias_idx:.3f}")

    if kappas:
        summary["multilabel"]["macro_kappa"] = float(np.mean(kappas))
        summary["multilabel"]["macro_AC1"] = float(np.mean(ac1s))
        print(f"Macro κ={summary['multilabel']['macro_kappa']:.3f}  "
              f"Macro AC1={summary['multilabel']['macro_AC1']:.3f}")

    a_1, b_1, sl_matched = extract_singlelabel_pairs(A, B)
    summary["single_label"] = {"matched": sl_matched}
    print(f"\n=== Single-label reliability (primary/most-salient) ===")
    print(f"Matched items with primary label present: {sl_matched}")
    if sl_matched > 0:
        k = cohen_kappa_score(a_1, b_1)
        pa = float((a_1 == b_1).mean())
        ac1_macro = ac1_macro_ovr(a_1, b_1, classes=(0,1,2,3))
        k_lo, k_hi = bootstrap_ci(lambda x,y: cohen_kappa_score(x,y), a_1, b_1, iters=args.boot_iters)
        ac_lo, ac_hi = bootstrap_ci(
            lambda x,y: ac1_macro_ovr(x, y, classes=(0,1,2,3)), a_1, b_1, iters=args.boot_iters
        )
        print(f"κ={k:.3f} [{k_lo:.3f},{k_hi:.3f}]  "
              f"AC1≈{ac1_macro:.3f} [{ac_lo:.3f},{ac_hi:.3f}]  "
              f"%agree={pa:.3f}")
        summary["single_label"].update({
            "kappa": k, "kappa_CI": [k_lo, k_hi],
            "AC1_macroOvR": ac1_macro, "AC1_macroOvR_CI": [ac_lo, ac_hi],
            "percent_agree": pa
        })
    else:
        print("No overlapping primary labels to evaluate.")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary → {args.out_json}")

if __name__ == "__main__":
    main()
