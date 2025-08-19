import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_first(path, pred):
    for name in sorted(os.listdir(path)):
        if pred(name):
            return os.path.join(path, name)
    return None

def find_pr_files(label_dir):
    pos = find_first(label_dir, lambda n: n.endswith(".json") and "_pos" in n)
    neg = find_first(label_dir, lambda n: n.endswith(".json") and "_neg" in n)
    return pos, neg

def find_tau_file(label_dir):
    best = None; best_mtime = -1
    for name in os.listdir(label_dir):
        if not name.endswith(".json"):
            continue
        p = os.path.join(label_dir, name)
        try:
            obj = load_json(p)
        except Exception:
            continue
        if "tau_pos" in obj and "tau_neg" in obj:
            m = os.path.getmtime(p)
            if m > best_mtime:
                best, best_mtime = p, m
    return best

def load_pr_arrays(pr_json_path):
    obj = load_json(pr_json_path)
    P = np.array(obj["precision"], dtype=float)
    R = np.array(obj["recall"], dtype=float)
    T = np.array(obj.get("thresholds", []), dtype=float)
    AP = float(obj.get("auprc", np.nan))
    return P, R, T, AP

def pr_point_for_tau(P, R, T, tau):
    if T.size == 0 or tau is None:
        return None
    k = int(np.argmin(np.abs(T - float(tau))))
    j = min(k + 1, len(P) - 1)
    return float(P[j]), float(R[j]), float(T[k])

def overlay_plot(curves, title, out_path=None, step=True, dpi=150, color_map=None,
                 linestyle="solid", xlim=(0,1), ylim=(0,1)):
    plt.figure(figsize=(6.8, 5.2))
    for c in curves:
        name = c["name"]
        color = (color_map.get(name) if color_map else None)
        label = f"{name} (AP={c['AP']:.3f})"
        ls = "--" if c.get("style") == "dashed" else "solid"

        if step:
            plt.step(c["R"], c["P"], where="post", linewidth=2, linestyle=ls, color=color, label=label)
        else:
            plt.plot(c["R"], c["P"], linewidth=2, linestyle=ls, color=color, label=label)

        if c.get("baseline") is not None:
            plt.axhline(float(c["baseline"]), linestyle=":", color=color, alpha=0.6)

        if c.get("tau") is not None:
            mark = pr_point_for_tau(c["P"], c["R"], c["T"], c["tau"])
            if mark:
                p, r, t = mark
                plt.scatter([r], [p], s=35, color=color)
                plt.annotate(f"{name}: τ≈{t:.3f}\nP={p:.2f}, R={r:.2f}",
                             (r, p), textcoords="offset points", xytext=(8, -12))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def build_color_map(names):
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])
    names_sorted = sorted(names)
    return {name: palette[i % len(palette)] for i, name in enumerate(names_sorted)}

def parse_range(s, default):
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return default
    try:
        a = float(parts[0]); b = float(parts[1])
        return (a, b)
    except ValueError:
        return default

def main():
    ap = argparse.ArgumentParser(
        description="Overlay PR curves for labels by pointing to their folders (auto-load pos/neg PR, tau, baselines)."
    )
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Label folders (e.g., .../race .../gender .../religion)")
    ap.add_argument("--title_pos", default="Positive PR, overlay")
    ap.add_argument("--title_neg", default="Negative PR, overlay")
    ap.add_argument("--out_pos", default=None)
    ap.add_argument("--out_neg", default=None)
    ap.add_argument("--step", action="store_true", help="Draw step curves")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--xlim_pos", default=None, help="xlim for positive plot as 'a,b' (default 0,1)")
    ap.add_argument("--ylim_pos", default=None, help="ylim for positive plot as 'a,b' (default 0,1)")
    ap.add_argument("--xlim_neg", default=None, help="xlim for negative plot as 'a,b' (default 0.5,1.0)")
    ap.add_argument("--ylim_neg", default=None, help="ylim for negative plot as 'a,b' (default 0.5,1.0)")
    args = ap.parse_args()

    pos_curves, neg_curves = [], []
    label_names = set()

    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"[warn] not a dir: {d} (skipping)")
            continue
        name = os.path.basename(os.path.normpath(d))
        pr_pos, pr_neg = find_pr_files(d)
        tau_path = find_tau_file(d)

        if not pr_pos or not pr_neg:
            print(f"[warn] missing PR files in {d} (found pos={bool(pr_pos)}, neg={bool(pr_neg)})")
            continue

        Pp, Rp, Tp, APp = load_pr_arrays(pr_pos)
        Pn, Rn, Tn, APn = load_pr_arrays(pr_neg)

        tau_pos = tau_neg = pos_base = neg_base = None
        if tau_path:
            tau = load_json(tau_path)
            name = tau.get("label_key", name)
            tau_pos = float(tau.get("tau_pos")) if "tau_pos" in tau else None
            tau_neg = float(tau.get("tau_neg")) if "tau_neg" in tau else None
            n = float(tau.get("n", 0.0)) or 0.0
            n_pos = float(tau.get("n_pos", 0.0)) or 0.0
            n_neg = float(tau.get("n_neg", 0.0)) or 0.0
            if n > 0:
                pos_base = n_pos / n
                neg_base = n_neg / n

        label_names.add(name)

        pos_curves.append({
            "name": name, "P": Pp, "R": Rp, "T": Tp, "AP": APp,
            "tau": tau_pos, "baseline": pos_base, "style": "solid"
        })
        neg_curves.append({
            "name": name, "P": Pn, "R": Rn, "T": Tn, "AP": APn,
            "tau": tau_neg, "baseline": neg_base, "style": "dashed"
        })

        print(f"[i] {name}: pos={os.path.basename(pr_pos)}, neg={os.path.basename(pr_neg)}, tau={os.path.basename(tau_path) if tau_path else 'none'}")

    color_map = build_color_map(label_names)

    xlim_pos = parse_range(args.xlim_pos, (0, 1))
    ylim_pos = parse_range(args.ylim_pos, (0, 1))
    xlim_neg = parse_range(args.xlim_neg, (0.6, 1.0))
    ylim_neg = parse_range(args.ylim_neg, (0.8, 1.0))

    if pos_curves:
        overlay_plot(pos_curves, args.title_pos, args.out_pos, step=args.step, dpi=args.dpi,
                     color_map=color_map, linestyle="solid", xlim=xlim_pos, ylim=ylim_pos)
    if neg_curves:
        overlay_plot(neg_curves, args.title_neg, args.out_neg, step=args.step, dpi=args.dpi,
                     color_map=color_map, linestyle="dashed", xlim=xlim_neg, ylim=ylim_neg)

if __name__ == "__main__":
    main()
