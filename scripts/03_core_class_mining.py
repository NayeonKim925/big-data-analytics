"""Phase 2: confident core class mining (TaxoClass Sect. 3.2.2).

Input : cache/sim_train.npz + cache/cand_train.json   (from Phase 1)
Output: results/core_classes_train.json  - per-doc confident core classes
        results/phase2_mining.json       - mining statistics + quality audit

Method (paper Eq. 2-3):
  conf(D, c)  = sim(D, c) - max_{c' in Par(c) ∪ Sib(c)} sim(D, c')
  keep c for D iff conf(D, c) >= median{ conf(D', c) | D' in D(c) },
  where D(c) = documents having c in their candidate set.
Docs may end up with ZERO core classes - the paper excludes them from the
classifier loss (Eq. 8); we keep them in the file as empty lists.

Ground-truth usage: train labels are read ONLY to audit silver-label quality
(precision etc.) for the report. They are never written into the outputs
used for training.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_doc2labels
from src.nli_topdown import SENTINEL


def rivals(taxo, c):
    """Par(c) ∪ Sib(c). Siblings = other children of c's parents.
    Top-level classes have no stored parent (virtual root), so their
    rivals are the other top-level classes."""
    out = set()
    ps = taxo.parents.get(c, ())
    if not ps:
        return taxo.roots - {c}
    for p in ps:
        out.add(p)
        out.update(taxo.children.get(p, ()))
    out.discard(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="observation-1 filter (paper Sect 3.2.2): keep c only "
                         "if conf(D,c) > min_conf, i.e. the doc is MORE similar "
                         "to c than to any of c's parents/siblings. Pass a very "
                         "negative value to disable (median-only ablation).")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated min_conf values, e.g. '0,0.05,0.1,0.2'. "
                         "Prints a precision/coverage trade-off table and saves "
                         "phase2_sweep.json WITHOUT writing core classes.")
    args = ap.parse_args()

    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    S = np.load(Path(args.cache_dir) / f"sim_{args.split}.npz")["scores"].astype(np.float32)
    cands = json.loads((Path(args.cache_dir) / f"cand_{args.split}.json").read_text())
    n = len(cands)

    # ---- Eq. 2: confidence of each (doc, candidate) pair ----
    print("computing confidence scores (Eq. 2) ...", flush=True)
    conf = {}                      # (i, c) -> conf score
    by_class = defaultdict(list)   # c -> [conf over docs in D(c)]
    for i in range(n):
        for c in cands[i]:
            riv = [S[i, r] for r in rivals(taxo, c) if S[i, r] != SENTINEL]
            v = float(S[i, c] - max(riv)) if riv else float(S[i, c])
            conf[(i, c)] = v
            by_class[c].append(v)

    # ---- Eq. 3 median filter AND observation-1 positivity filter ----
    med = {c: float(np.median(v)) for c, v in by_class.items()}
    true = load_doc2labels(d / args.split / "doc2labels.txt")[:n]
    lvl_names = {0: "root", 1: "mid", 2: "leaf"}

    def mine(minc):
        return [[c for c in cands[i]
                 if conf[(i, c)] > minc and conf[(i, c)] >= med[c]]
                for i in range(n)]

    def stats_of(cores, minc):
        sizes = [len(c) for c in cores]
        lvl_dist = {"root": 0, "mid": 0, "leaf": 0}
        for cs in cores:
            for c in cs:
                lvl_dist[lvl_names[taxo.level(c)]] += 1
        s = {"min_conf": minc,
             "docs_with_core_pct": round(100 * sum(x > 0 for x in sizes) / n, 2),
             "avg_cores_per_doc": round(float(np.mean(sizes)), 2),
             "core_level_dist": lvl_dist}
        pairs = [(i, c) for i in range(n) for c in cores[i]]
        if pairs:
            s["core_precision_pct"] = round(
                100 * float(np.mean([c in set(true[i]) for i, c in pairs])), 2)
            pos_hit, pos_n = 0, 0
            for i in range(n):
                if not cores[i]:
                    continue
                pos = set(cores[i])
                for c in cores[i]:
                    pos.update(taxo.parents.get(c, ()))
                pos_hit += len(pos & set(true[i]))
                pos_n += len(pos)
            s["eq7_positive_precision_pct"] = round(100 * pos_hit / pos_n, 2)
        return s

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        rows = []
        print(f"{'min_conf':>8} | {'docs%':>6} | {'cores/doc':>9} | "
              f"{'core_prec%':>10} | {'eq7_prec%':>9}")
        for minc in [float(x) for x in args.sweep.split(",")]:
            s = stats_of(mine(minc), minc)
            rows.append(s)
            print(f"{minc:>8} | {s['docs_with_core_pct']:>6} | "
                  f"{s['avg_cores_per_doc']:>9} | "
                  f"{s.get('core_precision_pct', 0):>10} | "
                  f"{s.get('eq7_positive_precision_pct', 0):>9}")
        (out / "phase2_sweep.json").write_text(json.dumps(rows, indent=2))
        print(f"saved -> {out}/phase2_sweep.json  "
              f"(no core file written in sweep mode)")
        return

    cores = mine(args.min_conf)
    stats = {"n_docs": n, **stats_of(cores, args.min_conf)}
    sizes = [len(c) for c in cores]
    stats["core_size_hist"] = {str(k): int(v) for k, v in
                               zip(*np.unique(sizes, return_counts=True))}
    print("mining stats:", stats)

    (out / f"core_classes_{args.split}.json").write_text(json.dumps(cores))
    (out / "phase2_mining.json").write_text(json.dumps(stats, indent=2))
    print(f"saved -> {out}/core_classes_{args.split}.json, {out}/phase2_mining.json")


if __name__ == "__main__":
    main()
