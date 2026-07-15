"""Phase 1b: zero-shot baseline from the cached similarity matrix.

Produces the first rows of the ablation table, all from ONE cache:
  1. hier-greedy   - Hier-0Shot-TC replication (top-down argmax path, len 3)
                     paper reference on Amazon-531: 0.4742 F1 / 0.7144 P@1
  2. top-3         - naive decoder: 3 highest-scoring classes (may be
                     hierarchy-inconsistent; theoretical max F1 = 0.985)
  3. path-auto     - structure-aware decoder over the 568 verified paths,
                     with automatic 2-vs-3 length decision
Plus ranking metrics (P@1/P@3/MRR both variants) and candidate-coverage
diagnostics (recall ceiling of the top-down selection, Sect. 3.2.1).

Run:  python scripts/02_zeroshot_baseline.py \
          --data_dir reference_data/Amazon-531 --cache cache/sim_test.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_doc2labels
from src.evaluate import (evaluate_all, example_f1,
                          scores_to_pred_sets_paths, scores_to_pred_sets_topk)
from src.nli_topdown import SENTINEL


def hier_greedy(S, taxo):
    """Hier-0Shot-TC style: argmax at each level, restricted to computed
    entries (i.e., to the candidate set the top-down search explored)."""
    preds = []
    roots = sorted(taxo.roots)
    for i in range(S.shape[0]):
        r = max(roots, key=lambda c: S[i, c])
        path = [r]
        kids = [c for c in taxo.children.get(r, ()) if S[i, c] != SENTINEL]
        if kids:
            m = max(kids, key=lambda c: S[i, c])
            path.append(m)
            gkids = [c for c in taxo.children.get(m, ()) if S[i, c] != SENTINEL]
            if gkids:
                path.append(max(gkids, key=lambda c: S[i, c]))
        preds.append(path)
    return preds


def coverage(true_sets, S):
    """Candidate-selection quality: was the truth even explored?"""
    mask = S != SENTINEL
    per_class = np.mean([mask[i, c] for i, t in enumerate(true_sets) for c in t])
    per_doc = np.mean([all(mask[i, c] for c in t)
                       for i, t in enumerate(true_sets)])
    return {"true_class_computed_pct": round(float(per_class) * 100, 2),
            "full_true_path_computed_pct": round(float(per_doc) * 100, 2),
            "avg_computed_classes_per_doc": round(float(mask.sum(1).mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache", default="cache/sim_test.npz")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/phase1_zeroshot.json")
    args = ap.parse_args()

    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    paths = taxo.enumerate_paths()
    true = load_doc2labels(d / args.split / "doc2labels.txt")

    S = np.load(args.cache)["scores"].astype(np.float32)
    true = true[:S.shape[0]]  # tolerate --limit dry runs

    report = {"cache": args.cache, "coverage": coverage(true, S)}
    print("candidate coverage:", report["coverage"])

    decoders = {
        "hier_greedy": hier_greedy(S, taxo),
        "top3": scores_to_pred_sets_topk(S, k=3),
        "path_auto": scores_to_pred_sets_paths(S, paths),
        "path_len3": scores_to_pred_sets_paths(S, paths, length_rule=3),
    }
    ranking = evaluate_all(true, score_matrix=S, name="ranking")
    report["ranking"] = ranking
    print(f"ranking: P@1={ranking['p@1']} P@3={ranking['p@3']} "
          f"MRR(literal)={ranking['mrr_literal']} "
          f"MRR(filtered)={ranking['mrr_filtered']}")

    report["decoders"] = {}
    for name, preds in decoders.items():
        f1 = round(example_f1(true, preds), 4)
        report["decoders"][name] = f1
        print(f"Example-F1 [{name:12s}] = {f1}")

    # error decomposition: which level does the greedy path get right?
    g = decoders["hier_greedy"]
    lvl = {}
    for li, lname in enumerate(["root", "mid", "leaf"]):
        ok = [g[i][li] in set(true[i]) for i in range(len(true)) if len(g[i]) > li]
        lvl[lname] = round(100 * float(np.mean(ok)), 2)
    report["greedy_level_accuracy_pct"] = lvl
    print("greedy per-level accuracy (%):", lvl)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
