"""Phase 0: dataset verification + evaluation sanity checks.

Run:  python scripts/00_verify_dataset.py --data_dir <path>/TELEClass/data/Amazon-531
Optionally pass --course_train / --course_test to verify your course files
are the same documents in the same order as the reference data.

Produces the numbers cited in the README's "Problem Analysis" section:
  1. label-set size distribution (2 or 3 labels)
  2. 100% of label sets are root->down ancestry chains ("paths")
  3. exactly one top-level class per document
  4. metric sanity: oracle = 1.0, random-path baseline as the floor
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_corpus, load_doc2labels, check_alignment
from src.evaluate import evaluate_all, example_f1

random.seed(42)
np.random.seed(42)


def is_chain(labs, taxo):
    """True iff every pair in the label set is in an ancestor relation."""
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            if a not in taxo.ancestors(b) and b not in taxo.ancestors(a):
                return False
    return True


def analyze_split(name, labels, taxo):
    from collections import Counter
    sizes = Counter(len(l) for l in labels)
    chains = sum(is_chain(l, taxo) for l in labels)
    tops = Counter(sum(1 for c in l if c in taxo.roots) for l in labels)
    print(f"\n[{name}] n={len(labels)}")
    print(f"  label-set sizes           : {dict(sizes)}")
    print(f"  ancestry chains (paths)   : {chains}/{len(labels)} "
          f"({chains / len(labels) * 100:.2f}%)")
    print(f"  top-level classes per doc : {dict(tops)}")
    return {"n": len(labels), "sizes": dict(sizes),
            "chain_pct": round(chains / len(labels) * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="path to TELEClass data/Amazon-531 directory")
    ap.add_argument("--course_train", default=None,
                    help="(optional) your course train_corpus.txt to check alignment")
    ap.add_argument("--course_test", default=None,
                    help="(optional) your course test_corpus.txt to check alignment")
    ap.add_argument("--out", default="results/phase0.json")
    args = ap.parse_args()
    d = Path(args.data_dir)

    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    levels = {}
    for c in taxo.names:
        levels.setdefault(taxo.level(c), 0)
        levels[taxo.level(c)] += 1
    print(f"taxonomy: {len(taxo.names)} classes, levels {dict(sorted(levels.items()))}, "
          f"{len(taxo.roots)} top-level, {len(taxo.leaves)} leaves, "
          f"{sum(1 for c in taxo.parents.values() if len(c) > 1)} multi-parent")

    paths = taxo.enumerate_paths()
    n2 = sum(1 for p in paths if len(p) == 2)
    print(f"candidate label-set space: {len(paths)} paths "
          f"({n2} of length 2, {len(paths) - n2} of length 3) "
          f"vs 2^531 unconstrained")

    report = {"levels": {str(k): v for k, v in sorted(levels.items())},
              "n_paths": len(paths)}

    train_labels = load_doc2labels(d / "train" / "doc2labels.txt")
    test_labels = load_doc2labels(d / "test" / "doc2labels.txt")
    report["train"] = analyze_split("train", train_labels, taxo)
    report["test"] = analyze_split("test", test_labels, taxo)

    # optional: confirm course corpus == reference corpus, doc-by-doc
    for tag, course, ref in (("train", args.course_train, d / "train" / "corpus.txt"),
                             ("test", args.course_test, d / "test" / "corpus.txt")):
        if course:
            ok, msg = check_alignment(course, ref)
            print(f"\nalignment check ({tag}): {'OK' if ok else 'FAILED'} - {msg}")
            report[f"alignment_{tag}"] = msg

    # ---- metric sanity checks on the test split ----
    print("\n[metric sanity checks - test split]")
    oracle = evaluate_all(test_labels, pred_sets=test_labels, name="oracle")
    print(f"  oracle Example-F1        : {oracle['example_f1']} (must be 1.0)")

    rand_paths = [list(random.choice(paths)) for _ in test_labels]
    rand_f1 = example_f1(test_labels, rand_paths)
    print(f"  random-path Example-F1   : {rand_f1:.4f} (floor / dummy baseline)")

    # a structure-only reference: most frequent train path predicted for all
    from collections import Counter
    top_path = Counter(tuple(sorted(l)) for l in train_labels).most_common(1)[0][0]
    const_f1 = example_f1(test_labels, [list(top_path)] * len(test_labels))
    print(f"  majority-path Example-F1 : {const_f1:.4f} "
          f"(path {[taxo.names[c] for c in top_path]})")

    report["oracle_f1"] = oracle["example_f1"]
    report["random_path_f1"] = round(rand_f1, 4)
    report["majority_path_f1"] = round(const_f1, 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
