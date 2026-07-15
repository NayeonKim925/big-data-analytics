"""Phase 4: final path decoding + evaluation from saved scores (no GPU).

Decoder: best-path-by-mean with length margin delta.
delta is calibrated WITHOUT labels: chosen so the predicted length-3 share
matches the dataset property verified in phase 0 (92.5% of label sets are
length-3 paths). Test-set-optimal delta exists but is NOT adopted
(would be test tuning); see README sensitivity analysis.

Usage: python scripts/05_decode_eval.py --data_dir reference_data/Amazon-531
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_doc2labels
from src.evaluate import example_f1


def decode(scores, path2, path3, delta):
    s2, s3 = scores[:, path2].mean(2), scores[:, path3].mean(2)
    b2, v2 = s2.argmax(1), s2.max(1)
    b3, v3 = s3.argmax(1), s3.max(1)
    use3 = v3 >= v2 - delta
    return [list(path3[b3[i]]) if use3[i] else list(path2[b2[i]])
            for i in range(len(scores))]


def calibrate_delta(scores, path2, path3, target=0.925):
    grid = np.arange(0.0, 1.0, 0.005)
    def len3_share(d):
        return np.mean([len(p) == 3 for p in decode(scores, path2, path3, d)])
    return float(min(grid, key=lambda d: abs(len3_share(d) - target)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--scores", default="results/clf_scores_test_epoch0.npz")
    args = ap.parse_args()

    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    paths = taxo.enumerate_paths()
    path2 = np.array([p for p in paths if len(p) == 2])
    path3 = np.array([p for p in paths if len(p) == 3])
    true = load_doc2labels(d / "test" / "doc2labels.txt")
    scores = np.load(args.scores)["scores"].astype(np.float32)

    f_auto = example_f1(true, decode(scores, path2, path3, 0.0))
    delta = calibrate_delta(scores, path2, path3)
    preds = decode(scores, path2, path3, delta)
    share = np.mean([len(p) == 3 for p in preds])
    print(f"path auto (delta=0)          : {f_auto:.4f}")
    print(f"path calibrated (delta={delta:.3f}): {example_f1(true, preds):.4f}  len3 share={share:.3f}")


if __name__ == "__main__":
    main()