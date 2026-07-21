"""Evaluation metrics for HMTC, matching TaxoClass (NAACL 2021) definitions.

All downstream experiments report through this single module so that
numbers in the README are produced by one auditable code path.

Metrics:
  - Example-F1: mean over docs of 2|T ∩ P| / (|T| + |P|)   (set prediction)
  - P@k:        mean over docs of |T ∩ top-k| / min(k, |T|) (ranking)
  - MRR:        mean over docs of (1/|T|) * sum_{c in T} 1/rank(c)
P@k and MRR require per-class scores; Example-F1 requires a predicted set.
"""
import numpy as np


def example_f1(true_sets, pred_sets):
    total = 0.0
    for t, p in zip(true_sets, pred_sets):
        t, p = set(t), set(p)
        denom = len(t) + len(p)
        total += (2 * len(t & p) / denom) if denom else 0.0
    return total / len(true_sets)


def precision_at_k(true_sets, score_matrix, k):
    """score_matrix: (n_docs, n_classes) array of per-class scores."""
    topk = np.argsort(-score_matrix, axis=1)[:, :k]
    total = 0.0
    for i, t in enumerate(true_sets):
        t = set(t)
        total += len(t & set(topk[i].tolist())) / min(k, len(t))
    return total / len(true_sets)


def _rank_matrix(score_matrix):
    order = np.argsort(-score_matrix, axis=1)
    ranks = np.empty_like(order)
    n = order.shape[1]
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]
    return ranks


def mrr(true_sets, score_matrix):
    """Literal Table-2 formula: rank over ALL classes.

    NOTE: on Amazon-531 a PERFECT ranking yields only ~0.6216 under this
    formula, because a doc's 2-3 true classes can at best occupy ranks
    1..3: e.g. (1 + 1/2 + 1/3)/3 = 0.611 per 3-label doc. TaxoClass reports
    0.6332 > ceiling, so the paper's implementation must use the filtered
    variant below. Report both, clearly labeled.
    """
    ranks = _rank_matrix(score_matrix)
    total = 0.0
    for i, t in enumerate(true_sets):
        total += sum(1.0 / ranks[i, c] for c in t) / len(t)
    return total / len(true_sets)


def mrr_filtered(true_sets, score_matrix):
    """Filtered-rank MRR: when ranking true class c, the OTHER true classes
    are removed from the candidate list (standard KG/IR convention).
    filtered_rank(c) = rank(c) - #(true classes ranked above c).
    Perfect ranking -> 1.0. This is almost certainly what Table 2 reports.
    """
    ranks = _rank_matrix(score_matrix)
    total = 0.0
    for i, t in enumerate(true_sets):
        r = sorted(ranks[i, c] for c in t)
        # after sorting, exactly j other true classes rank above the j-th one
        total += sum(1.0 / (rank - j) for j, rank in enumerate(r)) / len(t)
    return total / len(true_sets)


def evaluate_all(true_sets, pred_sets=None, score_matrix=None, name=""):
    """One-stop report. Provide pred_sets (for Example-F1) and/or scores."""
    out = {"name": name, "n_docs": len(true_sets)}
    if pred_sets is not None:
        out["example_f1"] = round(example_f1(true_sets, pred_sets), 4)
    if score_matrix is not None:
        out["p@1"] = round(precision_at_k(true_sets, score_matrix, 1), 4)
        out["p@3"] = round(precision_at_k(true_sets, score_matrix, 3), 4)
        out["mrr_literal"] = round(mrr(true_sets, score_matrix), 4)
        out["mrr_filtered"] = round(mrr_filtered(true_sets, score_matrix), 4)
    return out


def scores_to_pred_sets_topk(score_matrix, k=3):
    """Naive decoder: take top-k classes per doc (baseline decoder)."""
    topk = np.argsort(-score_matrix, axis=1)[:, :k]
    return [row.tolist() for row in topk]


def scores_to_pred_sets_paths(score_matrix, paths, length_rule="auto"):
    """Structure-aware decoder: pick the best root->down path.

    Uses the verified dataset property that every label set is a path of
    length 2 or 3. Scores each candidate path by the mean of its class
    scores and returns the argmax path per document.

    length_rule:
      - "auto": compare best length-2 path vs best length-3 path by mean score
      - 2 or 3: force a fixed length
    """
    path2 = [p for p in paths if len(p) == 2]
    path3 = [p for p in paths if len(p) == 3]
    p2 = np.array(path2)                      # (n2, 2)
    p3 = np.array(path3)                      # (n3, 3)
    s2 = score_matrix[:, p2].mean(axis=2)     # (n_docs, n2)
    s3 = score_matrix[:, p3].mean(axis=2)     # (n_docs, n3)
    best2, best2s = s2.argmax(axis=1), s2.max(axis=1)
    best3, best3s = s3.argmax(axis=1), s3.max(axis=1)
    preds = []
    for i in range(score_matrix.shape[0]):
        if length_rule == 2:
            preds.append(list(path2[best2[i]]))
        elif length_rule == 3:
            preds.append(list(path3[best3[i]]))
        else:
            preds.append(list(path3[best3[i]]) if best3s[i] >= best2s[i]
                         else list(path2[best2[i]]))
    return preds
