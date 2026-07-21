"""Top-down document-class similarity computation.

Implements TaxoClass (NAACL 2021) Sect. 3.1 (entailment-based similarity)
and the candidate-selection part of Sect. 3.2.1 (top-down search), producing
a cached (n_docs x 531) score matrix where uncomputed entries hold SENTINEL.

Selection procedure (paper, mapped to Amazon-531's 3 levels):
  - Stage A: score all 6 top-level classes; keep top k_root=2 per doc.
  - Stage B: score all children of the selected roots; per selected root keep
    top 3 children (l+2 with l=1); pool and keep top 4 (=(l+1)^2) by path score.
  - Stage C: score all children of the selected mids; per selected mid keep
    top 4 children (l+2 with l=2); pool and keep top 9 (=(2+1)^2) by path score.
  - Gap fill: additionally score every parent/sibling of each candidate that
    is still missing, so that confidence scores (Eq. 2) can be computed later
    from this cache alone (needed in Phase 2, avoids re-running the GPU job).

Path score (paper Eq. 1): ps(root)=sim(root); ps(c)=max_{p in Par(c)} ps(p)*sim(c).
"""
import numpy as np

SENTINEL = -1.0


def _fill(S, pairs, scores):
    for (i, c), s in zip(pairs, scores):
        S[i, c] = s


def _score_missing(S, texts, pairs, scorer):
    """Score only pairs not yet in the cache; return nothing (fills S)."""
    todo = [(i, c) for (i, c) in pairs if S[i, c] == SENTINEL]
    if todo:
        _fill(S, todo, scorer(texts, todo))


def topdown_shard(texts, taxo, scorer, k_root=2, k_child=(3, 4), k_pool=(4, 9)):
    """Run the 3-stage top-down selection for one shard of documents.

    texts:  list of document strings (shard)
    scorer: callable(texts, pairs) -> list/array of similarity scores in [0,1],
            where pairs is a list of (doc_index_within_shard, class_id).
    Returns (S, candidates): S is (len(texts) x n_classes) float32 with
    SENTINEL for uncomputed entries; candidates[i] is the ordered candidate
    core-class list (queue members) for doc i.
    """
    n, C = len(texts), len(taxo.names)
    S = np.full((n, C), SENTINEL, dtype=np.float32)
    roots = sorted(taxo.roots)

    # ---- Stage A: all top-level classes ----
    _score_missing(S, texts, [(i, c) for i in range(n) for c in roots], scorer)
    sel_roots = [sorted(roots, key=lambda c: -S[i, c])[:k_root] for i in range(n)]

    # ---- Stage B: children of selected roots -> select mids ----
    pairs = [(i, ch) for i in range(n) for r in sel_roots[i]
             for ch in taxo.children.get(r, ())]
    _score_missing(S, texts, pairs, scorer)

    sel_mids = []
    for i in range(n):
        pool = set()
        for r in sel_roots[i]:
            kids = sorted(taxo.children.get(r, ()), key=lambda c: -S[i, c])
            pool.update(kids[:k_child[0]])
        # path score: ps(mid) = max over computed root parents of sim(root)*sim(mid)
        def ps_mid(c):
            ps = [S[i, p] * S[i, c] for p in taxo.parents.get(c, ())
                  if S[i, p] != SENTINEL]
            return max(ps) if ps else -np.inf
        sel_mids.append(sorted(pool, key=lambda c: -ps_mid(c))[:k_pool[0]])

    # ---- Stage C: children of selected mids -> select leaves ----
    pairs = [(i, ch) for i in range(n) for m in sel_mids[i]
             for ch in taxo.children.get(m, ())]
    _score_missing(S, texts, pairs, scorer)

    sel_leaves = []
    for i in range(n):
        pool = set()
        for m in sel_mids[i]:
            kids = sorted(taxo.children.get(m, ()), key=lambda c: -S[i, c])
            pool.update(kids[:k_child[1]])
        def ps_leaf(c):
            best = -np.inf
            for p in taxo.parents.get(c, ()):
                if S[i, p] == SENTINEL:
                    continue
                ps_p = max((S[i, g] * S[i, p] for g in taxo.parents.get(p, ())
                            if S[i, g] != SENTINEL), default=S[i, p])
                best = max(best, ps_p * S[i, c])
            return best
        sel_leaves.append(sorted(pool, key=lambda c: -ps_leaf(c))[:k_pool[1]])

    candidates = [sel_roots[i] + sel_mids[i] + sel_leaves[i] for i in range(n)]

    # ---- Gap fill: parents & siblings of every candidate (for Eq. 2 later) ----
    pairs = []
    for i in range(n):
        need = set()
        for c in candidates[i]:
            for p in taxo.parents.get(c, ()):
                need.add(p)
                need.update(taxo.children.get(p, ()))   # siblings incl. c
        pairs.extend((i, c) for c in need)
    _score_missing(S, texts, pairs, scorer)

    return S, candidates


class MockScorer:
    """Deterministic fake scorer for pipeline dry runs (no GPU / no model).
    Gives mildly informative scores: overlap between class name words and
    document words, plus seeded noise - enough to exercise every code path."""
    def __init__(self, taxo, seed=42):
        self.names = {c: set(n.split()) for c, n in taxo.names.items()}
        self.rng = np.random.default_rng(seed)

    def __call__(self, texts, pairs):
        out = np.empty(len(pairs), dtype=np.float32)
        for k, (i, c) in enumerate(pairs):
            words = set(texts[i].split())
            ov = len(words & self.names[c]) / max(1, len(self.names[c]))
            out[k] = np.clip(0.5 * ov + 0.3 * self.rng.random(), 0, 1)
        return out


class NLIScorer:
    """Real entailment scorer (paper setting: roberta-large-mnli).

    premise  = document text (truncated first on overflow)
    hypothesis = template filled with the class surface name
    score    = P(entailment) from the MNLI head.
    """
    def __init__(self, model_name="roberta-large-mnli",
                 template="this product is about {}.",
                 batch_size=128, max_length=320, device=None, fp16=True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16 if fp16 else torch.float32
        ).to(self.device).eval()
        # locate the entailment logit index from the model config (robust
        # across MNLI checkpoints, which order labels differently)
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent_idx = label2id["entailment"]
        self.template = template
        self.batch_size = batch_size
        self.max_length = max_length
        self.class_names = None  # set externally: dict class_id -> name

    def __call__(self, texts, pairs):
        import numpy as np
        scores = np.empty(len(pairs), dtype=np.float32)
        bs = self.batch_size
        with self.torch.inference_mode():
            for s in range(0, len(pairs), bs):
                chunk = pairs[s:s + bs]
                prem = [texts[i] for i, _ in chunk]
                hyp = [self.template.format(self.class_names[c]) for _, c in chunk]
                enc = self.tok(prem, hyp, truncation="only_first",
                               max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                probs = logits.softmax(dim=-1)[:, self.ent_idx]
                scores[s:s + len(chunk)] = probs.cpu().numpy()
        return scores
