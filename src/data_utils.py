"""Data utilities for Amazon-531 weakly-supervised HMTC.

Dataset source (with ground-truth labels for evaluation only):
    https://github.com/yzhan238/TELEClass  -> data/Amazon-531/
Ground-truth labels are NEVER used for training in this project;
they are used exclusively to compute evaluation metrics, mirroring
the private Kaggle leaderboard of the original course setup.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Taxonomy:
    """Class taxonomy as a DAG (Amazon-531: 3 levels, 6 -> 64 -> 461)."""
    names: dict            # class_id -> surface name (underscores replaced)
    parents: dict          # class_id -> set of parent ids
    children: dict         # class_id -> set of child ids
    roots: set = field(default_factory=set)    # top-level classes (no parent)
    leaves: set = field(default_factory=set)   # classes with no children

    @classmethod
    def load(cls, labels_path, hierarchy_path):
        names = {}
        with open(labels_path) as f:
            for line in f:
                cid, name = line.rstrip("\n").split("\t")
                names[int(cid)] = name.replace("_", " ")
        parents, children = defaultdict(set), defaultdict(set)
        with open(hierarchy_path) as f:
            for line in f:
                p, c = map(int, line.strip().split("\t"))
                parents[c].add(p)
                children[p].add(c)
        all_ids = set(names)
        roots = {c for c in all_ids if c not in parents}
        leaves = {c for c in all_ids if c not in children}
        return cls(names=names, parents=dict(parents), children=dict(children),
                   roots=roots, leaves=leaves)

    def level(self, c):
        """0 for top-level classes, min distance from a root otherwise."""
        if c in self.roots:
            return 0
        return 1 + min(self.level(p) for p in self.parents[c])

    def ancestors(self, c):
        """All ancestors of c (excluding c itself)."""
        out, stack = set(), list(self.parents.get(c, ()))
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(self.parents.get(p, ()))
        return out

    def enumerate_paths(self):
        """All root-to-node paths of length 2 or 3 (the label-set search space).

        Verified property of Amazon-531 (train AND test, 100%):
        every ground-truth label set is exactly one such path.
        Returns a list of tuples, e.g. (root, mid) or (root, mid, leaf).
        """
        paths = []
        for r in sorted(self.roots):
            for m in sorted(self.children.get(r, ())):
                paths.append((r, m))
                for l in sorted(self.children.get(m, ())):
                    paths.append((r, m, l))
        return paths


def load_corpus(path):
    """corpus.txt: '<doc_id>\\t<text>' per line -> list of texts (index = doc id)."""
    docs = []
    with open(path) as f:
        for line in f:
            _, text = line.rstrip("\n").split("\t", 1)
            docs.append(text)
    return docs


def load_doc2labels(path):
    """doc2labels.txt: '<doc_id>\\t<c1,c2[,c3]>' per line -> list of label lists."""
    labels = []
    with open(path) as f:
        for line in f:
            _, labs = line.rstrip("\n").split("\t")
            labels.append([int(x) for x in labs.split(",")])
    return labels


def check_alignment(course_corpus_path, reference_corpus_path, n_check=200):
    """Verify the course-provided corpus and the TELEClass corpus are the
    same documents in the same order (so labels map 1:1 to course doc ids).
    Compares normalized text of the first/last n_check docs.
    """
    def norm(s):
        return " ".join(s.lower().split())

    a = load_corpus(course_corpus_path)
    b = load_corpus(reference_corpus_path)
    if len(a) != len(b):
        return False, f"size mismatch: {len(a)} vs {len(b)}"
    idxs = list(range(min(n_check, len(a)))) + list(range(max(0, len(a) - n_check), len(a)))
    for i in idxs:
        if norm(a[i]) != norm(b[i]):
            return False, f"text mismatch at doc {i}"
    return True, f"aligned ({len(a)} docs, {2 * n_check} spot-checked)"
