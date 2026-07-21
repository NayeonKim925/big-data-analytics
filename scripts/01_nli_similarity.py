"""Phase 1a: compute & cache document-class similarity matrices.

This is the single expensive GPU job of the whole project. Everything
downstream (zero-shot baselines, core class mining, decoders, ablations)
reads the cache instead of touching the GPU again.

Run on the L4 box (survives SSH disconnects):
  mkdir -p logs
  nohup python scripts/01_nli_similarity.py \
      --data_dir reference_data/Amazon-531 --split test  > logs/nli_test.log 2>&1 &
  tail -f logs/nli_test.log
  # then the same with --split train (bigger; can run overnight)

Dry run without GPU/model (pipeline check only):
  python scripts/01_nli_similarity.py --data_dir ... --split test --mock --limit 500

Resume: just rerun the same command - finished shards are skipped.
Output: cache/sim_{split}.npz  (scores float16, sentinel -1 = not computed)
        cache/cand_{split}.json (per-doc candidate core classes, Sect. 3.2.1)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_corpus
from src.nli_topdown import SENTINEL, MockScorer, NLIScorer, topdown_shard


def run_split(split, args, taxo):
    d = Path(args.data_dir)
    texts = load_corpus(d / split / "corpus.txt")
    if args.limit:
        texts = texts[:args.limit]
    n, C = len(texts), len(taxo.names)

    shard_dir = Path(args.cache_dir) / f"shards_{split}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        scorer = MockScorer(taxo)
    else:
        scorer = NLIScorer(model_name=args.model, batch_size=args.batch_size,
                           max_length=args.max_length)
        scorer.class_names = taxo.names

    n_shards = (n + args.shard_size - 1) // args.shard_size
    t0 = time.time()
    for si in range(n_shards):
        out = shard_dir / f"shard_{si:04d}.npz"
        if out.exists():
            print(f"[{split}] shard {si + 1}/{n_shards} exists, skip", flush=True)
            continue
        lo, hi = si * args.shard_size, min((si + 1) * args.shard_size, n)
        S, cands = topdown_shard(texts[lo:hi], taxo, scorer)
        np.savez_compressed(out, scores=S.astype(np.float16), lo=lo, hi=hi,
                            cands=json.dumps(cands))
        done = min(hi, n)
        rate = done / max(1e-9, time.time() - t0)
        eta = (n - done) / max(1e-9, rate)
        print(f"[{split}] shard {si + 1}/{n_shards} done "
              f"(docs {lo}-{hi}, {rate:.1f} docs/s, ETA {eta / 3600:.2f}h)",
              flush=True)

    # ---- merge shards ----
    S = np.full((n, C), SENTINEL, dtype=np.float16)
    all_cands = [None] * n
    for f in sorted(shard_dir.glob("shard_*.npz")):
        z = np.load(f, allow_pickle=True)
        lo, hi = int(z["lo"]), int(z["hi"])
        S[lo:hi] = z["scores"]
        for k, c in enumerate(json.loads(str(z["cands"]))):
            all_cands[lo + k] = c
    cache = Path(args.cache_dir)
    np.savez_compressed(cache / f"sim_{split}.npz", scores=S)
    (cache / f"cand_{split}.json").write_text(json.dumps(all_cands))
    computed = float((S != SENTINEL).mean())
    print(f"[{split}] merged -> {cache}/sim_{split}.npz "
          f"({n}x{C}, {computed * 100:.1f}% entries computed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", choices=["train", "test", "both"], default="test")
    ap.add_argument("--model", default="roberta-large-mnli")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--shard_size", type=int, default=2048)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--mock", action="store_true",
                    help="fake scorer, no GPU/model - pipeline dry run only")
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N docs (for dry runs); 0 = all")
    args = ap.parse_args()

    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    splits = ["test", "train"] if args.split == "both" else [args.split]
    for s in splits:
        run_split(s, args, taxo)


if __name__ == "__main__":
    main()
