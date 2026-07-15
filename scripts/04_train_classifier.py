"""Phase 3: core-class-guided classifier training (paper Sect. 3.3.2).

Trains the dual-encoder classifier on the mined silver labels:
  positives  C+ = cores ∪ parents(cores)                       (Eq. 7)
  negatives  C- = all - C+ - children(cores)
  docs with empty core sets are excluded from the loss          (Eq. 8, BCE)
Learning rates follow the paper: 5e-5 (BERT) / 4e-3 (GNN + B), AdamW,
batch 64 (reduce with --batch_size on smaller GPUs), seeds fixed to 42.

Each epoch: full test-set inference -> all metrics incl. path decoding,
appended to results/phase3_log.json; checkpoint saved to checkpoints/
(keep this folder on Drive - survives Colab session death). Resume with
--resume checkpoints/epoch_N.pt.

Reference targets (paper Table 2, Amazon-531):
  TaxoClass-NoST (this stage, before self-training)  0.5431 Example-F1
  TaxoClass-NoGNN                                    0.5271
Final scores matrix is saved to results/clf_scores_test.npz for later
analysis (case studies, decoder ablations) without re-running inference.

Colab: !python scripts/04_train_classifier.py --data_dir reference_data/Amazon-531
Debug (CPU, tiny random model, ~1 min): add --tiny_debug --limit 200
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_corpus, load_doc2labels
from src.evaluate import (evaluate_all, example_f1,
                          scores_to_pred_sets_paths, scores_to_pred_sets_topk)
from src.classifier import TaxoClassifier

random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def make_targets(cores, taxo):
    """Per-doc (positive multi-hot, loss mask) per Eq. 7. Mask zeroes out
    children of cores (neither positive nor negative)."""
    C = len(taxo.names)
    keep, Y, M = [], [], []
    for i, cs in enumerate(cores):
        if not cs:
            continue
        pos = set(cs)
        excl = set()
        for c in cs:
            pos.update(taxo.parents.get(c, ()))
            excl.update(taxo.children.get(c, ()))
        y = np.zeros(C, dtype=np.float32)
        y[list(pos)] = 1.0
        m = np.ones(C, dtype=np.float32)
        m[list(excl - pos)] = 0.0
        keep.append(i); Y.append(y); M.append(m)
    return keep, np.stack(Y), np.stack(M)


@torch.no_grad()
def predict_all(model, texts, tok, device, max_length, batch_size, amp):
    model.eval()
    out = np.zeros((len(texts), model.node_feat.shape[0]), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(enc["input_ids"], enc["attention_mask"])
        out[s:s + len(enc["input_ids"])] = torch.sigmoid(logits.float()).cpu().numpy()
    model.train()
    return out


def evaluate_epoch(scores, true, paths):
    r = evaluate_all(true, score_matrix=scores)
    return {
        "p@1": r["p@1"], "p@3": r["p@3"], "mrr_filtered": r["mrr_filtered"],
        "f1_top3": round(example_f1(true, scores_to_pred_sets_topk(scores, 3)), 4),
        "f1_path_auto": round(example_f1(
            true, scores_to_pred_sets_paths(scores, paths)), 4),
        "f1_path_len3": round(example_f1(
            true, scores_to_pred_sets_paths(scores, paths, length_rule=3)), 4),
        "f1_thresh05": round(example_f1(
            true, [np.where(row > 0.5)[0].tolist() for row in scores]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cores", default="results/core_classes_train.json")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--lr_bert", type=float, default=5e-5)
    ap.add_argument("--lr_head", type=float, default=4e-3)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiny_debug", action="store_true",
                    help="random tiny BERT, no download - pipeline check only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    paths = taxo.enumerate_paths()

    train_texts = load_corpus(d / "train" / "corpus.txt")
    test_texts = load_corpus(d / "test" / "corpus.txt")
    test_true = load_doc2labels(d / "test" / "doc2labels.txt")
    cores = json.loads(Path(args.cores).read_text())
    if args.limit:
        train_texts, cores = train_texts[:args.limit], cores[:args.limit]
        test_texts, test_true = test_texts[:args.limit], test_true[:args.limit]

    from transformers import AutoTokenizer, AutoModel, AutoConfig
    if args.tiny_debug:
        # fully offline: build a small WordPiece vocab from the corpus itself
        from collections import Counter
        from transformers import BertTokenizerFast, BertConfig, BertModel
        cnt = Counter(w for t in train_texts for w in t.split())
        name_words = [w for c in taxo.names.values() for w in c.split()]
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + \
                sorted(set(name_words)) + \
                [w for w, _ in cnt.most_common(8000) if w not in set(name_words)]
        vp = Path("results/_debug_vocab.txt")
        vp.parent.mkdir(exist_ok=True)
        vp.write_text("\n".join(vocab))
        tok = BertTokenizerFast(vocab_file=str(vp), do_lower_case=True)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=64,
                         num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128)
        bert = BertModel(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(args.model_name)
        bert = AutoModel.from_pretrained(args.model_name)

    model = TaxoClassifier(bert, taxo, tok, device=device).to(device)
    opt = torch.optim.AdamW([
        {"params": model.bert_parameters(), "lr": args.lr_bert},
        {"params": model.head_parameters(), "lr": args.lr_head},
    ])
    scaler = torch.amp.GradScaler(enabled=amp)

    keep, Y, M = make_targets(cores, taxo)
    print(f"training docs with cores: {len(keep)}/{len(cores)}", flush=True)
    Y, M = torch.tensor(Y), torch.tensor(M)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {args.resume} (next epoch {start_epoch})", flush=True)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(exist_ok=True)
    log_path = Path("results/phase3_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else []

    bce = nn.BCEWithLogitsLoss(reduction="none")
    order = list(range(len(keep)))
    for epoch in range(start_epoch, args.epochs):
        random.shuffle(order)
        t0, tot, nb = time.time(), 0.0, 0
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            texts = [train_texts[keep[i]] for i in idx]
            enc = tok(texts, truncation=True, padding=True,
                      max_length=args.max_length, return_tensors="pt").to(device)
            y, m = Y[idx].to(device), M[idx].to(device)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(enc["input_ids"], enc["attention_mask"])
                # Eq. 8: per-document SUM over classes (not mean) - with only
                # ~5 positives among 531 classes, mean-normalization dilutes
                # the positive gradient ~100x and the model collapses to
                # predicting all-zeros. Paper LRs assume the sum form.
                loss = (bce(logits, y) * m).sum(dim=1).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"  epoch {epoch} step {nb} loss {tot / nb:.4f} "
                      f"({(time.time() - t0) / nb:.2f}s/step)", flush=True)

        scores = predict_all(model, test_texts, tok, device,
                             args.max_length, args.batch_size * 4, amp)
        metrics = evaluate_epoch(scores, test_true, paths)
        # diagnostics: (a) can the model FIT its own silver labels? (b) collapse?
        fit_n = min(2000, len(keep))
        fit_scores = predict_all(model, [train_texts[keep[i]] for i in range(fit_n)],
                                 tok, device, args.max_length,
                                 args.batch_size * 4, amp)
        pos_sets = [set(np.where(Y[i].numpy() > 0)[0]) for i in range(fit_n)]
        metrics["train_fit_p@1"] = round(float(np.mean(
            [fit_scores[i].argmax() in pos_sets[i] for i in range(fit_n)])), 4)
        from collections import Counter
        top1 = Counter(scores.argmax(1).tolist())
        metrics["test_top1_share"] = round(top1.most_common(1)[0][1] / len(scores), 4)
        entry = {"epoch": epoch, "train_loss": round(tot / max(nb, 1), 4),
                 "minutes": round((time.time() - t0) / 60, 1), **metrics}
        print("EPOCH RESULT:", json.dumps(entry), flush=True)
        log.append(entry)
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, ckpt_dir / f"epoch_{epoch}.pt")
        np.savez_compressed("results/clf_scores_test.npz",
                            scores=scores.astype(np.float16))
    print("done. per-epoch log -> results/phase3_log.json", flush=True)


if __name__ == "__main__":
    main()
