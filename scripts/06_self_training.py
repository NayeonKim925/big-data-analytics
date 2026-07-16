"""Phase 4: multi-label self-training (paper Sect. 3.4, Eq. 9-10).

Starts from a Phase-3 checkpoint and refines it on the ENTIRE unlabeled
train corpus (no silver labels used here). Every 25 batches the target
distribution Q is recomputed from current predictions P (Eq. 10, per-class
normalized sharpening) and the model is trained with KL(Q||P) (Eq. 9).
Paper LRs for this stage: 1e-6 (BERT) / 5e-4 (head). Batch order is over
the full corpus; Q is computed on the fly for each batch from cached P
statistics (per-class sums), refreshed every --q_every batches.

Usage:
  python scripts/06_self_training.py --data_dir reference_data/Amazon-531 \
      --init checkpoints/epoch_0.pt --batches 800
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_corpus, load_doc2labels
from src.evaluate import evaluate_all, scores_to_pred_sets_paths, scores_to_pred_sets_topk
from src.classifier import TaxoClassifier

random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--init", required=True, help="phase-3 checkpoint")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--batches", type=int, default=800,
                    help="total ST batches B (Alg. 1)")
    ap.add_argument("--q_every", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=200)
    args = ap.parse_args()

    device = "cuda"
    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    paths = taxo.enumerate_paths()
    train_texts = load_corpus(d / "train" / "corpus.txt")
    test_texts = load_corpus(d / "test" / "corpus.txt")
    test_true = load_doc2labels(d / "test" / "doc2labels.txt")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(args.model_name)
    bert = AutoModel.from_pretrained(args.model_name)
    model = TaxoClassifier(bert, taxo, tok, device=device).to(device)
    ck = torch.load(args.init, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"initialized from {args.init} (epoch {ck['epoch']})", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.bert_parameters(), "lr": 1e-6},
        {"params": model.head_parameters(), "lr": 5e-4},
    ])
    amp = True
    scaler = torch.amp.GradScaler(enabled=amp)

    sys.path.insert(0, str(Path(__file__).parent))
    import importlib
    m04 = importlib.import_module("04_train_classifier")
    predict_all = m04.predict_all

    def q_from_p(P):
        """Eq. 10: per-class normalized sharpening, rows renormalized."""
        col = P.sum(0, keepdims=True) + 1e-9          # sum_i p_ij
        col_neg = (1.0 - P).sum(0, keepdims=True) + 1e-9
        a = (P ** 2) / col
        b = ((1.0 - P) ** 2) / col_neg
        return a / (a + b + 1e-9)

    order = list(range(len(train_texts)))
    random.shuffle(order)
    ptr, log = 0, []
    P_cache, Q_cache, cache_idx = None, None, None
    t0 = time.time()

    for b in range(1, args.batches + 1):
        if (b - 1) % args.q_every == 0:
            # refresh P/Q on a pool = next q_every batches' docs
            pool = [order[(ptr + k) % len(order)]
                    for k in range(args.q_every * args.batch_size)]
            with torch.no_grad():
                logits = predict_all(model, [train_texts[i] for i in pool],
                                     tok, device, args.max_length,
                                     args.batch_size * 4, amp)
            P_cache = 1 / (1 + np.exp(-logits.astype(np.float32)))
            Q_cache = q_from_p(P_cache)
            cache_idx = {doc: k for k, doc in enumerate(pool)}

        idx = [order[(ptr + k) % len(order)] for k in range(args.batch_size)]
        ptr = (ptr + args.batch_size) % len(order)
        texts = [train_texts[i] for i in idx]
        q = torch.tensor(np.stack([Q_cache[cache_idx[i]] for i in idx]),
                         device=device, dtype=torch.float32)
        enc = tok(texts, truncation=True, padding=True,
                  max_length=args.max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(enc["input_ids"], enc["attention_mask"])
            logp = torch.nn.functional.logsigmoid(logits)
            log1mp = torch.nn.functional.logsigmoid(-logits)
            # KL(Q||P) for per-class Bernoulli, dropping const entropy(Q):
            loss = -(q * logp + (1 - q) * log1mp).sum(1).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

        if b % 50 == 0:
            print(f"  batch {b}/{args.batches} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/b:.2f}s/batch)", flush=True)
        if b % args.eval_every == 0 or b == args.batches:
            scores = predict_all(model, test_texts, tok, device,
                                 args.max_length, args.batch_size * 4, amp)
            f_auto = m04.evaluate_epoch(scores, test_true, paths) \
                if hasattr(m04, "evaluate_epoch") else None
            entry = {"batch": b, **(f_auto or {})}
            print("ST RESULT:", json.dumps(entry), flush=True)
            log.append(entry)
            Path("results/phase4_st_log.json").write_text(json.dumps(log, indent=2))
            torch.save({"model": model.state_dict(), "epoch": f"st_{b}"},
                       f"checkpoints/st_{b}.pt")
            np.savez_compressed("results/clf_scores_test_st.npz",
                                scores=scores.astype(np.float16))


if __name__ == "__main__":
    main()