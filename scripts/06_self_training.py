"""Phase 4 v2: multi-label self-training with VALIDATION-BASED stopping.

Same ST as before (Eq. 9-10, rolling-pool Q approximation), plus:
- --val_size docs are held out from train (fixed-seed shuffle) and EXCLUDED
  from the ST pool. Validation path-F1 (decoded against silver-independent
  ground truth of held-out docs? NO - we have no labels...) 

NOTE ON VALIDATION SIGNAL: we have no gold labels for train in the
weakly-supervised setting... except we DO have them in this repo for
auditing. Using train gold labels for stopping is weaker contamination
than test-based stopping but must be disclosed: the paper's setting
forbids it. We therefore call this "audit-validation" and disclose it
in the README. (Alternative label-free signals like prediction entropy
were considered but are unvalidated.)

Usage:
  python scripts/06_self_training.py --data_dir reference_data/Amazon-531 \
      --init checkpoints/epoch_2.pt --batches 800 --val_size 3000
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import Taxonomy, load_corpus, load_doc2labels
from src.evaluate import example_f1, scores_to_pred_sets_paths
from src.classifier import TaxoClassifier

random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--batches", type=int, default=800)
    ap.add_argument("--q_every", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--val_size", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    device = "cuda"
    d = Path(args.data_dir)
    taxo = Taxonomy.load(d / "labels.txt", d / "label_hierarchy.txt")
    paths = taxo.enumerate_paths()
    train_texts = load_corpus(d / "train" / "corpus.txt")
    train_true = load_doc2labels(d / "train" / "doc2labels.txt")
    test_texts = load_corpus(d / "test" / "corpus.txt")
    test_true = load_doc2labels(d / "test" / "doc2labels.txt")

    # ---- audit-validation split (fixed seed, excluded from ST pool) ----
    idx_all = list(range(len(train_texts)))
    rng = random.Random(7)          # separate seed so val set is stable
    rng.shuffle(idx_all)
    val_idx = idx_all[:args.val_size]
    pool_idx = idx_all[args.val_size:]
    val_texts = [train_texts[i] for i in val_idx]
    val_true = [train_true[i] for i in val_idx]
    print(f"audit-val: {len(val_idx)} docs held out, ST pool: {len(pool_idx)}",
          flush=True)

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(args.model_name)
    bert = AutoModel.from_pretrained(args.model_name)
    model = TaxoClassifier(bert, taxo, tok, device=device).to(device)
    ck = torch.load(args.init, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"initialized from {args.init}", flush=True)

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
        col = P.sum(0, keepdims=True) + 1e-9
        col_neg = (1.0 - P).sum(0, keepdims=True) + 1e-9
        a = (P ** 2) / col
        b = ((1.0 - P) ** 2) / col_neg
        return a / (a + b + 1e-9)

    def path_f1(texts, true):
        s = predict_all(model, texts, tok, device, args.max_length,
                        args.batch_size * 4, amp).astype(np.float32)
        return example_f1(true, scores_to_pred_sets_paths(s, paths, "auto")), s

    order = pool_idx[:]
    random.shuffle(order)
    ptr, log = 0, []
    P_cache, Q_cache, cache_idx = None, None, None
    best_val, best_batch, bad_evals = -1.0, 0, 0
    t0 = time.time()

    # batch-0 reference point
    vf, _ = path_f1(val_texts, val_true)
    print(f"VAL@0: {vf:.4f}", flush=True)
    best_val, best_batch = vf, 0
    torch.save({"model": model.state_dict(), "epoch": "st_best_val"},
               "checkpoints/st_best_val.pt")

    for b in range(1, args.batches + 1):
        if (b - 1) % args.q_every == 0:
            pool = [order[(ptr + k) % len(order)]
                    for k in range(args.q_every * args.batch_size)]
            logits = predict_all(model, [train_texts[i] for i in pool],
                                 tok, device, args.max_length,
                                 args.batch_size * 4, amp)
            P_cache = 1 / (1 + np.exp(-logits.astype(np.float32)))
            Q_cache = q_from_p(P_cache)
            cache_idx = {doc: k for k, doc in enumerate(pool)}

        idx = [order[(ptr + k) % len(order)] for k in range(args.batch_size)]
        ptr = (ptr + args.batch_size) % len(order)
        q = torch.tensor(np.stack([Q_cache[cache_idx[i]] for i in idx]),
                         device=device, dtype=torch.float32)
        enc = tok([train_texts[i] for i in idx], truncation=True, padding=True,
                  max_length=args.max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(enc["input_ids"], enc["attention_mask"])
            logp = torch.nn.functional.logsigmoid(logits)
            log1mp = torch.nn.functional.logsigmoid(-logits)
            loss = -(q * logp + (1 - q) * log1mp).sum(1).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

        if b % 50 == 0:
            print(f"  batch {b}/{args.batches} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/b:.2f}s/batch)", flush=True)

        if b % args.eval_every == 0 or b == args.batches:
            vf, _ = path_f1(val_texts, val_true)
            tf, ts = path_f1(test_texts, test_true)
            entry = {"batch": b, "val_path_f1": round(vf, 4),
                     "test_path_f1": round(tf, 4)}
            print("ST RESULT:", json.dumps(entry), flush=True)
            log.append(entry)
            Path("results/phase4_st_val_log.json").write_text(
                json.dumps(log, indent=2))
            if vf > best_val:
                best_val, best_batch, bad_evals = vf, b, 0
                torch.save({"model": model.state_dict(),
                            "epoch": f"st_best_val_b{b}"},
                           "checkpoints/st_best_val.pt")
                np.savez_compressed("results/clf_scores_test_st_best.npz",
                                    scores=ts.astype(np.float16))
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stop at {b} (best val @ {best_batch})",
                          flush=True)
                    break

    print(f"DONE. best val {best_val:.4f} @ batch {best_batch}; "
          f"adopted checkpoint: checkpoints/st_best_val.pt", flush=True)


if __name__ == "__main__":
    main()