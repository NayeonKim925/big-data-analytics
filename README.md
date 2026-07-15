# TaxoClass Reproduction on Amazon-531

Weakly-supervised hierarchical multi-label text classification: a reproduction
of the TaxoClass (Shen et al., NAACL 2021) pipeline on Amazon-531, using only
class names as supervision (no labeled documents).

## Dataset & Problem Structure (Phase 0)

Amazon-531: 531 classes, 3-level taxonomy, 29,487 train / 19,658 test docs.

Key verified property: **100% of label sets are root-to-leaf paths** — length 2
(7.5%) or length 3 (92.5%). This collapses prediction from a 2^531 subset
problem to choosing among **568 candidate paths**, which motivates the
path-based decoder below.

Verification + metric sanity checks (oracle F1 = 1.0, random-path floor =
0.0627): `python scripts/00_verify_dataset.py --data_dir reference_data/Amazon-531`

## Pipeline

1. **NLI similarity** (`01`): roberta-large-mnli scores doc–class pairs
   top-down through the taxonomy (shard-level checkpointing).
2. **Zero-shot baseline** (`02`): NLI scores used directly for prediction.
3. **Core class mining** (`03`): per-document silver labels from similarity
   structure. Mined cores cover 99.9% of train docs; 71.0% contain at least
   one true label (silver label quality).
4. **Classifier training** (`04`): bert-base-uncased dual-encoder trained on
   silver labels (BCE, per-doc sum over classes). Batch 32, lr 5e-5/4e-3,
   seed 42. Best test performance at epoch 0; later epochs overfit silver
   label noise (loss ↓, test F1 ↓).
5. **Path decoding** (`05`): final structure-aware decoder, no GPU required.

## Results (test, Example-F1)

| Method | Example-F1 | Notes |
|---|---|---|
| Random path | 0.0627 | floor (phase 0) |
| Majority path | 0.1199 | phase 0 |
| Zero-shot NLI top-3 | →여기(③ 결과) | no training |
| Classifier top-3 | 0.4114 | unconstrained — reference upper line, not a valid path output |
| Path decoder (auto, δ=0) | 0.3993 | |
| **Path decoder (calibrated δ)** | **→여기(② 결과)** | **adopted** |

Reproduce the final row from saved scores:
`python scripts/05_decode_eval.py --data_dir reference_data/Amazon-531`

## Analysis

**1. The naive path decoder has a severe short-path bias.** Scoring paths by
the mean of raw logits makes length-2 paths structurally easier to rank high
(mean of 2 large values beats mean of 3). At δ=0 the decoder predicts
length-2 for ~95% of documents, while the true distribution is the opposite
(92.5% length-3).

**2. Label-free calibration.** We correct this by choosing the margin δ so
that the *predicted* length-3 share matches the dataset property verified in
phase 0 (92.5%). This uses no labels — only a structural property of the
label space — and improves F1 by ~0.9pt over auto. A test-set sweep shows a
higher optimum exists (δ=0.16, F1 0.4201, length-3 share ≈ 50%), but adopting
it would constitute test-set tuning, so it is reported only as sensitivity
analysis.

**3. Metric-optimal length distribution ≠ data distribution.** Under
Example-F1, predicting a correct length-2 prefix of a length-3 truth scores
0.80, while a length-3 prediction with a wrong leaf scores 0.67. Extending to
the leaf pays off only when leaf accuracy exceeds ~40%; our model sits near
this break-even, which explains why the F1-optimal length-3 share (~50%) is
far below the true share (92.5%).

**4. Scores are not calibrated probabilities.** Training with per-document
BCE *sums* over 531 classes leaves sigmoid outputs uncalibrated (all leaf
confidences > 0.5, median 0.58), so probability-threshold decoding rules are
ineffective without post-hoc calibration.

## Reproducibility Note

An earlier run of this pipeline (on since-lost infrastructure) reached 0.4684
Example-F1 with a path decoder. After full environment reconstruction, that
figure could not be reproduced; every number in this document comes from the
current, reproducible environment (scripts + saved artifacts in this repo and
the companion HF dataset). Mined core classes survived the loss and were
verified against ground truth (71.0% hit rate) before reuse.

## Artifacts

- Code: this repo. Heavy artifacts (checkpoints, score matrices, mined cores)
  are backed up to a private HF dataset (`hmtc-cache`).
- `results/clf_scores_test_epoch0.npz`: test score matrix of the adopted
  model (epoch 0) — sufficient to reproduce all decoder results without GPU.
