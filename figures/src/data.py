# All numbers sourced from the repo (results/*.json, logs/*.log, README.md).
# Provenance noted per field so captions can be defended in an interview.

# --- Page 2: Example-F1 journey (README §4 table; test Example-F1, path_auto) ---
JOURNEY = [
    # label, F1, role, silver_prec, delta_pt, note
    dict(key="zeroshot", label="Zero-shot\n(선생 NLI)",      f1=0.4623, role="base",  prec=None,  delta=None,  sub="path decoder"),
    dict(key="clf",      label="Classifier\n(초기)",          f1=0.3995, role="drop",  prec=30.3,  delta=-6.3,  sub="silver 30%"),
    dict(key="refine",   label="Data refine\n(min_conf·512)",  f1=0.4614, role="data",  prec=43.2,  delta=+6.2,  sub="silver 43%"),
    dict(key="llm",      label="LLM review\n10K calls",        f1=0.5180, role="data",  prec=48.8,  delta=+5.7,  sub="silver 49%"),
    dict(key="final",    label="Self-training\n+ audit-val",   f1=0.5488, role="final", prec=None,  delta=+3.1,  sub="채택"),
]
NOST = 0.5431          # 논문 TaxoClass-NoST (README §4 참조)
ZS_PAPER = 0.4742      # Hier-0Shot-TC
FULL_PAPER = 0.5934    # TaxoClass full (완전지도)
PURE_WS_BEST = 0.5180  # 논문 세팅 준수 최고치 (row 5)

# --- Page 3: metric reversal (README §3, §5.1; phase2_sweep min_conf=0) ---
HITRATE_DOC = 71.0     # "문서당 정답 1개 이상 포함" (초기 지표)
PREC_LABEL = 30.3      # 라벨 단위 precision (phase2_sweep min_conf 0: 30.32)
AVG_LABELS = 3.8       # avg_cores_per_doc @ min_conf 0
WRONG_LABELS = 2.7     # 3.8 * (1-0.303) ≈ 2.65 → 2.7 (README §5.1)
CORRECT_LABELS = round(AVG_LABELS * PREC_LABEL/100, 1)  # ≈1.1

# --- Page 3: Precision–Coverage sweep (results/phase2_sweep.json) ---
SWEEP = [
    dict(min_conf=0.00, cov=99.88, prec=30.32, avg=3.80),
    dict(min_conf=0.05, cov=95.77, prec=35.86, avg=2.67),
    dict(min_conf=0.10, cov=88.24, prec=39.78, avg=1.95),
    dict(min_conf=0.15, cov=78.60, prec=43.16, avg=1.45),  # selected knee
    dict(min_conf=0.20, cov=67.65, prec=46.31, avg=1.07),
    dict(min_conf=0.30, cov=45.13, prec=51.22, avg=0.57),
]
SWEEP_KNEE = 0.15

# --- Page 3: Hypothesis-1 control (batch size, <1pt) results/phase3_log_b16/b32 ---
BATCH_CTRL = dict(b16=0.4063, b32=0.3993)  # best f1_path_auto per run (≈ equal)

# --- Page 3: overfitting signature (initial config, results/phase3_log_b32) ---
# train_loss falls while test F1 stays flat (~0.40): memorising noisy silver.
OVERFIT_B32 = [
    dict(ep=0, loss=17.79, f1=0.3993, fit=0.85),
    dict(ep=1, loss=10.98, f1=0.3861, fit=0.88),
    dict(ep=2, loss=9.71,  f1=0.3973, fit=0.874),
]
# after the fix (len 512), test F1 rises then stabilises (results/phase3_log_512e3)
FIXED_512 = [
    dict(ep=0, loss=13.83, f1=0.4189, fit=0.74),
    dict(ep=1, loss=6.63,  f1=0.4474, fit=0.784),
    dict(ep=2, loss=5.64,  f1=0.4614, fit=0.8385),
]

# --- Page 4 귀속①: precision & F1 co-movement (README §4/§7) ---
COMOVE = [
    dict(stage="초기",       prec=30.3, f1=0.3995),
    dict(stage="정제 후",     prec=43.2, f1=0.4614),
    dict(stage="LLM 10K 후",  prec=48.8, f1=0.5180),
]

# --- Page 4 귀속②: LLM budget scaling, 1K vs 10K (README §5.3) ---
LLM_SCALE = dict(
    k1  = dict(calls="1K",  cov=3.2,  d_prec=+0.6, d_f1=0.0,  f1=0.4571, note="변화 없음"),
    k10 = dict(calls="10K", cov=31.0, d_prec=+5.6, d_f1=+5.7, f1=0.5180, note="+5.7pt"),
)

# --- Page 4 귀속③: self-training stopping (results/phase4_st_log*, st_val.log) ---
# Same row-5 base + same algorithm; only the stopping rule differs.
ST_FIXED = [   # run to completion (results/phase4_st_log.json) — test path F1
    dict(batch=0,   test=0.5180),   # row-5 base (epoch-2 checkpoint)
    dict(batch=200, test=0.5492),
    dict(batch=400, test=0.5245),
    dict(batch=600, test=0.5125),
    dict(batch=800, test=0.5002),
]
ST_VAL = [     # audit-validation run (results/phase4_st_val_log.json)
    dict(batch=0,   val=0.5233, test=0.5180),
    dict(batch=100, val=0.5357, test=0.5338),
    dict(batch=200, val=0.5447, test=0.5488),   # best val -> adopted
    dict(batch=300, val=0.5260, test=0.5313),
    dict(batch=400, val=0.5175, test=0.5204),   # early stop (patience 2)
]
ST_ADOPTED = 0.5488
ST_TEST_PEAK = 0.5492      # not adopted (would be test-set tuning) — README §5.4
ST_FIXED_END = 0.5002
ST_STOP_BATCH = 200
# Official table deltas (README §4): row6 fixed(row3 base) +1.2 -> 0.4729 ; row7 audit-val(row5) +3.1 -> 0.5488
ROW6_FIXED = dict(delta=+1.2, f1=0.4729, base="row3")
ROW7_VAL   = dict(delta=+3.1, f1=0.5488, base="row5")

# --- Phase 0 structure (results/phase0.json) ---
N_TRAIN, N_TEST = 29487, 19658
N_CLASSES, N_PATHS = 531, 568
LEVELS = dict(root=6, mid=64, leaf=461)

# --- Zero-shot decoder comparison (results/phase1_zeroshot.json) ---
DECODERS = dict(top3=0.2728, hier_greedy=0.3753, path_len3=0.4680, path_auto=0.4623)
RANDOM_PATH, MAJORITY_PATH = 0.0627, 0.1199
AVG_CLASSES_PER_DOC = 68.5   # top-down traversal (vs 531 exhaustive)

# --- Level-wise greedy accuracy (results/phase1_zeroshot.json) — error deepens ---
LEVEL_ACC = dict(root=59.16, mid=32.11, leaf=20.67)

# --- Core size histogram @ min_conf 0.15 (results/phase2_mining.json) ---
CORE_SIZE_HIST = {0:6309,1:9938,2:8326,3:3757,4:957,5:173,6:25,7:2}

# --- Reproduction notes (README §6): paper-vs-implementation issues ---
REPRO = [
  dict(t="Eq.8 loss 정규화", sym="mean → sum",
       find="consistency를 mean 정규화하면 positive gradient가 ~100배 희석 → 전 클래스 점수 균일화(collapse)",
       fix="per-document sum BCE"),
  dict(t="클래스 표현 anisotropy", sym="near-collinear",
       find="BERT 이름 임베딩으로 초기화한 클래스 feature가 near-collinear → bilinear가 클래스 구분 불가",
       fix="centering·normalize + learnable + GCN residual"),
  dict(t="Sect.3.2.2 positivity", sym="prec 18% → 30%",
       find="median 필터만 구현하면 silver precision 18%에 그침 — 논문의 두 조건이 모두 필요",
       fix="이중 조건 구현"),
  dict(t="MRR ceiling", sym="이론상한 0.621",
       find="Table 2 MRR을 문자 그대로 계산하면 이론 최대치 0.621 < 논문 보고 0.6332 → filtered-rank 추정",
       fix="filtered/unfiltered 병기"),
  dict(t="Eq.6 표기", sym="sigmoid(exp())",
       find="문자 그대로면 sigmoid(exp(·))∈(0.5,1)로 BCE가 ill-posed",
       fix="표준 bilinear-logit sigmoid로 구현"),
]
