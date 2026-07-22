# Technical Report — Weakly-Supervised Hierarchical Multi-Label Text Classification

이 문서는 [README.md](../README.md)에서 요약된 내용의 상세 버전입니다. 문제 정의, 전체 파이프라인, 발견한 구현 버그, 정량 분석, 그리고 한계를 담고 있습니다.

---

## 1. Problem — 라벨 0개, 클래스 531개

| 항목 | 내용 |
|---|---|
| 데이터 | Amazon-531 상품 리뷰 — train 29,487 / test 19,658 문서 |
| 클래스 | 531개, 3-level taxonomy (tree) |
| 사용 가능한 supervision | 클래스 이름 문자열 + 트리 구조뿐 — 문서–라벨 쌍 0개 |
| Task | 문서마다 정답 카테고리 집합(멀티라벨) 예측 |

일반적인 지도학습은 수만 개의 라벨링된 예시를 전제한다. 이 세팅에는 그것이 없다. 새 taxonomy는 도입됐지만 라벨링 예산이 없는 실무 상황과 동형이다.

**Phase 0 — 데이터 구조 검증.** 모델링에 앞서 정답 label set의 구조를 전수 검증했다. 결과: 모든 label set이 root-to-leaf 경로다. 길이 2가 7.5%, 길이 3이 92.5%, 예외 0건. 따라서 예측 문제는 2^531가지 부분집합 선택이 아니라 **568개 candidate path 중 하나를 고르는 문제**로 축소된다. 이 검증된 성질이 §2.5 path 디코더의 근거다.

> **Research Question**: 클래스 이름과 taxonomy 구조만으로 학습한 분류기가, 학습 없이 쓰는 zero-shot 추론을 상회할 수 있는가? 상회하지 못한다면 병목은 어느 단계인가?

---

## 2. Pipeline

1. **NLI similarity** (`01`) — `roberta-large-mnli`, "this product is about {}." 템플릿, taxonomy top-down 탐색 (문서당 평균 68.5 클래스만 계산, shard 체크포인팅으로 계산량 절감).
2. **Zero-shot baseline** (`02`) — NLI 점수를 그대로 path 예측에 사용.
3. **Core class mining** (`03`) — `conf(D,c) = sim(D,c) − max(부모/형제 sim)`로 상대 신뢰도를 정의하고, 클래스별 median + min_conf 이중 필터를 적용해 silver label을 채굴.
4. **Classifier** (`04`) — BERT `[CLS]` + GCN class encoder(ego-mean readout) + bilinear matching. positive = core classes ∪ 부모, 자식은 negative에서 제외. batch 32, len 512, seed 42.
5. **Path decoding** (`05`) — 저장된 점수 행렬만으로 GPU 없이 전체 결과 재현 가능.
6. **Self-training** (`06`) — pseudo-label pool `Q`를 rolling pool로 근사 (§5 한계 참조).
7. **LLM refinement** (`07`) — 결정 경계(uncertainty 최소)에 걸린 문서를 GPT-4o-mini로 재판정.

---

## 3. 최종 결과 (test, Example-F1)

| # | 단계 | Silver Precision | Example-F1 | Δ |
|---|---|---|---|---|
| 0 | random / majority path | — | 0.063 / 0.120 | — |
| 1 | zero-shot NLI + path decoder | — | 0.4623 | — |
| 2 | + classifier (초기: silver 30%, len 256) | 30.3% | 0.3995 | −6.3 |
| 3 | + silver 정제 (min_conf 0.15) + len 512 | 43.2% | 0.4614 | +6.2 |
| 4 | + LLM refine 1K calls (커버리지 3.2%) | 43.8% | 0.4571 | ±0 |
| 5 | + LLM refine 10K calls (누적, scaling 실험) | 48.8% | **0.5180** | +5.7 |
| 6 | + self-training (행 3 저품질 silver 기준, 고정 800 steps) | — | 0.4729 | +1.2 |
| 7 | + self-training (행 5 고품질 silver 기준 + audit-validation stopping) | — | **0.5488** | +3.1 |

참조: 논문 Hier-0Shot-TC 0.4742 / TaxoClass-NoST 0.5431 / TaxoClass-full 0.5934.

**정직성 공개**: 행 7(최종 0.5488)의 stopping 신호는 train에서 분리한 gold label 3,000개(audit-validation)를 사용하므로 순수 weak-supervision 세팅을 벗어난다. 이 사실은 README와 본 문서에 명시한다. **논문 세팅을 그대로 준수한 최고치는 행 5 (0.5180)** — zero-shot 대비 +5.6pt, 논문 TaxoClass-NoST(0.5431)와 2.5pt 차이다.

---

## 4. 핵심 분석

**4.1 병목은 모델이 아니라 silver label이었다.**
초기 classifier가 zero-shot보다 6.3pt 낮은 원인을 추적하는 과정에서 batch 크기(효과 <1pt), 디코딩 방식, 학습 길이를 순차적으로 통제 실험해 배제했다. 이후 silver label 품질 감사에서 진짜 원인을 발견했다. "문서당 정답 1개 이상 포함" 지표는 71%로 양호해 보였지만, 문서당 평균 3.8개 라벨을 채굴하며 그중 2.7개가 오답인 과잉 생성을 가리고 있었다 — 라벨 단위 precision은 실제로 30.3%였다. 지표 선택이 데이터 문제를 은폐한 사례다.

**4.2 Precision–coverage trade-off 측정과 선택.**
min_conf 스윕 결과 (0 → 0.3에서 precision 30% → 51%, coverage 100% → 45%), 균형점 0.15(precision 43%, coverage 79%)를 채택했다. 여기에 document truncation 설정 오류(256 → 512 토큰) 수정을 더해 F1 +6.2pt, 그리고 학습 중 관찰되던 과적합 패턴(train loss 하락 + test F1 하락)이 사라졌다.

**4.3 LLM 예산의 수확 체감/스케일 곡선.**
결정 경계(uncertainty 최소) 문서에만 API 예산을 집중했다. 1K calls → 코퍼스 커버리지 3.2%, precision +0.7pt, **F1 변화 없음** (재검수 규모가 전체 노이즈 대비 너무 작았기 때문). 10K calls로 확장 → precision 48.8%(+5.6pt), F1 0.5180(+5.7pt). silver precision과 최종 F1이 거의 1:1로 동행함을 확인했고, 이는 "병목은 데이터 품질"이라는 진단의 정량적 확증이 되었다.

---

## 5. Limitations

- **행 7의 audit-validation stopping은 순수 weak supervision을 벗어난다** — train gold label 3,000개를 사용한다. entropy 등 label-free stopping 신호로 대체하는 것은 future work.
- **Self-training의 pseudo-label pool `Q`를 rolling pool로 근사**했다 — 논문의 전체 재계산 방식과 다르며, TaxoClass-full(0.5934)에 도달하지 못한 원인 후보 중 하나다.
- **silver precision 49%에서 중단** — LLM 예산 곡선의 상단(10K calls 이상)과, refine과 mining threshold의 상호작용은 미탐색이다.
- **path 디코더는 Amazon-531의 검증된 성질(전 정답 = root-to-leaf 경로)에 의존**한다. DAG이거나 비경로 라벨이 존재하는 taxonomy에는 그대로 이식할 수 없고, 해당 성질의 사전 검증이 선행되어야 한다.
- **LLM refinement의 판정 품질 자체는 감사하지 않았다** — GPT-4o-mini의 재판정을 silver에 병합했을 때의 precision 증분으로 간접 확인했을 뿐, 판정 단위 정확도는 별도 측정이 필요하다.

---

## 6. Repository Structure

```
weakly-supervised-hmtc/
├── README.md
├── docs/
│   └── technical_report.md          # 본 문서
├── scripts/
│   ├── 00_verify_dataset.py         # Phase 0: 데이터 검증 (path property 전수 확인)
│   ├── 01_nli_similarity.py         # NLI similarity 계산 (shard 체크포인팅) 
│   ├── 02_zeroshot_baseline.py      # 행 1: zero-shot baseline
│   ├── 03_core_class_mining.py      # core class mining (silver label 채굴) 
│   ├── 04_train_classifier.py       # classifier 학습 
│   ├── 05_decode_eval.py            # path 디코딩 + 평가 (캐시만으로 GPU 없이 재현)
│   ├── 06_self_training.py          # self-training (--val_size 옵션으로 행 6/7 전환)
│   └── 07_llm_refinement.py         # LLM refinement (GPT-4o-mini) 
├── src/                             # 공용 모듈
├── cache/                           # NLI 점수 행렬 (.npz, shard 단위)
├── results/                         # 실험 결과 JSON
├── checkpoints/                     # 모델 checkpoint (st_best_val.pt 등)
└── reference_data/Amazon-531/       # TELEClass 공개 데이터 (gold label 포함)
```


---

## 8. Reproduction

```bash
# Phase 0: 데이터 검증
python scripts/00_verify_dataset.py --data_dir reference_data/Amazon-531

# 행 1: zero-shot baseline
python scripts/02_zeroshot_baseline.py --data_dir reference_data/Amazon-531

# 디코더/평가 (캐시 기반, GPU 불필요)
python scripts/05_decode_eval.py --data_dir reference_data/Amazon-531

# 행 2~7: scripts/01 → 03 → 04 (→ 06, 07)
# 각 스크립트의 커맨드 옵션은 docstring 참조
```

