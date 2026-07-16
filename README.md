# TaxoClass 재현 및 개선: 클래스 이름만으로 하는 531-클래스 계층 분류

> 데이터: Amazon-531 (49K 리뷰, 531 클래스, 3-level taxonomy) · 논문: TaxoClass (Shen et al., NAACL 2021)

**요약: 레이블된 문서가 0개인 조건에서, 클래스 이름과 taxonomy 구조만으로 상품 리뷰 계층 분류기를 학습한다. 재현 과정에서 분류기가 학습 없는 zero-shot baseline보다 오히려 6pt 낮은 현상을 발견했고, 원인을 추적해 자동 채굴된 silver label의 precision이 30%에 불과함을 규명했다. 채굴 필터 강화(30→43%)와 문서 truncation 교정(256→512)으로 +6.2pt, LLM API 예산의 불확실성 기반 할당(10K calls)으로 silver를 49%까지 정제해 최종 Example-F1 0.5180을 달성 — zero-shot 대비 +5.6pt, 논문 보고치(NoST 0.5431)와 2.5pt 차이다. 핵심 교훈: 병목은 모델이 아니라 데이터 품질이었고, 잘못된 지표 선택("문서당 정답 1개 이상 포함" 71%)이 그 문제(라벨 단위 precision 30%)를 은폐하고 있었다.**

## 최종 결과 (test, Example-F1)

| # | 단계 | silver prec. | path F1 | Δ |
|---|---|---|---|---|
| 0 | random / majority path | — | 0.063 / 0.120 | — |
| 1 | zero-shot NLI + path decoder | — | 0.4623 | — |
| 2 | + classifier (초기: silver 30%, len 256) | 30.3% | 0.3995 | −6.3 |
| 3 | + silver 정제 (min_conf 0.15) + len 512 | 43.2% | 0.4614 | +6.2 |
| 4 | + LLM refine 1K calls (과제 예산) | 43.8% | 0.4571 | ±0 |
| 5 | + LLM refine 10K calls 누적 (scaling 실험) | 48.8% | **0.5180** | +5.7 |
| 6 | + self-training (행 3 기준, 800 batches) | — | 0.4729 | +1.2 |

참조: 논문 Hier-0Shot-TC 0.4742 / TaxoClass-NoST 0.5431 / full 0.5934.
최종 채택 모델은 행 5 — zero-shot 대비 **+5.6pt**. (행 6 ST는 행 3 silver
기준이며, 행 5 + ST 조합은 future work.)

재현:python scripts/00_verify_dataset.py --data_dir reference_data/Amazon-531   # phase 0
python scripts/02_zeroshot_baseline.py --data_dir reference_data/Amazon-531  # 행 1
python scripts/05_decode_eval.py --data_dir reference_data/Amazon-531        # 디코더
행 2~6: scripts/01→03→04(→06,07), 커맨드는 각 스크립트 docstring 참조

## 문제 구조 (Phase 0)

Amazon-531: 531 클래스, train 29,487 / test 19,658. 검증된 핵심 성질:
**모든 label set이 root-to-leaf 경로** (길이 2 7.5% / 길이 3 92.5%) →
예측이 2^531 부분집합 선택이 아니라 **568개 candidate path 중 택일**로
축소. 이것이 path 디코더의 근거다.

## 파이프라인

1. **NLI similarity** (`01`) — roberta-large-mnli, "this product is about
   {}." 템플릿, taxonomy top-down 탐색 (문서당 평균 68.5 클래스만 계산,
   shard 체크포인팅)
2. **Zero-shot baseline** (`02`) — NLI 점수 직접 예측
3. **Core class mining** (`03`) — conf(D,c) = sim(D,c) − max(부모/형제 sim),
   클래스별 median + min_conf 이중 필터 (Eq. 2–3)
4. **Classifier** (`04`) — BERT [CLS] + GCN class encoder(ego-mean readout)
   + bilinear matching. positive = cores ∪ 부모, 자식은 negative 제외
   (Eq. 7–8). batch 32, len 512, seed 42
5. **Path decoding** (`05`) — 저장된 점수에서 GPU 없이 전 결과 재현
6. **Self-training** (`06`) — Eq. 9–10, Q를 rolling pool로 근사 (한계 참조)
7. **LLM refinement** (`07`) — 결정 경계 문서를 GPT-4o-mini로 재판정

## 핵심 분석

**1. 병목은 모델이 아니라 silver label이었다.** 초기 classifier가
zero-shot보다 6pt 낮은 원인을 추적: batch 크기(효과 <1pt), 디코딩, 학습
길이를 순차 배제한 뒤 silver 품질 감사에서 원인 발견. "문서당 정답 1개
이상 포함" 지표는 71%로 양호해 보였으나, 문서당 평균 3.8개를 캐며
2.7개가 오답인 과잉 생성을 가리고 있었다 — 라벨 단위 precision은 **30%**.
지표 선택이 데이터 문제를 은폐한 사례.

**2. Precision–coverage trade-off 측정과 선택.** min_conf 스윕 결과
(0→0.3에서 precision 30→51%, 커버리지 100→45%), 균형점 0.15 (43%/79%)
채택. truncation 256→512 교정과 합쳐 F1 +6.2pt, 과적합 패턴(loss↓·test
F1↓) 소멸.

**3. LLM 예산의 수확 체감 곡선.** 결정 경계(uncertainty 최소) 문서에
예산 할당: 1K calls → 코퍼스 3.2% 커버, precision +0.7pt, **F1 변화
없음**. 10K calls 누적(2차분 9,000 중 7,486 문서 수정 — 경계 문서의 83%
에서 rule과 LLM 판정이 갈림) → precision +5.0pt, **F1 +5.7pt**. 결론:
문서 단위 정제는 예산이 코퍼스 규모에 비례해야 효과가 나타나며,
소예산에서는 클래스 단위 개입(531개 클래스명 설명문 생성 등)이 합리적
대안.

**4. Naive path 디코더의 short-path 편향과 레이블 없는 캘리브레이션.**
raw logit 평균으로 path를 스코어링하면 길이 2가 구조적으로 유리
(δ=0에서 길이 2 예측 ~95% vs 실제 7.5%). 마진 δ를 레이블이 아니라
Phase 0에서 검증한 성질(길이 3 비율 92.5%)에 예측 분포를 맞춰 결정
(δ=0.265). test 스윕상 더 높은 최적점이 존재하나 test tuning이므로
민감도 분석으로만 보고.

**5. Metric 최적 ≠ 분포 일치.** Example-F1에서 길이 3 정답에 대해
올바른 길이 2 prefix는 0.80점, leaf 오답 길이 3은 0.67점 — leaf
정확도가 ~40%를 넘어야 leaf 예측이 이득. 초기 classifier는 이 손익분기
근처였고, 이것이 F1 최적 길이 분포(~50%)가 실제 분포(92.5%)와 다른
이유를 설명한다.

**6. Self-training은 랭킹을 개선하고 캘리브레이션을 파괴했다.**
ST(+1.2pt path F1) 후 p@1과 threshold 기반 F1은 급락 — Eq. 10의 corpus
합산을 rolling pool로 근사하며 절대 확률 스케일이 왜곡된 탓. path
디코딩은 상대 순위만 사용해 영향을 받지 않았다. ST 중간 지점(batch 400)
에서 0.489 피크가 관측되었으나 validation 없는 test 기반 체크포인트
선택을 피해 사전 설정 예산의 최종 모델(0.473)을 채택.

## 재현성 노트

원 실험 인프라(클라우드 인스턴스)의 갑작스러운 유실로 전체 환경을
재구축했다. 초기 관찰치 0.4684의 출처를 classifier로 오인해 재학습으로
추적했으나, zero-shot 파이프라인 재실행으로 해당 수치가 zero-shot +
path 디코딩의 결과임을 확인 (재현값 0.4680, top-3 0.2728 일치). 유실을
피한 silver label은 재사용 전 ground truth 감사로 무결성을 검증했다.
본 문서의 모든 수치는 재구축 환경에서 재현된 값이며, 재현되지 않은
수치는 포함하지 않았다.

## 한계와 다음 단계

- **논문과의 잔여 갭 (0.518 vs NoST 0.543):** 템플릿·Eq.2-3·entailment
  확률 추출의 일치는 확인했으나 mining precision이 논문 수준 미달 —
  후보 선정(Sect 3.2.1) 세부 구현 차이로 추정, 공식 repo와의 diff는
  future work
- **행 5 + self-training 조합 미실행** — 가장 유력한 다음 실험
- ST의 Q 근사를 corpus 전체 통계로 교체, validation split 도입
- similarity 모델 업그레이드 (DeBERTa-v3-MNLI 등)

## 산출물

코드는 이 저장소, 무거운 산출물(checkpoints, 점수 행렬, NLI 캐시,
silver label 각 버전)은 private HF dataset `hmtc-cache`에 백업.
`results/`의 npz + json만으로 모든 디코더/평가 결과가 GPU 없이 재현됨.
