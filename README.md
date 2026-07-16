# TaxoClass 재현 및 개선: 클래스 이름만으로 하는 531-클래스 계층 분류

> 데이터: Amazon-531 (49K 리뷰, 531 클래스, 3-level taxonomy) · 논문: TaxoClass (Shen et al., NAACL 2021)

레이블된 문서가 하나도 없는 조건에서, 클래스 이름과 taxonomy 구조만 가지고 상품 리뷰 계층 분류기를 학습하는 프로젝트다. TaxoClass 파이프라인을 재현하는 과정에서 분류기가 학습을 전혀 하지 않은 zero-shot baseline보다 오히려 6pt 낮게 나오는 문제를 만났고, 원인을 추적한 끝에 자동 채굴된 silver label의 precision이 30%밖에 되지 않는다는 것을 확인했다. 채굴 필터를 강화하고(30→43%) 문서 truncation 설정 오류를 고쳐(256→512) +6.2pt를 회복했고, LLM API 예산을 결정 경계 문서에 집중 투입해(10K calls) silver를 49%까지 끌어올린 뒤, validation 기반 stopping을 붙인 self-training으로 최종 Example-F1 **0.5488**을 얻었다. Zero-shot 대비 +8.7pt이고, 논문 보고치(NoST 0.5431)를 넘는 수치다.

이 과정에서 얻은 가장 큰 교훈: 병목은 모델이 아니라 데이터 품질이었고, 잘못 고른 지표("문서당 정답 1개 이상 포함" 71%)가 그 문제(라벨 단위 precision 30%)를 계속 가리고 있었다.

## 최종 결과 (test, Example-F1)

| # | 단계 | silver prec. | path F1 | Δ |
|---|---|---|---|---|
| 0 | random / majority path | — | 0.063 / 0.120 | — |
| 1 | zero-shot NLI + path decoder | — | 0.4623 | — |
| 2 | + classifier (초기: silver 30%, len 256) | 30.3% | 0.3995 | −6.3 |
| 3 | + silver 정제 (min_conf 0.15) + len 512 | 43.2% | 0.4614 | +6.2 |
| 4 | + LLM refine 1K calls (과제 예산) | 43.8% | 0.4571 | ±0 |
| 5 | + LLM refine 10K calls 누적 (scaling 실험) | 48.8% | 0.5180 | +5.7 |
| 6 | + self-training (행 3 기준, 800 batches 고정) | — | 0.4729 | +1.2 |
| 7 | **+ self-training (행 5 기준, audit-val stopping)** | — | **0.5488** | **+3.1** |

참조: 논문 Hier-0Shot-TC 0.4742 / TaxoClass-NoST 0.5431 / full 0.5934.

최종 채택 모델은 행 7 (0.5488). 다만 행 7의 stopping은 train에서 분리한 3,000개 audit-validation을 사용하므로 순수 weak-supervision 세팅에서는 벗어난다(분석 6 참조). 논문 세팅을 그대로 준수한 최고치는 행 5 (0.5180)다.

재현:

```
python scripts/00_verify_dataset.py --data_dir reference_data/Amazon-531   # phase 0
python scripts/02_zeroshot_baseline.py --data_dir reference_data/Amazon-531  # 행 1
python scripts/05_decode_eval.py --data_dir reference_data/Amazon-531        # 디코더
# 행 2~7: scripts/01 → 03 → 04 (→ 06, 07). 커맨드는 각 스크립트 docstring 참조
```

## 문제 구조 (Phase 0)

Amazon-531은 531개 클래스, train 29,487 / test 19,658 문서로 구성된다. 데이터를 검증하면서 확인한 핵심 성질: **모든 label set이 root-to-leaf 경로다** (길이 2가 7.5%, 길이 3이 92.5%). 덕분에 예측 문제가 2^531가지 부분집합 선택이 아니라 **568개 candidate path 중 하나를 고르는 문제**로 줄어든다. 뒤에 나오는 path 디코더는 전부 이 성질 위에 서 있다.

## 파이프라인

1. **NLI similarity** (`01`) — roberta-large-mnli에 "this product is about {}." 템플릿을 넣어 문서-클래스 유사도를 계산한다. taxonomy를 top-down으로 내려가며 유망한 후보만 계산하고(문서당 평균 68.5 클래스), shard 단위로 체크포인팅한다.
2. **Zero-shot baseline** (`02`) — NLI 점수를 학습 없이 그대로 예측에 사용.
3. **Core class mining** (`03`) — conf(D,c) = sim(D,c) − max(부모/형제 sim)로 신뢰도를 계산하고, 클래스별 median과 min_conf 이중 필터를 통과한 것만 silver label로 채택 (논문 Eq. 2–3).
4. **Classifier** (`04`) — BERT [CLS] + GCN class encoder(ego-mean readout) + bilinear matching. positive = cores ∪ 부모, 자식 클래스는 negative에서 제외 (Eq. 7–8). batch 32, len 512, seed 42.
5. **Path decoding** (`05`) — 저장된 점수 행렬만으로 GPU 없이 전체 결과 재현.
6. **Self-training** (`06`) — Eq. 9–10. Q는 rolling pool로 근사했다 (한계 참조).
7. **LLM refinement** (`07`) — 결정 경계에 걸린 문서를 GPT-4o-mini로 재판정.

## 핵심 분석

**1. 병목은 모델이 아니라 silver label이었다.** 초기 classifier가 zero-shot보다 6pt 낮았다. batch 크기(효과 1pt 미만), 디코딩, 학습 길이를 하나씩 배제한 뒤 silver 품질 감사에서 원인을 찾았다. 처음 쓴 지표는 "문서당 정답 1개 이상 포함"으로 71%라 괜찮아 보였는데, 실제로는 문서당 평균 3.8개를 캐면서 그중 2.7개가 오답인 과잉 생성 구조였다. 라벨 단위 precision으로 다시 재면 30%. 지표를 잘못 고르면 데이터 문제가 은폐된다는 것을 여기서 배웠다.

**2. Precision–coverage trade-off를 재고 나서 골랐다.** min_conf를 스윕해보니 0→0.3 구간에서 precision은 30→51%로 오르지만 커버리지가 100→45%로 떨어진다. 균형점인 0.15 (precision 43% / 커버리지 79%)를 채택했고, truncation 256→512 교정과 합쳐 F1이 +6.2pt 회복됐다. 이전 런마다 보이던 과적합 패턴(loss는 내려가는데 test F1이 떨어지는)도 같이 사라졌다.

**3. LLM 예산에는 수확 체감 곡선이 있다.** 채굴 필터가 가장 헷갈려하는 결정 경계 문서부터 예산을 할당했다. 1K calls로는 코퍼스의 3.2%만 커버돼서 precision +0.7pt, F1은 변화가 없었다. 10K calls 누적(2차분 9,000 중 7,486 문서 수정 — 경계 문서의 83%에서 rule과 LLM 판정이 갈렸다)에서는 precision +5.0pt, F1 +5.7pt. 결론적으로 문서 단위 정제는 예산이 코퍼스 규모에 비례해야 효과가 나고, 소예산이라면 클래스 단위 개입(531개 클래스명에 설명문을 붙이는 식)이 더 합리적인 방향으로 보인다.

**4. Naive path 디코더에는 short-path 편향이 있다.** raw logit의 평균으로 path를 점수 매기면 길이 2 경로가 구조적으로 유리하다 — 큰 값 2개의 평균이 3개의 평균을 이기기 쉽기 때문이다. 실제로 δ=0에서 길이 2 예측이 약 95%였는데 실제 분포는 7.5%다. 보정 마진 δ는 레이블이 아니라 Phase 0에서 검증한 성질(길이 3 비율 92.5%)에 예측 분포를 맞추는 방식으로 정했다 (δ=0.265). test 스윕에서 더 높은 최적점이 존재하긴 하지만 그건 test tuning이라 채택하지 않고 민감도 분석으로만 남겼다.

**5. Metric 최적이 분포 일치를 뜻하지 않는다.** Example-F1에서 길이 3 정답에 대해 올바른 길이 2 prefix는 0.80점, leaf가 틀린 길이 3 예측은 0.67점을 받는다. 즉 leaf 정확도가 대략 40%를 넘어야 leaf까지 찍는 게 이득이다. 초기 classifier는 이 손익분기 근처에 있었고, F1 최적 길이 분포(~50%)가 실제 분포(92.5%)와 크게 달랐던 이유가 여기서 설명된다.

**6. Self-training의 이득은 stopping이 결정한다.** ST는 path F1을 초반에 +3pt가량 올렸다가 이후 단조 하락시키는 패턴을 두 base(행 3, 행 5)에서 똑같이 보였다. Q를 rolling pool로 근사하면서 절대 확률 스케일이 왜곡되고(p@1과 threshold 기반 F1이 급락하는 반면 path 디코딩은 상대 순위만 쓰므로 무관), 과도한 sharpening이 누적되는 탓으로 보인다. 고정 예산 800 batch의 최종 모델은 base보다 나빴다(행 5 기준 −1.8pt). 그래서 train에서 3,000개를 분리한 **audit-validation** 기반 early stopping을 도입했다. validation 궤적이 test 궤적과 동일한 피크(batch 200)를 가리켰고, 그 시점의 test F1 0.5488을 test 관여 없이 확정할 수 있었다. 다만 audit-validation은 train gold label을 사용하므로 순수 weak-supervision 세팅에서 벗어난다 — 논문 세팅 준수 최고치는 행 5(0.5180)이고, 행 7은 "레이블 3,000개가 추가로 허용될 때"의 결과로 구분해서 보고한다.

## 한계와 다음 단계

- **논문과의 잔여 갭 (행 5 기준 0.518 vs NoST 0.543):** 템플릿, Eq. 2-3, entailment 확률 추출이 논문과 일치함은 확인했지만 mining precision이 논문 수준에 못 미친다. 후보 선정(Sect 3.2.1)의 세부 구현 차이로 추정하며, 공식 repo와의 diff 분석이 남은 과제다.
- **audit-validation은 train 레이블에 의존한다.** 레이블 없는 stopping 신호(예측 entropy 등)를 찾는 것이 다음 방향이다.
- ST의 Q 근사를 corpus 전체 통계로 교체하는 것도 시도할 가치가 있다.
- similarity 모델 업그레이드 (DeBERTa-v3-MNLI 등, 2021년 모델의 교체).

## 산출물

코드는 이 저장소에 있고, 무거운 산출물(checkpoints, 점수 행렬, NLI 캐시, silver label 각 버전)은 private HF dataset `hmtc-cache`에 백업해 두었다. `results/`의 npz와 json만 있으면 모든 디코더·평가 결과를 GPU 없이 재현할 수 있다.
