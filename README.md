# Weakly-Supervised Hierarchical Multi-Label Text Classification — TaxoClass 재현과 데이터 품질 진단 (Amazon-531)

> Korea University DATA304 (2025-2) 과제 프로젝트의 개인 확장
> 논문: TaxoClass (Shen et al., NAACL 2021) · 데이터: Amazon-531 — TELEClass (WWW 2025) 공개 reference data ✏️링크 · 모델: [`roberta-large-mnli`](https://huggingface.co/FacebookAI/roberta-large-mnli), [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)

**요약: 문서–라벨 쌍이 0개인 조건에서 클래스 이름과 taxonomy 구조만으로 531-클래스 계층 멀티라벨 분류기를 학습하는 TaxoClass 파이프라인을 재현했다. 재현 과정에서 학습된 분류기가 학습을 전혀 하지 않은 zero-shot baseline보다 6.3pt 낮게 나오는 역전 현상을 만났고, 후보 원인을 순차 배제한 끝에 자동 채굴된 silver label의 라벨 단위 precision이 30%에 불과함을 규명했다 — 이 문제는 처음 사용하던 품질 지표("문서당 정답 1개 이상 포함" 71%)에 가려져 있었다. 채굴 필터 강화(precision 30→43%)와 입력 truncation 교정(256→512 토큰)으로 +6.2pt를 회복하고, LLM API 예산을 결정 경계 문서에 집중 투입해(10K calls, 약 $5) silver를 49%까지 정제한 뒤, validation 기반 stopping을 붙인 self-training으로 최종 Example-F1 0.5488을 달성했다. zero-shot 대비 +8.7pt이며 논문 보고치(TaxoClass-NoST 0.5431)를 상회한다(단, 최종 행의 stopping 신호는 순수 weak-supervision 세팅에서 벗어난다 — §5.4에 공개). 핵심 결론: 병목은 모델이 아니라 학습 데이터 품질이었고, 같은 노력을 모델이 아닌 답지(silver label)의 구축·정제·검수에 투입하는 것이 성능 개선의 정답이었다.**

---

## 1. Problem — 라벨 0개, 클래스 531개

| 항목 | 내용 |
|---|---|
| 데이터 | Amazon-531 상품 리뷰 — **train 29,487 / test 19,658 문서** |
| 클래스 | 531개, 3-level taxonomy (tree) |
| 사용 가능한 supervision | **클래스 이름 문자열 + 트리 구조뿐.** 사람이 라벨링한 문서–라벨 쌍 0개 |
| Task | 문서마다 정답 카테고리 집합(멀티라벨) 예측 |

일반적인 지도학습이라면 수만 개의 라벨링된 예시로 학습하지만, 이 세팅에서는 라벨이 존재하지 않는다. 실무에서 새 taxonomy가 도입되었지만 라벨링 예산이 없는 상황과 동형이다.

**Phase 0 — 데이터 구조 검증 (`scripts/00_verify_dataset.py`).** 파이프라인 설계에 앞서 정답 label set의 구조를 전수 검증했다. 확인된 성질: **모든 label set이 root-to-leaf 경로다** — 길이 2가 7.5%, 길이 3이 92.5%, 예외 0건. 이 성질 덕분에 예측 문제가 2^531가지 부분집합 선택이 아니라 **568개 candidate path 중 하나를 고르는 문제**로 축소된다. §2.5의 path 디코더는 전부 이 검증된 성질 위에 서 있으며, 모델링 전에 데이터의 구조적 제약을 먼저 확인하는 것이 이후 모든 설계 결정의 출발점이었다.

> **Research Question**: 사람 라벨 없이 클래스 이름과 taxonomy 구조만으로 학습한 분류기가, 같은 재료를 학습 없이 사용하는 zero-shot 추론을 상회할 수 있는가? 상회하지 못한다면 병목은 파이프라인의 어느 단계인가?

---

## 2. Method — 선생(zero-shot NLI) · 답지(silver label) · 학생(classifier)

파이프라인은 TaxoClass의 4단계(similarity → core class mining → classifier → self-training)에 path 디코딩을 더한 구조다. 세 구성 요소의 관계가 이 프로젝트를 읽는 열쇠다: **같은 NLI 점수 행렬에서 세 갈래가 나온다.** 점수를 학습 없이 그대로 예측에 쓰면 zero-shot(선생), 점수에서 학습용 라벨을 채굴하면 silver label(답지), 그 답지로 학습하면 classifier(학생)다. 선생은 기준선이고, 학생이 선생을 넘는지가 파이프라인 전체의 성패 지표다.

### 2.1 Document–Class Similarity — NLI (`01`)

`roberta-large-mnli`에 (premise = 문서, hypothesis = `"this product is about {class name}."`) 쌍을 입력해 entailment 확률을 문서–클래스 유사도로 사용한다. 전 클래스를 계산하면 문서당 531회 forward가 필요하므로(약 2,600만 회), taxonomy를 **top-down으로 내려가며 유망한 가지만 확장** — 실측 문서당 평균 **68.5 클래스**만 계산했다. 결과는 shard 단위로 체크포인팅하며 train/test 양쪽 점수 행렬을 `.npz`로 캐시해, 이후의 모든 단계(mining·디코딩·평가)가 GPU 없이 재현된다.

### 2.2 Core Class Mining — Silver Label 채굴 (`03`, 논문 Eq. 2–3)

모든 identity의 채굴 신호는 절대 점수가 아니라 **주변 대비 상대 확신도**다.

```
conf(D, c) = sim(D, c) − max( sim(D, parent(c)), sim(D, siblings(c)) )
```

문서 D에서 클래스 c가 부모·형제보다 뚜렷이 튀는 경우에만 core 후보가 된다. 여기에 **이중 필터** — (1) 클래스별 median 기준, (2) `min_conf` 절대 하한 — 를 통과한 core에 트리상 조상을 붙인 것이 silver label이다(트리에서 부모는 유일하므로 조상은 자동 결정). 필터를 통과하지 못한 문서는 답지 없이 학습에서 제외되며, 이 precision–coverage trade-off의 측정과 선택이 §5.2다. 논문 Sect. 3.2.2의 positivity 조건이 median 필터와 **함께** 필요하다는 점을 재현 중 확인했다(§6).

### 2.3 Classifier (`04`, 논문 Eq. 7–8)

- **문서 인코더**: `bert-base-uncased` [CLS]
- **클래스 인코더**: GCN (ego-mean readout, residual connection) — taxonomy 이웃 정보를 클래스 표현에 주입해 tail class가 부모의 신호를 빌릴 수 있게 한다. 클래스 feature는 centering/normalize 후 learnable 파라미터로 학습(§6의 anisotropy 이슈 대응)
- **매칭**: bilinear — 문서 표현과 클래스 표현의 상호작용 점수
- **라벨 구성**: positive = cores ∪ 부모, **core의 자식 클래스는 negative에서 제외** (미채굴이 곧 오답 증거가 아니므로)
- **Loss**: per-document **sum** BCE — mean 정규화 시 positive gradient가 희석되어 collapse가 발생한다(§6)
- 셋업: batch 32, max seq len 512, seed 42

### 2.4 Self-Training (`06`, 논문 Eq. 9–10)

학생이 자기 예측 분포를 sharpening한 soft target Q로 스스로를 재학습한다. Q의 갱신은 **rolling pool로 근사**했다(논문의 전체 재계산과 다름 — §8 한계). Stopping 전략을 두 가지로 실험했다: (a) 고정 800 batches, (b) audit-validation 기반 best checkpoint + patience 2. 두 전략의 차이가 §5.4의 핵심이다.

### 2.5 Path Decoding (`05`)

모델은 531개 클래스에 점수만 출력하지만, 정답은 항상 경로 모양이다(§1의 검증된 성질). 따라서 점수 상위 k개를 무제약으로 찍는 대신, **568개 candidate path 각각의 경로 점수를 집계해 argmax 경로 하나를 제출**한다. zero-shot 기준으로 구조를 무시한 top-3 예측 0.2729 ✏️ 대비 path 디코딩 0.4623 — 출력 구조 제약을 아는 것만으로 얻는 개선이다. NLI든 학생이든 동일한 디코더를 통과하므로 모든 행이 같은 조건에서 비교된다.

---

## 3. Evaluation Protocol — Ground Truth의 역할 분리

이 프로젝트에서 가장 오해되기 쉬운 지점이라 명시한다. train/test의 gold label은 TELEClass 공개 reference data로 확보되어 있으나, **파이프라인과 평가에서의 역할이 엄격히 분리**된다.

| 데이터 | GT 존재 | GT의 용도 |
|---|---|---|
| test 19,658 | O | **채점 전용** — 파이프라인 어디에도 입력되지 않음 |
| train 29,487 | O | **오프라인 감사(audit) 전용** — silver label 품질 측정에만 사용, 학습 신호로 미사용 |

즉 "라벨 0개"의 정확한 의미는 GT가 세상에 없다는 것이 아니라, **모델을 만드는 과정(채굴·학습·self-training)이 GT를 볼 수 없다**는 것이다. 실험자는 GT를 진단 도구로 사용한다 — silver precision 30%/43%/49%라는 수치가 전부 train GT 대조 감사에서 나온다. 실제 라벨이 없는 현장이라면 이 감사는 소량 샘플의 수동 검수로 대체된다. 유일한 예외는 행 7의 audit-validation stopping이며 §5.4에 별도로 공개한다.

### 지표

| 지표 | 역할 | 계산 대상 |
|---|---|---|
| **Example-F1** | 주 지표 — 디코딩된 예측 경로 vs GT | test 19,658 |
| **silver precision (라벨 단위)** | 답지 품질 — 채굴된 라벨 중 GT에 실재하는 비율 | train silver label |
| **silver coverage** | 답지 보유 문서 비율 (필터 강도의 대가) | train |
| MRR (filtered / unfiltered 병기) | 논문 Table 2 대조 — §6의 ceiling 이슈로 두 변형 모두 계산 | test |

**지표 설계 자체가 이 프로젝트의 교훈 중 하나다.** 초기에는 silver 품질을 "문서당 정답 1개 이상 포함" 비율(71%)로 측정했고 문제가 보이지 않았다. 그러나 이 지표는 문서당 평균 3.8개를 캐며 그중 2.7개가 오답인 과잉 생성 구조를 가린다 — 라벨 단위 precision으로 재측정하자 30%가 드러났다(§5.1). 이후 품질 관리는 **라벨 단위 precision + 문서 coverage의 2축**으로 수행했다.

---

## 4. Experiments — 누적 개선표

각 행은 이전 행에 대한 가설–검증 단위다. (test, Example-F1)

| # | 단계 | silver prec. | path F1 | Δ |
|---|---|---|---|---|
| 0 | random / majority path | — | 0.063 / 0.120 | — |
| 1 | zero-shot NLI + path decoder (선생) | — | 0.4623 | — |
| 2 | + classifier (초기: silver 30%, len 256) | 30.3% | 0.3995 | **−6.3** |
| 3 | + silver 정제 (min_conf 0.15) + len 512 | 43.2% | 0.4614 | +6.2 |
| 4 | + LLM refine 1K calls (과제 예산) | 43.8% | 0.4571 | ±0 |
| 5 | + LLM refine 10K calls 누적 (scaling 실험) | 48.8% | 0.5180 | +5.7 |
| 6 | + self-training (행 3 기준, 800 batches 고정) | — | 0.4729 | +1.2 |
| 7 | **+ self-training (행 5 기준, audit-val stopping)** | — | **0.5488** | **+3.1** |

참조 (논문 보고치): Hier-0Shot-TC 0.4742 / TaxoClass-NoST 0.5431 / TaxoClass full 0.5934

**행별 가설.** 행 2: "선생의 점수로 학습한 학생은 패턴 일반화로 선생을 넘을 것" → **기각**(−6.3), 진단 시작. 행 3: "병목이 silver 품질이라면 채굴 필터 강화만으로 회복될 것" → 지지(+6.2). 행 4–5: "부분 정제의 효과는 투입 규모의 함수일 것" → 1K 무효 / 10K 유효로 임계 규모 존재 확인. 행 6–7: "self-training의 이득은 stopping 시점에 민감할 것" → 고정 스텝 +1.2 vs validation 기반 +3.1.

최종 채택 모델은 **행 7 (0.5488)**. 단, 행 7의 stopping은 train에서 분리한 3,000개 audit-validation을 사용하므로 순수 weak-supervision 세팅에서 벗어난다(§5.4에 공개). **논문 세팅을 그대로 준수한 최고치는 행 5 (0.5180)**다.

---

## 5. Results & Analysis

### 5.1 병목 진단 — 지표가 데이터 문제를 가리고 있었다

초기 classifier(행 2)가 zero-shot(행 1)보다 6.3pt 낮았다. 학습이 성능을 깎는 역전이므로 어딘가에 구조적 결함이 있다는 신호다. 후보를 순차 배제했다: **batch 크기**(변화 1pt 미만) → **디코딩 방식**(선생·학생 동일 디코더 확인) → **학습 길이** → 마지막으로 **silver label 품질 감사**에서 원인 발견.

감사가 늦어진 이유가 핵심 교훈이다. 당시 품질 지표 "문서당 정답 1개 이상 포함"은 71%로 양호해 보였다. 그러나 라벨 단위로 재면 precision **30.3%** — 문서당 3.8개를 캐서 하나만 맞고 2.7개가 틀린 과잉 생성이 문서 단위 지표에 완전히 가려져 있었다. 학생은 70%가 오답인 답지로 성실히 학습한 것이고, 배울수록 나빠지는 것이 오히려 정합적인 결과였다. **지표를 잘못 고르면 데이터 문제가 지표상 존재하지 않는다** — 이후 모든 품질 검수를 라벨 단위 precision + coverage 2축으로 전환했다.

### 5.2 Precision–Coverage Trade-off — 측정하고 선택한다

필터 강화는 공짜가 아니다. `min_conf`를 0→0.3으로 스윕하면 precision은 30→51%로 오르지만 coverage(답지 보유 문서 비율)는 100→45%로 떨어진다. 이 곡선을 측정한 뒤 **균형점 0.15 (precision 43.2% / coverage 79%)를 채택**했다 — 임의 선택이 아니라 trade-off 곡선 위의 의사결정이다. 같은 시점에 발견한 truncation 설정 오류(문서를 256 토큰에서 잘라 긴 리뷰의 앞 절반만 학습)를 512로 교정했고, 두 수정을 합쳐 **+6.2pt** 회복(행 3). 수정 전 관찰되던 과적합 패턴(train loss 감소·test F1 하락)이 수정 후 소멸한 것이 부수 증거다.

### 5.3 LLM 예산의 임계 규모 — 1K는 무효, 10K는 유효

silver 43%는 여전히 절반 이상이 오답이다. 필터로 걸러지지 않는 **결정 경계 문서**(confidence가 threshold 근방인 문서)를 GPT-4o-mini로 재판정하는 실험을 설계했다: uncertainty가 큰 문서부터 예산을 할당한다.

- **1K calls** (과제 예산): 코퍼스의 3.2% 커버, precision +0.7pt — **F1 변화 없음** (행 4)
- **10K calls 누적** (약 $5, scaling 실험): precision 48.8% — **F1 +5.7pt, 0.5180** (행 5)

같은 개입이라도 임계 규모 아래에서는 효과가 0이다. "LLM으로 데이터를 정제하면 좋아진다"가 아니라 **"얼마 이상 투입해야 좋아지는가"의 곡선을 얻은 것**이 이 실험의 산출물이며, 정제 예산 의사결정은 이 곡선 없이는 성립하지 않는다.

### 5.4 Self-Training — Stopping이 이득의 크기를 결정한다

고정 800 batches의 ST(행 6)는 +1.2pt에 그쳤고, 학습 로그에서 성능이 피크 후 하락하는 패턴이 관찰됐다 — confirmation bias가 누적되는 ST의 알려진 위험과 부합한다. 이에 stopping을 validation 기반으로 재설계했다(행 7): train에서 3,000개를 고정 seed로 분리(ST pool에서도 제외해 누수 차단), eval마다 validation path F1로 best checkpoint를 선택, patience 2.

결과 0.5488 (+3.1pt), 논문 NoST(0.5431) 상회. 두 가지를 공개한다:

1. **Audit-validation은 순수 weak supervision 위반이다.** stopping 신호가 train gold label 3,000개에서 나오므로, test 기반 선택보다는 약하지만 논문 세팅이 금지하는 오염이다. 그래서 이 값을 "audit-validation"으로 명명해 공개하고, 논문 세팅 준수 최고치(행 5, 0.5180)를 병기한다. label-free stopping 신호(예측 entropy 등)는 미검증 future work다.
2. **test 피크(0.5492)는 채택하지 않았다.** ST 로그상 test F1 최고점은 0.5492였으나, 이는 결과를 본 후 고르는 값이라 평가가 아니라 test set 튜닝이 된다. 보고치는 결과를 보기 전에 규칙(validation best)이 정한 0.5488이다.

### 5.5 논문 대비 위치

- zero-shot: 본 재현 0.4623 vs 논문 Hier-0Shot-TC 0.4742 — 근접하나 소폭 하회
- 학습 파이프라인: 행 5 0.5180 (논문 세팅 준수) vs NoST 0.5431 — 2.5pt 차. silver precision이 49%에서 멈춘 것이 주된 잔여 격차로 추정된다
- ST 포함: 행 7 0.5488 vs NoST 0.5431 (상회, 단 §5.4의 세팅 차이 공개) / full 0.5934 미도달 — Q rolling-pool 근사와 ST 세부 구현 차이가 후보 원인(§8)

---

## 6. 재현 노트 — 논문과 구현 사이에서 확인한 것들

재현 과정에서 논문 기술과 실제 구현 사이의 간극을 확인하고 문서화했다. 각 항목은 "논문을 문자 그대로 구현하면 무엇이 깨지는가"에 대한 기록이다.

| # | 이슈 | 확인한 사실 | 대응 |
|---|---|---|---|
| 1 | **Eq. 8 loss 정규화** | consistency 대상을 mean으로 정규화하면 positive gradient가 약 100배 희석되어 모델이 collapse (전 클래스 점수 균일화) | per-document **sum** BCE로 교정 → 학습 정상화 |
| 2 | **클래스 표현 anisotropy** | BERT 이름 임베딩으로 초기화한 클래스 feature가 near-collinear — bilinear 매칭이 클래스를 구분하지 못함 | centering/normalize + learnable 파라미터화 + GCN residual |
| 3 | **Sect. 3.2.2 positivity 조건** | median 필터만 구현하면 silver precision 18%에 그침 — 논문의 두 조건이 모두 필요 | 이중 조건 구현 → 30%대 확보 (이후 §5.2의 정제로 43%) |
| 4 | **MRR ceiling** | 논문 Table 2의 MRR 정의를 문자 그대로 계산하면 이 데이터셋의 이론적 최대치가 약 0.621로, 논문 보고치 0.6332보다 낮음 → 논문은 filtered-rank MRR 사용으로 추정 | 본 레포는 filtered / unfiltered 두 변형을 모두 계산·병기 |
| 5 | **Eq. 6 표기** | 문자 그대로면 sigmoid(exp(·)) ∈ (0.5, 1)로 BCE가 ill-posed | 표기 이슈로 판단, bilinear logit에 sigmoid를 적용하는 표준형으로 구현 ✏️ |

이 목록은 "구현이 안 돌아가는" 버그가 아니라 **학습은 돌아가되 결과가 조용히 왜곡되는** 유형이라는 공통점이 있다. 재현 실험에서 수치 불일치가 났을 때 모델 탓으로 돌리기 전에 논문–구현 diff를 원인 후보로 감사하는 절차가 필요함을 보여준다.

---

## 7. Conclusion

> **모델이 나쁜 것이 아니라 답지가 나빴다. 답지를 측정하고 정제하는 데 노력을 쓰는 것이 정답이었다.**

1. **병목은 데이터 품질이었다** — 동일 모델·동일 아키텍처에서 silver precision을 30→43→49%로 올리는 것만으로 F1이 0.40→0.46→0.52로 동행했다. 데이터 축의 개선이 모델 축의 개선 없이 성능을 견인한다.
2. **지표 선택이 데이터 문제를 은폐할 수 있다** — 문서 단위 71%라는 안심되는 숫자 뒤에 라벨 단위 30%가 숨어 있었다. 품질 검수 지표는 실패 유형(여기서는 과잉 생성)을 드러내는 단위로 설계해야 한다.
3. **정제 투자에는 임계 규모가 있다** — 1K calls는 효과 0, 10K calls는 +5.7pt. 개입의 효과가 아니라 개입-규모 곡선을 측정해야 예산 의사결정이 가능하다.
4. **보고치는 결과를 보기 전의 규칙이 정한다** — test 피크 0.5492 대신 validation 규칙이 정한 0.5488을, 논문 세팅 위반이 있는 행 7 옆에는 준수 최고치 0.5180을 병기했다.

### 8. Limitations & Future Work

- **단일 seed(42), 단일 encoder family** — 차이의 방향은 행 전반에서 일관되나, 3-seed 반복과 분산 보고가 필요하다. seed 민감도 미측정은 본 레포의 가장 큰 통계적 한계다.
- **행 7의 audit-validation stopping은 순수 weak supervision을 벗어난다** (train gold 3,000개 사용, §5.4 공개). 예측 entropy 등 label-free stopping 신호의 검증이 future work.
- **Self-training의 Q를 rolling pool로 근사**했다 — 논문의 전체 재계산 방식과 다르며, full(0.5934) 미도달의 후보 원인이다.
- **silver precision 49%에서 중단** — LLM 예산 곡선의 상단(10K 이상)과, refine과 mining threshold의 상호작용은 미탐색이다.
- **path 디코더는 Amazon-531의 검증된 성질(전 정답 = root-to-leaf 경로)에 의존**한다. DAG이거나 비경로 라벨이 존재하는 taxonomy에는 그대로 이식할 수 없고, 해당 성질의 검증이 선행되어야 한다.
- **LLM refinement의 판정 품질 자체는 감사하지 않았다** — GPT-4o-mini의 재판정을 silver에 병합했을 때의 precision 증분으로 간접 확인했을 뿐, 판정 단위 정확도는 별도 측정이 필요하다.

---

## Repository

```
weakly-supervised-hmtc/
├── README.md                        # 이 문서
├── scripts/                         # 단계별 실행 스크립트 (번호 = 실행 순서)
│   ├── 00_verify_dataset.py         #   Phase 0: 데이터 검증 (path property 전수 확인)
│   ├── 01_...py                     #   NLI similarity 계산 (shard 체크포인팅) ✏️
│   ├── 02_zeroshot_baseline.py      #   행 1: zero-shot baseline
│   ├── 03_...py                     #   core class mining (silver label 채굴) ✏️
│   ├── 04_...py                     #   classifier 학습 ✏️
│   ├── 05_decode_eval.py            #   path 디코딩 + 평가 (캐시만으로 GPU 없이 재현)
│   ├── 06_self_training.py          #   self-training (--val_size 옵션으로 행 6/7 전환)
│   └── 07_...py                     #   LLM refinement (GPT-4o-mini) ✏️
├── src/                             # 공용 모듈
├── cache/                           # NLI 점수 행렬 (.npz, shard 단위)
├── results/                         # 실험 결과 JSON
├── checkpoints/                     # 모델 checkpoint (st_best_val.pt 등)
└── reference_data/Amazon-531/       # TELEClass 공개 데이터 (gold label 포함)
```

<details>
<summary>재현하기 (커맨드 요약)</summary>

```bash
# Phase 0: 데이터 검증
python scripts/00_verify_dataset.py --data_dir reference_data/Amazon-531

# 행 1: zero-shot baseline
python scripts/02_zeroshot_baseline.py --data_dir reference_data/Amazon-531

# 디코더/평가 (캐시 기반, GPU 불필요)
python scripts/05_decode_eval.py --data_dir reference_data/Amazon-531

# 행 2~7: scripts/01 → 03 → 04 (→ 06, 07). 커맨드는 각 스크립트 docstring 참조
```

</details>

---
