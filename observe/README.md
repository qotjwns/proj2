# 저장된 Drainage 모델의 선택적 분류 평가

추가 학습 없이 깨끗한 CIFAR-10 테스트 10,000개에 한 번 추론합니다.
기존 모델 구조와 정규화 상수를 재사용하고, 학습 데이터 생성이나 crop·flip은 수행하지 않습니다.
`model.eval()`과 `torch.inference_mode()`를 사용합니다.

## 실행

저장소 최상위 폴더에서 실행합니다. 기존 `venv`에 학습용 의존성이 설치되어 있다고 가정합니다.

```bash
# 시각화용 추가 의존성
./venv/bin/python -m pip install -r observe/requirements.txt

# 체크포인트 로드 → 한 번 추론 → logits 저장 → 세 점수 평가
./venv/bin/python observe/evaluate.py

# 저장된 평가 결과로 전체 시각화 생성 (모델·데이터 로드 없음)
./venv/bin/python observe/visualize.py
```

기본 체크포인트는 `experiment/cifar10/asym/alpha_dl/n0.45/seed1_epoch120_KOITBALW.pt`,
설정은 `config/cifar10_alpha_dl.json`, 결과 폴더는 `observe/results/noise_0.45`입니다.
이 체크포인트는 **120 epoch 종료 가중치**입니다. 학습 로그의 best epoch 가중치를 뜻하지 않습니다.
체크포인트의 경로·SHA-256, 설정, 정규화, 장치 정보를 logits와 함께 기록합니다.

`--device auto`는 CUDA → MPS → CPU 순서로 선택합니다. `--device cpu`, `--batch-size 128`,
`--num-workers 0`, `--num-threads 4` 등으로 실행 환경을 바꿀 수 있습니다.
데이터가 없을 때만 `--download`를 지정하면 됩니다.

다른 체크포인트를 평가하려면 결과 폴더도 별도로 지정합니다.

```bash
./venv/bin/python observe/evaluate.py \
  --checkpoint experiment/cifar10/asym/alpha_dl/n0.45/seed1_epoch120_KOITBALW.pt \
  --config config/cifar10_alpha_dl.json \
  --output-dir observe/results/another_run

./venv/bin/python observe/visualize.py --results-dir observe/results/another_run
```

이미 저장된 logits를 재사용할 때는 다음 명령을 사용합니다. 모델·체크포인트·데이터를 다시 읽지 않습니다.
원본 logits는 보존하고, 표·곡선·점수 파일만 다시 만듭니다.

```bash
./venv/bin/python observe/evaluate.py \
  --cache observe/results/noise_0.45/logits.pt
./venv/bin/python observe/visualize.py
```

같은 폴더에 `logits.pt`가 있는데 다시 추론을 요청하면 중단하고 `--cache` 사용을 안내합니다.
다른 결과 폴더의 캐시를 평가할 때는 `--output-dir`도 함께 지정하세요.

## 계산 방식

- 클래스 예측: Drainage를 제외한 10개 logits의 argmax로 고정합니다.
- Drainage: 전체 11개 softmax에서 `1 - p_d`.
- MSP: 클래스 10개에만 softmax를 적용한 뒤 최대 확률.
- Logit margin: 클래스 10개 중 가장 큰 logit에서 두 번째로 큰 logit을 뺀 값.
- 모든 점수는 클수록 수용 우선이며, CPU float64로 계산하고 정렬 전에 반올림하지 않습니다.
- 현재 학습 설정의 Drainage 인덱스는 `10`입니다. 학습 라벨은 logits `0..9`에 대응합니다.

점수의 **정확한 동점**은 해석적으로 처리합니다. 이미 수용한 `a`개에 오답 `E`개가 있고,
다음 동점 그룹 `m`개 중 오답이 `e`개라면, 그 그룹에서 `t`개를 수용할 때 기대 오류율은
`(E + t * e / m) / (a + t)`입니다. 정답 여부나 샘플 순서로 동점을 깨지 않습니다.
부동소수점 계산 결과가 같은 경우도 동점으로 처리합니다.

90%, 95%, 99%에서는 각각 상위 9,000·9,500·9,900개를 수용합니다.
일반적인 N에서는 `max(1, floor(N * coverage))`개를 사용하고 실제 개수를 JSON에 기록합니다.
전체 곡선은 `k=1,...,N`, 수용률 `k/N`에서 계산합니다. AURC는 이 오류율들의 **이산 평균**이며
사다리꼴 적분을 사용하지 않습니다. 세 행의 전체 정확도와 100% 수용률의 오류율은 같습니다.

이 실험은 테스트 점수의 순위를 비교합니다. 실제 서비스 임계값은 별도의 validation에서 정하고
테스트에 그대로 적용해야 합니다. 그래프의 무작위 거절 기준선은 전체 오류율입니다.

## 생성 파일

| 파일 | 내용 |
| --- | --- |
| `logits.pt` | 원래 logits `[10000, 11]`, 정답, 테스트 샘플 인덱스, 추론 메타데이터 |
| `scores.npz` | 동일한 클래스 예측·오답 여부, 세 점수, 점수별 내림차순 인덱스 |
| `curves.npz` | 수용 개수, 수용률, 세 점수의 동점을 반영한 기대 오류율 |
| `metrics.csv` / `metrics.json` | 전체 정확도, 오류율 @90·95·99%, AURC (모두 0~1 비율) |
| `results.md` | 정확도·오류율을 %로 표시한 결과 표, AURC는 0~1 비율 |
| `risk_coverage.png` / `risk_coverage.pdf` | 전체 곡선 및 90~100% 수용률 확대 그래프 |
| `score_definitions.png` / `.pdf` | Drainage·MSP·Logit margin 계산 방식 |
| `metrics_comparison.png` / `.pdf` | 전체 결과 표, 수용률별 오류율·AURC 막대그래프 |
| `score_distributions.png` / `.pdf` | 정답·오답 그룹별 세 점수의 분포 |
| `sorted_scores.png` / `.pdf` | 점수별로 정렬한 전체 10,000개의 정답·오답 표시 |
| `overview.png` / `.pdf` | 점수 정의·결과 표·분포·Risk–coverage를 합친 요약 그림 |
| `visualizations.md` | 모든 그림을 볼 수 있는 목차와 한국어 설명 |

시각화는 `metrics.json`, `curves.npz`, `scores.npz`를 읽어서 생성합니다.
`./venv/bin/python observe/visualize.py` 한 번으로 위의 모든 PNG·PDF와 목차를 갱신합니다.
점수 분포는 정답·오답 그룹을 각각 100%로 정규화하므로 그룹 크기의 영향을 분리해서 볼 수 있습니다.
정렬 산점도에는 표본 추출 없이 모든 평가 이미지를 표시합니다.

`scores.npz`의 `*_order`는 점수 순서를 살펴보기 위한 인덱스입니다. 동점 내 개별 순서에는
의미가 없으며, 그 순서로 오답을 단순 누적하면 동점의 기대 오류율과 달라질 수 있습니다.
평가에는 `curves.npz`에 저장한 기대 오류율을 사용하세요.

## MSP·Drainage 조건부 비교

```bash
./venv/bin/python observe/conditional_analysis.py
```

기준 점수를 10개 분위 구간으로 나누고 각 구간에서 다른 점수의 하위·상위 25% 오류율을 비교합니다.
MSP를 기준으로 Drainage를 비교하는 방향과, Drainage를 기준으로 MSP를 비교하는 방향을 모두 계산합니다.
저장된 `scores.npz`를 사용하며 결과는 `observe/results/noise_0.45/conditional`에 생성됩니다.

- `conditional_results.md`: 한국어 설명, 전체 요약, 구간별 표, 그림.
- `conditional_comparison.png` / `.pdf`: 하위·상위 오류율 및 차이, 95% bootstrap 구간.
- `control_balance.png` / `.pdf`: 두 그룹에 남아 있는 기준 점수 평균 차이.
- `conditional_bins.csv`, `conditional_summary.csv`, `conditional_results.json`: 수치 및 설정.
- `conditional_groups.npz`: 이미지별 구간과 하위·상위 그룹의 기대 포함 가중치.

`--bins`, `--tail-fraction`, `--bootstrap`, `--seed`, `--output-dir`로 설정을 바꿀 수 있습니다.
기본 bootstrap은 2,000회이며, 기준 구간 경계는 고정하고 각 반복에서 비교 그룹을 다시 나눕니다.
동점은 기대 포함 가중치로 계산합니다. 전체 차이는 같은 구간 가중치를 적용한 하위 오류율−상위 오류율입니다.
같은 구간 안에서도 기준 점수가 차이날 수 있으므로 `control_balance`를 함께 확인해야 합니다.

## 계산 검증

```bash
./venv/bin/python -m unittest observe.test_metrics -v
./venv/bin/python -m unittest observe.test_conditional_metrics -v
```

동점의 모든 순열에 대한 평균과 구현 결과를 비교하고, 수용률 경계에서의 동점 처리,
클래스 예측 고정, 점수 정의, 이산 AURC를 검사합니다.
