# PyBullet Closed-Loop Rollout

이 실험은 one-step PyBullet state prediction을 반복적으로 `WorldState`에 다시
적용했을 때 state가 안정적으로 유지되는지 보는 진단이다. Simulator와 매 step
재동기화하는 physics benchmark가 아니라 state-integrity diagnostic이다.

Source CSVs:

- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_closed_loop_rollout_clipped.csv`

## 프로토콜

- Simulator: 초기 objectified state 생성을 위한 PyBullet `DIRECT` rigid-body rollout.
- Training: clean one-step PyBullet cup samples.
- Rollout: predicted object delta를 반복적으로 `WorldState`에 적용.
- Horizon: `5, 10, 25`.
- Model: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`.
- Seed: `11, 13`.
- Background object: `32`.
- Metric: branch flip rate, branch entropy, raw predicted delta norm,
  constraint violations per step, selected K.

## Unclipped Rollout 요약

| horizon | model | branch flip | violations/step | entropy | raw delta norm | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | graph-transformer | 0.063 | 0.046 | 0.858 | 4.163 | 36.417 |
| 5 | wpu-cws-indexed-local-dense | 0.146 | 0.067 | 0.922 | 2.789 | 4.417 |
| 5 | wpu-cws-indexed-sparse | 0.125 | 0.200 | 1.061 | 0.799 | 4.417 |
| 10 | graph-transformer | 0.151 | 0.140 | 0.891 | 6.107 | 36.417 |
| 10 | wpu-cws-indexed-local-dense | 0.148 | 0.204 | 0.951 | 2.649 | 4.417 |
| 10 | wpu-cws-indexed-sparse | 0.204 | 0.531 | 1.026 | 3.534 | 4.417 |
| 25 | graph-transformer | 0.056 | 0.273 | 0.890 | 8.410 | 36.417 |
| 25 | wpu-cws-indexed-local-dense | 0.056 | 0.499 | 0.881 | 2.745 | 4.417 |
| 25 | wpu-cws-indexed-sparse | 0.076 | 3.374 | 0.423 | 1,958,877.608 | 4.417 |

## Delta-Clipped H=25 요약

두 번째 run은 `WorldState`를 업데이트하기 전에 object별 delta-vector norm을 `0.25`로
clip한다.

| model | unclipped violations/step | clipped violations/step | clipped branch flip | raw delta norm after clipping run |
| --- | ---: | ---: | ---: | ---: |
| graph-transformer | 0.273 | 0.253 | 0.055 | 8.364 |
| wpu-cws-indexed-local-dense | 0.499 | 0.314 | 0.049 | 2.810 |
| wpu-cws-indexed-sparse | 3.374 | 0.785 | 0.083 | 1,939,290.234 |

## 해석

이 결과는 WPU에 불리하지만 중요하다. One-step WPU sparse가 그럴듯해 보여도,
predicted delta를 반복 적용하면 horizon 25에서 catastrophic state explosion이
발생할 수 있다. Local-dense WPU는 이 진단에서 sparse WPU보다 안정적이지만,
unclipped run에서는 graph baseline보다 constraint violation이 많다.

Delta clipping은 violation을 줄인다. 특히 WPU sparse에서 효과가 크다. 하지만
근본적인 model instability를 해결하지는 않는다. Clipped run에서도 raw predicted
delta norm은 여전히 매우 크다. Clamp는 state update 직전에만 적용되기 때문이다.
따라서 clipping은 safety layer이지 학습된 long-horizon solution이 아니다.

## 설계 결론

WPU에는 명시적 state-integrity loop가 필요하다.

```text
predict delta -> verify constraints -> clip or reject unsafe delta
-> expand K or run local dense -> update branch uncertainty
```

이 결과는 WPU를 one-step branch accuracy만으로 평가하면 안 된다는 v2 방향을
강화한다. State integrity, rollout drift, uncertainty, correction/escalation
frequency를 함께 보고해야 한다.

## 다음 단계

- one-step branch/delta loss뿐 아니라 rollout consistency loss로 학습한다.
- 같은 one-step event를 반복하지 말고 branch-specific delta trajectory를 만든다.
- constraint-aware delta head 또는 verifier-guided rejection을 추가한다.
- 일부 rollout state를 PyBullet과 재동기화해 model drift를 simulator ground truth와
  비교한다.
