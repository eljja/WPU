# PyBullet Shift Generalization and Calibration

이 benchmark는 nominal PyBullet cup dynamics에서 학습한 뒤 held-out mechanism
family에서 평가한다. Cross-generator-family generalization과 calibration metric을
first-class output으로 추가한다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization.csv`

## 프로토콜

- Train mechanism: `nominal`.
- Eval mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Model: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seed: `11, 13`.
- Background objects: `32`.
- Training steps: `20`.
- Eval samples: seed/mechanism마다 `36`.
- Calibration metrics: ECE, Brier score, NLL.

## 요약

| eval mechanism | model | accuracy | ECE | Brier | NLL | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | wpu-cws-indexed-sparse | 0.486111 | 0.216639 | 0.638486 | 1.044608 | 4.3625 |
| nominal | wpu-cws-indexed-local-dense | 0.416666 | 0.221161 | 0.666338 | 1.065497 | 4.3625 |
| nominal | graph-transformer | 0.361111 | 0.217941 | 0.719271 | 1.137500 | 4.3625 |
| nominal | serialized-token | 0.402778 | 0.100940 | 0.630391 | 1.009570 | 4.3625 |
| high_force | wpu-cws-indexed-sparse | 0.444445 | 0.110487 | 0.643738 | 1.063413 | 4.3625 |
| high_force | wpu-cws-indexed-local-dense | 0.444445 | 0.369688 | 0.804497 | 1.376495 | 4.3625 |
| high_force | graph-transformer | 0.430555 | 0.393302 | 0.846081 | 1.453774 | 4.3625 |
| high_force | serialized-token | 0.458334 | 0.188049 | 0.686918 | 1.124732 | 4.3625 |
| edge_shift | wpu-cws-indexed-sparse | 0.597222 | 0.171235 | 0.624835 | 1.037711 | 4.3625 |
| edge_shift | wpu-cws-indexed-local-dense | 0.527778 | 0.168204 | 0.626533 | 1.051272 | 4.3625 |
| edge_shift | graph-transformer | 0.472222 | 0.195111 | 0.667463 | 1.130049 | 4.3625 |
| edge_shift | serialized-token | 0.472222 | 0.119432 | 0.644900 | 1.072424 | 4.3625 |
| catch_heavy | wpu-cws-indexed-sparse | 0.194445 | 0.248698 | 0.686505 | 1.106100 | 4.8125 |
| catch_heavy | wpu-cws-indexed-local-dense | 0.277778 | 0.383692 | 0.779896 | 1.207206 | 4.8125 |
| catch_heavy | graph-transformer | 0.361112 | 0.355464 | 0.789159 | 1.211036 | 4.8125 |
| catch_heavy | serialized-token | 0.402778 | 0.198037 | 0.637309 | 1.004722 | 4.8125 |

## 해석

결과는 mixed지만 유용하다.

Positive regime은 `edge_shift`다. Sparse WPU accuracy는 `0.597222`로 local-dense
WPU, graph, serialized-token baseline보다 높다. Event-local object graph가 식별 가능하고
mechanism shift가 여전히 local한 경우 WPU premise와 맞는 결과다.

Negative regime은 `catch_heavy`다. WPU sparse는 `0.194445`까지 떨어지고,
serialized-token baseline은 `0.402778`을 기록한다. 현재 WPU의 state/retrieval/branch
head가 바뀐 catch-action prior를 충분히 사용하지 못한다는 뜻이다. 이 결과는 숨기면 안
된다. WPU claim에 필요한 regime boundary다.

Calibration도 아직 해결되지 않았다. ECE는 model과 mechanism에 따라 크게 달라진다.
Sparse WPU는 `high_force`에서 낮은 ECE(`0.110487`)를 보이지만 `catch_heavy`에서는
정확도가 낮다. Serialized-token은 이 짧은 pilot에서 더 잘 calibrate된 경우가 많다.
따라서 향후 WPU claim은 accuracy와 calibration을 함께 보고해야 한다.

## 결과

Priority 4와 5는 이제 instrumented 상태지만 해결된 것은 아니다.

- Mechanism-family shift를 통한 cross-generator evaluation이 생겼다.
- ECE, Brier, NLL이 first-class output이 되었다.
- Robust world-state generalization을 주장하려면 mechanism-aware branch prior,
  uncertainty-gated fallback, shift-aware calibration이 더 필요하다.
