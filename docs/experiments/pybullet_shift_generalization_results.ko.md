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
- Seed: `11, 13, 17, 19, 23, 29, 31`.
- Background objects: `32`.
- Training steps: `20`.
- Eval samples: seed/mechanism마다 `36`.
- Calibration metrics: ECE, Brier score, NLL.

## 요약

| eval mechanism | model | accuracy | ECE | Brier | NLL | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | wpu-cws-indexed-sparse | 0.444445 | 0.118385 | 0.628308 | 1.033729 | 4.350000 |
| nominal | wpu-cws-indexed-local-dense | 0.388889 | 0.196291 | 0.646500 | 1.035757 | 4.350000 |
| nominal | graph-transformer | 0.444444 | 0.187391 | 0.648265 | 1.036947 | 4.350000 |
| nominal | serialized-token | 0.500000 | 0.131319 | 0.609825 | 0.982766 | 4.350000 |
| high_force | wpu-cws-indexed-sparse | 0.428571 | 0.146666 | 0.624832 | 1.036275 | 4.350000 |
| high_force | wpu-cws-indexed-local-dense | 0.432540 | 0.313820 | 0.736105 | 1.222798 | 4.350000 |
| high_force | graph-transformer | 0.452381 | 0.313074 | 0.758171 | 1.291259 | 4.350000 |
| high_force | serialized-token | 0.460318 | 0.210711 | 0.663026 | 1.092347 | 4.350000 |
| edge_shift | wpu-cws-indexed-sparse | 0.527778 | 0.212969 | 0.607690 | 1.012082 | 4.350000 |
| edge_shift | wpu-cws-indexed-local-dense | 0.456349 | 0.195355 | 0.598732 | 0.992563 | 4.350000 |
| edge_shift | graph-transformer | 0.531746 | 0.210478 | 0.604425 | 1.022356 | 4.350000 |
| edge_shift | serialized-token | 0.571428 | 0.187070 | 0.592148 | 0.987664 | 4.350000 |
| catch_heavy | wpu-cws-indexed-sparse | 0.321429 | 0.241991 | 0.655905 | 1.067589 | 4.803571 |
| catch_heavy | wpu-cws-indexed-local-dense | 0.408730 | 0.264469 | 0.688479 | 1.076357 | 4.803571 |
| catch_heavy | graph-transformer | 0.341270 | 0.294828 | 0.718105 | 1.111870 | 4.803571 |
| catch_heavy | serialized-token | 0.349206 | 0.219188 | 0.650279 | 1.020129 | 4.803571 |

## 해석

결과는 mixed지만 유용하다.

7-seed 결과는 같은 regime boundary를 유지한다. WPU는 `catch_heavy`에서 local-dense path로
`0.408730` accuracy를 기록해 best non-WPU `0.349206`보다 높다. 하지만 `edge_shift`와
`high_force`에서는 baseline에 밀린다. 따라서 2-seed에서 보였던 `edge_shift` 우위는
강한 shift claim으로 쓰기에는 안정적이지 않다.

Calibration은 aggregate 기준으로 약하게 유리하다. Dashboard 기준 평균 WPU ECE는 평균
baseline ECE보다 낮지만 7-seed 재실행 후 ratio는 `0.963449`로 약해졌다. Accuracy가
여전히 mixed이고 multi-step rollout calibration이 아니므로 calibration 문제가 해결됐다고
주장하면 안 된다.

## 결과

Priority 4와 5는 이제 instrumented 상태지만 해결된 것은 아니다.

- Mechanism-family shift를 통한 cross-generator evaluation이 생겼다.
- ECE, Brier, NLL이 first-class output이 되었다.
- Robust world-state generalization을 주장하려면 mechanism-aware branch prior,
  uncertainty-gated fallback, shift-aware calibration이 더 필요하다.
