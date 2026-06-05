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
- Seed: `11, 13, 17, 19, 23`.
- Background objects: `32`.
- Training steps: `20`.
- Eval samples: seed/mechanism마다 `36`.
- Calibration metrics: ECE, Brier score, NLL.

## 요약

| eval mechanism | model | accuracy | ECE | Brier | NLL | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | wpu-cws-indexed-sparse | 0.427778 | 0.124159 | 0.636457 | 1.046428 | 4.3600 |
| nominal | wpu-cws-indexed-local-dense | 0.394444 | 0.200036 | 0.658903 | 1.054336 | 4.3600 |
| nominal | graph-transformer | 0.411111 | 0.221604 | 0.675927 | 1.077000 | 4.3600 |
| nominal | serialized-token | 0.455556 | 0.140774 | 0.628618 | 1.012293 | 4.3600 |
| high_force | wpu-cws-indexed-sparse | 0.416667 | 0.112684 | 0.638699 | 1.055790 | 4.3600 |
| high_force | wpu-cws-indexed-local-dense | 0.416667 | 0.342607 | 0.777555 | 1.295899 | 4.3600 |
| high_force | graph-transformer | 0.416667 | 0.382426 | 0.828363 | 1.420038 | 4.3600 |
| high_force | serialized-token | 0.433333 | 0.232135 | 0.703167 | 1.154525 | 4.3600 |
| edge_shift | wpu-cws-indexed-sparse | 0.522222 | 0.178233 | 0.622615 | 1.033803 | 4.3600 |
| edge_shift | wpu-cws-indexed-local-dense | 0.477778 | 0.185243 | 0.603099 | 0.999774 | 4.3600 |
| edge_shift | graph-transformer | 0.522222 | 0.224137 | 0.614147 | 1.033391 | 4.3600 |
| edge_shift | serialized-token | 0.555555 | 0.167665 | 0.603833 | 0.997661 | 4.3600 |
| catch_heavy | wpu-cws-indexed-sparse | 0.300000 | 0.252880 | 0.662538 | 1.078916 | 4.8050 |
| catch_heavy | wpu-cws-indexed-local-dense | 0.366667 | 0.313698 | 0.728429 | 1.135705 | 4.8050 |
| catch_heavy | graph-transformer | 0.322222 | 0.355048 | 0.763415 | 1.185473 | 4.8050 |
| catch_heavy | serialized-token | 0.327778 | 0.229288 | 0.670691 | 1.058393 | 4.8050 |

## 해석

결과는 mixed지만 유용하다.

5-seed 결과는 regime boundary를 바꾼다. WPU는 `catch_heavy`에서 local-dense path로
`0.366667` accuracy를 기록해 best non-WPU `0.327778`보다 높다. 하지만 `edge_shift`와
`high_force`에서는 baseline에 밀린다. 따라서 2-seed에서 보였던 `edge_shift` 우위는
강한 shift claim으로 쓰기에는 안정적이지 않다.

Calibration은 aggregate 기준으로 개선됐다. Dashboard 기준 평균 WPU ECE는 평균 baseline
ECE보다 낮아졌다. 하지만 accuracy가 여전히 mixed이고 multi-step rollout calibration이
아니므로 calibration 문제가 해결됐다고 주장하면 안 된다.

## 결과

Priority 4와 5는 이제 instrumented 상태지만 해결된 것은 아니다.

- Mechanism-family shift를 통한 cross-generator evaluation이 생겼다.
- ECE, Brier, NLL이 first-class output이 되었다.
- Robust world-state generalization을 주장하려면 mechanism-aware branch prior,
  uncertainty-gated fallback, shift-aware calibration이 더 필요하다.
