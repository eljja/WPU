# PyBullet Mixture 학습 Shift/Calibration Probe

이 probe는 7-seed nominal-only shift benchmark를 보완한다. 네 가지 mechanism
family 전체를 섞어 학습하고, post-hoc temperature calibration을 적용한 뒤 같은
mechanism들을 평가한다. 현재 iteration에서는 full 7-seed calibrated mixture run이
너무 무거웠기 때문에 3-seed probe로 명시해 보고한다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_mixture_calibrated.csv`

## Protocol

- Train mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Eval mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seeds: `11, 13, 17`.
- Background objects: `32`.
- Training steps: `16`.
- Eval samples: seed/mechanism마다 `36`.
- Calibration: training-family held-out sample에서 scalar temperature를 post-hoc fitting.

## 요약

| eval mechanism | best WPU accuracy | best baseline accuracy | best WPU ECE | best baseline ECE | 해석 |
|---|---:|---:|---:|---:|---|
| nominal | 0.472222 | 0.407407 | 0.107388 | 0.143710 | 이 작은 probe에서는 WPU가 nominal accuracy와 calibration을 개선한다. |
| high_force | 0.444444 | 0.444444 | 0.204216 | 0.182570 | Accuracy는 동률이지만 baseline calibration이 더 좋다. |
| edge_shift | 0.546297 | 0.388889 | 0.157736 | 0.093222 | WPU local-dense accuracy는 좋아지지만 graph-transformer calibration이 더 좋다. |
| catch_heavy | 0.333333 | 0.481481 | 0.295547 | 0.202056 | Serialized-token baseline이 명확히 더 강하다. |

Mean WPU ECE는 `0.208404`, non-WPU baseline ECE는 `0.183805`이므로 calibrated-mixture
ECE ratio는 `1.133834`다. 이는 7-seed nominal-only benchmark에서 보였던 약한 ECE
advantage를 뒤집는 결과다.

## 해석

Mixture training은 보편적 해결책이 아니다. 명시적 object geometry와 local relation
propagation이 유용한 `edge_shift`에서는 WPU가 좋아지지만, branch prior 자체가 크게
바뀌는 `catch_heavy`는 해결하지 못한다. Calibration 역시 post-hoc temperature만으로
해결되지 않는다. WPU는 일부 mechanism에서 accuracy를 얻으면서도 dense/token baseline보다
덜 calibrated될 수 있다.

다음 P4/P5 개선은 threshold tuning이 아니라 mechanism-aware branch prior,
uncertainty-gated fallback, multi-step calibration loss를 WPU branch/rollout model 내부에
넣는 방향이어야 한다.
