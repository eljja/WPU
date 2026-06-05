# PyBullet Local-Law Revision

이 실험은 local-law revision 아이디어를 synthetic hidden mechanism에서
PyBullet-derived objectified state로 옮긴 것이다. Hand impulse, edge distance,
hand distance, catch action, relation strength 같은 `WorldState` field로 cup
`delta_x`를 설명하는 단순하고 해석 가능한 후보 법칙을 맞춘다.

Source CSV:

- `docs/experiments/pybullet_local_law_revision.csv`

## 프로토콜

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Target: cup position delta의 x축.
- Training distribution: nominal PyBullet cup samples.
- Calibration: mechanism별 small calibration set.
- Evaluation mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Seed: `11, 13`.
- Train/calibration/eval samples: `64/24/48`.
- Policy:
  - `base_law`: nominal sample에서 fit한 linear law.
  - `gain_calibrated_law`: base law에 scalar gain correction.
  - `form_revised_law`: calibration set에서 candidate form 선택.
  - `oracle_form_law`: test-set upper bound이며 deployable policy가 아님.

## 요약

| mechanism | policy | selected form | delta MSE | relative improvement | decision |
| --- | --- | --- | ---: | ---: | --- |
| nominal | base_law | base | 0.025154 | 0.000000 | baseline |
| nominal | gain_calibrated_law | gain_scaled_base | 0.026402 | 0.000000 | keep_base_or_collect_data |
| nominal | form_revised_law | edge_form | 0.035359 | 0.000000 | keep_base_or_collect_data |
| nominal | oracle_form_law | edge_form | 0.010712 | 0.574098 | accept_revision |
| high_force | base_law | base | 0.086944 | 0.000000 | baseline |
| high_force | gain_calibrated_law | gain_scaled_base | 0.077267 | 0.111118 | accept_revision |
| high_force | form_revised_law | catch_form | 0.037060 | 0.573785 | accept_revision |
| high_force | oracle_form_law | catch_form | 0.018673 | 0.785224 | accept_revision |
| edge_shift | base_law | base | 0.680045 | 0.000000 | baseline |
| edge_shift | gain_calibrated_law | gain_scaled_base | 0.641828 | 0.056191 | keep_base_or_collect_data |
| edge_shift | form_revised_law | catch_form | 0.563311 | 0.171645 | accept_revision |
| edge_shift | oracle_form_law | edge_form | 0.086721 | 0.872477 | accept_revision |
| catch_heavy | base_law | base | 0.012977 | 0.000000 | baseline |
| catch_heavy | gain_calibrated_law | gain_scaled_base | 0.012939 | 0.003018 | keep_base_or_collect_data |
| catch_heavy | form_revised_law | quadratic_form | 0.016676 | 0.000000 | keep_base_or_collect_data |
| catch_heavy | oracle_form_law | catch_form | 0.009083 | 0.300059 | accept_revision |

## 해석

결과는 좁지만 중요한 주장을 지지한다. Objectified simulator state 위에서 local-law
revision은 작동 가능한 실험 단위이며, OOD residual은 revised form이 필요한 조건을
드러낼 수 있다. `high_force`와 `edge_shift`가 positive regime이다. 두 경우 모두
deployable revised form이 nominal base law보다 낮은 MSE를 냈다.

하지만 다음 병목도 분명하다. `edge_shift`에서 oracle form은 deployed form보다 훨씬
좋다. `nominal`과 `catch_heavy`에서는 revision이 과적합하거나 거의 개선하지 못했다.
따라서 WPU가 “물리 법칙을 발견했다”고 주장하면 안 된다. 현재 방어 가능한 주장은
다음이다.

```text
Objectified state는 local law hypothesis를 명시적이고 측정 가능하며 stress 가능하고
제한된 후보 family 안에서 revision 가능한 단위로 만든다.
```

## 발견한 문제

- 24개 calibration sample만으로 candidate form을 고르는 것은 불안정하다.
- `edge_shift`에서 deployable form과 test-set oracle form이 달라 mechanism-selection
  gap이 드러났다.
- 아직 unknown-law discovery가 아니다. Candidate form은 hand-provided다.
- Target은 하나의 scalar cup delta이며 full rigid-body state evolution은 아니다.

## 다음 단계

- calibration MSE 최저값이 아니라 validation split 또는 risk-adjusted law-form
  selection을 사용한다.
- position, velocity, branch probability를 함께 예측하는 multi-output law로 확장한다.
- law residual을 closed-loop verifier와 연결해 unsafe delta가 law revision 또는
  local dense fallback을 trigger하게 한다.
- hand-provided form 대신 objectified variable 위의 generated candidate descriptor나
  symbolic regression을 도입한다.
