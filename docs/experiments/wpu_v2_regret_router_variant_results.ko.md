# WPU V2 Regret Router Variant Results

Source CSVs:

- `docs/experiments/wpu_v2_regret_router_variant_paired.csv`
- `docs/experiments/wpu_v2_regret_router_variant_summary.csv`

이 실험은 staged regret router의 입력을 더 state-native하게 만들면 routing 품질이
개선되는지 검증한다. CSV는 `catch_action`과 physical object-state scalar를
physics/state regret head에 전달하는 route-state contract 수정 이후 재생성했다.

## 질문

이전 post-hoc probe에서는 physical state feature가 raw sparse diagnostic보다
held-out seed에서 더 잘 일반화되는 신호를 보였다. 하지만 같은 feature를 WPU 모델 내부
router에 넣었을 때도 도움이 되는지는 별도 검증이 필요하다. 이 실험은 세 internal router
variant를 비교한다.

- `internal`: sparse hidden summary, K pressure, selector confidence,
  interaction density를 쓰는 기존 staged regret head.
- `physics_hidden`: sparse hidden summary에 pair geometry, target physical scalar,
  selected-set physical scalar, `force`, `catch_action`을 더한 router.
- `state_only`: hidden summary 없이 K pressure, interaction density, pair geometry,
  target/selected physical scalar, `force`, `catch_action`만 쓰는 router.

## 프로토콜

- Dataset: pairwise CWS synthetic physics.
- `N = 2048`.
- `K in {8, 16, 32}`.
- Seeds: `11, 13, 17, 19, 23`.
- Model size: hidden dim `128`, one local dense layer, four heads.
- Training: 40 propagation steps, then 80 route-regret steps.
- Evaluation: 90 held-out samples per condition.
- Compute cost: `0.05`.

Artifacts:

- `docs/experiments/wpu_v2_staged_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_physics_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_state_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_regret_router_variant_summary.csv`
- `docs/experiments/wpu_v2_regret_router_variant_paired.csv`
- `scripts/compare_regret_router_variants.py`

## Aggregate Results

| router | routed loss | delta vs sparse | oracle excess | dense compute | regret corr | routed accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal | 0.963 | -0.025 | 0.045 | 0.237 | 0.351 | 0.493 |
| physics hidden | 0.963 | -0.022 | 0.049 | 0.240 | 0.332 | 0.503 |
| state only | 0.983 | 0.001 | 0.074 | 0.297 | 0.161 | 0.495 |

K-specific routed loss:

| router | K=8 | K=16 | K=32 |
| --- | ---: | ---: | ---: |
| internal | 0.960 | 0.949 | 0.979 |
| physics hidden | 0.958 | 0.942 | 0.989 |
| state only | 0.981 | 0.971 | 0.995 |

## 해석

`state_only` internal route head는 현재 구조에서는 기각된다. Hidden summary를 너무
강하게 제거해 regret correlation이 `0.351`에서 `0.161`로 떨어지고, routed loss는
sparse-only보다 약간 나빠지며 oracle excess는 `0.045`에서 `0.074`로 증가한다.

확장된 physical/action feature context를 hidden-summary router에 추가하는 것은 대체로
neutral하다. Routed accuracy는 약간 좋아지지만 overall loss는 거의 같고 `K=32`에서는
더 나쁘다. 즉 route-state contract 수정은 correctness를 위해 필요했지만, 그 자체로
P1을 닫지는 않는다. Post-hoc physical-state 신호는 단순 concatenation만으로 내부
router에 전이되지 않는다.

현재 deployed router 중 가장 나은 것은 여전히 기존 staged internal regret router다.
최종 구조는 아니지만 bounded dense compute에서 안정적으로 loss를 낮춘 유일한 internal
route head다.

## V2에 대한 결론

다음 scheduler는 hand-chosen state scalar 위의 단순 MLP가 아니어야 한다. 더 유망한
방향은 structured verifier다.

- Hidden local propagation evidence로 sparse execution이 충분한지 추정한다.
- Explicit physical state constraint로 그 추정을 veto 또는 verify한다.
- 두 신호가 충돌하면 K expansion 또는 uncertainty increase를 trigger한다.
- Diagnostic residual 또는 단일 deployed threshold는 cross-seed calibration이 확인되기
  전까지 직접 의존하지 않는다.

이는 WPU thesis와 일치한다. Router는 state-native여야 하지만, state signal은 작은
scalar vector가 아니라 structured constraint와 verification으로 표현되어야 한다. 다음
PyBullet mechanism-shift test에는 explicit route-regret training도 필요하다. 단순히
regret hybrid model을 PyBullet script에 넣으면 미학습 route head를 비교하게 된다.
