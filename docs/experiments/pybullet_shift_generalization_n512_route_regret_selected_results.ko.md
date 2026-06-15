# PyBullet N_bg=512 Selected Route-Regret Shift Screen

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n512_route_regret_selected.csv`

이 실험은 large-state PyBullet mechanism-shift screen에 validation-selected
route-regret WPU를 추가한 것이다. Explicit sparse/dense counterfactual route
supervision과 threshold selection이 large-`N`, small-`K` mechanism-shift regime을
개선하는지 확인한다.

## Protocol

- 전체 object 수: `N=517` (`N_bg=512` plus cup, table, hand, edge, goal).
- Train mechanism: `nominal`.
- Eval mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`, `edge_high_force`, `edge_catch_heavy`.
- Seeds: `11`, `13`, `17`.
- 비교 모델: sparse WPU, local-dense WPU, selected route-regret WPU, graph
  transformer, serialized token.
- Route-regret threshold 후보: `-0.75`, `-0.5`, `-0.25`, `0.0`, `0.25`.
  Held-out nominal validation split에서 compute-adjusted NLL로 선택했다.

## Aggregate Result

| model | macro accuracy | macro ECE | dense compute ratio |
|---|---:|---:|---:|
| graph-transformer | 0.508929 | 0.244585 | 1.000000 |
| serialized-token | 0.425595 | 0.268479 | 1.000000 |
| wpu-cws-indexed-local-dense | 0.377976 | 0.186405 | 1.000000 |
| wpu-cws-indexed-physics-regret-hybrid | 0.366071 | 0.243926 | 0.071429 |
| wpu-cws-indexed-sparse | 0.351190 | 0.209293 | 0.000000 |

7개 mechanism에서 best WPU 대 best non-WPU는 `2 / 1 / 4` win/tie/loss다.
Selected route-regret WPU는 dense recompute를 적게 사용하지만(`0.071429`), macro
accuracy에서 best baseline을 넘지 못한다.

## Per-Mechanism Boundary

| mechanism | best WPU | best baseline | selected route-regret WPU |
|---|---:|---:|---:|
| catch_heavy | 0.500000 | 0.645833 | 0.041667 |
| edge_catch_heavy | 0.375000 | 0.520833 | 0.187500 |
| edge_high_force | 0.458333 | 0.458333 | 0.458333 |
| edge_shift | 0.437500 | 0.416667 | 0.375000 |
| high_force | 0.354167 | 0.562500 | 0.354167 |
| no_catch | 0.770833 | 0.520833 | 0.770833 |
| nominal | 0.500000 | 0.604167 | 0.375000 |

## 해석

이 결과는 유용한 negative/mixed result다. Validation-selected route-regret routing은
mechanism-shift generalization을 해결하지 못한다. `no_catch`에서는 강하고
`edge_high_force`에서는 baseline과 동률이지만, `catch_heavy`에서 크게 실패하며
graph-transformer macro accuracy를 회복하지 못한다.

핵심 결론은 구조적이다. Route-regret threshold selection만으로는 부족하다. 다음 P4/P5
단계는 sparse/dense route selector만 더 조정하는 것이 아니라 mechanism-aware
propagation 또는 adaptation을 학습해야 한다. 특히 `catch_heavy`는 local working-set
route만으로 처리하기 어려운 branch-prior/mechanism law 변화의 강한 반례로 남는다.
