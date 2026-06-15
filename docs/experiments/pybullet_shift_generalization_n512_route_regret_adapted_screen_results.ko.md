# PyBullet N_bg=512 Route-Regret With Mechanism-Prior Adaptation Screen

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n512_route_regret_adapted_screen.csv`

이 follow-up은 selected route-regret WPU가 baselines와 같은 mechanism-prior adaptation
interface를 받을 때 개선되는지 확인한다. 이는 adapted setting이며 zero-shot mechanism
generalization은 아니다.

## Protocol

- 전체 objects: `N=517`.
- Train mechanism: `nominal`.
- Eval mechanisms: `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- Seeds: `11`, `13`, `17`.
- Models: selected route-regret WPU, graph transformer, serialized token.
- Mechanism-prior strengths: `0.0`, `0.25`, `0.5`, `0.75`, `1.0`.
- Selection metric: mechanism-specific calibration sample에서 `nll_ece`.

## Aggregate Result

| model | macro accuracy | macro ECE | macro Brier | dense compute ratio |
|---|---:|---:|---:|---:|
| graph-transformer | 0.527778 | 0.337050 | 0.601170 | 1.000000 |
| serialized-token | 0.416667 | 0.275198 | 0.658986 | 1.000000 |
| wpu-cws-indexed-physics-regret-hybrid | 0.312500 | 0.274279 | 0.663490 | 0.097222 |

4개 shifted mechanism에서 selected route-regret WPU 대 best non-WPU는 `0 / 0 / 4`
win/tie/loss다. Dense compute는 낮게 유지하지만 accuracy는 유지하지 못한다.

## Per-Mechanism Boundary

| mechanism | route-regret WPU | best baseline | delta |
|---|---:|---:|---:|
| catch_heavy | 0.333333 | 0.944444 | -0.611111 |
| edge_shift | 0.194445 | 0.222222 | -0.027777 |
| high_force | 0.333333 | 0.500000 | -0.166667 |
| no_catch | 0.388889 | 0.527778 | -0.138889 |

## 해석

이는 negative adapted-screen result다. Mechanism-prior adaptation은 유용한
interface지만, 같은 adaptation evidence가 dense/token baseline에도 주어지면 selected
route-regret WPU를 구하지 못한다. 특히 `catch_heavy` gap은 branch-prior correction이
dense baseline에는 크게 도움이 될 수 있지만, sparse route-regret model에는
mechanism-aware propagation capacity가 부족함을 보여준다.

따라서 다음 P4/P5 단계는 또 다른 prior-bias 또는 threshold selection이 아니다. WPU
propagation model 자체를 바꿔야 한다. Mechanism state는 post-hoc route/output bias가
아니라 local transition dynamics, branch logits, uncertainty를 직접 condition해야 한다.
