# N_bg=512 Shuffled Multi-Mechanism Factorized Adapter Results

이 보고서는 이전 multi-mechanism screen의 중요한 training-protocol 문제를 수정한다.
Training dataset은 `ConcatDataset`으로 만들어졌지만 DataLoader가 shuffle을 하지 않았다.
작은 step budget에서는 mechanism을 균등하게 보는 것이 아니라 dataset 순서대로 볼 수
있었다. 이제 training DataLoader는 seed-fixed shuffle을 사용한다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv`

## Model Change

새 `wpu-cws-indexed-mechanism-factorized` 모델은 sparse execution을 유지하면서
object-wise residual adapter를 factorized mechanism adapter로 바꾼다. Route physics
context를 encode하고, 각 selected object embedding에 대한 per-object scale/shift update를
만든다.

이는 여전히 state-native propagation model이다. World를 token sequence로 직렬화하지 않고,
dense fallback도 쓰지 않는다.

## Corrected 5-Seed Result

| Model | Mean branch accuracy | Mean ECE | Mean Brier | Mean NLL | Mean dense compute |
|---|---:|---:|---:|---:|---:|
| `graph-transformer` | 0.548571 | 0.254194 | 0.581732 | 0.960382 | 1.000000 |
| `serialized-token` | 0.394286 | 0.256186 | 0.638318 | 1.050593 | 1.000000 |
| `wpu-cws-indexed-mechanism-factorized` | 0.497143 | 0.256011 | 0.639074 | 1.056679 | 0.000000 |

3-seed shuffled screen에서는 factorized adapter가 좋아 보였지만, 5-seed에서는 유지되지
않는다. Graph-transformer가 macro accuracy, Brier, NLL에서 더 좋다. WPU는 dense compute
`0.000000`을 유지하지만, 이는 compute property이지 accuracy win이 아니다.

## Per-Mechanism Boundary

| Mechanism | Factorized WPU | Best baseline | Delta | 경계 |
|---|---:|---:|---:|---|
| `catch_heavy` | 0.720000 | 0.720000 | +0.000000 | 동률이다. |
| `edge_catch_heavy` | 0.270000 | 0.450000 | -0.180000 | 실패다. Edge+catch composition이 약하다. |
| `edge_high_force` | 0.370000 | 0.570000 | -0.200000 | 실패다. Edge+force composition이 약하다. |
| `edge_shift` | 0.420000 | 0.540000 | -0.120000 | 실패다. Edge law가 여전히 약하다. |
| `high_force` | 0.440000 | 0.520000 | -0.080000 | 실패다. |
| `no_catch` | 0.590000 | 0.470000 | +0.120000 | positive다. |
| `nominal` | 0.670000 | 0.570000 | +0.100000 | positive다. |

Win/tie/loss는 `2/1/4`, mean margin은 `-0.051429`다.

## Interpretation

이는 유용한 negative result다. 이전 optimistic multi-mechanism 결과를 보정하며,
mechanism-aware sparse adapter만으로는 robust composition generalization이 아직 충분하지
않음을 보여준다. 주요 실패는 여전히 edge-conditioned family, 특히 composed edge
mechanism이다.

현재 WPU v2 방향은 유효하지만, 주장은 더 좁아져야 한다.

- `K`가 작고 indexed될 때 WPU는 sparse compute advantage를 가진다.
- WPU는 mechanism state를 local object propagation에 주입할 수 있다.
- 그러나 현재 small training budget에서는 held-out edge mechanism에 대한 robust
  local-law composition을 아직 학습하지 못한다.

## Next Step

다음 실험은 branch-label supervision에만 의존하면 안 된다. Edge-distance auxiliary target,
force/edge/catch factor loss, 또는 한 번에 하나의 mechanism factor만 분리하는
simulator-derived counterfactual pair 같은 explicit local-law/composition supervision을
추가해야 한다.
