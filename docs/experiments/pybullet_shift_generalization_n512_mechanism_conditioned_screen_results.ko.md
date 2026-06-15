# N_bg=512 Mechanism-Conditioned Propagation Screen

이 screen은 selected route-regret와 matched mechanism-prior adaptation이 실패한 뒤의
다음 가설을 검증한다. 핵심 가설은 mechanism-relevant object/event state가 route
selector나 post-hoc branch prior가 아니라 local propagation dynamics 자체를 condition해야
한다는 것이다.

Source CSV:
`docs/experiments/pybullet_shift_generalization_n512_mechanism_conditioned_screen.csv`

## Protocol

- Training mechanism: `nominal`
- Evaluation mechanisms: `high_force`, `edge_shift`, `catch_heavy`, `no_catch`
- Seeds: `11`, `13`, `17`
- Background objects: `512`
- Total objects: `517`
- Samples per mechanism/seed: `12`
- Training steps: `4`
- Simulation steps: `80`
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-mechanism-conditioned`,
  `graph-transformer`, `serialized-token`
- Mechanism prior adaptation: 사용하지 않음
- Mechanism-conditioned WPU의 dense fallback: 사용하지 않음

## Aggregate Result

| Model | Mean branch accuracy | Mean ECE | Mean dense compute |
|---|---:|---:|---:|
| `graph-transformer` | 0.500000 | 0.242072 | 1.000000 |
| `serialized-token` | 0.430555 | 0.242987 | 1.000000 |
| `wpu-cws-indexed-mechanism-conditioned` | 0.541667 | 0.203094 | 0.000000 |
| `wpu-cws-indexed-sparse` | 0.500000 | 0.158891 | 0.000000 |

Mechanism-conditioned WPU는 이 screen에서 macro accuracy가 가장 높고 dense recompute를
쓰지 않는다. Sparse WPU 대비 macro accuracy도 `+0.041667` 개선한다. 다만 이는 작은
screen이므로 최종 mechanism-shift evidence로 보면 안 된다.

## Per-Mechanism Boundary

| Mechanism | Mechanism-conditioned WPU | Best baseline | Delta | 해석 |
|---|---:|---:|---:|---|
| `catch_heavy` | 0.777778 | 0.638889 | +0.138889 | 명확한 positive screen이다. Mechanism context가 prior-shift branch에 도움을 준다. |
| `edge_shift` | 0.361111 | 0.444444 | -0.083333 | 실패 경계다. Geometry/edge-law shift는 아직 해결되지 않았다. |
| `high_force` | 0.500000 | 0.500000 | +0.000000 | 동률이다. |
| `no_catch` | 0.527778 | 0.527778 | +0.000000 | 동률이다. |

Best non-WPU baseline 대비 win/tie/loss는 `1/2/1`이다. 이 결과는 route-regret adapted
screen의 `0/0/4`보다 낫지만, large-N zero-shot mechanism generalization을 주장하기에는
부족하다.

## Interpretation

이 결과는 현재 연구 방향을 바꾼다. Negative route-regret adaptation screen은 threshold
routing이나 post-hoc prior가 missing mechanism이 아님을 시사했다. 이번 screen은 missing
mechanism이 transition function 자체일 가능성을 높인다. 즉 local propagation이 force,
action, target physical scalar, selected-set physical scalar, pair geometry 같은 explicit
mechanism context를 받아야 한다.

한계도 명확하다. 이 실험은 3 seeds, 4 shifts의 작은 screen이고, `edge_shift`는 여전히
serialized-token baseline보다 낮다. 이 CSV 하나로 latency, power, broad superiority를
주장할 수 없다. 방어 가능한 주장은 mechanism-conditioned propagation이 WPU v2의 유망한
다음 방향이라는 것이다.

## Next Experiment

다음 실험은 같은 model family를 더 많은 seed, 전체 mechanism, 최소 하나의 더 큰
background regime으로 확장해야 한다. 같은 training budget에서 mechanism-conditioned sparse
propagation, local-dense, selected route-regret, adapted route-regret, graph-transformer,
serialized-token baseline을 비교하고 accuracy, ECE, Brier, NLL, dense compute, selected
`K`, latency, route decision을 함께 보고해야 한다.
