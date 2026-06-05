# PyBullet Systems Profile

이 실험은 PyBullet-derived objectified state에서 WPU의 systems 측면을 계측한다.
모델을 학습하지 않고 accuracy도 측정하지 않는다. 목적은 full-state tensorization
비용, pre-tensor indexed WPU 비용, branch-overlay memory 비용을 분리하는 것이다.

Source CSV:

- `docs/experiments/pybullet_system_profile.csv`

## 프로토콜

- Simulator: PyBullet `DIRECT` cup scene.
- Sample: seed/background setting마다 `8`.
- Seed: `11, 13`.
- Background objects: `0, 32, 128, 512, 2048`.
- Branch counts: `1, 3, 8`.
- Indexed WPU query: event target과 relation frontier, `max_nodes=12`,
  `max_depth=1`.
- Metric:
  - `full_tensor_bytes`: 전체 objectified state batch를 tensorize한 byte.
  - `selected_tensor_bytes`: pre-tensor indexed projection 이후 tensor byte.
  - `tensor_byte_reduction`: full tensorization 대비 indexed WPU tensorization
    감소율.
  - `tensorize_latency_reduction`: 전체 state `StateGraphBatch` 구성 대비
    selected-state 구성의 실제 CPU latency 감소율.
  - `branch_memory_reduction`: `BaseState + branch delta`가 branch별 full state
    copy 대비 줄이는 memory proxy.
  - `work_proxy_reduction`: dense `N^2 * B` object work proxy 대비 selected
    `K * E_K * B` sparse work proxy 감소율.

## 요약

| background objects | branches | total objects | selected objects | tensor byte reduction | tensorize latency reduction | branch memory reduction | work proxy reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 4.562 | 4.562 | 0.000000 | 0.171420 | 0.000000 | 0.000000 |
| 0 | 3 | 4.562 | 4.562 | 0.000000 | 0.075183 | 0.269627 | 0.000000 |
| 0 | 8 | 4.562 | 4.562 | 0.000000 | 0.090003 | 0.477961 | 0.000000 |
| 32 | 1 | 36.562 | 4.562 | 0.859694 | 0.814191 | 0.000000 | 0.984326 |
| 32 | 3 | 36.562 | 4.562 | 0.859694 | 0.832416 | 0.617715 | 0.984326 |
| 32 | 8 | 36.562 | 4.562 | 0.859694 | 0.837179 | 0.826048 | 0.984326 |
| 128 | 1 | 132.562 | 4.562 | 0.960764 | 0.947560 | 0.000000 | 0.998804 |
| 128 | 3 | 132.562 | 4.562 | 0.960764 | 0.947610 | 0.653167 | 0.998804 |
| 128 | 8 | 132.562 | 4.562 | 0.960764 | 0.949431 | 0.861500 | 0.998804 |
| 512 | 1 | 516.562 | 4.562 | 0.989891 | 0.984421 | 0.000000 | 0.999921 |
| 512 | 3 | 516.562 | 4.562 | 0.989891 | 0.983728 | 0.663202 | 0.999921 |
| 512 | 8 | 516.562 | 4.562 | 0.989891 | 0.984730 | 0.871536 | 0.999921 |
| 2048 | 1 | 2052.562 | 4.562 | 0.997454 | 0.995549 | 0.000000 | 0.999995 |
| 2048 | 3 | 2052.562 | 4.562 | 0.997454 | 0.995649 | 0.665795 | 0.999995 |
| 2048 | 8 | 2052.562 | 4.562 | 0.997454 | 0.995509 | 0.874128 | 0.999995 |

## 해석

이 결과는 현재 WPU의 large-`N` systems premise를 가장 깨끗하게 보여주는
evidence다. Irrelevant background state가 `N≈4.6`에서 `N≈2052.6`으로 커져도,
pre-tensor indexed WPU path는 neural state를 `K≈4.6` 수준으로 유지한다. 그 결과
tensor-byte reduction은 `0.997454`, sparse object-work proxy reduction은
`0.999995`까지 올라간다. 실제 CPU tensorization latency reduction도 가장 큰
`N`의 B=1 조건에서 `0.995549`까지 도달해 byte proxy가 preprocessing time과 같은
방향임을 보여준다.

Branch 결과도 WPU memory thesis와 맞다. `B=8`에서 `BaseState + branch delta`는
가장 큰 `N`에서 branch별 full state copy 대비 branch memory proxy를 `0.874128`
줄인다.

하지만 이것은 hardware speedup이나 lower power를 증명하지 않는다. Python-level
tensorization 측정이며 model-forward latency, irregular sparse-kernel overhead,
cache behavior, GPU occupancy, 실제 energy를 포함하지 않는다. 방어 가능한 주장은 더 좁다.

```text
Causal working set K가 tensorization 전에 선택된다면, WPU는 token/full-state graph
baseline이 같은 state index를 구현하지 않는 한 지불해야 하는 큰 systems cost를
줄일 수 있는 구조를 노출한다.
```

## 발견한 문제

- Profiler는 이제 CPU tensorization latency를 측정하지만, trained model forward
  latency, CUDA allocator traffic, energy는 아직 측정하지 않는다.
- `sys.getsizeof` 기반 state memory는 Python-object approximation이지 allocator-level
  memory measurement가 아니다.
- Indexed frontier는 이 PyBullet scene에서 relation-derived라 상대적으로 쉽다.
  더 어려운 perception/distractor setting에서는 effective `K` 품질이 떨어질 수 있다.
- Branch overlay는 synthetic delta record이며 rollback, correction,
  uncertainty-gated branch pruning을 아직 포함하지 않는다.

## 다음 단계

- 같은 `N` setting에서 trained/randomized model forward pass의 runtime latency와
  CUDA memory를 추가한다.
- Objectification corruption을 넣어 relation error가 selected `K`, tensor reduction,
  downstream loss를 어떻게 바꾸는지 측정한다.
- Python-object memory estimate를 serialized byte size 및 allocator-level measurement로
  대체한다.
- WPU 주장은 systems profile과 accuracy를 함께 보고해야 한다. 즉 acceptable prediction
  quality와 lower state-processing work가 동시에 필요하다.
