# PyBullet Route-Regret Training Smoke

Source CSVs:

- `docs/experiments/pybullet_route_regret_training_smoke.csv`
- `docs/experiments/pybullet_route_regret_training_smoke_untrained.csv`
- `docs/experiments/pybullet_route_regret_training_smoke_threshold.csv`

이 smoke test는 PyBullet mechanism-shift 실험에서 WPU route-regret head를 명시적으로
학습하고 측정할 수 있음을 검증한다. Staged-regret 결과의 교훈은 명확하다.
Regret-hybrid model을 PyBullet shift 평가에 넣을 때 route head가 미학습 상태이면 안
된다.

## 변경 사항

- `scripts/pybullet_shift_generalization.py`에 `--route-regret-loss-weight`,
  `--route-regret-compute-cost`, `--route-regret-threshold`를 추가했다.
- 학습 중 CWS model은 `dense_loss - sparse_loss + compute_cost` counterfactual target을
  `model.route_regret_loss`로 받을 수 있다.
- 평가 CSV는 `route_regret_mean`, `route_regret_negative_ratio`,
  `route_regret_loss_weight`, `route_regret_compute_cost`,
  `route_regret_threshold`를 기록한다.
- `CausalWorkingSetProcessor`는 predicted regret이 0보다 작으면 dense route를 쓰는
  hard-coded 정책 대신 configurable internal `route_regret_threshold`를 갖는다.

## Smoke Results

| condition | threshold | nominal dense compute | high-force dense compute | nominal accuracy | high-force accuracy |
|---|---:|---:|---:|---:|---:|
| untrained route head | 0.0 | 1.0 | 1.0 | 0.625 | 0.125 |
| trained route head | 0.0 | 1.0 | 1.0 | 0.625 | 0.125 |
| trained route head | -0.5 | 0.0 | 0.0 | 0.625 | 0.125 |

## 해석

이 결과는 mechanism-shift 성능 결과가 아니라 infrastructure evidence다. 작은 run은 두
가지를 보인다.

- Explicit route-regret training이 PyBullet path에 연결됐고 route metric을 출력한다.
- 이 작은 설정에서는 zero threshold가 all-dense routing으로 붕괴할 수 있으므로
  configurable route threshold가 필요하다.

다음 실제 실험은 route-regret threshold를 sweep하거나 validation split에서 선택한 뒤,
sparse, local-dense, regret-hybrid, graph, serialized-token baseline을 같은
mechanism-shift protocol에서 비교해야 한다.
