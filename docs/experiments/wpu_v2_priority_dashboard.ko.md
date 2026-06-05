# WPU v2 우선순위 대시보드

이 문서는 기존 실험 CSV에서 v2 우선순위 1~7의 현재 상태를 보수적으로 재계산한다. 목적은 WPU 주장이 실험 증거를 초과하지 않도록 만드는 것이다.

| 우선순위 | 항목 | 상태 | 관측값 | 목표 | 지표 |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.328025 | 0.500000 | `gap_closure_fraction` |
| 2 | 장기 state integrity | partial | 0.964322 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 5.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.333333 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration과 uncertainty | partial | 0.963449 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile과 memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality와 propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## 해석

현재 dashboard는 WPU v2가 유망하지만 아직 완결된 우월성 주장이 아님을 보여준다. 가장 강한 주장은 large-N 자체가 아니라, objectified state에서 작은 causal working set K를 tensorization 전에 식별할 수 있을 때 WPU가 계산량과 메모리 측면에서 유리해진다는 조건부 주장이다.

- P1 Candidate-oracle gap: Candidate-regret deployment sweep은 margin-only gate보다 강하지만, 논문용 observed 값은 test-best sweep이 아니라 train-selected deployment를 우선 사용한다. 현재 train-selected closure는 0.328025로 목표 0.5에 못 미치고 harmful accept도 0.251111로 threshold 근처에 남아 있어 P1은 fail이다. Harmful-accept/ranking penalty 학습은 harmful accept를 0.088889까지 낮췄지만 closure가 0.081253으로 크게 떨어져, 안전 penalty만으로는 candidate-oracle gap을 닫지 못한다.
- P2 장기 state integrity: 최고 WPU H=25 integrity는 guarded state-store projection에서 나온다. Unsafe-delta rejection은 sparse raw 폭주를 완화하지만 rejection rate가 높고, rollout-consistency와 state-validity regularization도 sparse raw 안정성을 해결하지 못한다. 따라서 P2는 applied-state integrity와 raw delta stability를 분리해서 주장해야 하며 rollback/correction 계층이 필요하다.
- P3 Simulator-backed benchmark: PyBullet benchmark는 5개 seed와 background N_bg=128까지 확장됐다. 다만 mechanism 다양성, training scale, long-horizon simulator rollout은 아직 부족하다.
- P4 Mechanism-family shift generalization: 7-seed shift benchmark에서 WPU는 catch_heavy에서 앞서지만 edge_shift와 high_force에서는 baseline에 밀린다. 3-seed calibrated mixture probe에서는 edge_shift가 개선되지만 catch_heavy는 serialized-token baseline이 더 강하다. Shift generalization은 부분적으로만 성립한다.
- P5 Calibration과 uncertainty: 7-seed 평균 WPU ECE는 0.211243, baseline ECE는 0.219257로 ratio가 0.963449다. 그러나 3-seed calibrated mixture probe에서는 WPU ECE ratio가 1.133834로 악화된다. 따라서 calibration advantage는 안정된 결론이 아니며 multi-step/shift calibration이 필요하다.
- P6 Systems profile과 memory traffic: Tensor-byte reduction은 mean total objects 2052.6에서 0.997454까지 도달하고 CPU tensorization latency reduction도 0.996035까지 도달한다. Random-model CPU sparse-forward latency reduction은 0.996975, CUDA sparse-forward latency reduction은 0.996216까지 관측됐다. 다만 CUDA peak-memory reduction은 0.304080 수준이고 energy 및 matched-accuracy speedup 증거는 아직 없다.
- P7 Objectification quality와 propagation loss: Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. Objectification metric은 있지만 downstream loss 연결은 미완성이다.

## 다음 조치

- P1: Candidate-regret 학습에 calibrated uncertainty, harmful-accept penalty, cross-seed perturbation을 더 강하게 넣는다.
- P2: 단순 delta-norm, rollout-consistency, state-validity regularization은 부족하다. Guarded state-store projection을 유지하되 rollback/correction과 uncertainty escalation을 모델-메모리 계층에 넣는다.
- P3: Seed, mechanism, training scale, long-horizon simulator rollout을 늘린다.
- P4: Leave-family-out training, 더 어려운 shift, mechanism-aware branch prior를 추가한다.
- P5: Temperature head, branch calibration loss, multi-step ECE/Brier/NLL, uncertainty-gated recompute를 추가한다.
- P6: Energy, allocator traffic, sparse-kernel behavior, matched-accuracy speedup을 측정한다.
- P7: Controlled objectification corruption에서 propagation을 학습/평가하고 report component와 downstream loss의 관계를 회귀 분석한다.
