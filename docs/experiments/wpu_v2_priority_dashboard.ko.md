# WPU v2 우선순위 대시보드

이 문서는 기존 실험 CSV에서 v2 우선순위 1~7의 현재 상태를 보수적으로 재계산한다. 목적은 WPU 주장이 실험 증거를 초과하지 않도록 만드는 것이다.

| 우선순위 | 항목 | 상태 | 관측값 | 목표 | 지표 |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.328025 | 0.500000 | `gap_closure_fraction` |
| 2 | 장기 state integrity | partial | 0.988647 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 7.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.333333 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration과 uncertainty | partial | 0.963449 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile과 memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality와 propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## 해석

현재 dashboard는 WPU v2가 유망하지만 아직 완결된 우월성 주장이 아님을 보여준다. 가장 강한 주장은 large-N 자체가 아니라, objectified state에서 작은 causal working set K를 tensorization 전에 식별할 수 있을 때 WPU가 계산량과 메모리 측면에서 유리해진다는 조건부 주장이다.

- P1 Candidate-oracle gap: Candidate-regret deployment sweep은 margin-only gate보다 강하지만, 논문용 observed 값은 test-best sweep이 아니라 train-selected deployment를 우선 사용한다. 현재 train-selected closure는 0.328025로 목표 0.5에 못 미치고 harmful accept도 0.251111로 threshold 근처에 남아 있어 P1은 fail이다. Harmful-accept/ranking penalty 학습은 안전하지만 closure가 0.081253으로 떨어지고, feature perturbation은 test-sweep safe closure를 0.329756까지 조금 올리지만 train-selected closure는 0.312586에 머문다.
- P2 장기 state integrity: Rollback/correction memory layer는 sparse WPU H=25 integrity를 0.988647까지 올리지만 rollback rate가 0.812500으로 매우 높다. Guarded projection과 rollback은 applied state를 보호한다는 증거이지 raw delta model이 안정적이라는 증거가 아니다. 따라서 P2는 raw delta stability와 memory-layer safety를 분리해 주장해야 한다.
- P3 Simulator-backed benchmark: PyBullet benchmark는 7개 seed와 background N_bg=128까지 확장됐다. N=133에서 WPU sparse accuracy가 0.547619로 serialized-token 0.539683보다 약간 높지만, serialized-token은 여전히 가장 빠르다. Simulator-backed evidence는 강화됐지만 규모와 mechanism 다양성은 아직 부족하다.
- P4 Mechanism-family shift generalization: 7-seed nominal-shift benchmark는 mixed이고, 3-seed leave-family-out probe는 win-rate 0.750000을 보인다. WPU는 nominal/high_force/edge_shift holdout에서는 앞서지만 catch_heavy branch-prior shift에서는 baseline에 진다. 따라서 shift generalization은 조건부다.
- P5 Calibration과 uncertainty: 7-seed 평균 WPU ECE ratio는 0.963449이고, leave-family-out 평균 ECE ratio는 0.972745로 양호하지만, calibrated mixture probe에서는 1.133834로 악화된다. Calibration advantage는 안정된 결론이 아니며 mechanism-aware uncertainty가 필요하다.
- P6 Systems profile과 memory traffic: Tensor-byte reduction은 0.997454, CPU sparse-forward reduction은 0.996975, CUDA sparse-forward reduction은 0.996216까지 관측됐다. Matched-speedup audit은 N=5에서는 accuracy matched지만 WPU가 token보다 느리고, N=133에서는 WPU가 빠르지만 accuracy tolerance 조건을 넘는다. Energy와 strict matched-accuracy speedup은 아직 미해결이다.
- P7 Objectification quality와 propagation loss: Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. Objectification metric은 있지만 downstream loss 연결은 미완성이다.

## 다음 조치

- P1: Candidate-regret 학습에 calibrated uncertainty, harmful-accept penalty, cross-seed perturbation을 더 강하게 넣는다.
- P2: 단순 delta-norm, rollout-consistency, state-validity regularization은 부족하다. Guarded state-store projection을 유지하되 rollback/correction과 uncertainty escalation을 모델-메모리 계층에 넣는다.
- P3: 더 많은 mechanism, long-horizon simulator rollout, parameter-matched 7-seed benchmark를 추가한다.
- P4: Catch-heavy류 branch-prior shift를 겨냥한 mechanism-aware branch prior와 uncertainty-gated fallback을 추가한다.
- P5: Post-hoc temperature가 아니라 학습 가능한 calibration head, multi-step ECE/Brier/NLL, uncertainty-gated recompute를 추가한다.
- P6: Energy, allocator traffic, sparse-kernel behavior, strict matched-accuracy speedup을 측정한다.
- P7: Controlled objectification corruption에서 propagation을 학습/평가하고 report component와 downstream loss의 관계를 회귀 분석한다.
