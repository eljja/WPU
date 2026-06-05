# WPU v2 우선순위 대시보드

이 문서는 기존 실험 CSV에서 v2 우선순위 1~7의 현재 상태를 보수적으로 재계산한다. 목적은 WPU 주장이 실험 증거를 초과하지 않도록 만드는 것이다.

| 우선순위 | 항목 | 상태 | 관측값 | 목표 | 지표 |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.244220 | 0.500000 | `gap_closure_fraction` |
| 2 | 장기 state integrity | partial | 0.964322 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 2.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.333333 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration과 uncertainty | partial | 1.068727 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile과 memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality와 propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## 해석

현재 dashboard는 WPU v2가 유망하지만 아직 완결된 우월성 주장이 아님을 보여준다. 가장 강한 주장은 large-N 자체가 아니라, objectified state에서 작은 causal working set K를 tensorization 전에 식별할 수 있을 때 WPU가 계산량과 메모리 측면에서 유리해진다는 조건부 주장이다.

- P1 Candidate-oracle gap: 최고 deployed closure는 0.244220이고 평균 closure는 0.160601이다. Decomposition 결과 aggregate policy 하나를 더 고르는 방식으로는 gap이 닫히지 않는다.
- P2 장기 state integrity: 최고 WPU H=25 integrity는 0.964322이고 guarded sparse는 0.958508이다. 하지만 raw sparse는 0.084722로 남아 있어 state-store guard가 적용 state를 보호한 것이지 raw delta model 안정성이 해결된 것은 아니다.
- P3 Simulator-backed benchmark: PyBullet benchmark는 2개 seed와 background N_bg=128까지 존재하지만, 논문급 강한 주장에는 seed와 mechanism 수가 부족하다.
- P4 Mechanism-family shift generalization: WPU는 edge_shift에서 앞서지만 high_force와 catch_heavy에서는 baseline에 밀린다. Shift generalization은 부분적으로만 성립한다.
- P5 Calibration과 uncertainty: 평균 WPU ECE는 0.236226, baseline ECE는 0.221034로 WPU가 약 1.068727배 높다. Calibration은 측정됐지만 개선됐다고 보기 어렵다.
- P6 Systems profile과 memory traffic: Proxy tensor-byte reduction은 mean total objects 2052.6에서 0.997454까지 도달한다. 다만 실제 hardware/runtime/energy 증거는 아직 없다.
- P7 Objectification quality와 propagation loss: Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. Objectification metric은 있지만 downstream loss 연결은 미완성이다.

## 다음 조치

- P1: Aggregate policy selection 아래로 내려가 per-candidate uncertainty, sample-level no-harm gate, regret target을 추가한다.
- P2: Guarded state-store projection을 유지하되, rollout-consistency loss와 unsafe-delta rejection을 학습 단계로 끌어올린다.
- P3: Seed, mechanism, training scale, long-horizon simulator rollout을 늘린다.
- P4: Leave-family-out training, 더 어려운 shift, mechanism-aware branch prior를 추가한다.
- P5: Temperature head, branch calibration loss, multi-step ECE/Brier/NLL, uncertainty-gated recompute를 추가한다.
- P6: 실제 CPU/GPU latency, CUDA memory, allocator traffic, sparse-kernel behavior, matched-accuracy speedup을 측정한다.
- P7: Controlled objectification corruption에서 propagation을 학습/평가하고 report component와 downstream loss의 관계를 회귀 분석한다.
