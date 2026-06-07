# WPU v2 우선순위 대시보드

이 문서는 기존 실험 CSV에서 v2 우선순위 1~7의 현재 상태를 보수적으로 재계산한다. 목적은 WPU 주장이 실험 증거를 초과하지 않도록 만드는 것이다.

| 우선순위 | 항목 | 상태 | 관측값 | 목표 | 지표 |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.328025 | 0.500000 | `gap_closure_fraction` |
| 2 | 장기 state integrity | partial | 0.988647 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 7.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.666667 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration과 uncertainty | partial | 0.963449 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile과 memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality와 propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## 해석

현재 dashboard는 WPU v2가 유망하지만 아직 완결된 우월성 주장이 아님을 보여준다. 가장 강한 주장은 large-N 자체가 아니라, objectified state에서 작은 causal working set K를 tensorization 전에 식별할 수 있을 때 WPU가 계산량과 메모리 측면에서 유리해진다는 조건부 주장이다.

- P1 Candidate-oracle gap: Candidate-regret deployment sweep은 margin-only gate보다 강하지만, 논문용 observed 값은 test-best sweep이 아니라 train-selected deployment를 우선 사용한다. 현재 train-selected closure는 0.328025로 목표 0.5에 못 미치고 harmful accept도 0.251111로 threshold 근처에 남아 있어 P1은 fail이다. Harmful-accept/ranking penalty 학습은 안전하지만 closure가 0.081253으로 떨어지고, feature perturbation은 test-sweep safe closure를 0.329756까지 조금 올리지만 train-selected closure는 0.312586에 머문다. 별도 safety/utility head도 negative result다. Best closure는 0.147450, safe best는 0.090719, train-selected closure는 0.144863에 그친다. Cross-fit ensemble regret gate도 train-selected overfit 가설을 부정하는 negative result다. 최고 closure는 0.287268, safe best는 0.279738, cross-fit selected closure는 0.270989로 direct regret gate보다 낮다.
- P2 장기 state integrity: Rollback-only memory layer는 sparse WPU H=25 integrity를 0.988647까지 올리지만 rollback rate가 0.812500으로 매우 높다. Corrected rollback은 rollback rate를 0.564167까지 낮추지만 integrity가 0.900288로 떨어진다. Escalated corrected rollback은 local-dense fallback을 사용해 integrity를 0.914831로 올리고 rollback rate를 0.000000으로 낮춘다. 따라서 P2는 sparse-first/dense-when-needed safety layer가 유효할 수 있음을 보이지만, raw delta stability가 해결된 것은 아니다.
- P3 Simulator-backed benchmark: PyBullet benchmark는 7개 seed와 background N_bg=128까지 확장됐다. N=133에서 WPU sparse accuracy가 0.547619로 serialized-token 0.539683보다 약간 높지만, serialized-token은 여전히 가장 빠르다. Simulator-backed evidence는 강화됐지만 규모와 mechanism 다양성은 아직 부족하다.
- P4 Mechanism-family shift generalization: 7-seed nominal-shift benchmark는 mixed이고, 3-seed leave-family-out probe는 win-rate 0.750000을 보인다. 새 composition-shift stress에서는 WPU가 accuracy 기준 3/3에서 baseline 이상이며 평균 accuracy delta가 0.123457이다. Branch-prior audit은 catch_heavy가 prior-dominated shift임을 보인다. Mechanism-prior adaptation은 shifted WPU win-rate를 0.333333에서 0.666667로 올리고 prior-dominated shift를 1개에서 0개로 줄인다. Prior-strength sweep의 accuracy-best 설정은 strength=0.75, mean WPU accuracy 0.601852지만 shifted win-rate는 0.666667에 머문다. Calibration-selected prior는 mean accuracy/ECE를 개선하지만 shifted win-rate는 0.333333에 머문다. Few-shot mechanism adaptation은 shifted WPU win-rate 1.000000, mean margin change 0.050264까지 도달하지만 mechanism별 calibration set을 쓰는 adapted protocol이다. 따라서 P4는 adapted regime에서 크게 개선됐지만 zero-shot solved는 아니다.
- P5 Calibration과 uncertainty: 7-seed 평균 WPU ECE ratio는 0.963449이고, leave-family-out 평균 ECE ratio는 0.972745로 양호하지만, calibrated mixture probe에서는 1.133834로 악화된다. Composition-shift stress의 평균 ECE ratio는 1.327702이고 no_catch에서 2.362081까지 악화된다. Temperature+bias calibration은 no_catch를 개선하지만 3개 mechanism 중 1개만 ECE ratio가 개선되어 보편 해결책은 아니다. Branch-prior audit은 catch_heavy에서 majority prior 0.753968이 best WPU 0.408730을 크게 앞선다는 점을 보여준다. Mechanism-prior adaptation은 accuracy를 개선하지만 shifted mean ECE를 0.024819 악화시킨다. Prior-strength sweep에서도 win-rate를 유지/개선하면서 ECE를 악화시키지 않는 비영점 strength가 없었다. Calibration-selected prior는 shifted mean ECE를 -0.046204, Brier를 -0.105470 개선하지만 baseline win-rate는 올리지 못한다. Few-shot mechanism adaptation도 ECE를 -0.055342 개선한다. 따라서 branch probability adaptation은 개선됐지만 zero-shot robust mechanism generalization과 분리해서 보고해야 한다.
- P6 Systems profile과 memory traffic: Tensor-byte reduction은 0.997454, CPU sparse-forward reduction은 0.996975, CUDA sparse-forward reduction은 0.996216까지 관측됐다. Screening-only energy proxy도 추가됐지만 실제 전력 측정은 아니다. Matched-speedup audit의 판정 기준을 corrected matched-or-better로 고치면 N=133에서는 best-accuracy non-WPU baseline 대비 WPU가 더 정확하고 더 빠르다. Pareto audit에서도 WPU는 N=133에서 frontier에 올라가지만 N=5에서는 token에 지배된다. Real energy와 sparse-kernel behavior는 아직 미해결이다.
- P7 Objectification quality와 propagation loss: Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. Objectification metric은 있지만 downstream loss 연결은 미완성이다.

## 다음 조치

- P1: Post-hoc gate를 더 튜닝하기보다 retrieval/propagation과 candidate scoring을 joint objective로 묶고, no-harm/calibration target을 cross-seed 전이에 맞게 학습한다.
- P2: 단순 delta-norm, rollout-consistency, state-validity regularization은 부족하다. Guarded state-store projection을 유지하되 rollback/correction과 uncertainty escalation을 모델-메모리 계층에 넣는다.
- P3: 더 많은 mechanism, long-horizon simulator rollout, parameter-matched 7-seed benchmark를 추가한다.
- P4: Catch-heavy류 branch-prior shift를 겨냥한 mechanism-aware branch prior와 uncertainty-gated fallback을 추가한다.
- P5: Post-hoc temperature가 아니라 학습 가능한 calibration head, multi-step ECE/Brier/NLL, uncertainty-gated recompute를 추가한다.
- P6: Energy, allocator traffic, sparse-kernel behavior, Pareto frontier, trained matched-or-better speedup을 측정한다.
- P7: Controlled objectification corruption에서 propagation을 학습/평가하고 report component와 downstream loss의 관계를 회귀 분석한다.
