# PyBullet uncertainty-gated recompute 결과

Source CSV: `docs/experiments/pybullet_uncertainty_gated_recompute.csv`

이 실험은 sparse WPU가 낮은 branch confidence를 보일 때 같은 WPU family의 local-dense recompute로만 넘기는 정책을 평가한다. Token 또는 graph baseline으로 fallback하지 않기 때문에 WPU 내부 uncertainty routing 검증이다.

| 정책 | Accuracy | ECE | Brier | NLL | Dense recompute rate |
|---|---:|---:|---:|---:|---:|
| Sparse WPU (`wpu_sparse`) | 0.451058 | 0.184273 | 0.658191 | 1.087398 | 0.000000 |
| Local-dense WPU (`wpu_local_dense`) | 0.522486 | 0.169800 | 0.634327 | 1.063696 | 1.000000 |
| Best ECE-safe gate (`wpu_gated_t0.45`) | 0.522486 | 0.167877 | 0.634919 | 1.063278 | 0.985450 |
| Best low-cost gate (`wpu_gated_t0.34`) | 0.460318 | 0.189668 | 0.656412 | 1.085302 | 0.025132 |
| Best NLL gate (`wpu_gated_t0.40`) | 0.522486 | 0.175164 | 0.636546 | 1.061715 | 0.867725 |

## 해석

- ECE-safe gate는 sparse 대비 accuracy를 +0.071428, ECE를 -0.016396, dense recompute rate를 0.985450로 만든다.
- NLL-selected gate는 sparse 대비 NLL을 -0.025683, ECE를 -0.009109 변화시킨다.
- Low-cost gate는 dense recompute rate 0.025132에서 accuracy를 +0.009260, ECE를 +0.005395 변화시킨다. 따라서 현재 threshold gate는 calibration을 개선할 수 있지만, 저비용 sparse routing 해법으로는 아직 부족하다.
- 이 결과는 WPU의 calibration 개선 방향이 token fallback이 아니라 state-native uncertainty routing일 수 있음을 검증한다. 다만 threshold는 아직 hand policy이고, 유의미한 개선은 거의 full recompute에 가까우므로 학습 가능한 gate와 held-out threshold selection이 다음 단계다.

## Mechanism별 요약

| Mechanism | Sparse acc | Sparse ECE | Best gate | Gate acc | Gate ECE | Dense rate |
|---|---:|---:|---|---:|---:|---:|
| edge_catch_heavy | 0.345238 | 0.132180 | `wpu_gated_t0.34` (not ECE-safe) | 0.341270 | 0.146288 | 0.039683 |
| edge_high_force | 0.492064 | 0.128242 | `wpu_gated_t0.34` | 0.492064 | 0.128242 | 0.000000 |
| no_catch | 0.515873 | 0.292398 | `wpu_gated_t0.45` | 0.603174 | 0.193446 | 1.000000 |
