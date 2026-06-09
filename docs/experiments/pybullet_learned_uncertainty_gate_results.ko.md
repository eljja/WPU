# PyBullet learned uncertainty gate 결과

Source CSV: `docs/experiments/pybullet_learned_uncertainty_gate.csv`

이 실험은 sparse WPU 출력과 event feature만으로 local-dense recompute의 NLL benefit을 예측하는 작은 gate를 학습한다. Source gate는 train mechanisms로만 학습하고, few-shot gate는 eval mechanism calibration samples를 사용한다.

| Policy | Accuracy | ECE | Brier | NLL | Dense recompute rate |
|---|---:|---:|---:|---:|---:|
| Sparse WPU (`wpu_sparse`) | 0.510582 | 0.201734 | 0.646535 | 1.069807 | 0.000000 |
| Local-dense WPU (`wpu_local_dense`) | 0.584656 | 0.187422 | 0.582624 | 0.986987 | 1.000000 |
| Source low-cost (`source_learned_p0.12`) | 0.563492 | 0.212503 | 0.612986 | 1.024772 | 0.205027 |
| Few-shot lowest-rate over budget (`fewshot_learned_p0.12`) | 0.600529 | 0.243247 | 0.595602 | 1.000422 | 0.292328 |
| Source best ECE candidate (not safe) (`source_learned_p0.12`) | 0.563492 | 0.212503 | 0.612986 | 1.024772 | 0.205027 |
| Few-shot best ECE candidate (not safe) (`fewshot_learned_p0.00`) | 0.601852 | 0.222013 | 0.582460 | 0.982130 | 0.671958 |

## 해석

- Source low-cost: accuracy 변화 +0.052910, ECE 변화 +0.010769, dense recompute rate 0.205027.
- Few-shot lowest-rate over budget: accuracy 변화 +0.089947, ECE 변화 +0.041513, dense recompute rate 0.292328.
- Source best ECE candidate (not safe): accuracy 변화 +0.052910, ECE 변화 +0.010769, dense recompute rate 0.205027.
- Few-shot best ECE candidate (not safe): accuracy 변화 +0.091270, ECE 변화 +0.020279, dense recompute rate 0.671958.
- Source gate는 저비용 accuracy를 개선하지만 aggregate ECE를 악화시킨다. Few-shot gate는 accuracy/NLL을 더 개선하지만 low-cost budget을 넘고 ECE도 악화시킨다. 따라서 sparse-output gate만으로 calibration-safe 저비용 routing은 아직 해결되지 않았다.

## Mechanism별 low-cost 요약

| Mechanism | Sparse acc | Sparse ECE | Source acc/ECE/rate | Few-shot acc/ECE/rate |
|---|---:|---:|---:|---:|
| edge_catch_heavy | 0.408730 | 0.130788 | 0.432540/0.135096/0.119048 | 0.440476/0.134581/0.226191 |
| edge_high_force | 0.571429 | 0.196511 | 0.623016/0.191999/0.321429 | 0.670635/0.226442/0.444444 |
| no_catch | 0.551587 | 0.277903 | 0.634921/0.310413/0.174603 | 0.686508/0.347333/0.357143 |
