# PyBullet Systems Claim-Boundary Audit

이 파생 감사는 P6 systems evidence를 claim별로 분리한다. 목표는 WPU의 systems 장점을 숨기는 것이 아니라, tensorization/latency proxy와 실제 hardware claim의 경계를 명확히 하는 것이다.

Source CSV: `docs/experiments/pybullet_system_claim_boundary.csv`

## Summary

- Supported proxy 축은 `4`개이고 partial trained 축은 `2`개다.
- Branch-overlay memory proxy의 최대 reduction은 `0.874128`이다.
- CUDA peak-memory proxy는 최대 `0.304080`라서 latency proxy보다 훨씬 약하다.
- Screening/weak proxy 축은 `2`개이고, real power/sparse-kernel 축은 아직 `1`개 미측정이다.

## Interpretation

현재 P6의 강한 주장은 pre-tensor working-set selection, branch-overlay memory accounting, large-N random-forward latency proxy, 그리고 제한된 trained matched-speedup이다. 반면 GPU peak memory, 실제 memory traffic, allocator telemetry, real power, custom sparse kernel evidence는 아직 부족하다. 따라서 WPU는 hardware result가 아니라 hardware로 가야 할 systems hypothesis로 써야 한다.

## Boundary Rows

| axis | status | observed | target | N | B | evidence | limitation |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| pre_tensor_working_set_tensor_bytes | supported_proxy | 0.997454 | 0.950000 | 2052.562 | 1 | CPU tensorization proxy | This is a tensorization proxy, not hardware memory-traffic telemetry. |
| random_cpu_sparse_forward_latency | supported_proxy | 0.996975 | 0.950000 | 2052.562 | 3 | random-model CPU forward proxy | Random-model latency is an upper-bound signal, not trained matched-accuracy speedup. |
| random_cuda_sparse_forward_latency | supported_proxy | 0.996216 | 0.950000 | 2052.375 | 3 | random-model CUDA forward proxy | This still uses generic PyTorch modules, not a custom sparse frontier kernel. |
| random_cuda_peak_memory | weak_proxy | 0.304080 | 0.950000 | 2052.375 | 1 | CUDA peak-memory proxy | P6 cannot claim broad GPU-memory dominance from the current PyTorch profile. |
| branch_overlay_memory | supported_proxy | 0.874128 | 0.800000 | 2052.562 | 8 | state-store memory proxy | This is byte accounting from state objects, not allocator-level resident-memory telemetry. |
| trained_matched_or_better_speedup | partial_matched | 0.500000 | 1.000000 | 133.0 |  | trained benchmark audit | Current trained speedup evidence has only two N values and is not universal latency dominance. |
| accuracy_latency_pareto_frontier | partial_pareto | 0.500000 | 1.000000 | 133 |  | trained Pareto audit | This separates matched-speedup evidence from full Pareto dominance. |
| screening_energy_proxy | screening_only | 0.999990 | 0.950000 | 2052.562 | 8 | derived energy proxy | This is not wall-plug power, GPU power telemetry, or hardware energy measurement. |
| real_power_or_sparse_kernel | not_measured | 0.000000 | 1.000000 |  |  | missing hardware measurement | Hardware/chip/IP claims remain unsupported until this row changes. |
