# PyBullet Uncertainty-Gated Recompute Results

Source CSV: `docs/experiments/pybullet_uncertainty_gated_recompute.csv`

This experiment routes low-confidence sparse WPU predictions to a local-dense recompute path within the WPU family. It is not a token or graph fallback; it tests state-native uncertainty routing.

| Policy | Accuracy | ECE | Brier | NLL | Dense recompute rate |
|---|---:|---:|---:|---:|---:|
| Sparse WPU (`wpu_sparse`) | 0.451058 | 0.184273 | 0.658191 | 1.087398 | 0.000000 |
| Local-dense WPU (`wpu_local_dense`) | 0.522486 | 0.169800 | 0.634327 | 1.063696 | 1.000000 |
| Best ECE-safe gate (`wpu_gated_t0.45`) | 0.522486 | 0.167877 | 0.634919 | 1.063278 | 0.985450 |
| Best low-cost gate (`wpu_gated_t0.34`) | 0.460318 | 0.189668 | 0.656412 | 1.085302 | 0.025132 |
| Best NLL gate (`wpu_gated_t0.40`) | 0.522486 | 0.175164 | 0.636546 | 1.061715 | 0.867725 |

## Interpretation

- The ECE-safe gate changes accuracy by +0.071428, ECE by -0.016396, with dense recompute rate 0.985450.
- The NLL-selected gate changes NLL by -0.025683 and ECE by -0.009109 versus sparse WPU.
- The low-cost gate has dense recompute rate 0.025132, changing accuracy by +0.009260 and ECE by +0.005395. The current threshold gate can improve calibration, but it is not yet a low-cost sparse-routing solution.
- The result tests whether WPU calibration can be improved by state-native uncertainty routing rather than by returning to token processing. The remaining gap is that the threshold is still a hand policy and the useful aggregate improvement is close to full recompute; the next step is a learned gate with held-out threshold selection.

## Per-Mechanism Summary

| Mechanism | Sparse acc | Sparse ECE | Best gate | Gate acc | Gate ECE | Dense rate |
|---|---:|---:|---|---:|---:|---:|
| edge_catch_heavy | 0.345238 | 0.132180 | `wpu_gated_t0.34` (not ECE-safe) | 0.341270 | 0.146288 | 0.039683 |
| edge_high_force | 0.492064 | 0.128242 | `wpu_gated_t0.34` | 0.492064 | 0.128242 | 0.000000 |
| no_catch | 0.515873 | 0.292398 | `wpu_gated_t0.45` | 0.603174 | 0.193446 | 1.000000 |
