# PyBullet State-Integrity Audit

This audit derives long-horizon state-integrity metrics from the PyBullet
closed-loop rollout results. It does not resynchronize to the simulator;
it evaluates whether repeated `DeltaState` overlays keep object state
within simple validity bounds.

Source CSVs:

- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_closed_loop_rollout_clipped.csv`
- `docs/experiments/pybullet_closed_loop_rollout_guarded.csv`
- `docs/experiments/pybullet_closed_loop_rollout_regularized.csv`
- `docs/experiments/pybullet_closed_loop_rollout_rejected.csv`
- `docs/experiments/pybullet_closed_loop_rollout_consistency.csv`
- `docs/experiments/pybullet_closed_loop_rollout_validity.csv`
- `docs/experiments/pybullet_closed_loop_rollout_validity_strong.csv`
- `docs/experiments/pybullet_closed_loop_rollout_rollback.csv`
- `docs/experiments/pybullet_closed_loop_rollout_corrected_rollback.csv`
- `docs/experiments/pybullet_closed_loop_rollout_escalated_corrected_rollback.csv`
- `docs/experiments/pybullet_closed_loop_rollout_finite_clamped.csv`
- `docs/experiments/pybullet_closed_loop_rollout_finite_corrected.csv`
- `docs/experiments/pybullet_closed_loop_rollout_selective_corrected.csv`
- `docs/experiments/pybullet_closed_loop_rollout_selective_corrected_stride2.csv`
- `docs/experiments/pybullet_closed_loop_rollout_selective_corrected_margin1.csv`

Derived CSV:

- `docs/experiments/pybullet_state_integrity_audit.csv`

## Summary

| run | model | H | violations/step | delta norm | flip rate | reject rate | correction rate | corrected objects | rollback rate | escalation rate | escalation success | integrity score | low-disruption score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clipped | graph-transformer | 25 | 0.253333 | 8.363635 | 0.054688 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.557002 | 0.557002 |
| clipped | wpu-cws-indexed-local-dense | 25 | 0.314167 | 2.809900 | 0.048611 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.719139 | 0.719139 |
| clipped | wpu-cws-indexed-sparse | 25 | 0.785000 | 1939290.233702 | 0.082465 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.201757 | 0.201757 |
| consistency | wpu-cws-indexed-local-dense | 25 | 0.811667 | 1.965078 | 0.059896 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.472827 | 0.472827 |
| consistency | wpu-cws-indexed-sparse | 25 | 3.360834 | 1775082.311771 | 0.077257 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.084549 | 0.084549 |
| corrected_rollback | graph-transformer | 25 | 0.000000 | 5.756884 | 0.056424 | 0.000000 | 0.268334 | 0.268334 | 0.000000 | 0.000000 | 0.000000 | 0.787224 | 0.679891 |
| corrected_rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 2.263392 | 0.055556 | 0.000000 | 0.499166 | 0.499166 | 0.000000 | 0.000000 | 0.000000 | 0.909670 | 0.710004 |
| corrected_rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 2.392552 | 0.079862 | 0.000000 | 0.812500 | 0.812500 | 0.564167 | 0.000000 | 0.000000 | 0.900288 | 0.406038 |
| escalated_corrected_rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 1.942319 | 0.085938 | 0.000000 | 0.710833 | 0.710833 | 0.000000 | 0.805833 | 0.116107 | 0.914831 | 0.549915 |
| finite_clamped | graph-transformer | 25 | 0.253333 | 2.096666 | 0.054688 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.776346 | 0.776346 |
| finite_clamped | wpu-cws-indexed-local-dense | 25 | 0.314167 | 0.741596 | 0.048611 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.791530 | 0.791530 |
| finite_clamped | wpu-cws-indexed-sparse | 25 | 0.784166 | 0.709270 | 0.082465 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.527391 | 0.527391 |
| finite_corrected | graph-transformer | 25 | 0.000000 | 2.095547 | 0.054688 | 0.000000 | 0.253333 | 0.253333 | 0.000000 | 0.000000 | 0.000000 | 0.915718 | 0.814385 |
| finite_corrected | wpu-cws-indexed-local-dense | 25 | 0.000000 | 0.735348 | 0.048611 | 0.000000 | 0.314167 | 0.314167 | 0.000000 | 0.000000 | 0.000000 | 0.964541 | 0.838874 |
| finite_corrected | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.697858 | 0.084201 | 0.000000 | 0.784166 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.958735 | 0.645068 |
| guarded | graph-transformer | 25 | 0.000000 | 2.096666 | 0.054688 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.915679 | 0.915679 |
| guarded | wpu-cws-indexed-local-dense | 25 | 0.000000 | 0.741597 | 0.048611 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.964322 | 0.964322 |
| guarded | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.709288 | 0.083334 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.958508 | 0.958508 |
| raw | graph-transformer | 5 | 0.045833 | 4.162467 | 0.062500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.816605 | 0.816605 |
| raw | graph-transformer | 10 | 0.139583 | 6.106724 | 0.150463 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.679401 | 0.679401 |
| raw | graph-transformer | 25 | 0.272500 | 8.409911 | 0.056424 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.544493 | 0.544493 |
| raw | wpu-cws-indexed-local-dense | 5 | 0.066667 | 2.788842 | 0.145834 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.836557 | 0.836557 |
| raw | wpu-cws-indexed-local-dense | 10 | 0.204167 | 2.648499 | 0.148148 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.765381 | 0.765381 |
| raw | wpu-cws-indexed-local-dense | 25 | 0.499166 | 2.744688 | 0.055556 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.618283 | 0.618283 |
| raw | wpu-cws-indexed-sparse | 5 | 0.200000 | 0.799330 | 0.125000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.837023 | 0.837023 |
| raw | wpu-cws-indexed-sparse | 10 | 0.531250 | 3.534072 | 0.203704 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.543379 | 0.543379 |
| raw | wpu-cws-indexed-sparse | 25 | 3.374166 | 1958877.607881 | 0.076389 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.084722 | 0.084722 |
| regularized | wpu-cws-indexed-local-dense | 25 | 0.536667 | 1.915983 | 0.044271 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.628920 | 0.628920 |
| regularized | wpu-cws-indexed-sparse | 25 | 3.316667 | 1797100.815468 | 0.064237 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.087153 | 0.087153 |
| rejected | graph-transformer | 25 | 0.271666 | 3.406922 | 0.053819 | 0.359166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.720577 | 0.648744 |
| rejected | wpu-cws-indexed-local-dense | 25 | 0.499166 | 2.277815 | 0.055556 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.634624 | 0.634624 |
| rejected | wpu-cws-indexed-sparse | 25 | 0.785834 | 0.635544 | 0.076389 | 0.640000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.530270 | 0.402270 |
| rollback | graph-transformer | 25 | 0.000000 | 4.140561 | 0.057292 | 0.000000 | 0.000000 | 0.000000 | 0.261667 | 0.000000 | 0.000000 | 0.843622 | 0.765122 |
| rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 1.225809 | 0.052951 | 0.000000 | 0.000000 | 0.000000 | 0.499166 | 0.000000 | 0.000000 | 0.946506 | 0.796756 |
| rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.150753 | 0.030382 | 0.000000 | 0.000000 | 0.000000 | 0.812500 | 0.000000 | 0.000000 | 0.988647 | 0.744897 |
| selective_corrected | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.697858 | 0.084201 | 0.000000 | 0.784166 | 0.027461 | 0.000000 | 0.000000 | 0.000000 | 0.958735 | 0.758574 |
| selective_corrected_margin1 | wpu-cws-indexed-sparse | 25 | 0.784166 | 0.709270 | 0.082465 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.527391 | 0.527391 |
| selective_corrected_stride2 | wpu-cws-indexed-sparse | 25 | 0.770000 | 0.709051 | 0.082465 | 0.000000 | 0.014166 | 0.027371 | 0.000000 | 0.000000 | 0.000000 | 0.535190 | 0.527543 |
| validity | wpu-cws-indexed-local-dense | 25 | 0.605833 | 2.222097 | 0.054688 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.578081 | 0.578081 |
| validity | wpu-cws-indexed-sparse | 25 | 3.374166 | 1785671.546102 | 0.076389 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.084722 | 0.084722 |
| validity_strong | wpu-cws-indexed-local-dense | 25 | 0.710833 | 2.212986 | 0.055556 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.520476 | 0.520476 |
| validity_strong | wpu-cws-indexed-sparse | 25 | 3.374166 | 1785671.546102 | 0.076389 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.084722 | 0.084722 |

## Interpretation

The audit confirms that one-step branch accuracy is not enough for a
world-state processor. The sparse WPU path can keep a small selected
`K`, but repeated raw deltas can still create invalid state. Guarded
state-store projection can protect the applied state and lift WPU
H=25 integrity above the dashboard threshold, but it does not prove
the underlying delta model is stable. Future reports must distinguish
raw model deltas from guarded state-store deltas.

The regularized run adds a training-time target-relative delta-norm
penalty. It is intentionally reported as a raw rollout, not as a
guarded state-store result. In the current evidence it only slightly
improves raw WPU sparse H=25 integrity, so simple delta-norm
regularization is not sufficient to solve model-delta instability.

The unsafe-delta rejection run is a state-store safety mechanism,
not proof that the raw transition model is stable. It must be
reported together with rejection rate: high integrity with high
rejection means the memory layer protected the state by declining
unsafe updates.

The correction run applies a bounded state projection after a predicted
delta increases validity violations, and only then falls back to
rollback if the corrected state is still worse than the previous
state. It tests whether memory-layer repair can reduce rollback
frequency while preserving applied-state integrity.

The finite-clamped run first sanitizes non-finite or extreme
predicted deltas, then applies norm clipping. It removes the
sparse WPU delta-norm explosion seen in the earlier clipped run,
but it does not eliminate validity violations by itself. This
separates numerical delta safety from state validity.

The finite-corrected run combines finite-safe delta clipping with
correction-only projection. It is a stronger memory-safety result:
sparse WPU reaches H=25 integrity comparable to guarded projection
with zero rollback and zero dense escalation, but at a high
correction rate. This still does not prove raw dynamics stability;
it shows that bounded local correction can protect applied state
without declining or recomputing most updates.

The selective-correction run uses the same finite-safe correction
trigger as finite-corrected rollout but only projects objects that
actually violate validity bounds. It preserves sparse H=25 integrity
while reducing the corrected-object fraction. The stride-2 and
margin-1 variants show the current boundary: reducing correction
trigger frequency directly causes validity violations to return.
This narrows the P2 problem to learning a more stable transition or
a better correction trigger, not merely shrinking the correction
projection itself.

The escalation run tests sparse-first, dense-when-needed memory
safety: when the sparse delta increases validity violations, the
state is restored and a local-dense WPU fallback recomputes the
update before correction or rollback. In the current evidence this
reduces rollback frequency to zero for sparse H=25 while improving
corrected-rollback integrity, but it is still a safety-layer result
rather than proof of stable raw sparse dynamics.

The rollout-consistency run adds a second-step delta-growth penalty
during training. In the current evidence it does not solve sparse
raw-delta instability, so rollout consistency needs a stronger
state-validity objective or correction mechanism before it can
replace guarded memory safety.

The state-validity runs add training losses for predicted position,
velocity, and cup-floor bounds. In the current evidence they also
do not solve sparse raw-delta instability: both validity and
strong-validity sparse H=25 integrity remain at 0.084722.
Local-dense validity also falls below the raw local-dense score.
Validity losses therefore need rollback/correction and uncertainty
escalation rather than acting as a standalone fix.

This makes state integrity a first-class WPU metric:

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

Future WPU rollout claims should report this score or its components next
to accuracy and latency.
