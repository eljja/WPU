# arXiv Draft Materials

- `state_is_all_you_need_en.tex`: English arXiv-style LaTeX manuscript with five schematic figures embedded as TikZ.
- `state_is_all_you_need_ko.md`: Korean companion manuscript with the same thesis and results.

The English manuscript is the submission-oriented source. The Korean manuscript
is intended for review, discussion, and translation alignment.

## Follow-up Validation Plan

The paper text intentionally avoids a roadmap-style section. Future work and
validation planning are tracked here instead.

- Build balanced synthetic benchmarks with controllable `N`, `Delta N`,
  fanout, propagation depth, and branch count.
- Compare sparse-only, dense-only, Graph Transformer, Set Transformer, and
  serialized-token baselines under matched training and inference budgets.
- Stress-test irrelevant relations, global affected regions, identity swaps,
  distractor objects, and long-horizon branch divergence.
- Replace fixed `rho` routing with learned accuracy-latency-aware routing.
- Add perception-to-state front ends using slot/object discovery or supervised
  segmentation.
- Add checkpoint, rollback, uncertainty gating, and state-integrity mechanisms
  for persistent delta updates.
- Report calibration, long-horizon rollout error, branch collapse, CPU/GPU
  runtime, and branch-overlay memory.
- Validate on real or simulator-backed object dynamics datasets once the small
  synthetic regime is stable.
