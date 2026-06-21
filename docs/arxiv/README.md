# arXiv Draft Materials

- `state_is_all_you_need_en.tex`: English arXiv-style LaTeX manuscript.
- `state_is_all_you_need_en.pdf`: generated English PDF.
- `state_is_all_you_need_ko.md`: Korean companion manuscript with the same
  thesis, narrowed claim boundary, and current experiment interpretation.
- `../research_thesis.md` and `../research_thesis.ko.md`: repository-level
  thesis, novelty statement, evidence posture, and publication boundary.
- `../objectification.md`: formal WPU objectification definition used by the
  manuscript and claim ledger.

The English manuscript is the submission-oriented source. The Korean manuscript
is intended for review, discussion, and translation alignment.

`state_is_all_you_need_en.tex` is the authoritative submission source. The
committed PDF is a convenience artifact for review, and local rebuilds may
change PDF metadata even when the manuscript source is unchanged. Submission
edits should therefore be judged against the LaTeX source, the evidence tables,
and the CI paper build.

## Current Paper Shape

The main paper keeps the argument compact:

- token versus state primitives;
- objectification as the construction of persistent, addressable state objects;
- objectification quality as a measurable pre-propagation contract;
- claimed contributions as a state-native execution hypothesis: objectification
  contract, event frontier, delta overlays, sparse/hybrid/dense regime surface,
  and negative-result discipline;
- relation repair as auditable hypothesis generation for missed local edges;
- relation to object-centric, graph, world-model, and accelerator literature;
- WPU architecture and sparse/hybrid/dense routing;
- propagation as the state analogue of attention;
- small robot-cup validation;
- reviewer-driven regime evidence and robust baseline comparison;
- v2 retrieval evidence: generated candidates, regret-distilled retrieval,
  invariant candidate descriptors, and risk-adjusted mechanism selection;
- current large-state boundary: baseline-complete positive but small-margin
  cup-family evidence through total `N=517`, and WPU-only sparse-feasibility
  evidence at total `N=4101`;
- discussion that explicitly rejects universal superiority claims.

Detailed sweep tables, extra stress figures, and branch-memory schematics are
kept in the supplementary section of the PDF or in `docs/experiments/`.

## Follow-up Validation Plan

The paper text intentionally avoids a roadmap-style section. Future work and
validation planning are tracked here instead.

- Build balanced synthetic benchmarks with controllable `N`, `Delta N`,
  fanout, propagation depth, and branch count.
- Compare sparse-only, dense-only, Graph Transformer, Set Transformer, and
  serialized-token baselines under matched training and inference budgets.
- Stress-test irrelevant relations, global affected regions, identity swaps,
  distractor objects, and long-horizon branch divergence.
- Log `ObjectificationReport` metrics alongside accuracy, latency, work proxy,
  and branch calibration.
- Evaluate relation-repair precision/recall and downstream loss before treating
  repaired edges as useful.
- Replace fixed `rho` routing with learned accuracy-latency-aware routing.
- Improve large-`N` sparse stability with stronger propagation and regional
  dense correction.
- Add hierarchical world-copy indexing with causal-slice recall/precision
  metrics under object creation, deletion, region movement, noisy relations,
  and stale state.
- Add perception-to-state front ends using slot/object discovery or supervised
  segmentation.
- Treat perception as an object-state adapter layer rather than as solved WPU
  core functionality.
- Add checkpoint, rollback, uncertainty gating, and state-integrity mechanisms
  for persistent delta updates.
- Report calibration, long-horizon rollout error, branch collapse, CPU/GPU
  runtime, and branch-overlay memory.
- Validate on real or simulator-backed object dynamics datasets once the small
  synthetic regime is stable.

The decisive v2 target is to move the WPU accuracy crossover beyond the runtime
crossover. In the current dense N sweep, WPU-family accuracy advantage ends
around `N≈120`, while routed runtime advantage begins around `N≈124` versus
serialized-token and around `N≈178` versus dense-graph.

The strongest current v2 cross-seed result is risk-adjusted mechanism selection
over explicit role/geometry/family descriptors. At `N=2048`, it improves
held-out mean loss over static learned selection for `K=8,16,32`. Earlier
regret-distilled retrieval remains important because it wins 14 of 15 seed/K
conditions in same-seed validation-to-test evaluation, but the stricter
cross-seed conclusion is narrower: explicit state exposes object-level
working-set control and mechanism routing before propagation; it does not yet
close the candidate-oracle gap or prove broad WPU superiority.

## Application Boundary

Near-term applications should be framed as software runtime or middleware work:
digital twin state updates, simulation backends, game/server synchronization,
and robotics world-model maintenance. Hardware, chiplet/IP, or edge-processor
claims remain future hypotheses until real sparse-kernel, memory, and
matched-accuracy speedup evidence exists.
