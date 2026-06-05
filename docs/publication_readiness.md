# WPU Publication Readiness and Gap Register

This register states what is ready for external publication, what is not ready,
and what evidence would change that status. It is intentionally conservative:
the goal is to make WPU scientifically useful by making the remaining gaps
explicit and falsifiable.

## Current Readiness

| Target | Status | Reason |
|---|---|---|
| GitHub public repository | Ready | Code, tests, AGPL license, README, reproducibility guide, experiment index, claim ledger, wheel build, and CI are in place. |
| arXiv preprint | Ready with constrained claims | The paper can be posted as an early research prototype if it states a regime-specific hypothesis rather than broad superiority. |
| Workshop / position paper | Ready | The repository provides a coherent architecture, negative results, and a falsifiable experimental program. |
| Top-tier ML systems or robotics paper | Not ready | Needs stronger matched baselines, long-horizon evaluation, simulator or real-world tasks, and larger seed/model sweeps. |
| Hardware/chiplet/accelerator claim | Not ready | No sparse-kernel, memory-traffic, power, or matched-accuracy hardware evidence exists yet. |
| Nature/Science-style broad-impact claim | Not ready | The idea may be high-impact, but current evidence is synthetic and regime-limited. |

## Defensible External Claim

The current defensible claim is:

```text
WPU is a state-native execution model for objectified world state. It is useful
in regimes where total state N is large, the causal working set K is small and
identifiable before tensorization, updates are local and relation-mediated, and
branch/uncertainty state is reused across events.
```

This claim is supported by the implementation, v1 regime studies, and v2
working-set-control experiments. It does not imply that WPU is generally more
accurate than token, graph, or latent world models.

The objectification definition is maintained in `docs/objectification.md`.
Objectification is the step that turns raw observations or simulator state into
persistent objects with attributes, relations, uncertainty, deltas, and branch
overlays. Without this step, WPU reduces to ordinary tensor processing over
unstructured inputs. The implementation now exposes this boundary through
`ObjectificationReport` and uses the score as a scheduler safety signal, but
that does not yet validate perception-to-state construction.

## Gap Register

| Gap | Why it matters | Current evidence | Required next evidence |
|---|---|---|---|
| Broad baseline superiority is not shown | Reviewers will reject universal WPU claims without matched token/graph/world-model baselines. | v1 shows WPU loses at large `N`; v2 shows working-set-control gains but not broad dominance. | Parameter-matched, compute-matched token/graph baselines over controlled `N`, `K`, branch count, and horizon. |
| Candidate-oracle gap remains open | WPU v2 exposes a useful control surface, but deployed selectors still leave substantial oracle performance unused. | The latest gap audit shows risk-adjusted mechanism routing closes only `0.195451`, `0.244220`, and `0.042131` of the oracle gain at `K=8,16,32`. | Joint retriever-propagator training, calibrated regret targets, selector uncertainty, and transfer-stable candidate scoring. |
| Cross-seed and cross-task transfer is incomplete | Synthetic gains can be seed-specific if selection policies overfit generation artifacts. | Several cross-seed rerankers and gates fail or only partially improve. A PyBullet mechanism-family shift benchmark now exists and shows mixed results: WPU sparse leads on `edge_shift` but loses badly on `catch_heavy`. | Larger seed sweeps, new synthetic generators, leave-generator-family-out validation, and mechanism-aware branch priors. |
| Long-horizon state integrity is not proven | Persistent state is only valuable if delta overlays do not accumulate unrecoverable corruption. | A PyBullet state-integrity audit now tracks constraint validity, bounded delta drift, and branch stability. Raw WPU sparse falls to integrity `0.084722` at horizon 25; clipping improves it to `0.201757` but does not solve delta instability. | Rollout training with rollback, correction, calibration, uncertainty escalation, and state-consistency losses. |
| Real-world or simulator-backed grounding is absent | A world-processing claim needs evidence beyond toy object physics. | Current evidence is synthetic robot-cup and CWS data. | MuJoCo/Isaac/robotics/game-server/digital-twin benchmarks with explicit state extraction. |
| Perception-to-state is not solved | WPU assumes explicit state exists; external users will ask how pixels become objects and relations. | Documents correctly frame perception adapters as future work. | Object-state adapter baseline using supervised segmentation, slot discovery, or simulator-provided object labels. |
| Hardware claims are unsupported | WPU as a processing unit requires systems evidence, not only PyTorch models. | A PyBullet systems profile now separates full-state tensorization, indexed WPU tensorization, sparse work proxy, and branch-overlay memory proxy; at `N≈2052.6`, indexed tensor bytes drop by `0.997454` while `K≈4.6`, but this is still a Python proxy. | Sparse frontier kernel profiling, real memory-traffic accounting, allocator-level branch-overlay measurements, and matched-accuracy speedups. |
| Calibration and uncertainty are shallow | Branch probabilities matter only if calibrated under distribution shift. | PyBullet shift generalization now reports ECE, Brier, and NLL under held-out mechanism families, but calibration remains mixed and sometimes favors serialized-token. | Multi-step ECE/Brier/NLL, branch collapse tests, temperature/calibration heads, and uncertainty-gated recompute experiments. |
| Objectification quality is not fully solved | WPU performance depends on correct identity, relation, and delta construction. | `ObjectificationReport` now includes frontier completeness and semantic identity consistency, and the PyBullet quality benchmark logs identity recall, relation precision/recall, frontier recall, selected `K`, and component scores. | Connect objectification components to downstream loss and evaluate real perception/state adapters. |
| Relation repair can add false hypotheses | Repair can recover missing local connectivity, but spurious edges can expand `K` and hurt sparse precision. | The relation-repair probe shows ungated repair recovers frontier recall but drops precision to `0.078994` with near distractors and `0.013244` with dense distractors. Type-gated and learned-scorer repair restore precision to `1.000000` in-distribution. The learned scorer transfers across aliased type names when role/affordance state is preserved and improves toy downstream branch accuracy from `0.343750` to `0.671875`, but fails when both type and role information are removed. Ungated dense-distractor repair restores frontier recall but worsens downstream loss. | Repair precision/recall against simulator relations, downstream loss with and without repair, and learned gates that reject harmful repaired edges under cross-generator and hidden-mechanism shift. |
| Unknown-theory discovery is only a long-term program | Learning relations that expose unknown regularities is a stronger claim than using known object relations. | Synthetic evidence now covers relation transfer, local-law transfer, OOD stress, and revision. The revision probe reduces gain-shift MSE from `0.115978` to `0.000342` and power-shift MSE from `0.054596` to `0.008887`, while oracle relation revision reaches `0.000232`. | Simulator-backed held-out-rule or hidden-mechanism benchmarks where learned object relations improve prediction and produce falsifiable new structure. |

## Immediate Improvement Priorities

1. Close the candidate-oracle gap beyond the current best gap-closure fraction
   of `0.244220` without returning to token processing.
2. Improve long-horizon state integrity, not only report it: add rollback,
   correction, uncertainty escalation, and consistency losses.
3. Add a simulator-backed benchmark where explicit object state is available.
4. Expand the first PyBullet mechanism-family shift benchmark to more seeds,
   harder mechanisms, and leave-generator-family-out validation.
5. Improve calibration, not only report it: add temperature/calibration heads,
   uncertainty-gated fallback, and multi-step ECE/Brier/NLL.
6. Extend the PyBullet systems profile from proxy bytes to runtime, CUDA memory,
   allocator-level memory, and matched-accuracy speedups.
7. Connect `ObjectificationReport` component metrics to downstream propagation
   loss and real perception/state adapter quality.
8. Evaluate relation repair downstream impact before treating repaired edges as
   useful; compare ungated, type-gated, role-aware learned scoring, and
   no-repair baselines.
9. Add hidden-rule benchmarks where the model must infer relation families not
   supplied by the generator, then test whether learned relations reduce
   prediction error under held-out mechanisms.

## Public Communication Rules

- Say "state-native execution regime," not "universal Transformer replacement."
- Say "software research prototype," not "completed hardware accelerator."
- Say "local-causal propagation prior," not "real physical understanding."
- Say "candidate-oracle gap remains," not "retrieval is solved."
- Say "large `N` helps only when `K` is small and identifiable," not "WPU wins when `N` is large."
- Say "objectified state is required," not "raw images or tokens are WPU input."
- Say "unknown-theory discovery is a long-term target," not "WPU has discovered new laws."
- Say "objectification quality is measured by a contract report," not "objectification is solved."
- Say "relation repair proposes hypotheses," not "relation repair discovers physics."

## Minimum Bar for the Next Stronger Claim

The next stronger claim should only be made if a new experiment shows all of the
following:

- WPU preserves or improves accuracy at the point where sparse runtime becomes
  favorable.
- The improvement holds across at least five seeds and at least one generator or
  simulator shift.
- The causal working set is selected before tensorization with sublinear or
  indexed access to total state.
- Token and graph baselines are matched for parameter count, training budget,
  and available state information.
- Negative regimes remain reported rather than filtered out.
