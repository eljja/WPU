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
| Candidate-oracle gap remains open | WPU v2 exposes a useful control surface, but deployed selectors still leave substantial oracle performance unused. | Risk-adjusted descriptor selection improves held-out loss at `N=2048`, but candidate oracle is substantially better. | Joint retriever-propagator training, calibrated regret targets, and transfer-stable candidate scoring. |
| Cross-seed and cross-task transfer is incomplete | Synthetic gains can be seed-specific if selection policies overfit generation artifacts. | Several cross-seed rerankers and gates fail or only partially improve. | Larger seed sweeps, new synthetic generators, and leave-generator-family-out validation. |
| Long-horizon state integrity is not proven | Persistent state is only valuable if delta overlays do not accumulate unrecoverable corruption. | Current tests cover rollout normalization and short synthetic prediction. | Multi-step branch rollout benchmarks with rollback, correction, calibration, and state-consistency metrics. |
| Real-world or simulator-backed grounding is absent | A world-processing claim needs evidence beyond toy object physics. | Current evidence is synthetic robot-cup and CWS data. | MuJoCo/Isaac/robotics/game-server/digital-twin benchmarks with explicit state extraction. |
| Perception-to-state is not solved | WPU assumes explicit state exists; external users will ask how pixels become objects and relations. | Documents correctly frame perception adapters as future work. | Object-state adapter baseline using supervised segmentation, slot discovery, or simulator-provided object labels. |
| Hardware claims are unsupported | WPU as a processing unit requires systems evidence, not only PyTorch models. | Current code is a reference implementation. | Sparse frontier kernel profiling, memory-traffic accounting, branch-overlay memory measurements, and matched-accuracy speedups. |
| Calibration and uncertainty are shallow | Branch probabilities matter only if calibrated under distribution shift. | v1/v2 reports include branch accuracy and some calibration discussion, but not a full uncertainty benchmark. | ECE/Brier/NLL over multi-step rollouts, branch collapse tests, and uncertainty-gated recompute experiments. |
| Objectification quality is not benchmarked | WPU performance depends on correct identity, relation, and delta construction. | `evaluate_objectification` now measures the object contract, but current experiments still use synthetic object state where objectification is mostly given. | Object construction benchmarks measuring missed objects, identity swaps, relation errors, and downstream propagation loss. |
| Relation repair can add false hypotheses | Repair can recover missing local connectivity, but spurious edges can expand `K` and hurt sparse precision. | `repair_objectification_relations` is deterministic and tested only on a minimal frontier-recovery case. | Repair precision/recall against simulator relations, downstream loss with and without repair, and gates that reject harmful repaired edges. |
| Unknown-theory discovery is only a long-term program | Learning relations that expose unknown regularities is a stronger claim than using known object relations. | Current work uses hand-designed synthetic relations and learned selectors. | Held-out-rule or hidden-mechanism benchmarks where learned object relations improve prediction and produce falsifiable new structure. |

## Immediate Improvement Priorities

1. Close the candidate-oracle gap without returning to token processing.
2. Add long-horizon CWS rollout evaluation with state-integrity metrics.
3. Add a simulator-backed benchmark where explicit object state is available.
4. Expand cross-seed evaluation into cross-generator-family evaluation.
5. Add calibrated branch/uncertainty metrics as first-class reported outputs.
6. Profile sparse frontier and branch-overlay memory costs separately from dense tensor compute.
7. Use `ObjectificationReport` in experiment logs and add objectification-quality
   benchmarks: identity stability, relation precision, relation recall, delta
   locality, and downstream error from objectification mistakes.
8. Evaluate relation repair precision/recall and downstream impact before
   treating repaired edges as useful.
9. Add hidden-rule benchmarks where the model must infer relation families not
   supplied by the generator.

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
