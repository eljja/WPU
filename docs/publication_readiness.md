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
| Top-tier ML systems or robotics paper | Not ready | Needs stronger matched baselines, trained long-horizon dynamics, broader simulator or real-world tasks, and larger seed/model sweeps. |
| Hardware/chiplet/accelerator claim | Not ready | Systems proxies and one matched-or-better large-N speedup point are now present, but sparse-kernel, power, allocator-level memory, and Pareto-frontier evidence are still missing. |
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
| Candidate-oracle gap remains open | WPU v2 exposes a useful control surface, but deployed selectors still leave substantial oracle performance unused. | Risk-adjusted mechanism routing previously closed at most `0.244220` of oracle gain. Margin-only sample-level gates failed with best closure `0.082804`. Direct candidate-regret deployment reaches `0.329950` in the test sweep and `0.328025` under train-selected deployment, but remains below the `0.5` target and harmful accepts stay near the safety limit. The safety-frontier audit shows why this is not a threshold-only problem: at harmful limit `0.25`, best closure is about `0.327-0.330`; at harmful limit `0.10`, direct closure drops to `0.081898` and perturbed closure to `0.154320`. Harmful-accept/ranking penalty training lowers train-selected harmful accept to `0.088889` but collapses closure to `0.081253`. Feature perturbation improves safe test-sweep closure to `0.329756`, but train-selected deployment falls to `0.312586`. | Joint retriever-propagator training, stronger calibrated candidate-regret targets, selector uncertainty, harmful-accept penalties, no-harm rejection losses, and transfer-stable candidate scoring. |
| Cross-seed and cross-task transfer is incomplete | Synthetic gains can be seed-specific if selection policies overfit generation artifacts. | Several cross-seed rerankers and gates fail or only partially improve. The 7-seed PyBullet benchmark reduces seed fragility and WPU sparse slightly leads branch accuracy at `N=133` (`0.547619` vs serialized-token `0.539683`), but serialized-token is still faster. The 3-seed leave-family-out probe gives WPU win-rate `0.750000`, with failure on the held-out `catch_heavy` branch-prior shift. A 3-seed composition-shift stress probe is accuracy-positive for WPU (`1.000000` win-rate, mean delta `0.123457`) but still exposes calibration weakness. | Larger seed sweeps, new synthetic generators, harder simulator mechanisms, leave-generator-family-out validation, and mechanism-aware branch priors/calibrators. |
| Long-horizon state integrity is not proven | Persistent state is only valuable if delta overlays do not accumulate unrecoverable corruption. | A PyBullet state-integrity audit now tracks constraint validity, bounded delta drift, branch stability, unsafe-delta rejection, correction, and rollback. Raw WPU sparse falls to integrity `0.084722` at horizon 25. Guarded state-store projection raises applied-state integrity to `0.958508` for sparse WPU and `0.964322` for local-dense WPU, but raw delta instability remains. A target-relative delta-norm regularized raw rollout only improves sparse H=25 integrity to `0.087153`; naive rollout-consistency gives `0.084549`; state-validity and strong state-validity regularization both remain at `0.084722`; unsafe-delta rejection improves sparse integrity to `0.530270` only by rejecting `0.640000` of updates. Rollback raises sparse applied-state integrity to `0.988647`, but with rollback rate `0.812500`. Corrected rollback lowers rollback rate to `0.564167`, but integrity drops to `0.900288`, exposing a correction-vs-rollback tradeoff. | Rollout training with learned correction, calibration, uncertainty escalation, lower-rollback policies, unsafe-delta rejection, and state-consistency losses. |
| Real-world or simulator-backed grounding is still narrow | A world-processing claim needs evidence beyond toy object physics. | Current evidence includes synthetic robot-cup/CWS data and PyBullet cup rollouts with explicit simulator state. The simulator task is useful but still narrow, mostly single-object manipulation with synthetic labels. | MuJoCo/Isaac/robotics/game-server/digital-twin benchmarks with explicit state extraction and perception-to-state adapters. |
| Perception-to-state is not solved | WPU assumes explicit state exists; external users will ask how pixels become objects and relations. | Documents correctly frame perception adapters as future work. | Object-state adapter baseline using supervised segmentation, slot discovery, or simulator-provided object labels. |
| Hardware claims are unsupported | WPU as a processing unit requires systems evidence, not only PyTorch models. | A PyBullet systems profile now separates full-state tensorization, indexed WPU tensorization, sparse work proxy, branch-overlay memory proxy, CPU tensorization latency, random CPU forward proxy, and random CUDA forward/peak-memory proxy. At `N≈2052.6`, indexed tensor bytes drop by `0.997454` while `K≈4.6`, CPU tensorization latency reduction reaches `0.996035`, CPU sparse-forward latency reduction reaches `0.996975`, and CUDA sparse-forward latency reduction reaches `0.996216`. CUDA peak-memory reduction is only `0.304080`. The corrected matched-or-better audit shows WPU is slower at `N=5`, but at `N=133` WPU is more accurate than the best-accuracy non-WPU baseline and `19.184067x` faster. This is positive large-N evidence, not Pareto dominance over every baseline; serialized-token remains faster at lower accuracy. A screening-only energy proxy reaches large reductions, but it is not a power measurement. This is still not power, sparse-kernel, or hardware evidence. | Sparse frontier kernel profiling, real memory-traffic accounting, allocator-level branch-overlay measurements, power/energy telemetry, Pareto-frontier analysis, and trained matched-or-better speedups. |
| Calibration and uncertainty are shallow | Branch probabilities matter only if calibrated under distribution shift. | PyBullet shift generalization now reports ECE, Brier, and NLL under held-out mechanism families. The 7-seed aggregate ECE ratio is `0.963449` for WPU over baselines, but this is still single-step and accuracy remains mixed. A 3-seed calibrated mixture probe worsens WPU-vs-baseline ECE ratio to `1.133834`. A 3-seed composition-shift stress probe is accuracy-positive for WPU but has mean ECE ratio `1.327702`, with `no_catch` reaching `2.362081`. Temperature+bias calibration improves `no_catch` ECE to `0.960054` and reduces mean ECE ratio by `0.217855`, but improves only 1/3 composition mechanisms. Post-hoc calibration is useful diagnostics, not a solved uncertainty model. | Multi-step ECE/Brier/NLL, branch collapse tests, learned calibration heads, mechanism-aware uncertainty, and uncertainty-gated recompute experiments. |
| Objectification quality is not fully solved | WPU performance depends on correct identity, relation, and delta construction. | `ObjectificationReport` now includes frontier completeness and semantic identity consistency, and the PyBullet quality benchmark logs identity recall, relation precision/recall, frontier recall, selected `K`, and component scores. | Connect objectification components to downstream loss and evaluate real perception/state adapters. |
| Relation repair can add false hypotheses | Repair can recover missing local connectivity, but spurious edges can expand `K` and hurt sparse precision. | The relation-repair probe shows ungated repair recovers frontier recall but drops precision to `0.078994` with near distractors and `0.013244` with dense distractors. Type-gated and learned-scorer repair restore precision to `1.000000` in-distribution. The learned scorer transfers across aliased type names when role/affordance state is preserved and improves toy downstream branch accuracy from `0.343750` to `0.671875`, but fails when both type and role information are removed. Ungated dense-distractor repair restores frontier recall but worsens downstream loss. | Repair precision/recall against simulator relations, downstream loss with and without repair, and learned gates that reject harmful repaired edges under cross-generator and hidden-mechanism shift. |
| Unknown-theory discovery is only a long-term program | Learning relations that expose unknown regularities is a stronger claim than using known object relations. | Synthetic evidence now covers relation transfer, local-law transfer, OOD stress, and revision. The revision probe reduces gain-shift MSE from `0.115978` to `0.000342` and power-shift MSE from `0.054596` to `0.008887`, while oracle relation revision reaches `0.000232`. | Simulator-backed held-out-rule or hidden-mechanism benchmarks where learned object relations improve prediction and produce falsifiable new structure. |

## V2 Priority Dashboard

The current machine-generated dashboard is
`docs/experiments/wpu_v2_priority_dashboard.md`. It should be regenerated after
material v2 experiments:

```bash
python scripts/audit_v2_priority_dashboard.py
```

Current status is conservative: priority 1 candidate-oracle gap still fails the
dashboard threshold. Priority 2 long-horizon state integrity has moved from fail
to partial because guarded state-store projection protects applied state, but it
does not solve raw delta instability. Priorities 3 to 7 are partial rather than
solved. This is the correct publication posture:
the repository supports a falsifiable WPU regime hypothesis, not a completed
claim of broad superiority.

## Immediate Improvement Priorities

1. Close the candidate-oracle gap beyond the current conservative gap-closure
   fraction of `0.328025` without returning to token processing.
   Candidate-regret targets now improve over aggregate policy selection, but
   harmful accepts remain too frequent and perturbation does not improve
   train-selected deployment. The next step is joint retriever-propagator
   training and calibrated accept/reject losses, not another margin-only gate.
2. Improve long-horizon state integrity, not only report it. Simple delta-norm,
   rollout-consistency, state-validity, and rejection-only losses are
   insufficient. Rollback protects applied state but fires too often, so the next
   step is a learned correction layer with uncertainty escalation and lower
   rollback frequency.
3. Broaden the simulator-backed benchmark beyond the current PyBullet cup task:
   more mechanisms, longer rollouts, and explicit object state from at least one
   additional simulator or digital-twin environment.
4. Expand leave-family-out evaluation to more seeds, harder mechanisms, and
   branch-prior shifts such as `catch_heavy`, where WPU currently fails.
5. Improve calibration, not only report it: add temperature/calibration heads,
   uncertainty-gated fallback, and multi-step ECE/Brier/NLL.
6. Extend the PyBullet systems profile beyond random CPU/CUDA forward proxies
   to trained matched-or-better Pareto frontiers, allocator-level memory,
   sparse-kernel behavior, energy, and speedups.
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
