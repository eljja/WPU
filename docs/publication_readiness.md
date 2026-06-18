# WPU Publication Readiness and Gap Register

This register states what is ready for external publication, what is not ready,
and what evidence would change that status. It is intentionally conservative:
the goal is to make WPU scientifically useful by making the remaining gaps
explicit and falsifiable.

For the process-unit-specific release audit, see
`docs/process_unit_release_audit.md`.

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
| Candidate-oracle gap remains open | WPU v2 exposes a useful control surface, but deployed selectors still leave substantial oracle performance unused. | Risk-adjusted mechanism routing previously closed at most `0.244220` of oracle gain. Margin-only sample-level gates failed with best closure `0.082804`. Direct candidate-regret deployment reaches `0.329950` in the test sweep and `0.328025` under train-selected deployment, but remains below the `0.5` target and harmful accepts stay near the safety limit. The safety-frontier audit shows why this is not a threshold-only problem: at harmful limit `0.25`, best direct closure is about `0.327-0.330`; at harmful limit `0.10`, direct closure drops to `0.081898` and perturbed closure to `0.154320`. Cross-fit ensemble candidate-regret gating is also negative: best closure `0.287268`, safe best `0.279738`, and cross-fit selected closure `0.270989`. Harmful-accept/ranking penalty training lowers train-selected harmful accept to `0.088889` but collapses closure to `0.081253`. Feature perturbation improves safe test-sweep closure to `0.329756`, but train-selected deployment falls to `0.312586`. A separate safety/utility-head gate is negative: best closure `0.147450`, safe best `0.090719`, train-selected closure `0.144863`. Descriptor standardization plus group-DRO no-harm training is also negative as a standalone fix: best closure and safe best are `0.110889`, and train-selected closure is `0.093863`. A joint object-set candidate gate is also negative: best and safe closure are `0.101454`, train-selected closure is `0.072167`, and a regression-heavy K=16 ablation reaches only `0.034751`. Fixed-candidate/fixed-propagator downstream-loss selector training is also negative: best closure is `0.106927`, no deployment satisfies harmful accept <= `0.25`, and train-selected closure is `0.096833`. Learned candidate generation creates oracle headroom (`0.361251` at `K=16`) but deployed evaluator closure is only `0.042951`. Label-free sparse/local-dense verification signatures are also weaker than direct regret gating: best closure is `0.024989`, safe best is `0.023029`, and train-selected closure is `0.024989`. A shallow candidate-aware branch-logit propagation adapter is also weaker than direct regret gating: best/safe closure is `0.092185`, and train-selected closure is `0.069911`. | Joint retriever-propagator training, stronger calibrated candidate-regret targets, selector uncertainty, harmful-accept penalties, no-harm rejection losses, and transfer-stable candidate scoring learned with candidate generation, verification, and propagation dynamics rather than post-hoc object-set, generator-only, verification-feature-only, or shallow output-adapter gates. |
| Route-state contract was too compressed | Route decisions cannot become mechanism-aware if the route head receives only compressed geometry while objectified action/physical state exists elsewhere. | `docs/experiments/wpu_v2_route_physics_contract_smoke_results.md` verifies that physics/state regret heads now receive pair geometry, target physical scalars, selected-set physical scalars, `force`, and `catch_action`. The smoke run is not performance evidence, but it removes a structural mismatch before further P1/P4 experiments. | Re-run full staged-regret and mechanism-shift sweeps with the expanded route-state context, then compare route-regret correlation, dense-compute rate, harmful accepts, and shifted accuracy/calibration. |
| Cross-seed and cross-task transfer is incomplete | Synthetic gains can be seed-specific if selection policies overfit generation artifacts. | Several cross-seed rerankers and gates fail or only partially improve. The 7-seed PyBullet benchmark reduces seed fragility and WPU sparse slightly leads branch accuracy at `N=133` (`0.547619` vs serialized-token `0.539683`), but serialized-token is still faster. The 3-seed leave-family-out probe gives WPU win-rate `0.750000`, with failure on the held-out `catch_heavy` branch-prior shift. The 7-seed branch-prior audit confirms the mechanism: `catch_heavy` majority prior accuracy is `0.753968`, while best WPU is `0.408730`. A 7-seed mechanism-prior adaptation probe raises shifted WPU win-rate from `0.333333` to `0.666667`, but still does not reach full shift generalization. A prior-strength sweep keeps the best shifted win-rate at `0.666667`; its accuracy-best setting is `strength=0.75` with mean WPU accuracy `0.601852`. Calibration-selected prior strength improves mean accuracy/ECE but leaves shifted WPU-vs-baseline win-rate at `0.333333`. Few-shot mechanism adaptation is positive in an adapted protocol: shifted WPU win-rate reaches `1.000000` and mean WPU-baseline margin changes by `0.050264`, but it uses mechanism-specific calibration samples. A mechanism-aware adaptive policy strengthens the adapted regime: shifted WPU win-rate is `1.000000`, mean accuracy change is `0.198412`, mean margin change is `0.058201`, mean ECE change is `-0.099347`, and mean Brier change is `-0.155443`. A calibration-statistic shift detector recovers the same safe policy from base ECE and majority-prior gap with nominal false adaptation `0`, reducing mechanism-name oracle dependence. This is detect-and-adapt evidence, not zero-shot evidence. A 7-seed composition-shift stress probe is accuracy-positive for WPU (`1.000000` win-rate, mean delta `0.071428`) but still exposes calibration weakness. The N_bg=512 mechanism-diversity audit is mixed: original screens were negative (`2/1/4`, margin `-0.047619` for nominal-train; `2/0/5`, margin `-0.095238` for multi-mechanism), revealing that large N and small K do not imply mechanism generalization. After preserving action and physical state scalars during tensorization, the nominal-train screen recovers to `4/0/3` with margin `+0.002976`, but the multi-mechanism screen remains mixed/negative at `2/2/3` with margin `-0.032738`. | Larger seed sweeps, new synthetic generators, harder simulator mechanisms, leave-generator-family-out validation, mechanism-aware propagation, explicit mechanism-shift detectors, and selective adaptation policies. |
| Long-horizon state integrity is partially solved | Persistent state is only valuable if delta overlays do not accumulate unrecoverable corruption. | A PyBullet state-integrity audit now tracks constraint validity, bounded delta drift, branch stability, unsafe-delta rejection, correction, rollback, dense escalation, corrected-object fraction, low-disruption integrity, simulator-resynchronized trajectory MSE, and final branch accuracy. Earlier memory guards showed that projection/rollback can protect applied state but not learned dynamics. The relation-conditioned rollout audit is the first raw-transition positive: bounded delta at `0.05` reaches H=25 integrity `0.870264` without correction, rollback, rejection, or dense fallback. It also improves simulator-resynchronized prediction: trajectory MSE `0.707117` and branch accuracy `0.729167`, versus finite projection trajectory MSE `1.695024` and branch accuracy `0.250000`. The remaining gap is target-object trajectory MSE `361.358309`, so this is a usable sparse rollout baseline, not high-fidelity learned physics. | Adaptive per-feature/per-relation bounds, simulator-resynchronized trajectory training, target-object error reduction, state-validity objectives, calibrated uncertainty escalation, unsafe-delta rejection, and state-consistency losses. |
| Real-world or simulator-backed grounding is still narrow | A world-processing claim needs evidence beyond toy object physics. | Current evidence includes synthetic robot-cup/CWS data and PyBullet state. A simulator coverage audit separates breadth from superiority claims: full-training baseline-complete cup evidence reaches 7 seeds and `N=133`; matched `N=261` evidence now includes both a low-training screen and a medium-training baseline-complete run. In the N=261 medium run, best WPU reaches accuracy `0.466667` versus best baseline `0.450000`, with `60.629526x` lower forward latency than that best-accuracy baseline. Matched `N=517` evidence now includes a low-training micro-screen, a 5-seed medium run, and a higher-budget 5-seed run. In the N=517 higher-budget run, best WPU reaches accuracy `0.433333` versus best baseline `0.425000`, with `57.595711x` lower forward latency than that best-accuracy baseline. This strengthens P3 evidence, but the margin shrinks and it remains a single cup-family one-step benchmark. Large-state mechanism coverage now reaches 7 mechanisms at total `N=517`. It shows a recoverable input-contract gap for nominal-train shift once action/physical state is preserved, but it still exposes a multi-mechanism law-learning gap. Rollout diagnostics reach horizon 25; objectification-quality evidence covers 7 corruption settings; systems profiles reach `N≈2052`. The larger N_bg=512 cup extension reaches total `N=517` only for WPU models because the dense graph baseline did not complete under the attempted protocol, so it remains systems feasibility evidence. The simulator family remains narrow and mostly single-object manipulation. | MuJoCo/Isaac/robotics/game-server/digital-twin benchmarks with explicit state extraction, perception-to-state adapters, mechanism-aware propagation, and baseline-complete large-N comparisons. |
| Perception-to-state is not solved | WPU assumes explicit state exists; external users will ask how pixels become objects and relations. | Documents correctly frame perception adapters as future work. | Object-state adapter baseline using supervised segmentation, slot discovery, or simulator-provided object labels. |
| Hardware claims are unsupported | WPU as a processing unit requires systems evidence, not only PyTorch models. | A PyBullet systems profile now separates full-state tensorization, indexed WPU tensorization, sparse work proxy, branch-overlay memory proxy, CPU tensorization latency, random CPU forward proxy, and random CUDA forward/peak-memory proxy. At `N≈2052.6`, indexed tensor bytes drop by `0.997454` while `K≈4.6`, CPU tensorization latency reduction reaches `0.996035`, CPU sparse-forward latency reduction reaches `0.996975`, and CUDA sparse-forward latency reduction reaches `0.996216`. CUDA peak-memory reduction is only `0.304080`. The corrected matched-or-better audit shows WPU is slower at `N=5`, but at `N=133` WPU is more accurate than the best-accuracy non-WPU baseline and `19.184067x` faster. A Pareto audit places WPU on the accuracy-latency frontier at `N=133`, but WPU is dominated by serialized-token at `N=5`. A screening-only energy proxy reaches large reductions, but it is not a power measurement. The systems claim-boundary audit separates `4` supported proxy axes, `2` partial trained axes, and `1` unmeasured real-power/sparse-kernel axis; it also records branch-overlay memory proxy reduction `0.874128` and weak CUDA peak-memory proxy reduction `0.304080`. This is still not power, sparse-kernel, or hardware evidence. | Sparse frontier kernel profiling, real memory-traffic accounting, allocator-level branch-overlay measurements, power/energy telemetry, broader Pareto-frontier analysis, and trained matched-or-better speedups. |
| Calibration and uncertainty are shallow | Branch probabilities matter only if calibrated under distribution shift. | PyBullet shift generalization now reports ECE, Brier, and NLL under held-out mechanism families. The 7-seed aggregate ECE ratio is `0.963449` for WPU over baselines, but this is still single-step and accuracy remains mixed. A 3-seed calibrated mixture probe worsens WPU-vs-baseline ECE ratio to `1.133834`. The newer 7-seed composition-shift stress probe is accuracy-positive for WPU but has mean ECE ratio `1.014879`, with `no_catch` at `1.166073`; this is much less severe than the earlier 3-seed estimate but still not calibration dominance. Temperature+bias calibration improves `no_catch` ECE to `0.960054` and reduces mean ECE ratio by `0.217855`, but improves only 1/3 composition mechanisms. The branch-prior audit shows why post-hoc confidence scaling is not enough: one shifted mechanism is dominated by a changed label prior rather than propagation accuracy alone. Mechanism-prior adaptation removes that prior-dominated failure but worsens shifted mean WPU ECE by `0.024819`. The prior-strength sweep confirms this is not only a bad default strength: no nonzero strength preserves/improves shifted WPU win-rate relative to `strength=0` without increasing ECE. Calibration-selected prior strength is the first positive follow-up: shifted mean WPU accuracy changes by `0.145503`, ECE by `-0.046204`, and Brier by `-0.105470`, but baseline win-rate remains unchanged. Few-shot mechanism adaptation also improves shifted mean WPU ECE by `-0.055342` and Brier by `-0.103932`. A mechanism-aware adaptive policy and its calibration-statistic detector variant improve shifted WPU accuracy by `0.198412`, margin by `0.058201`, ECE by `-0.099347`, and Brier by `-0.155443`; this is positive for detect-and-adapt calibration, not zero-shot calibration. A WPU-only uncertainty-gated recompute probe improves aggregate accuracy by `0.071428` and ECE by `-0.016396`, but only at dense recompute rate `0.985450`; the low-cost gate uses rate `0.025132`, improves accuracy by only `0.009260`, and worsens ECE by `0.005395`. A learned sparse-output benefit gate improves source low-cost accuracy by `0.052910` at recompute rate `0.205027`, but worsens ECE by `0.010769`; few-shot gating improves accuracy more but is not low-cost/calibration-safe. A calibration-cost frontier audit makes the boundary explicit: global gates still fail, but a mechanism-selective calibration gate yields `1` non-reference calibration-safe policy under `cost_proxy <= 0.25` with accuracy delta `0.029100`, ECE delta `-0.001652`, Brier delta `-0.030758`, and cost `0.247355`. | Multi-step ECE/Brier/NLL, branch collapse tests, calibration-aware mechanism uncertainty, learned calibration heads, branch-prior shift detectors, and selective adaptation policies. |
| Objectification quality is not fully solved | WPU performance depends on correct identity, relation, and delta construction. | `ObjectificationReport` now includes frontier completeness and semantic identity consistency, and the PyBullet quality benchmark logs identity recall, relation precision/recall, frontier recall, selected `K`, and component scores. The objectification-loss coupling audit connects these metrics to downstream stress-test degradation: WPU sparse under heavy relation drop has the largest MSE increase (`0.087356`), selected-K deficit is the strongest MSE predictor (`|r|=0.481851`), and relation-confidence deficit is the strongest accuracy predictor (`|r|=0.352431`). Branch-accuracy movement is still small, so this is partial coupling evidence rather than a solved perception/state interface. | Closed-loop or multi-horizon corruption, stronger branch/rollout loss effects, component-to-loss regression at larger scale, and real perception/state adapters. |
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

The route-state contract follow-up closes one implementation gap but not P1.
After regenerating the 5-seed staged-regret router comparison, expanded
physical/action context is neutral by simple concatenation: `physics_hidden`
routed loss is `0.962987` versus `0.962894` for the internal route head, while
`state_only` remains worse at `0.982804`. The next mechanism-shift rerun must
therefore add explicit route-regret training to the PyBullet path rather than
only inserting a regret-hybrid model.
That infrastructure is now present: the PyBullet route-regret smoke wires
counterfactual sparse/dense route supervision and configurable route thresholds
into `scripts/pybullet_shift_generalization.py`. The smoke is not performance
evidence; it shows that zero-threshold routing can collapse to all-dense and
that validation-selected thresholds are required before a full shift claim. A
selected-threshold smoke now avoids all-dense/all-sparse endpoints, but the
accuracy is unchanged in the tiny run.
The N_bg=512 selected route-regret mechanism screen is mixed/negative:
selected route-regret WPU keeps dense compute low (`0.071429`) but best-WPU
versus best-baseline win/tie/loss is only `2/1/4`, and graph-transformer keeps
higher macro accuracy. This reinforces that P4/P5 needs mechanism-aware
propagation or adaptation, not another threshold-only route selector.
A matched mechanism-prior adaptation screen is also negative for the route-regret
WPU (`0/0/4` versus best baseline over four shifted mechanisms), so the next
step should condition propagation dynamics directly rather than add another
post-hoc prior or route threshold.
The first mechanism-conditioned propagation screen is a positive follow-up but
not yet a solved result. With dense fallback disabled at N_bg=512,
`wpu-cws-indexed-mechanism-conditioned` reaches macro accuracy `0.541667`
versus `0.500000` for the best non-WPU baseline and wins/ties/loses `1/2/1`
over four shifted mechanisms. The remaining `edge_shift` failure means this is
now a priority direction for larger sweeps, not a publication-level superiority
claim by itself.
The larger follow-up narrows that statement. Nominal-only mechanism-conditioned
expansion over 5 seeds and 7 mechanisms is negative, and the object-wise
mechanism adapter is also negative under nominal-only training. The positive
regime appears only when the adapter is trained on primitive mechanisms: at
N_bg=512 it reaches macro accuracy `0.497143` versus `0.472857` for the best
baseline, with dense compute `0.000000` and win/tie/loss `3/1/3`. This is a
conditional accuracy/compute result, not a broad zero-shot generalization
claim.
This has now been corrected further. The multi-mechanism training DataLoader was
not shuffled, making small-step screens mechanism-order sensitive. After adding
seed-fixed training shuffle and testing a factorized sparse mechanism adapter,
the 5-seed N_bg=512 result is negative: macro accuracy `0.497143` versus
graph-transformer `0.548571`, dense compute `0.000000`, and win/tie/loss
`2/1/4`. The next claim should therefore not be "primitive multi-mechanism
training solves composition"; it should be "composition remains the bottleneck,
and WPU needs explicit local-law/composition supervision."
The direct target-local supervision audit narrows this again. Adding target-local
delta MSE fixes the loss-alignment measurement problem but not the branch
composition problem: at weight `1.0`, WPU target-state MSE improves while macro
branch accuracy falls to `0.418571` versus `0.494286` for graph-transformer in
the matched run. Therefore the next publishable improvement must be architectural:
branch-conditioned or mechanism-specific transition dynamics, not another scalar
loss-weight sweep.
That architectural step now has a first positive screen. The
`wpu-cws-indexed-mechanism-branch` model adds a mechanism-conditioned branch
transition head while keeping dense compute at `0.000000`. On the 5-seed N_bg=512
shuffled multi-mechanism screen, it reaches macro accuracy `0.568571` versus
`0.548571` for graph-transformer, with win/tie/loss `4/0/3`. This improves the
publication story, but it remains a screen: three mechanisms are still below the
best dense baseline, and larger step/sample/N sweeps are required before making
a strong claim.
The first larger step/sample stress audit is negative for accuracy. After adding
explicit training-pool control, h32 WPU reaches `0.534524` versus `0.598810` for
the best baseline, and a fair h64 check reaches `0.603571` versus `0.622619` for
serialized-token. The efficiency story remains intact because WPU dense compute
is still `0.000000`, but the accuracy story now depends on improving sparse
transition-head expressivity rather than simply scaling the current head.
The first expressivity attempt is also negative. Branch-specific output experts
reach only `0.505952` macro accuracy under the h32 stress protocol, below the
prior mechanism-branch head and graph-transformer. This suggests the missing
mechanism is not branch-logit capacity alone; relation-conditioned local
propagation must be improved before another broad accuracy claim.
That propagation-level fix now has a strong positive screen. The
`wpu-cws-indexed-mechanism-relation` route reaches 5-seed h32 stress macro
accuracy `0.639286` versus `0.597143` for graph-transformer with dense compute
`0.000000`, and win/tie/loss `5/0/2` against the best baseline. The 3-seed h64
capacity check remains positive at `0.678571` versus `0.622619` for
serialized-token. A larger 5-seed N=1029 distractor screen is also positive:
`0.639286` versus `0.577143` for graph-transformer, dense compute `0.000000`,
and win/tie/loss `6/0/1`. The N=2053 3-seed distractor screen further improves
the scaling evidence: `0.644048` versus `0.516667` for graph-transformer, dense
compute `0.000000`, and win/tie/loss `7/0/0`. This substantially improves publication readiness because it
connects the WPU claim to relation-conditioned state propagation, not merely a
new classifier head. It is not yet final evidence: it is synthetic single-step
PyBullet evidence and needs calibration, rollout, and harder causal large-N tests.
The new relation closed-loop rollout diagnostic prevents overclaiming: raw
relation WPU H=25 integrity is only `0.250368`, trajectory MSE is `6.975125`,
and branch accuracy is `0.208333`. Finite projection raises integrity to
`0.876760` but remains weak as prediction (`1.695024` trajectory MSE,
`0.250000` branch accuracy). Bounded delta parameterization is the first
transition-level positive and turns the model into a usable sparse rollout
baseline, but target-object trajectory error remains high. The next step must
therefore improve adaptive per-feature/per-relation transition bounds, not only
the transition-training target set.

## Immediate Improvement Priorities

1. Close the candidate-oracle gap beyond the current conservative gap-closure
   fraction of `0.328025` without returning to token processing.
   Candidate-regret targets now improve over aggregate policy selection, but
   harmful accepts remain too frequent, perturbation does not improve
   train-selected deployment, cross-fit ensemble gating lowers closure, and
   descriptor-standardized group-DRO gating is weaker than direct regret gating.
   Joint object-set gating and fixed-candidate/fixed-propagator downstream-loss
   selector training are also weaker. A learned candidate generator creates
   oracle headroom (`0.361251` closure at `K=16`) but its deployed evaluator
   recovers only `0.042951`, so the bottleneck is not merely missing
   candidate-state features, selector-loss replacement, candidate generation
   alone, post-hoc sparse/dense verification signatures, or a shallow
   branch-logit output adapter. A joint utility verifier that adds candidate
   object-set tensors, verification signatures, uncertainty, and no-harm safety
   also remains weak (`0.097845` best/safe closure, `0.077781` train-selected
   closure). The next step is deeper joint candidate generation, retrieval,
   propagation verification, propagation dynamics, and calibrated accept/reject
   losses, not another fixed-propagator post-hoc gate.
2. Improve long-horizon state integrity, not only report it. Simple delta-norm,
   rollout-consistency, state-validity, and rejection-only losses are
   insufficient. Selective correction lowers how much state is modified when
   correction fires, but entropy/raw-delta/stride/margin trigger frontiers do
   not preserve integrity at low correction frequency. A learned trigger also
   collapses to high correction frequency on the hard seed split. A
   stable-transition loss sweep is partially positive: `delta_norm_strong`
   raises raw finite-clamped integrity to `0.633398`, raises selective
   low-disruption score to `0.809071`, and lowers correction rate to
   `0.598333`, but still has `0` rows meeting integrity >= `0.8` and
   correction_rate <= `0.25`. Bounded delta parameterization is the first
   raw-model positive: relation WPU H=25 integrity reaches `0.870264` at bound
   `0.05` without correction, rollback, rejection, or dense fallback. The new
   simulator-resynchronized audit also reduces the under-update concern:
   trajectory MSE is `0.707117` and branch accuracy is `0.729167`, versus
   finite projection trajectory MSE `1.695024` and branch accuracy `0.250000`.
   This improves publication readiness, but target-object trajectory MSE remains
   high at `361.358309`. Learned adaptive bounds, split position/velocity
   bounds, and target-object delta loss do not reduce that bottleneck. The next
   step is unrolled branch/trajectory-consistent transition training.
3. Broaden the simulator-backed benchmark beyond the current PyBullet cup task:
   more mechanisms, longer rollouts, explicit object state from at least one
   additional simulator or digital-twin environment, and baseline-complete
   large-N comparisons. WPU-only large-state runs are useful systems
   diagnostics, but they must not be promoted into accuracy-superiority claims.
4. Expand leave-family-out evaluation to more seeds, harder mechanisms, and
   branch-prior shifts such as `catch_heavy`, where WPU currently fails.
5. Improve calibration, not only report it: static and sparse-output learned
   gates improve accuracy but are not calibration-safe low-cost policies. Add
   calibration-aware mechanism uncertainty, temperature/calibration heads, and
   multi-step ECE/Brier/NLL.
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
