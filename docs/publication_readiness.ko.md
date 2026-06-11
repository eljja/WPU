# WPU 외부 공표 준비도와 Gap Register

이 문서는 지금 무엇을 외부에 공표할 수 있고, 무엇은 아직 주장하면 안 되며,
어떤 증거가 추가되어야 상태가 바뀌는지를 정리한다. 의도적으로 보수적으로 쓴다.
WPU를 학문적으로 의미 있게 만들려면 남은 gap을 명시하고 반증 가능하게 만들어야 한다.

## 현재 준비도

| 대상 | 상태 | 이유 |
|---|---|---|
| GitHub public repository | 준비됨 | 코드, 테스트, AGPL 라이선스, README, 재현성 가이드, 실험 index, claim ledger, wheel build, CI가 갖춰져 있다. |
| arXiv preprint | 제한된 주장으로 준비됨 | broad superiority가 아니라 regime-specific hypothesis로 쓰면 early research prototype으로 공개 가능하다. |
| Workshop / position paper | 준비됨 | 구조, negative result, 반증 가능한 실험 프로그램이 있다. |
| Top-tier ML systems 또는 robotics paper | 아직 부족함 | 더 강한 matched baseline, 학습된 long-horizon dynamics, 더 넓은 simulator/real-world task, 더 큰 seed/model sweep이 필요하다. |
| Hardware/chiplet/accelerator claim | 아직 부족함 | systems proxy와 large-N matched-or-better speedup 1개 지점은 생겼지만 sparse kernel, power, allocator-level memory, Pareto-frontier evidence가 아직 부족하다. |
| Nature/Science식 broad-impact claim | 아직 부족함 | 아이디어는 high-impact 가능성이 있지만 현재 증거는 synthetic, regime-limited다. |

## 방어 가능한 외부 주장

현재 방어 가능한 주장은 다음이다.

```text
WPU는 objectified world state를 위한 state-native execution model이다. 유리한
regime은 전체 state N은 크지만 causal working set K가 작고 tensorization 전에
식별 가능하며, update가 local/relation-mediated이고 branch/uncertainty state가
event 사이에서 재사용되는 경우다.
```

이 주장은 구현, v1 regime study, v2 working-set-control 실험으로 지지된다. 하지만
token, graph, latent world model보다 일반적으로 더 정확하다는 뜻은 아니다.

객체화의 정의는 `docs/objectification.ko.md`에서 관리한다. 객체화는 raw observation
또는 simulator state를 attributes, relations, uncertainty, deltas, branch overlays를
가진 persistent object로 바꾸는 단계다. 이 단계가 없으면 WPU는 unstructured input에
대한 일반 tensor processing으로 약해진다. 구현은 이 경계를 `ObjectificationReport`로
노출하고 score를 scheduler safety signal로 사용한다. 하지만 이것이 perception-to-state
construction을 검증했다는 뜻은 아니다.

## Gap Register

| Gap | 왜 중요한가 | 현재 증거 | 필요한 다음 증거 |
|---|---|---|---|
| broad baseline superiority가 없음 | matched token/graph/world-model baseline 없이 보편 우월 주장은 reject될 가능성이 높다. | v1은 large `N`에서 WPU가 지는 결과를 보였고, v2는 working-set-control gain은 보이나 broad dominance는 아니다. | `N`, `K`, branch count, horizon을 통제한 parameter-matched, compute-matched token/graph baseline. |
| candidate-oracle gap이 남아 있음 | v2는 유용한 control surface를 보여주지만 deployed selector는 oracle 성능을 충분히 사용하지 못한다. | Risk-adjusted mechanism routing의 기존 최고 closure는 `0.244220`이었다. Margin-only sample-level gate는 최고 `0.082804`로 실패했다. Direct candidate-regret deployment는 test sweep에서 `0.329950`, train-selected deployment에서 `0.328025`까지 도달했지만 목표 `0.5`에는 못 미치고 harmful accept가 safety limit 근처에 남아 있다. Safety-frontier audit은 이것이 단순 threshold 문제가 아님을 보인다. Harmful limit `0.25`에서는 direct best closure가 약 `0.327-0.330`이지만, harmful limit `0.10`에서는 direct closure `0.081898`, perturbed closure `0.154320`으로 낮아진다. Cross-fit ensemble candidate-regret gate도 negative result다. 최고 closure `0.287268`, safe best `0.279738`, cross-fit selected closure `0.270989`로 direct gate보다 낮다. Harmful-accept/ranking penalty 학습은 train-selected harmful accept를 `0.088889`까지 낮추지만 closure를 `0.081253`으로 떨어뜨린다. Feature perturbation은 safe test-sweep closure를 `0.329756`까지 올리지만 train-selected deployment는 `0.312586`으로 떨어진다. 별도 safety/utility-head gate도 negative result다. Best closure `0.147450`, safe best `0.090719`, train-selected closure `0.144863`에 그친다. Descriptor standardization과 group-DRO no-harm training도 단독 해결책으로는 negative result다. Best closure와 safe best는 모두 `0.110889`, train-selected closure는 `0.093863`이다. | joint retriever-propagator training, 더 강한 calibrated candidate-regret target, selector uncertainty, harmful-accept penalty, no-harm rejection loss, propagation과 함께 학습되는 transfer-stable candidate scoring. |
| cross-seed/cross-task transfer가 불완전함 | synthetic gain은 generator artifact에 overfit될 수 있다. | 여러 cross-seed reranker/gate가 실패하거나 부분 개선에 그쳤다. 7-seed PyBullet benchmark는 seed fragility를 줄였고 `N=133`에서 WPU sparse branch accuracy가 `0.547619`로 serialized-token `0.539683`보다 약간 높지만, serialized-token이 여전히 더 빠르다. 3-seed leave-family-out probe는 WPU win-rate `0.750000`을 보이나 held-out `catch_heavy` branch-prior shift에서는 실패한다. 7-seed branch-prior audit은 이 원인을 확인한다. `catch_heavy` majority prior accuracy는 `0.753968`이고 best WPU는 `0.408730`이다. 7-seed mechanism-prior adaptation probe는 shifted WPU win-rate를 `0.333333`에서 `0.666667`로 올리지만, full shift generalization에는 도달하지 못한다. Prior-strength sweep도 best shifted win-rate는 `0.666667`에 머문다. Accuracy-best 설정은 `strength=0.75`, mean WPU accuracy `0.601852`다. Calibration-selected prior strength는 mean accuracy/ECE를 개선하지만 shifted WPU-vs-baseline win-rate는 `0.333333`에 머문다. Few-shot mechanism adaptation은 adapted protocol에서 긍정적이다. Shifted WPU win-rate는 `1.000000`, mean WPU-baseline margin 변화는 `0.050264`지만 mechanism-specific calibration sample을 사용한다. Mechanism-aware adaptive policy는 adapted regime을 더 강화한다. Shifted WPU win-rate는 `1.000000`, mean accuracy 변화는 `0.198412`, margin 변화는 `0.058201`, ECE 변화는 `-0.099347`, Brier 변화는 `-0.155443`이다. 이는 detect-and-adapt evidence이지 zero-shot evidence는 아니다. 7-seed composition-shift stress는 WPU에 accuracy-positive 결과를 보인다(win-rate `1.000000`, mean delta `0.071428`) 하지만 calibration 약점은 남아 있다. | 더 큰 seed sweep, 새 synthetic generator, 더 어려운 simulator mechanism, leave-generator-family-out validation, 명시적 mechanism-shift detector, selective adaptation policy. |
| long-horizon state integrity가 증명되지 않음 | persistent state는 delta overlay가 장기적으로 망가지지 않을 때만 장점이다. | PyBullet state-integrity audit이 constraint validity, bounded delta drift, branch stability, unsafe-delta rejection, correction, rollback, dense escalation, corrected-object fraction, low-disruption integrity를 추적한다. Raw WPU sparse는 horizon 25에서 integrity `0.084722`까지 떨어진다. Guarded state-store projection은 sparse WPU의 applied-state integrity를 `0.958508`, local-dense WPU를 `0.964322`까지 올리지만 raw delta instability는 남아 있다. Target-relative delta-norm regularized raw rollout도 sparse H=25 integrity를 `0.087153`까지만 올린다. Naive rollout-consistency는 `0.084549`, state-validity와 strong state-validity regularization은 모두 `0.084722`에 그친다. Unsafe-delta rejection은 sparse integrity를 `0.530270`까지 올리지만 update의 `0.640000`을 거부한다. Rollback은 sparse applied-state integrity를 `0.988647`까지 올리지만 rollback rate가 `0.812500`이다. Corrected rollback은 rollback rate를 `0.564167`까지 낮추지만 integrity가 `0.900288`로 떨어진다. Escalated corrected rollback은 sparse integrity를 `0.914831`로 올리고 rollback rate를 `0.000000`으로 낮추지만 fallback rate가 `0.805833`이다. Finite-corrected sparse는 rollback과 escalation 없이 integrity `0.958735`를 달성하지만 correction rate가 `0.784166`으로 높다. Selective correction은 같은 integrity `0.958735`를 유지하면서 corrected-object fraction을 `0.027461`, low-disruption integrity를 `0.758574`까지 개선한다. 하지만 stride-2와 margin-1 correction gate는 sparse integrity를 `0.535190`, `0.527391`로 무너뜨린다. 이는 stable raw sparse dynamics가 아니라 memory-layer safety다. | stable transition objective, learned correction trigger, calibration, uncertainty escalation, 낮은 correction frequency, unsafe-delta rejection, state-consistency loss를 포함한 rollout training. |
| real-world 또는 simulator-backed grounding이 아직 좁음 | world processing claim은 toy object physics 밖의 증거가 필요하다. | 현재 evidence는 synthetic robot-cup/CWS data와 PyBullet state를 포함한다. 새 simulator coverage audit는 breadth와 superiority claim을 분리한다. 본훈련 baseline-complete cup evidence는 7 seeds와 `N=133`까지이고, 저훈련 matched screen은 `N=261`까지 도달하지만 feasibility evidence이지 강한 accuracy superiority evidence는 아니다. Shift evidence는 4개 mechanism family, rollout diagnostic은 horizon 25, objectification-quality evidence는 7개 corruption setting, systems profile은 `N≈2052`까지 포함한다. N_bg=512 cup extension은 total `N=517`에서 WPU model만 완료됐고 dense graph baseline이 같은 protocol에서 완료되지 않았으므로 matched-baseline accuracy evidence가 아니라 systems feasibility evidence다. Simulator family는 여전히 좁고 대부분 single-object manipulation에 가깝다. | MuJoCo/Isaac/robotics/game-server/digital-twin benchmark, explicit state extraction, perception-to-state adapter, baseline-complete large-N comparison. |
| perception-to-state가 해결되지 않음 | WPU는 explicit state가 있다고 가정한다. 외부 사용자는 pixels가 어떻게 object/relation이 되는지 물을 것이다. | 문서에서는 perception adapter를 future work로 제한하고 있다. | supervised segmentation, slot discovery, simulator-provided object label 기반 object-state adapter baseline. |
| hardware claim이 뒷받침되지 않음 | Processing unit 주장은 PyTorch 모델만으로 부족하고 systems evidence가 필요하다. | PyBullet systems profile이 full-state tensorization, indexed WPU tensorization, sparse work proxy, branch-overlay memory proxy, CPU tensorization latency, random CPU forward proxy, random CUDA forward/peak-memory proxy를 분리했다. `N≈2052.6`에서 indexed tensor byte는 `0.997454` 줄고 `K≈4.6`을 유지하며, CPU tensorization latency reduction은 `0.996035`, CPU sparse-forward latency reduction은 `0.996975`, CUDA sparse-forward latency reduction은 `0.996216`에 도달한다. CUDA peak-memory reduction은 `0.304080`에 그친다. 수정된 matched-or-better audit에서는 `N=5`에서 WPU가 느리지만, `N=133`에서는 best-accuracy non-WPU baseline보다 더 정확하고 `19.184067x` 빠르다. Pareto audit에서도 WPU는 `N=133` accuracy-latency frontier에 있지만 `N=5`에서는 serialized-token에 지배된다. Screening-only energy proxy는 큰 reduction을 보이지만 실제 전력 측정이 아니다. 아직 power, sparse-kernel, hardware evidence는 아니다. | sparse frontier kernel profiling, 실제 memory-traffic accounting, allocator-level branch-overlay measurement, power/energy telemetry, 더 넓은 Pareto-frontier analysis, trained matched-or-better speedup. |
| calibration/uncertainty가 얕음 | branch probability는 distribution shift에서 calibration이 맞아야 의미가 있다. | PyBullet shift generalization은 held-out mechanism family에서 ECE, Brier, NLL을 보고한다. 7-seed aggregate ECE ratio는 WPU가 baseline 대비 `0.963449`지만, 아직 single-step이고 accuracy가 mixed다. 3-seed calibrated mixture probe에서는 WPU-vs-baseline ECE ratio가 `1.133834`로 악화된다. 새 7-seed composition-shift stress는 WPU에 accuracy-positive지만 평균 ECE ratio가 `1.014879`이고 `no_catch`에서는 `1.166073`까지 악화된다. 이는 이전 3-seed 추정보다 훨씬 완화됐지만 calibration dominance는 아니다. Temperature+bias calibration은 `no_catch` ECE를 `0.960054`까지 낮추고 mean ECE ratio를 `0.217855` 줄였지만 composition mechanism 3개 중 1개만 개선했다. Branch-prior audit은 post-hoc confidence scaling만으로 부족한 이유를 보여준다. 하나의 shifted mechanism은 propagation accuracy만이 아니라 label prior 변화에 지배된다. Mechanism-prior adaptation은 prior-dominated failure를 제거하지만 shifted mean WPU ECE를 `0.024819` 악화시킨다. Prior-strength sweep은 이것이 단순 default strength 문제가 아님을 확인한다. `strength=0` 대비 win-rate를 유지/개선하면서 ECE를 악화시키지 않는 비영점 strength는 없었다. Calibration-selected prior strength는 첫 positive follow-up이다. Shifted mean WPU accuracy는 `0.145503`, ECE는 `-0.046204`, Brier는 `-0.105470` 변하지만, baseline win-rate는 그대로다. Few-shot mechanism adaptation도 shifted mean WPU ECE를 `-0.055342`, Brier를 `-0.103932` 개선한다. Mechanism-aware adaptive policy는 shifted WPU accuracy를 `0.198412`, margin을 `0.058201`, ECE를 `-0.099347`, Brier를 `-0.155443` 개선한다. 이는 detect-and-adapt calibration에는 긍정적이지만 zero-shot calibration은 아니다. WPU-only uncertainty-gated recompute probe는 aggregate accuracy를 `0.071428`, ECE를 `-0.016396` 개선하지만 dense recompute rate가 `0.985450`이다. Low-cost gate는 rate `0.025132`에서 accuracy를 `0.009260`만 올리고 ECE를 `0.005395` 악화시킨다. Learned sparse-output benefit gate는 source low-cost accuracy를 `0.052910` 개선하지만 recompute rate `0.205027`에서 ECE를 `0.010769` 악화시킨다. Few-shot gate는 accuracy를 더 올리지만 low-cost/calibration-safe가 아니다. Calibration-cost frontier audit은 이 경계를 명확히 한다. `cost_proxy <= 0.25`에서 non-reference calibration-safe policy는 `0`개이고, 최저 비용 non-reference calibration-safe policy의 cost proxy는 `0.867725`이다. | multi-step ECE/Brier/NLL, branch collapse test, calibration-aware mechanism uncertainty, learned calibration head, branch-prior shift detector, selective adaptation policy. |
| 객체화 품질이 완전히 해결되지 않음 | WPU 성능은 올바른 identity, relation, delta construction에 의존한다. | `ObjectificationReport`는 이제 frontier completeness와 semantic identity consistency를 포함하고, PyBullet quality benchmark는 identity recall, relation precision/recall, frontier recall, selected `K`, component score를 기록한다. | Objectification component를 downstream loss 및 실제 perception/state adapter quality와 연결한다. |
| relation repair가 false hypothesis를 추가할 수 있음 | Repair는 누락된 local connectivity를 복구할 수 있지만, spurious edge는 `K`를 키우고 sparse precision을 낮출 수 있다. | relation-repair probe는 ungated repair가 frontier recall은 복구하지만 near distractor에서 precision `0.078994`, dense distractor에서 `0.013244`로 떨어짐을 보인다. Type-gated 및 learned-scorer repair는 in-distribution에서 precision `1.000000`을 회복한다. Learned scorer는 role/affordance state가 보존되면 aliased type name을 넘어 transfer하고 toy downstream branch accuracy를 `0.343750`에서 `0.671875`로 올리지만, type과 role 정보가 모두 제거되면 실패한다. Ungated dense-distractor repair는 frontier recall을 복구해도 downstream loss를 악화시킨다. | simulator relation 대비 repair precision/recall, repair 전후 downstream loss, cross-generator 및 hidden-mechanism shift에서 harmful repaired edge를 reject하는 learned gate. |
| unknown-theory discovery는 장기 연구 프로그램일 뿐임 | 아직 모르는 규칙성을 드러내는 learned relation은 알려진 relation 사용보다 훨씬 강한 주장이다. | Synthetic evidence는 이제 relation transfer, local-law transfer, OOD stress, revision을 포함한다. Revision probe는 gain-shift MSE를 `0.115978`에서 `0.000342`로, power-shift MSE를 `0.054596`에서 `0.008887`로 낮춘다. Oracle relation revision은 `0.000232`에 도달한다. | simulator-backed held-out rule 또는 hidden-mechanism benchmark에서 learned object relation이 prediction을 개선하고 반증 가능한 새 구조를 제시하는 증거. |

## V2 Priority Dashboard

현재 machine-generated dashboard는
`docs/experiments/wpu_v2_priority_dashboard.ko.md`다. 중요한 v2 실험을 추가한
뒤에는 다음 명령으로 재생성한다.

```bash
python scripts/audit_v2_priority_dashboard.py
```

현재 상태는 보수적이다. Priority 1 candidate-oracle gap은 dashboard threshold를
통과하지 못한다. Priority 2 long-horizon state integrity는 guarded state-store
projection 덕분에 fail에서 partial로 올라갔지만 raw delta instability를 해결한 것은
아니다. Priority 3~7도 solved가 아니라 partial이다. 따라서 올바른 외부 공표 태도는
보편 우월성 주장이 아니라 반증 가능한 WPU regime hypothesis다.

## 즉시 개선 우선순위

1. token processing으로 돌아가지 않고 현재 conservative gap-closure fraction
   `0.328025`를 넘어 candidate-oracle gap을 줄인다. Candidate-regret target은
   aggregate policy selection보다 개선됐지만 harmful accept가 아직 많고
   perturbation은 train-selected deployment를 개선하지 못했으며 cross-fit ensemble
   gate도 closure를 낮췄고 descriptor-standardized group-DRO gate도 direct regret
   gate보다 약하다. 다음 단계는 margin-only gate가 아니라 joint
   retriever-propagator training과 calibrated accept/reject loss다.
2. long-horizon state integrity를 단순히 보고하는 것을 넘어 개선한다. 단순
   delta-norm, rollout-consistency, state-validity, rejection-only loss는
   부족하다. Selective correction은 correction이 발생했을 때 수정되는 state 범위를
   줄였지만 correction trigger frequency는 줄이지 못했다. 다음 단계는 learned stable
   transition/correction trigger, uncertainty escalation, 낮은 correction frequency다.
3. 현재 PyBullet cup task를 넘어 simulator-backed benchmark를 넓힌다. 더 많은
   mechanism, longer rollout, 최소 1개의 추가 simulator 또는 digital-twin
   environment에서 explicit object state를 평가하고, baseline-complete large-N
   comparison을 추가한다. WPU-only large-state run은 systems diagnostic으로는
   유용하지만 accuracy-superiority claim으로 승격하면 안 된다.
4. Leave-family-out evaluation을 더 많은 seed, 더 어려운 mechanism,
   `catch_heavy` 같은 branch-prior shift로 확장한다.
5. Calibration을 단순 보고가 아니라 개선한다. Static gate와 sparse-output learned
   gate는 accuracy를 개선하지만 calibration-safe 저비용 policy는 아니다.
   Calibration-aware mechanism uncertainty, temperature/calibration head,
   multi-step ECE/Brier/NLL을 추가한다.
6. PyBullet systems profile을 random CPU/CUDA forward proxy를 넘어 trained
   matched-or-better Pareto frontier, allocator-level memory, sparse-kernel
   behavior, energy, speedup으로 확장한다.
7. `ObjectificationReport` component metric을 downstream propagation loss와 실제
   perception/state adapter quality에 연결한다.
8. Repaired edge를 유용하다고 간주하기 전에 downstream impact를 평가한다. No-repair, ungated, type-gated, role-aware learned scoring을 비교한다.
9. generator가 relation family를 직접 주지 않는 hidden-rule benchmark를 추가하고, learned relation이 held-out mechanism에서 prediction error를 줄이는지 검증한다.

## 외부 커뮤니케이션 규칙

- "universal Transformer replacement"가 아니라 "state-native execution regime"이라고 말한다.
- "completed hardware accelerator"가 아니라 "software research prototype"이라고 말한다.
- "real physical understanding"이 아니라 "local-causal propagation prior"라고 말한다.
- "retrieval is solved"가 아니라 "candidate-oracle gap remains"라고 말한다.
- "N이 크면 WPU가 이긴다"가 아니라 "K가 작고 식별 가능할 때 large `N`이 유리하다"라고 말한다.
- "raw image나 token이 WPU input"이 아니라 "objectified state가 필요하다"라고 말한다.
- "WPU가 새 법칙을 발견했다"가 아니라 "unknown-theory discovery는 장기 목표"라고 말한다.
- "objectification이 해결됐다"가 아니라 "objectification quality를 contract report로 측정한다"라고 말한다.
- "relation repair가 physics를 발견했다"가 아니라 "relation repair는 hypothesis를 제안한다"라고 말한다.

## 다음 단계의 더 강한 주장을 위한 최소 기준

다음의 조건을 모두 만족하는 새 실험이 있을 때만 더 강한 주장을 해야 한다.

- sparse runtime이 유리해지는 지점에서 WPU가 accuracy를 유지하거나 개선한다.
- 결과가 최소 5 seeds와 최소 1개의 generator 또는 simulator shift에서 유지된다.
- causal working set이 tensorization 전에 선택되며 total state에 대해 sublinear 또는 indexed access를 사용한다.
- token/graph baseline이 parameter count, training budget, available state information에서 공정하게 matched 되어 있다.
- negative regime을 숨기지 않고 계속 보고한다.
