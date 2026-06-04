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
| Top-tier ML systems 또는 robotics paper | 아직 부족함 | matched baseline, long-horizon evaluation, simulator/real-world task, 더 큰 seed/model sweep이 필요하다. |
| Hardware/chiplet/accelerator claim | 아직 부족함 | sparse kernel, memory traffic, power, matched-accuracy hardware evidence가 없다. |
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
| candidate-oracle gap이 남아 있음 | v2는 유용한 control surface를 보여주지만 deployed selector는 oracle 성능을 충분히 사용하지 못한다. | risk-adjusted descriptor selection은 `N=2048` held-out loss를 개선하지만 candidate oracle은 여전히 훨씬 좋다. | joint retriever-propagator training, calibrated regret target, transfer-stable candidate scoring. |
| cross-seed/cross-task transfer가 불완전함 | synthetic gain은 generator artifact에 overfit될 수 있다. | 여러 cross-seed reranker/gate가 실패하거나 부분 개선에 그쳤다. | 더 큰 seed sweep, 새 synthetic generator, leave-generator-family-out validation. |
| long-horizon state integrity가 증명되지 않음 | persistent state는 delta overlay가 장기적으로 망가지지 않을 때만 장점이다. | 현재는 rollout normalization과 짧은 synthetic prediction 중심이다. | multi-step branch rollout benchmark, rollback/correction/calibration/state-consistency metric. |
| real-world 또는 simulator-backed grounding이 없음 | world processing claim은 toy object physics 밖의 증거가 필요하다. | 현재 evidence는 synthetic robot-cup 및 CWS data다. | MuJoCo/Isaac/robotics/game-server/digital-twin benchmark와 explicit state extraction. |
| perception-to-state가 해결되지 않음 | WPU는 explicit state가 있다고 가정한다. 외부 사용자는 pixels가 어떻게 object/relation이 되는지 물을 것이다. | 문서에서는 perception adapter를 future work로 제한하고 있다. | supervised segmentation, slot discovery, simulator-provided object label 기반 object-state adapter baseline. |
| hardware claim이 뒷받침되지 않음 | Processing unit 주장은 PyTorch 모델만으로 부족하고 systems evidence가 필요하다. | 현재 코드는 reference implementation이다. | sparse frontier kernel profiling, memory-traffic accounting, branch-overlay memory measurement, matched-accuracy speedup. |
| calibration/uncertainty가 얕음 | branch probability는 distribution shift에서 calibration이 맞아야 의미가 있다. | v1/v2는 branch accuracy와 일부 calibration 논의가 있으나 완전한 uncertainty benchmark는 아니다. | ECE/Brier/NLL multi-step rollout, branch collapse test, uncertainty-gated recompute experiment. |
| 객체화 품질 benchmark가 없음 | WPU 성능은 올바른 identity, relation, delta construction에 의존한다. | `evaluate_objectification`은 object contract를 측정하지만, 현재 실험은 여전히 synthetic object state가 주어지는 조건이 많다. | missed object, identity swap, relation error, downstream propagation loss를 측정하는 object construction benchmark. |
| relation repair가 false hypothesis를 추가할 수 있음 | Repair는 누락된 local connectivity를 복구할 수 있지만, spurious edge는 `K`를 키우고 sparse precision을 낮출 수 있다. | relation-repair probe는 ungated repair가 frontier recall은 복구하지만 near distractor에서 precision `0.078994`, dense distractor에서 `0.013244`로 떨어짐을 보인다. Type-gated 및 learned-scorer repair는 in-distribution에서 precision `1.000000`을 회복한다. Learned scorer는 role/affordance state가 보존되면 aliased type name을 넘어 transfer하고 toy downstream branch accuracy를 `0.343750`에서 `0.671875`로 올리지만, type과 role 정보가 모두 제거되면 실패한다. Ungated dense-distractor repair는 frontier recall을 복구해도 downstream loss를 악화시킨다. | simulator relation 대비 repair precision/recall, repair 전후 downstream loss, cross-generator 및 hidden-mechanism shift에서 harmful repaired edge를 reject하는 learned gate. |
| unknown-theory discovery는 장기 연구 프로그램일 뿐임 | 아직 모르는 규칙성을 드러내는 learned relation은 알려진 relation 사용보다 훨씬 강한 주장이다. | Synthetic evidence는 이제 relation transfer, local-law transfer, OOD stress, revision을 포함한다. Revision probe는 gain-shift MSE를 `0.115978`에서 `0.000342`로, power-shift MSE를 `0.054596`에서 `0.008887`로 낮춘다. Oracle relation revision은 `0.000232`에 도달한다. | simulator-backed held-out rule 또는 hidden-mechanism benchmark에서 learned object relation이 prediction을 개선하고 반증 가능한 새 구조를 제시하는 증거. |

## 즉시 개선 우선순위

1. token processing으로 돌아가지 않고 candidate-oracle gap을 줄인다.
2. state-integrity metric을 포함한 long-horizon CWS rollout 평가를 추가한다.
3. explicit object state를 사용할 수 있는 simulator-backed benchmark를 추가한다.
4. cross-seed 평가를 cross-generator-family 평가로 확장한다.
5. calibrated branch/uncertainty metric을 주요 결과로 보고한다.
6. sparse frontier와 branch-overlay memory cost를 dense tensor compute와 분리해 profile한다.
7. `ObjectificationReport`를 실험 log에 포함하고, identity stability, relation precision/recall, delta locality, objectification mistake로 인한 downstream error benchmark를 추가한다.
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
