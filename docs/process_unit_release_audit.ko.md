# WPU Process-Unit 공개 감사

이 문서는 현재 저장소가 원래 WPU 계획과 얼마나 일치하는지, 어디서 의도와
달라졌는지, 외부 공개 시 어떤 주장을 하면 안 되는지를 점검한다. 목적은 홍보가
아니라 논문/공개용 claim boundary를 고정하는 것이다.

## 공개 분류

| 영역 | 현재 수준 | 공개 문구 |
|---|---|---|
| 핵심 아이디어 | 일관된 연구 가설 | "객체화된 world state를 위한 state-native execution" |
| 소프트웨어 프로토타입 | 구현됨 | "PyTorch reference implementation 및 experimental runtime" |
| 과학적 증거 | regime-limited | "큰 `N` 안에 작고 식별 가능한 causal working set `K`가 있을 때 유망함" |
| 하드웨어/process-unit 증거 | 초기 systems proxy | "process-unit 방향과 workload abstraction이지 완성된 chip/IP 주장이 아님" |
| 현실 자율성 | 아직 미확립 | "object-state adapter, 더 넓은 simulator, real deployment test가 필요함" |

현재 저장소는 제한된 public research preview와 arXiv preprint에는 적합하다.
하지만 WPU가 완성된 processor, 보편 Transformer 대체재, 일반적으로 우월한
accelerator라는 증거는 아니다.

## 원래 계획 대비 일치 여부

| 계획 항목 | 현재 상태 | 감사 판단 |
|---|---|---|
| Python/PyTorch 연구 패키지 | package, scripts, tests, demos, docs가 구현됨. | 일치. |
| 명시적 world-state type | `wpu/core/`에 구현되고 store, batch, rollout, demo에서 사용됨. | 일치. |
| StateGraphBatch와 model API | train/eval/demo 경로에서 사용 가능. | prototype scale에서 일치. |
| sparse, hybrid, dense routing | scheduler, model variants, CWS experiments로 구현 및 테스트됨. | 일치하지만 learned routing은 미완. |
| Base state plus delta branch rollout | software abstraction 수준에서 구현됨. | 일치하지만 long-horizon raw delta stability는 미해결. |
| 첫 도메인 object physics | synthetic 및 PyBullet cup-family benchmark 구현. | 일치하지만 domain breadth는 좁음. |
| 영문/한글 논문 문서 | claim ledger, readiness register, reproducibility guide, arXiv draft가 있음. | 제한 claim 기준으로 일치. |
| token/graph baseline에 대한 보편 우월성 | 방어 가능한 universal claim이 아님. | 반드시 regime-specific으로 유지. |
| hardware WPU evidence | silicon, sparse kernel, power study가 아님. | "process unit"을 hardware로 해석하면 계획과 어긋남. |
| perception-to-object-state adapter | 아직 해결되지 않음. | 현재 evidence scope 밖. |

## 발견한 문제와 수정

| 문제 | 위험 | 적용한 수정 |
|---|---|---|
| "End-to-end downstream-loss selector"가 full joint WPU training처럼 읽힐 수 있음. | 고정된 후보 생성기와 고정된 propagation model까지 함께 최적화한 것으로 오해할 수 있음. | 결과 문서와 dashboard에 fixed-candidate/fixed-propagator selector training이며 full joint retriever-propagator training이 아니라고 명시. |
| "Process unit"이 현재 hardware처럼 읽힐 수 있음. | 지원되지 않는 hardware/power claim으로 과장될 수 있음. | 본 감사와 readiness register에서 현재 증거를 software runtime plus systems proxy로 분류. |
| Large-`N` 문구가 "N이 크면 WPU가 이긴다"로 흐를 수 있음. | 사실이 아님. `K`가 작고, 식별 가능하며, tensorization 전에 검색되어야 함. | claim ledger와 dashboard의 조건부 문구를 유지하고 본 감사에서 release rule로 반복. |
| P1 negative result가 gate 추가로 가려질 수 있음. | 연구가 bottleneck 해결이 아니라 threshold tuning으로 흐를 수 있음. | 다음 행동을 candidate generation, retrieval, propagation, low-harm calibration의 joint objective로 좁힘. |
| P2 높은 integrity가 heavy correction을 가릴 수 있음. | memory-layer correction이 안정적 learned dynamics처럼 보일 수 있음. | raw-delta instability와 높은 correction-trigger frequency를 readiness 문서에 계속 노출. |

## 현재 과학적 주장

현재 가장 강하게 방어 가능한 주장은 다음이다.

```text
WPU는 token-serialized 또는 dense graph processor가 native하게 노출하지 않는
execution regime을 노출한다. 큰 explicit world state 안에 작고 식별 가능한 causal
working set이 있을 때, full tensorization 이전에 object identity, relation, delta,
uncertainty, branch를 기준으로 계산을 scheduling할 수 있다.
```

이것은 representational universality 주장이 아니다. Token model도 원칙적으로 state를
encode할 수 있다. WPU의 주장은 operational claim이다. Objectification 이후에만
object identity, relation traversal, delta patching, branch overlay, causal
working-set selection이 native execution operation이 된다.

## 현재 blocker

1. Candidate selection은 아직 oracle gain의 대부분을 회수하지 못한다. Best deployed
   P1 closure는 목표보다 낮고, fixed-candidate downstream-loss selector도 negative다.
2. Long-horizon raw sparse dynamics는 memory-layer correction, rollback, fallback
   없이는 불안정하다.
3. Calibration-safe low-cost adaptation은 좁은 mechanism-selective regime에서만
   확인됐다.
4. Baseline-complete simulator evidence는 아직 대부분 PyBullet cup family에 묶여 있다.
5. Systems result는 latency/memory proxy이지 real sparse-kernel, allocator,
   wall-power, silicon evidence가 아니다.
6. Objectification quality는 측정하지만 perception-to-object construction은 아직
   해결되지 않았다.

## 외부 공개 규칙

외부 문서에서 말해도 되는 것:

- WPU는 state-native software research prototype이자 process-unit 방향이다.
- WPU는 large-state/small-causal-working-set regime에서 유망하다.
- 현재 WPU evidence는 broad model-quality dominance보다 execution structure에 더 강하다.
- WPU의 장기 목표는 relation/propagation 구조가 알려진 물리 근사뿐 아니라 아직
  명시적으로 모르는 regularity까지 학습하게 만드는 것이다.

외부 문서에서 말하면 안 되는 것:

- WPU는 완성된 hardware accelerator, chiplet, IP block, power-saving processor다.
- WPU는 token, graph, GPU, TPU, NPU, LPU system을 일반적으로 이긴다.
- WPU는 perception-to-state construction, 실제 물리 이해, unknown-law discovery를
  해결했다.
- 큰 `N`만으로 WPU가 우월해진다.

## 다음 completion bar

더 강한 process-unit claim에는 다음이 모두 필요하다.

- Random-forward proxy가 아니라 trained model 기준으로 matched accuracy에서
  token/graph baseline보다 빠른 non-empty regime.
- Candidate generation, retrieval, propagation을 함께 학습하고 held-out seed selection과
  no-harm calibration을 통과하는 P1 결과.
- Correction-heavy projection이 아니라 낮은 correction frequency에서도 유지되는
  long-horizon state integrity.
- 현재 cup family를 넘어선 baseline-complete large-`N` simulator task.
- Sparse kernel, allocator traffic, peak memory, power/energy telemetry에 대한 실제
  systems measurement.
