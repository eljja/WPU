# WPU Version 3 Plan

WPU v3는 sparse object-physics prototype에서 world-copy processor로 넘어가는 단계다.
이 문서는 추가 구현 전에 v3의 정의, benchmark plan, success criteria, roadmap을 고정한다.

## V3 Thesis

WPU v3는 다음 주장을 검증해야 한다.

> World-processing system은 world를 token stream으로 반복 직렬화하는 대신 persistent
> object-state copy로 유지해야 한다. Event가 발생하면 tensorization 전에 event-local
> causal working set을 검색하고, learned relation-conditioned mechanism으로 state를
> 전파하며, 시간에 따라 copy를 correction해야 한다. Large-`N`, local-event regime에서는
> 이 방식이 full token serialization 또는 dense graph recompute보다 더 좋은
> accuracy/latency/memory/consistency trade-off를 만들 수 있다.

이는 여전히 조건부 주장이다. v3도 WPU가 LPU, GPU, NPU, TPU, graph transformer, token
model을 항상 이긴다고 주장하면 안 된다.

## Alignment Audit

고정된 v3 계획은 매 작업 cycle이 다음 제약을 지킬 때만 WPU의 궁극 목표와 정렬된다.

- Primary representation은 persistent objectified state로 유지된다.
- Token baseline은 비교 대상이지 WPU의 숨은 구현 경로가 아니다.
- 모든 large-`N` 개선은 selected `K`가 bounded인지, sublinear인지, `N`과 함께 커지는지
  보고한다.
- 모든 accuracy claim은 latency, memory traffic, dense fallback, correction cost 중
  하나 이상의 증거와 함께 제시된다.
- 모든 world-copy claim은 one-step prediction뿐 아니라 horizon 또는 streaming evidence를
  포함한다.
- 모든 failure는 objectification, retrieval, propagation, uncertainty, correction,
  systems overhead 중 어디서 발생했는지 분류된다.

제안된 개선이 이 제약을 위반하면 WPU v3 progress가 아니라 baseline 또는 ablation으로
취급해야 한다.

## V2에서 달라지는 점

| 영역 | v2 | v3 목표 |
|---|---|---|
| State | Synthetic/PyBullet task용 object graph | Region, recency, stale state, streaming delta를 가진 hierarchical world copy |
| Retrieval | Event/frontier 및 CWS probe | Noisy, missing, moving state에서 recall/precision을 측정하는 causal index |
| Propagation | Relation-conditioned sparse message | Validity, uncertainty, correction feedback을 가진 learned local mechanism |
| Horizon | 대부분 one-step과 일부 rollout diagnostic | H=25/H=100 world-copy stability와 correction cost |
| Baseline | Token/graph accuracy 및 latency screen | state-update/sec, memory traffic, identity continuity, rollout stability까지 포함한 matched 비교 |
| Claim | small identifiable `K` regime에서 WPU가 유리 | executable world copy를 위한 객체지향적 state-processing runtime |

## 고정 정의

World copy의 canonical definition은 `docs/world_copy_model.ko.md`에 둔다.

v3 핵심 용어는 다음이다.

- `N`: 유지되는 전체 world state 크기.
- `K`: 선택된 event-local causal working set.
- `K_ref`: simulator 또는 oracle label이 있을 때의 reference causal working set.
- `causal slice`: tensor projection 전에 선택되는 object/relation subset.
- `object-oriented processing`: persistent identity, mutable state, typed relation,
  local delta, branch overlay에 대한 직접 처리.
- `natural propagation`: relation channel 위의 learned local-causality update.
- `correction cost`: copy를 valid하게 유지하기 위해 필요한 external observation, dense
  recompute, state rewrite 비용.

## Architecture Target

```text
Observation / simulator / state source
        |
        v
Objectification adapter
        |
        v
Hierarchical WorldState
        |
        v
WorldCausalIndex(event) -> causal slice K
        |
        v
Relation-conditioned propagation core
        |
        v
DeltaState + Branch overlays
        |
        v
Integrity / uncertainty / correction loop
        |
        v
Updated WorldCopy(t+1)
```

구현은 한 가지 규칙을 지켜야 한다. Causal slice가 충분히 신뢰 가능할 때만 sparse
execution을 허용한다. 그렇지 않으면 hybrid 또는 dense recompute로 escalate하고 그 이유를
기록해야 한다.

## Benchmark Program

### P1. Causal Index Stress

목표: `N`이 커져도 v3가 full-state recompute 없이 `K`를 검색할 수 있음을 검증한다.

필수 sweep:

- `N`: 128, 256, 512, 1024, 2048, 4096, 8192, 16384.
- `K_ref`: 4, 8, 16, 32.
- relation missing rate: 0, 0.1, 0.25, 0.5.
- relation false-positive rate: 0, 0.1, 0.25.
- region movement: none, local, cross-region.
- object churn: create/delete 0%, 1%, 5%.

Metrics:

- causal slice recall/precision;
- selected `K`;
- affected fraction;
- retrieval latency;
- touched bytes;
- false non-causal selection rate.

Success:

- moderate noise에서도 recall이 충분히 높다.
- selected `K`가 `N`에 대해 sublinear로 유지된다.
- retrieval cost가 serialized full-state scan보다 낮다.

### P2. Learned Mechanism Propagation

목표: hand-shaped sparse update를 learned relation-conditioned local mechanism으로 대체한다.

필수 model:

- relation-conditioned propagation을 가진 sparse WPU;
- mechanism-specific local head를 가진 sparse WPU;
- uncertainty-triggered local dense recompute를 가진 hybrid WPU;
- serialized-token baseline;
- graph-transformer baseline;
- dense graph baseline.

Metrics:

- next-state MSE/classification accuracy;
- branch accuracy/NLL/ECE;
- propagation no-harm rate;
- accuracy per byte touched;
- accuracy per millisecond.

Success:

- 최소 하나의 large-`N` local-event regime에서 WPU가 best token/graph baseline과 같거나
  더 좋다.
- WPU가 event latency 또는 memory traffic에서 의미 있게 낮다.
- 실패가 aggregate collapse로 숨지 않고 retrieval 또는 propagation 문제로 귀속된다.

### P3. Streaming World Store

목표: 단일 static graph가 아니라 stream 전체에서 world copy를 유지한다.

필수 event:

- object attribute update;
- object creation/deletion;
- identity merge/split;
- region migration;
- relation creation/deletion;
- confidence decay;
- stale-state eviction.

Metrics:

- identity continuity;
- stale-object rate;
- correction frequency;
- delta-log growth;
- branch-overlay memory;
- state-integrity score.

Success:

- full rewrite 없이 긴 stream에서 state가 valid하게 유지된다.
- correction cost가 full-state recompute cost보다 낮다.
- identity와 relation consistency가 측정 가능하고 auditable하다.

### P4. Long-Horizon World-Copy Rollout

목표: WPU state가 one-step뿐 아니라 장기적으로도 유용한지 검증한다.

필수 horizon:

- H=10, H=25, H=50, H=100.

Modes:

- no correction;
- uncertainty-triggered correction;
- scheduled observation correction;
- hybrid/dense escalation.

Metrics:

- trajectory MSE;
- target-object MSE;
- horizon branch accuracy;
- state-integrity score;
- correction cost;
- rollback/escalation rate.

Success:

- WPU가 현재 bounded-delta rollout보다 개선된다.
- constant global recompute 없이 long-horizon integrity가 안정적이다.
- aggregate background error만이 아니라 target-object error가 줄어든다.

### P5. Objectification Adapter Baseline

목표: perfect objectified state 가정을 줄인다.

초기 허용 source:

- simulator-provided object label;
- supervised segmentation/tracking;
- slot/object discovery는 더 어려운 optional path.

Metrics:

- identity recall;
- relation precision/recall;
- causal frontier recall;
- objectification score;
- adapter error 아래 downstream WPU loss.

Success:

- WPU가 objectification failure와 propagation failure를 분리한다.
- state repair와 uncertainty escalation이 missing causal object를 숨기지 않으면서
  downstream behavior를 개선한다.

### P6. Token/LPU-Oriented Comparison

목표: WPU를 token processing과 올바른 축에서 비교한다.

필수 metrics:

- event latency;
- state updates/sec;
- memory bytes/update;
- matched parameter budget accuracy;
- matched latency budget accuracy;
- long-horizon consistency;
- identity persistence;
- branch rollout/sec.

Success:

- WPU가 lower-work일 뿐 아니라 matched budget에서 task quality도 같거나 더 좋은 regime을
  보인다.
- token baseline이 다른 regime에서 이기는 것은 허용한다.

## V3 완료 조건

다음 조건을 모두 만족하기 전에는 다음 milestone을 "WPU v3 complete"라고 부르면 안 된다.

- World-copy formalism이 public API로 구현되어 있다.
- Causal index benchmark가 최소 `N=8192`까지 recall/precision과 latency를 보고한다.
- 최소 하나의 learned propagation benchmark가 baseline-complete다.
- 최소 하나의 streaming benchmark가 object churn과 region migration을 포함한다.
- 최소 하나의 H=25 이상 rollout이 현재 v2 stability baseline을 개선한다.
- 문서는 win과 함께 negative result와 failure mode를 보고한다.

## 즉시 구현 순서

1. Streaming world-copy event용 benchmark scaffolding 추가.
2. `WorldCausalIndex`가 candidate relation path와 retrieval cost를 보고하도록 확장.
3. Noisy-relation causal-index stress script 추가.
4. Causal slice를 입력으로 받는 relation-conditioned mechanism module 추가.
5. Correction accounting을 포함한 long-horizon world-copy rollout runner 추가.
6. 같은 world-copy stream에 대한 token/graph matched baseline 추가.
7. Benchmark evidence가 생긴 뒤에만 paper/claim docs 업데이트.

## 현재 V3 Evidence Boundary

현재 v3 substrate는 causal indexing, noisy-index stress, escalation-region correction
candidate, 그리고 첫 learned local-delta correction probe에 대한 controlled evidence를
포함한다. 최신 learned-correction probe는 P2 substrate positive다. Low-confidence relation
escalation 이후 bounded local region candidate는 max selected `K=16`에서 synthetic delta
MSE를 `0.275312`에서 `0.006365`로 낮춘다.

업데이트된 same-task baseline-comparison screen은 조건부 positive다. `wpu-region-guard`는
max selected `K=16`, mean work proxy `9.333333`, mean bytes proxy `336.000000`을
유지하면서 raw delta MSE `0.002646`을 달성한다. 이는 dense graph `0.003810`,
serialized token `0.003223`보다 낮다. Negative ablation도 중요하다.
`wpu-hybrid-context`는 MSE `0.020904`로 여전히 약하므로, 유효한 mechanism은 generic
context concatenation이 아니라 bounded local-region guard다.

첫 streaming region-guard screen은 이 결과를 object churn과 region migration이 포함된
controlled H=25 world-copy setting으로 확장한다. `wpu-region-guard`는 max selected `K=8`,
trajectory MSE `0.000000`, integrity `1.000000`, correction cost `0.000000`을 유지한다.
Dense state copy도 integrity는 같지만 `N`과 함께 증가하는 full-state work/bytes를
사용한다. 이는 아직 oracle-law evidence이며, 실제 learned dynamics와 잘못 objectified된
region 문제는 미해결이다.

하지만 이것은 P2 완료가 아니다. 다음에 필요한 것은 같은 world-copy stream에서 WPU와
token/graph/dense baseline을 state accuracy, latency, memory traffic, long-horizon
integrity 기준으로 비교하는 baseline-complete learned propagation benchmark다.

## Iteration Protocol

반복 개선 cycle은 항상 같은 loop를 따라야 한다.

1. 이 문서에서 가장 중요한 failing criterion을 선택한다.
2. 수정 전에 현재 code, test, report, claim docs를 확인한다.
3. 그 실패를 공격하는 가장 작은 WPU-native change를 구현한다.
4. 해당 change를 반증할 수 있는 benchmark 또는 test를 추가/수정한다.
5. Benchmark를 실행하고 positive/negative result를 모두 기록한다.
6. Superiority claim이 포함되면 token/graph baseline과 비교한다.
7. Public claim을 강화하기 전에 claim boundary를 먼저 업데이트한다.
8. 관련 test를 실행하고 관련 파일만 commit/push한다.

Cycle마다 산출물은 다음을 포함해야 한다.

- code 또는 benchmark change;
- 재현 가능한 command;
- CSV 또는 markdown report;
- claim이 바뀌면 영어/한글 문서 업데이트;
- 무엇이 개선됐고, 무엇이 실패했고, 다음 남은 문제가 무엇인지 명시.

다음 cycle은 개선하기 쉬운 figure가 아니라 world-copy 목표에 가장 큰 영향을 주는 남은
failure에서 시작해야 한다.

## Reusable Continuation Prompt

새 session에서 WPU v3 작업을 계속할 때는 다음 prompt를 사용한다.

```text
Continue WPU v3 toward the ultimate goal: an object-oriented world-state
processing unit that maintains an executable world copy through persistent
objectified state, causal indexing, learned relation-conditioned propagation,
delta/branch overlays, uncertainty, and correction.

Do not turn WPU back into a token processor. Token/graph/LPU-style models are
baselines only. WPU progress must preserve object identity, relation traversal,
event-local causal working sets, state patching, and long-horizon world-copy
integrity as native operations.

First inspect docs/world_copy_model.md and docs/versions/wpu_v3_plan.md, then
inspect the current code, tests, and latest experiment reports. Pick the
highest-impact unresolved v3 criterion, implement the smallest WPU-native
improvement, add a falsifiable benchmark or test, run it, record positive and
negative results, update English/Korean docs if claims change, run the relevant
tests, and commit/push the work.

Always report:
- what WPU failure was targeted;
- what changed in code or benchmark;
- whether selected K remains bounded or sublinear as N grows;
- whether accuracy, latency, memory traffic, correction cost, or state integrity
  improved;
- whether token/graph baselines still win in any regime;
- what remains the next highest-priority failure.
```

## Claim Boundary

공표 가능한 v3 주장은 다음이어야 한다.

> WPU v3는 executable world copy를 위한 객체지향적 state-processing runtime을 구현한다.
> Objectification과 causal retrieval이 신뢰 가능한 경우, 큰 persistent world를
> event-local causal slice로 갱신할 수 있음을 보이고, 이 방식이 token 또는 dense graph
> processing보다 latency, memory traffic, state consistency에서 유리한 regime을 측정한다.

공표하면 안 되는 주장은 다음이다.

> WPU가 실제 물리 세계 이해를 해결했거나 token/LPU-style processing을 보편적으로
> 대체한다.

selective-region stress 결과는 구체적인 경계를 추가한다. 오염된 region 후보를
순위화하고 제한하면 K 증가를 막지만, raw accuracy에서는 dense state copy가
여전히 이기며 membership과 relation evidence의 동시 누락은 해결되지 않았다.
