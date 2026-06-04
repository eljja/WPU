# Objectification in WPU

This document defines objectification for WPU and explains why it is central to
state-native world processing. It also sets the long-term research target:
relations between objectified state entities should first approximate known
local physical theories and, eventually, expose learnable structure for
regularities that are not yet explicitly understood.

## Definition

In WPU, objectification is the conversion of an observed or simulated world into
persistent, addressable state entities:

```text
Objectification(x) =
  identity
  + typed attributes
  + typed relations
  + time/history
  + uncertainty
  + admissible deltas
  + branch-local overlays
```

An objectified entity is not just a detected region, token span, row in a table,
or embedding. It is a state-bearing unit that can be queried, patched,
propagated through relations, branched into futures, and checked for
consistency across time.

## Minimal Object Contract

A WPU object should satisfy the following contract:

- `identity`: the object can be referred to across events and time steps.
- `attributes`: the object carries typed state such as position, velocity,
  role, health, ownership, temperature, or confidence.
- `relations`: the object participates in typed edges such as `on`, `near`,
  `touching`, `supports`, `connected_to`, `causes`, or `depends_on`.
- `uncertainty`: the object and its attributes can be uncertain.
- `delta semantics`: events update the object through explicit patches rather
  than by rewriting the whole world.
- `branch semantics`: alternative futures can overlay different deltas on the
  same base object.

The current code expresses this contract through `WorldObject`, `Relation`,
`Event`, `DeltaState`, `Branch`, and `WorldState`.

The public API also exposes `evaluate_objectification(state, ...)`, which
returns an `ObjectificationReport`. The report measures whether a supplied
state satisfies the operational contract before propagation: identity coverage,
relation endpoint validity, object/relation confidence, delta validity, and
optional delta locality against an expected causal working set.

## What Objectification Is Not

Objectification does not mean that WPU already solves perception. The current
WPU core assumes that object state is supplied by a simulator, database,
supervised state extractor, tracker, or future perception adapter.

Objectification also does not mean that tokens cannot describe objects. Tokens
can encode object information. The WPU claim is operational: explicit objects
make identity, update, relation traversal, delta overlays, and branch-local
state first-class operations.

## Why It Matters for Performance

Objectification makes sparse execution meaningful. If the world contains total
state size `N`, but an event touches a causal working set `K`, then WPU can aim
to execute:

```text
retrieve(K) + propagate(K) + patch(K)
```

instead of repeatedly processing all of `N`. This can improve event latency,
effective state-update throughput, branch-rollout efficiency, and memory
traffic only when `K << N` and `K` is identifiable without scanning all state.

The relevant metrics are therefore not raw token/sec but:

- event/sec;
- useful causal state updates/sec;
- branch rollout/sec;
- state-patch latency;
- bytes moved per causal update;
- accuracy retained at the sparse runtime crossover.

Objectification quality is a prerequisite for these metrics. If identity is
unstable, relation endpoints are invalid, or deltas hit non-causal objects, WPU
can spend less compute while becoming less correct. Therefore performance
reports should include both execution metrics and objectification metrics.

## Relation to Physical Approximation

WPU propagation should be understood as a simplified local-causality prior. It
does not solve physical law. It says that many world changes are mediated by
relations among persistent entities: contact, support, containment, proximity,
connectivity, ownership, dependency, and constraint.

The near-term scientific target is:

```text
object relations + learned propagation
  -> approximate local physical rules
  -> maintain predictive accuracy under sparse state updates
```

Examples include:

- contact and force transfer;
- support and falling;
- containment and spill risk;
- connectivity and flow;
- collision and local constraint violation;
- dependency and cascading failures.

This is analogous to how simple physical models can approximate a limited
regime without being universal physical truth. WPU should make the approximation
explicit, measurable, and falsifiable.

## Long-Term Target: Unknown Regularities

The stronger long-term ambition is not to hand-code every relation. It is to
learn object relation structures that discover useful latent regularities:

```text
observed object histories
  -> candidate relations
  -> propagation rules
  -> falsifiable predictions
  -> revised object/relation theory
```

For known domains, these learned relations may approximate physical theories.
For less understood domains, they may expose stable interaction patterns before
humans can name the underlying theory. This should be treated as a research
program, not as a current result.

## Improvement Path

The concrete path to improve WPU through objectification is:

1. Strengthen object contracts with schema validation, `ObjectificationReport`,
   and state-integrity tests.
2. Add relation families for contact, support, containment, flow, dependency,
   ownership, and constraint.
3. Train propagation with local conservation, consistency, and no-spurious-delta
   losses where domain knowledge is available.
4. Add long-horizon rollout tests that measure whether object identity and
   relation consistency survive repeated deltas.
5. Add simulator-backed benchmarks where ground-truth objects and relations are
   available.
6. Learn candidate relations from object histories and evaluate whether they
   improve prediction under held-out regimes.
7. Couple retriever/projection budgets to objectification quality: low relation
   validity or poor delta locality should trigger wider retrieval, dense
   recompute, or state repair rather than blind sparse propagation. The current
   scheduler implements a first version by escalating low objectification
   scores away from sparse routing.
8. Report failures where objectification is wrong: missed objects, identity
   swaps, relation hallucinations, and global events where `K` is not small.

## Claim Boundary

Current WPU evidence supports objectification as an explicit computational
interface. It does not yet prove:

- end-to-end perception-to-object construction;
- broad physical understanding;
- discovery of unknown physical laws;
- hardware-level energy or speed advantage;
- universal accuracy superiority over token or graph models.
