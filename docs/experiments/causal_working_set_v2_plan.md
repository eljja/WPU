# Causal Working Set WPU v2 Experiment Plan

This plan defines the next comparison suite for testing the large-`N` WPU claim.
It supersedes any informal claim that WPU automatically improves when `N`
increases.

## Hypothesis

Large `N` alone does not favor WPU. WPU should become favorable when total world
state size `N` grows while the event-conditioned causal working set `K` remains
small, identifiable, and sufficient for prediction.

```text
C_t = Select(S_t, e_t)
|C_t| = K, K << N

Delta S_t, p(branch) = WPU_Core(C_t, e_t)
S_{t+1} = S_t + Delta S_t
```

The target result is not "WPU always wins." The target is a non-empty regime
where WPU preserves accuracy while latency and memory scale with `K` more than
with `N`.

## Implemented Components

- `wpu.models.CausalWorkingSetProcessor`
- `wpu.data.WorkingSetPhysicsDataset`
- `scripts/causal_working_set_experiment.py`

The processor supports these selector modes:

- `wpu-cws-target`: target object only.
- `wpu-cws-frontier`: target plus relation frontier.
- `wpu-cws-oracle`: known causal-role objects, used to separate core capacity
  from selector failure.
- `wpu-cws-learned`: learned relevance top-`K`.

## Required Comparisons

Primary models:

- `wpu-cws-frontier`
- `wpu-cws-oracle`
- `wpu-cws-learned`
- `serialized-token`
- `graph-transformer`

Recommended parameter scale:

- `hidden_dim=512`, `layers=2`, `num_heads=8`
- WPU-CWS: about 6.85M parameters
- serialized-token: about 6.33M parameters
- graph-transformer: about 7.38M parameters

This is close enough for a first 8M-class comparison. A larger follow-up can use
`hidden_dim=768`, but runtime and memory cost will be much higher.

## Sweeps

N-sweep with fixed K:

```text
K = 8
N = 64, 128, 256, 512, 1024, 2048, 4096
```

K-sweep with fixed N:

```text
N = 2048
K = 4, 8, 16, 32, 64
```

Adversarial distractor sweep:

```text
N fixed, K fixed
fake cups / fake hands / fake edges / noisy relations increase
```

## Metrics

- branch accuracy
- next-state MSE
- selected `K`
- causal working-set recall
- forward latency
- peak GPU memory
- OOM or timeout boundary
- calibration, once the basic comparison is stable
- long-horizon branch consistency, in a later phase

## Example Smoke Run

```bash
python scripts/causal_working_set_experiment.py \
  --models wpu-cws-frontier serialized-token \
  --n-values 32 64 \
  --fixed-k 8 \
  --hidden-dim 32 \
  --num-heads 4 \
  --layers 1 \
  --working-set-size 8 \
  --steps 2 \
  --samples 8 \
  --batch-size 2 \
  --runtime-repeats 2 \
  --seeds 11 13 \
  --out-dir artifacts/cws_smoke
```

## Example 8M-Class Run

```bash
python scripts/causal_working_set_experiment.py \
  --models wpu-cws-frontier wpu-cws-oracle wpu-cws-learned serialized-token graph-transformer \
  --n-values 64 128 256 512 1024 2048 4096 \
  --fixed-k 8 \
  --hidden-dim 512 \
  --num-heads 8 \
  --layers 2 \
  --working-set-size 16 \
  --steps 500 \
  --samples 512 \
  --batch-size 8 \
  --runtime-repeats 30 \
  --seeds 11 13 17 19 23 \
  --out-dir artifacts/causal_working_set_v2
```

## Example Distractor Sweep

```bash
python scripts/causal_working_set_experiment.py \
  --mode distractor-sweep \
  --models wpu-cws-frontier wpu-cws-oracle wpu-cws-learned serialized-token graph-transformer \
  --n-values 2048 \
  --fixed-k 8 \
  --distractor-values 0 8 16 32 64 128 \
  --hidden-dim 512 \
  --num-heads 8 \
  --layers 2 \
  --working-set-size 16 \
  --steps 500 \
  --samples 512 \
  --batch-size 8 \
  --runtime-repeats 30 \
  --seeds 11 13 17 19 23 \
  --out-dir artifacts/causal_working_set_v2_distractors
```

## Interpretation Rules

If oracle WPU succeeds at large `N` but learned/frontier WPU fails, the WPU core
is plausible and the selector is the bottleneck.

If oracle WPU fails, the current propagation/branch core is insufficient even
with correct causal access.

If WPU preserves accuracy but does not improve latency or memory, the execution
abstraction is not yet efficient enough.

If token/graph baselines preserve accuracy and remain faster at matched
parameters, WPU has no demonstrated advantage in this regime.

If learned/frontier selectors have low causal recall while oracle remains
strong, selector learning is the primary research problem.
