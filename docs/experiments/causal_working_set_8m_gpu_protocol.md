# Causal Working Set 8M GPU Protocol

This protocol is the intended next evidence-generating run for the WPU large-`N`
claim. It requires CUDA PyTorch. The current local environment used during
development had CPU-only PyTorch, so the committed CPU reports are pipeline
checks rather than paper evidence.

## Goal

Test whether a parameter-matched Causal Working Set WPU can preserve branch
accuracy while latency and memory grow more slowly with total state size `N`.

The falsifiable hypothesis is:

```text
If K is fixed and identifiable while N grows, WPU-CWS should scale with K more
than N. If it cannot preserve accuracy at matched parameter scale, the large-N
claim is not supported.
```

## Environment

Recommended:

- CUDA-capable GPU with at least 16 GB VRAM.
- Python 3.11.
- CUDA-enabled PyTorch.

Verify:

```powershell
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Primary Run

```powershell
.\scripts\run_cws_8m_gpu.ps1 -Python python
```

The runner executes:

- models: `wpu-cws-frontier`, `wpu-cws-oracle`, `wpu-cws-learned`,
  `serialized-token`, `graph-transformer`
- `N = 64, 128, 256, 512, 1024, 2048, 4096`
- fixed `K = 8`
- hidden dim `512`, 2 layers, 8 heads
- 5 seeds: `11, 13, 17, 19, 23`
- selector loss weight `0.1`

Expected parameter scale:

- WPU-CWS: about 6.85M parameters
- serialized-token: about 6.33M parameters
- graph-transformer: about 7.38M parameters

## Output

- Raw CSV: `artifacts/causal_working_set_8m_gpu/n-sweep.csv`
- Report: `docs/experiments/causal_working_set_8m_gpu_results.md`

## Required Interpretation

The run supports the large-`N` WPU claim only if all are true:

- WPU-CWS accuracy remains above majority baseline as `N` grows.
- `wpu-cws-oracle` remains strong at large `N`.
- `wpu-cws-learned` approaches oracle causal recall.
- At matched or acceptable accuracy, WPU-CWS latency or memory grows slower than
  serialized-token and graph-transformer baselines.

The run weakens the claim if any are true:

- `wpu-cws-oracle` fails at large `N`.
- learned selector causal recall remains low even with selector supervision.
- token/graph baselines preserve accuracy and remain faster.
- WPU is faster only when it is wrong.

## Follow-Up

After the primary N-sweep, run the distractor sweep:

```powershell
python scripts/causal_working_set_experiment.py `
  --mode distractor-sweep `
  --models wpu-cws-frontier wpu-cws-oracle wpu-cws-learned serialized-token graph-transformer `
  --n-values 2048 `
  --fixed-k 8 `
  --distractor-values 0 8 16 32 64 128 `
  --hidden-dim 512 `
  --num-heads 8 `
  --layers 2 `
  --working-set-size 16 `
  --steps 500 `
  --samples 512 `
  --batch-size 8 `
  --runtime-repeats 30 `
  --seeds 11 13 17 19 23 `
  --selector-loss-weight 0.1 `
  --device cuda `
  --out-dir artifacts/causal_working_set_8m_gpu_distractors
```
