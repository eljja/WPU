# N=512 Mechanism-Branch Step/Sample Stress Audit

This audit stress-tests the positive mechanism-branch screen. The previous 5-seed short-budget result showed `wpu-cws-indexed-mechanism-branch` slightly ahead of graph-transformer while using zero dense compute. This follow-up asks whether that result survives larger training/evaluation budgets and fairer capacity checks.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_h64_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_baselines_h64_trainpool40_steps16_samples40_3seed.csv`

## Implementation Correction

The first stress run exposed an experiment-control issue. In `scripts/pybullet_shift_generalization.py`, `--samples` controls evaluation sample count, while the training pool was implicitly derived from `steps * batch_size`. The script now adds `--train-samples-per-mechanism`, making training-pool size explicit.

## Results

| condition | WPU macro accuracy | best baseline macro accuracy | WPU dense compute | WPU win/tie/loss | mean margin |
|---|---:|---:|---:|---:|---:|
| short, 3 seeds, h32 | 0.626191 | 0.540476 | 0.000000 | 6/0/1 | +0.066667 |
| short, 5 seeds, h32 | 0.568571 | 0.548571 | 0.000000 | 4/0/3 | +0.020000 |
| steps16/eval40, h32 | 0.538095 | 0.552381 | 0.000000 | 2/1/4 | -0.038095 |
| trainpool40/steps16/eval40, h32 | 0.534524 | 0.598810 | 0.000000 | 2/0/5 | -0.064286 |
| trainpool40/steps16/eval40, h64 | 0.603571 | 0.622619 | 0.000000 | 0/3/4 | -0.053571 |

The h64 WPU improves substantially over h32 WPU under the stress protocol, so model capacity matters. However, h64 baselines also improve, and serialized-token reaches the strongest macro accuracy. The fair h64 stress comparison is therefore negative for WPU accuracy, although WPU keeps dense compute at zero.

## Interpretation

Mechanism-conditioned branch transition remains a useful architectural direction, but the accuracy advantage is not yet robust to larger training/evaluation budgets. The positive short-budget result should be treated as a sparse-efficiency screen, not as a stable accuracy superiority claim.

The important scientific update is that the bottleneck has moved: WPU can express mechanism-conditioned branch updates, but its current branch transition head does not scale as well as dense/token baselines when more samples and capacity are available. The next improvement should target transition-head expressivity and optimization, not simply add more data.

## Next Step

The next WPU v2 direction should be a stronger sparse transition model:

- branch-specific transition experts rather than one additive branch correction head;
- relation-type-conditioned local messages for edge/force/catch composition;
- regularized branch-prior controls so improvement is not only label-prior fitting;
- larger-N sweeps after the transition head is strengthened, because current stress evidence does not yet support a broad accuracy claim.
